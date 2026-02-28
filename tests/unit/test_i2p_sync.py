"""SuperBrain I2P Sync Test Suite

Tests SAM protocol transport (with framing), mock SAM sockets/bridge,
I2PSyncManager, and full sync over I2P transport.
All tests use mocks — no real I2P router or network access needed.
"""
import sys
import os
import asyncio
import struct
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    generate_node_keypair,
)
from sync.protocol.delta_sync import run_sync, PROTOCOL_VERSION
from sync.queue.sync_queue import SyncQueue
from sync.i2p.transport import I2PTransport, connect_to_destination, _sam_handshake, _sam_create_session, SAMError
from sync.i2p.discovery import I2PPeerInfo, StaticI2PPeers, I2PAddressBook
from sync.i2p.sync_manager import I2PSyncManager
from tests.mock.mock_i2p import (
    MockSAMSocket, create_mock_i2p_pair,
    MockSAMBridge, MockSAMBridgeSocket,
)

passed = failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


def make_chunk(content, node_id="test-node"):
    """Create a signed public chunk."""
    priv, _ = generate_node_keypair()
    chunk = KnowledgeChunk(content=content, origin_node_id=node_id)
    chunk.make_public().sign(priv)
    return chunk


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── 1. MockSAMSocket pair ────────────────────────────────────────

def test_mock_socket_pair():
    print("\n1. MockSAMSocket pair...")

    sock_a, sock_b = create_mock_i2p_pair()

    # A sends, B receives
    sock_a.send(b"hello from A")
    data = sock_b.recv(1024)
    check(data == b"hello from A", f"B received: {data}")

    # B sends, A receives
    sock_b.send(b"hello from B")
    data = sock_a.recv(1024)
    check(data == b"hello from B", f"A received: {data}")

    # Partial reads (stream semantics)
    sock_a.send(b"abcdefghij")
    part1 = sock_b.recv(5)
    part2 = sock_b.recv(5)
    check(part1 == b"abcde", f"Partial 1: {part1}")
    check(part2 == b"fghij", f"Partial 2: {part2}")

    # Close
    sock_a.close()
    sock_b.close()
    check(True, "Close OK")


# ── 2. MockSAMSocket close behavior ─────────────────────────────

def test_mock_socket_close():
    print("\n2. MockSAMSocket close behavior...")

    sock_a, sock_b = create_mock_i2p_pair()

    sock_a.close()

    # send on closed socket raises
    try:
        sock_a.send(b"fail")
        check(False, "Should raise on send after close")
    except ConnectionError:
        check(True, "ConnectionError on send after close")

    # recv on closed socket returns empty
    data = sock_a.recv(1024)
    check(data == b"", "Closed recv returns empty")

    sock_b.close()


# ── 3. MockSAMBridge SAM protocol ───────────────────────────────

def test_sam_bridge_protocol():
    print("\n3. MockSAMBridge SAM protocol...")

    bridge = MockSAMBridge()

    # HELLO
    sock = bridge.create_socket()
    sock.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
    resp = sock.recv(4096).decode('ascii')
    check("RESULT=OK" in resp, f"HELLO OK: {resp.strip()}")
    check("VERSION=3.1" in resp, "Version 3.1")

    # SESSION CREATE
    sock.send(b"SESSION CREATE STYLE=STREAM ID=test-session DESTINATION=TRANSIENT\n")
    resp = sock.recv(4096).decode('ascii')
    check("RESULT=OK" in resp, f"SESSION CREATE OK: {resp.strip()[:60]}...")
    check("DESTINATION=" in resp, "Has DESTINATION")

    # Extract destination
    dest = None
    for part in resp.strip().split():
        if part.startswith("DESTINATION="):
            dest = part[len("DESTINATION="):]
    check(dest is not None and len(dest) > 100, f"Destination length: {len(dest or '')}")

    # STREAM CONNECT (no pending accept — goes to data mode anyway)
    sock2 = bridge.create_socket()
    sock2.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
    sock2.recv(4096)
    sock2.send(b"SESSION CREATE STYLE=STREAM ID=test-client DESTINATION=TRANSIENT\n")
    sock2.recv(4096)
    sock2.send(f"STREAM CONNECT ID=test-client DESTINATION={dest}\n".encode())
    resp = sock2.recv(4096).decode('ascii')
    check("RESULT=OK" in resp, f"STREAM CONNECT OK: {resp.strip()}")

    # NAMING LOOKUP
    bridge.add_name("alice.i2p", "AAAA_fake_destination_BBBB")
    sock3 = bridge.create_socket()
    sock3.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
    sock3.recv(4096)
    sock3.send(b"NAMING LOOKUP NAME=alice.i2p\n")
    resp = sock3.recv(4096).decode('ascii')
    check("RESULT=OK" in resp, f"NAMING found: {resp.strip()}")
    check("AAAA_fake_destination_BBBB" in resp, "Destination value correct")

    # NAMING LOOKUP not found
    sock3.send(b"NAMING LOOKUP NAME=unknown.i2p\n")
    resp = sock3.recv(4096).decode('ascii')
    check("KEY_NOT_FOUND" in resp, f"NAMING not found: {resp.strip()}")

    sock.close()
    sock2.close()
    sock3.close()


# ── 4. MockSAMBridge CONNECT/ACCEPT pairing ─────────────────────

def test_sam_bridge_pairing():
    print("\n4. MockSAMBridge CONNECT/ACCEPT pairing...")

    bridge = MockSAMBridge()

    # Server: create session + STREAM ACCEPT
    server_sock = bridge.create_socket()
    server_sock.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
    server_sock.recv(4096)
    server_sock.send(b"SESSION CREATE STYLE=STREAM ID=server-sess DESTINATION=TRANSIENT\n")
    resp = server_sock.recv(4096).decode('ascii')
    server_dest = None
    for part in resp.strip().split():
        if part.startswith("DESTINATION="):
            server_dest = part[len("DESTINATION="):]

    # Issue ACCEPT on a new socket (reusing session)
    accept_sock = bridge.create_socket()
    accept_sock.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
    accept_sock.recv(4096)
    accept_sock.send(b"STREAM ACCEPT ID=server-sess\n")
    # Response comes when CONNECT matches

    # Client: connect to server
    client_sock = bridge.create_socket()
    client_sock.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
    client_sock.recv(4096)
    client_sock.send(b"SESSION CREATE STYLE=STREAM ID=client-sess DESTINATION=TRANSIENT\n")
    client_sock.recv(4096)
    client_sock.send(f"STREAM CONNECT ID=client-sess DESTINATION={server_dest}\n".encode())

    # Both should get RESULT=OK
    client_resp = client_sock.recv(4096).decode('ascii')
    accept_resp = accept_sock.recv(4096).decode('ascii')

    check("RESULT=OK" in client_resp, f"Client CONNECT OK: {client_resp.strip()}")
    check("RESULT=OK" in accept_resp, f"Server ACCEPT OK: {accept_resp.strip()}")

    # Data mode: client sends, server receives
    check(client_sock._data_mode, "Client in data mode")
    check(accept_sock._data_mode, "Server in data mode")

    client_sock.send(b"hello over I2P!")
    data = accept_sock.recv(4096)
    check(data == b"hello over I2P!", f"Server received: {data}")

    accept_sock.send(b"reply from server")
    data = client_sock.recv(4096)
    check(data == b"reply from server", f"Client received: {data}")

    server_sock.close()
    accept_sock.close()
    client_sock.close()


# ── 5. I2PTransport framing ─────────────────────────────────────

def test_transport_framing():
    print("\n5. I2PTransport framing...")

    async def _run():
        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        payload = b"test I2P message payload"
        await ta.send(payload)

        received = await tb.receive()
        check(received == payload, f"Received matches: {len(received)} bytes")
        check(ta.bytes_sent == len(payload), f"A bytes_sent: {ta.bytes_sent}")
        check(tb.bytes_received == len(payload), f"B bytes_received: {tb.bytes_received}")

        await ta.close()
        await tb.close()

    run(_run())


# ── 6. I2PTransport round-trip ───────────────────────────────────

def test_transport_roundtrip():
    print("\n6. I2PTransport round-trip...")

    async def _run():
        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        await ta.send(b"from A to B")
        data_at_b = await tb.receive()
        check(data_at_b == b"from A to B", "A->B OK")

        await tb.send(b"from B to A")
        data_at_a = await ta.receive()
        check(data_at_a == b"from B to A", "B->A OK")

        await ta.close()
        await tb.close()

    run(_run())


# ── 7. Multiple messages ─────────────────────────────────────────

def test_multiple_messages():
    print("\n7. Multiple messages...")

    async def _run():
        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        messages = [f"message {i}: {'x' * (i * 100)}".encode() for i in range(5)]

        for msg in messages:
            await ta.send(msg)

        for i, expected in enumerate(messages):
            received = await tb.receive()
            check(received == expected, f"Message {i}: {len(received)} bytes")

        await ta.close()
        await tb.close()

    run(_run())


# ── 8. Large message ─────────────────────────────────────────────

def test_large_message():
    print("\n8. Large message (1MB)...")

    async def _run():
        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        payload = os.urandom(1024 * 1024)
        await ta.send(payload)
        received = await tb.receive()

        check(len(received) == len(payload), f"Size: {len(received)}")
        check(received == payload, "Content matches")

        await ta.close()
        await tb.close()

    run(_run())


# ── 9. Transport close behavior ──────────────────────────────────

def test_transport_close():
    print("\n9. Transport close behavior...")

    async def _run():
        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")

        await ta.close()
        check(ta.closed, "Closed flag set")

        # send after close
        try:
            await ta.send(b"should fail")
            check(False, "Should raise on send after close")
        except ConnectionError:
            check(True, "ConnectionError on send after close")

        # receive after close
        try:
            await ta.receive()
            check(False, "Should raise on receive after close")
        except ConnectionError:
            check(True, "ConnectionError on receive after close")

        # close again is safe
        await ta.close()
        check(True, "Double close OK")

        sock_b.close()

    run(_run())


# ── 10. SAM handshake via transport helpers ──────────────────────

def test_sam_handshake():
    print("\n10. SAM handshake via transport helpers...")

    async def _run():
        bridge = MockSAMBridge()
        sock = bridge.create_socket()

        await _sam_handshake(sock)
        check(True, "SAM handshake OK")

        dest = await _sam_create_session(sock, "test-sess-1")
        check(len(dest) > 100, f"Session created, dest length: {len(dest)}")

        sock.close()

    run(_run())


# ── 11. connect_to_destination with mock ─────────────────────────

def test_connect_factory():
    print("\n11. connect_to_destination with mock bridge...")

    async def _run():
        bridge = MockSAMBridge()

        transport = await connect_to_destination(
            destination="AAAA_fake_dest_BBBB",
            _socket_factory=bridge.create_socket,
        )

        check(isinstance(transport, I2PTransport), "Returns I2PTransport")
        check(not transport.closed, "Transport is open")
        check("AAAA_fake_dest" in transport.name, f"Name: {transport.name}")

        await transport.close()

    run(_run())


# ── 12. StaticI2PPeers ───────────────────────────────────────────

def test_static_peers():
    print("\n12. StaticI2PPeers...")

    # (destination,) tuples
    sp = StaticI2PPeers([("dest_AAAA",), ("dest_BBBB",)])
    check(len(sp.peers) == 2, f"2 peers: {len(sp.peers)}")
    check(sp.peers[0].destination == "dest_AAAA", f"Dest: {sp.peers[0].destination}")

    # (destination, name) tuples
    sp2 = StaticI2PPeers([("dest_CCCC", "Alice")])
    check(sp2.peers[0].name == "Alice", f"Name: {sp2.peers[0].name}")

    # start/stop are no-ops
    async def _noop():
        await sp.start()
        await sp.stop()
    run(_noop())
    check(True, "start/stop no-ops")

    # scan returns peers
    async def _scan():
        return await sp.scan()
    result = run(_scan())
    check(len(result) == 2, f"scan returns 2: {len(result)}")


# ── 13. I2PAddressBook ───────────────────────────────────────────

def test_address_book():
    print("\n13. I2PAddressBook...")

    # Create temp address book file
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="i2p_ab_")
    with os.fdopen(fd, 'w') as f:
        f.write("# I2P Address Book\n")
        f.write("alice=AAAA_dest_alice_BBBB\n")
        f.write("bob=CCCC_dest_bob_DDDD\n")
        f.write("  \n")  # blank line
        f.write("# comment\n")
        f.write("EEEE_dest_no_name_FFFF\n")

    found = []
    ab = I2PAddressBook(path, on_peer_found=lambda p: found.append(p))

    async def _run():
        await ab.start()
        peers = ab.peers
        check(len(peers) == 3, f"3 peers: {len(peers)}")

        names = {p.name for p in peers}
        check("alice" in names, "Alice found")
        check("bob" in names, "Bob found")

        dests = {p.destination for p in peers}
        check("AAAA_dest_alice_BBBB" in dests, "Alice destination")
        check("EEEE_dest_no_name_FFFF" in dests, "No-name destination")

        check(len(found) == 3, f"on_peer_found called 3 times: {len(found)}")

        await ab.stop()

    run(_run())
    os.unlink(path)

    # Missing file
    ab2 = I2PAddressBook("/nonexistent/path.txt")
    async def _missing():
        await ab2.start()  # should not raise
    run(_missing())
    check(len(ab2.peers) == 0, "Missing file: 0 peers")


# ── 14. Full sync over I2P transport ─────────────────────────────

def test_full_sync():
    print("\n14. Full sync over I2P transport...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="i2p_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="i2p_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        c1 = KnowledgeChunk(content="I2P sync chunk alpha from node A", origin_node_id="i2p-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="I2P sync chunk beta from node A", origin_node_id="i2p-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        c3 = KnowledgeChunk(content="I2P sync chunk gamma from node B", origin_node_id="i2p-B")
        c3.make_public().sign(priv_b)
        c4 = KnowledgeChunk(content="I2P sync chunk delta from node B", origin_node_id="i2p-B")
        c4.make_public().sign(priv_b)
        queue_b.add_to_queue(c3)
        queue_b.add_to_queue(c4)

        check(queue_a.stats()["total"] == 2, "A starts with 2")
        check(queue_b.stats()["total"] == 2, "B starts with 2")

        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        result_a, result_b = await asyncio.gather(
            run_sync(queue_a, "i2p-A", priv_a, b"", ta),
            run_sync(queue_b, "i2p-B", priv_b, b"", tb),
        )

        check(result_a.success, f"A sync success (errors: {result_a.errors})")
        check(result_b.success, f"B sync success (errors: {result_b.errors})")
        check(result_a.chunks_sent == 2, f"A sent 2: {result_a.chunks_sent}")
        check(result_a.chunks_received == 2, f"A received 2: {result_a.chunks_received}")
        check(result_b.chunks_sent == 2, f"B sent 2: {result_b.chunks_sent}")
        check(result_b.chunks_received == 2, f"B received 2: {result_b.chunks_received}")

        check(queue_a.stats()["total"] == 4, f"A has 4: {queue_a.stats()['total']}")
        check(queue_b.stats()["total"] == 4, f"B has 4: {queue_b.stats()['total']}")

        check(queue_a.get_chunk(c3.content_hash) is not None, "A has B's chunk gamma")
        check(queue_b.get_chunk(c1.content_hash) is not None, "B has A's chunk alpha")

        await ta.close()
        await tb.close()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 15. Sync with empty queues ───────────────────────────────────

def test_sync_empty():
    print("\n15. Sync with empty queues (noop)...")

    async def _run():
        priv_a, _ = generate_node_keypair()
        priv_b, _ = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="i2p_empty_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="i2p_empty_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        result_a, result_b = await asyncio.gather(
            run_sync(queue_a, "i2p-A", priv_a, b"", ta),
            run_sync(queue_b, "i2p-B", priv_b, b"", tb),
        )

        check(result_a.success, "A empty sync OK")
        check(result_a.chunks_sent == 0, f"A sent 0: {result_a.chunks_sent}")
        check(result_a.chunks_received == 0, f"A received 0: {result_a.chunks_received}")

        await ta.close()
        await tb.close()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 16. I2PSyncManager with mock connect ─────────────────────────

def test_i2p_sync_manager():
    print("\n16. I2PSyncManager integration (mock connect)...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="i2p_mgr_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="i2p_mgr_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Add chunks to A
        c1 = KnowledgeChunk(content="I2P manager chunk one", origin_node_id="i2p-mgr-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="I2P manager chunk two", origin_node_id="i2p-mgr-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        async def mock_connect(destination, name=""):
            sock_a_local, sock_b_local = create_mock_i2p_pair("server", "client")

            # Start server-side run_sync as background task
            ta = I2PTransport(sock_a_local, name="server")
            asyncio.create_task(
                run_sync(queue_a, "i2p-mgr-A", priv_a, b"", ta)
            )

            return I2PTransport(sock_b_local, name="client")

        peer_dest = "FAKE_DEST_FOR_TESTING_1234567890"

        mgr_b = I2PSyncManager(
            sync_queue=queue_b,
            node_id="i2p-mgr-B",
            private_key=priv_b,
            public_key=pub_b,
            static_peers=[(peer_dest, "node-A")],
            mock_connect=mock_connect,
            _server_socket_factory=MockSAMBridge().create_socket,
        )
        await mgr_b.start()

        # Trigger sync
        results = await mgr_b.sync_all()

        check(peer_dest in results, "Synced with peer")
        result = results.get(peer_dest)
        check(result is not None, "Result not None")
        if result:
            check(result.success, f"Sync success (errors: {result.errors})")
            check(result.chunks_received == 2, f"Received 2: {result.chunks_received}")

        check(queue_b.stats()["total"] == 2, f"B has 2 chunks: {queue_b.stats()['total']}")
        check(queue_b.get_chunk(c1.content_hash) is not None, "B has chunk 1")

        # Check stats
        stats = mgr_b.stats
        check(stats["node_id"] == "i2p-mgr-B", "Stats node_id")
        check(stats["running"], "Stats running")
        check(peer_dest in stats["sync_records"], "Stats has peer record")
        if peer_dest in stats["sync_records"]:
            rec = stats["sync_records"][peer_dest]
            check(rec["sync_count"] == 1, f"Sync count: {rec['sync_count']}")
            check(rec["total_received"] == 2, f"Total received: {rec['total_received']}")

        await mgr_b.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 17. Rate limiting ────────────────────────────────────────────

def test_rate_limiting():
    print("\n17. Rate limiting...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="i2p_rate_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="i2p_rate_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        c = make_chunk("Rate limit test I2P")
        queue_a.add_to_queue(c)

        async def mock_connect(destination, name=""):
            sa, sb = create_mock_i2p_pair()
            ta = I2PTransport(sa, name="server")
            asyncio.create_task(run_sync(queue_a, "rate-A", priv_a, b"", ta))
            return I2PTransport(sb, name="client")

        peer_dest = "RATE_TEST_DEST"

        mgr = I2PSyncManager(
            sync_queue=queue_b, node_id="rate-B",
            private_key=priv_b, public_key=pub_b,
            static_peers=[(peer_dest,)],
            mock_connect=mock_connect,
            _server_socket_factory=MockSAMBridge().create_socket,
        )
        await mgr.start()

        peer = mgr._discovery.peers[0]
        r1 = await mgr.sync_with_peer(peer)
        check(r1 is not None, "First sync works")

        # Immediate second sync should be rate-limited
        r2 = await mgr.sync_with_peer(peer)
        check(r2 is None, "Second sync rate-limited (None)")

        await mgr.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 18. Bidirectional sync ───────────────────────────────────────

def test_bidirectional_sync():
    print("\n18. Bidirectional sync (both sides have chunks)...")

    async def _run():
        priv_a, _ = generate_node_keypair()
        priv_b, _ = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="i2p_bidi_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="i2p_bidi_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # A has unique chunk
        ca = make_chunk("Only on A side via I2P", "node-A")
        queue_a.add_to_queue(ca)

        # B has unique chunk
        cb = make_chunk("Only on B side via I2P", "node-B")
        queue_b.add_to_queue(cb)

        sock_a, sock_b = create_mock_i2p_pair()
        ta = I2PTransport(sock_a, name="i2p-A")
        tb = I2PTransport(sock_b, name="i2p-B")

        result_a, result_b = await asyncio.gather(
            run_sync(queue_a, "node-A", priv_a, b"", ta),
            run_sync(queue_b, "node-B", priv_b, b"", tb),
        )

        check(result_a.chunks_sent == 1, f"A sent 1: {result_a.chunks_sent}")
        check(result_a.chunks_received == 1, f"A received 1: {result_a.chunks_received}")
        check(result_b.chunks_sent == 1, f"B sent 1: {result_b.chunks_sent}")
        check(result_b.chunks_received == 1, f"B received 1: {result_b.chunks_received}")

        # Both have both chunks
        check(queue_a.stats()["total"] == 2, f"A has 2: {queue_a.stats()['total']}")
        check(queue_b.stats()["total"] == 2, f"B has 2: {queue_b.stats()['total']}")

        await ta.close()
        await tb.close()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 19. SAM bridge data mode through I2PTransport ────────────────

def test_bridge_data_via_transport():
    print("\n19. SAM bridge data mode through I2PTransport...")

    async def _run():
        bridge = MockSAMBridge()

        # Server side: HELLO + SESSION + ACCEPT
        srv = bridge.create_socket()
        srv.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
        srv.recv(4096)
        srv.send(b"SESSION CREATE STYLE=STREAM ID=srv DESTINATION=TRANSIENT\n")
        resp = srv.recv(4096).decode('ascii')
        dest = None
        for part in resp.strip().split():
            if part.startswith("DESTINATION="):
                dest = part[len("DESTINATION="):]

        accept = bridge.create_socket()
        accept.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
        accept.recv(4096)
        accept.send(b"STREAM ACCEPT ID=srv\n")

        # Client side: HELLO + SESSION + CONNECT
        cli = bridge.create_socket()
        cli.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
        cli.recv(4096)
        cli.send(b"SESSION CREATE STYLE=STREAM ID=cli DESTINATION=TRANSIENT\n")
        cli.recv(4096)
        cli.send(f"STREAM CONNECT ID=cli DESTINATION={dest}\n".encode())
        cli.recv(4096)  # RESULT=OK
        accept.recv(4096)  # RESULT=OK

        # Wrap in I2PTransport — framing on top of data-mode stream
        t_client = I2PTransport(cli, name="client")
        t_server = I2PTransport(accept, name="server")

        await t_client.send(b"framed I2P message")
        received = await t_server.receive()
        check(received == b"framed I2P message", f"Server got: {received}")

        await t_server.send(b"framed reply")
        received = await t_client.receive()
        check(received == b"framed reply", f"Client got: {received}")

        await t_client.close()
        await t_server.close()
        srv.close()

    run(_run())


# ── 20. Manager default init ────────────────────────────────────

def test_manager_defaults():
    print("\n20. Manager default init...")

    async def _run():
        _, path = tempfile.mkstemp(suffix=".db", prefix="i2p_def_")
        queue = SyncQueue(db_path=path)

        bridge = MockSAMBridge()
        mgr = I2PSyncManager(
            sync_queue=queue,
            _server_socket_factory=bridge.create_socket,
        )

        check(mgr._node_id is not None, f"Auto node_id: {mgr._node_id[:8]}...")
        check(mgr._private_key is not None, "Auto private_key")
        check(mgr._public_key is not None, "Auto public_key")
        check(len(mgr._discovery.peers) == 0, "No peers by default")

        await mgr.start()
        check(mgr._running, "Running after start")
        check(mgr.local_destination is not None, f"Local dest: {mgr.local_destination[:16]}...")

        stats = mgr.stats
        check(stats["running"], "Stats running")
        check(stats["peers_known"] == 0, "0 peers known")

        await mgr.stop()
        check(not mgr._running, "Stopped")

        queue.close()
        os.unlink(path)

    run(_run())


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain I2P Sync Tests")
    print("=" * 60)

    test_mock_socket_pair()
    test_mock_socket_close()
    test_sam_bridge_protocol()
    test_sam_bridge_pairing()
    test_transport_framing()
    test_transport_roundtrip()
    test_multiple_messages()
    test_large_message()
    test_transport_close()
    test_sam_handshake()
    test_connect_factory()
    test_static_peers()
    test_address_book()
    test_full_sync()
    test_sync_empty()
    test_i2p_sync_manager()
    test_rate_limiting()
    test_bidirectional_sync()
    test_bridge_data_via_transport()
    test_manager_defaults()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit to avoid blocking on lingering executor threads
    # (mock SAM sockets may have threads waiting in recv)
    os._exit(1 if failed else 0)
