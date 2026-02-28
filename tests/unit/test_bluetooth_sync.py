"""SuperBrain Bluetooth Sync Test Suite

Tests RFCOMM transport (with framing), mock Bluetooth sockets,
BluetoothSyncManager, and full sync over Bluetooth transport.
All tests use mock sockets — no real Bluetooth hardware needed.
"""
import sys
import os
import asyncio
import struct
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    generate_node_keypair,
)
from sync.protocol.delta_sync import run_sync, PROTOCOL_VERSION
from sync.queue.sync_queue import SyncQueue
from sync.bluetooth.rfcomm_transport import BluetoothTransport
from sync.bluetooth.discovery import BluetoothPeerInfo, StaticBluetoothPeers
from sync.bluetooth.bluetooth_sync import BluetoothSyncManager
from tests.mock.mock_bluetooth import MockBluetoothSocket, create_mock_bluetooth_pair

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


# ── 1. MockBluetoothSocket pair ───────────────────────────────────

def test_mock_socket_pair():
    print("\n1. MockBluetoothSocket pair...")

    sock_a, sock_b = create_mock_bluetooth_pair()

    # A sends, B receives
    sock_a.send(b"hello from A")
    data = sock_b.recv(1024)
    check(data == b"hello from A", f"B received: {data}")

    # B sends, A receives
    sock_b.send(b"hello from B")
    data = sock_a.recv(1024)
    check(data == b"hello from A"[:0] + b"hello from B", f"A received: {data}")

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


# ── 2. BluetoothTransport framing ────────────────────────────────

def test_transport_framing():
    print("\n2. BluetoothTransport framing...")

    async def _run():
        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")
        tb = BluetoothTransport(sock_b, name="bt-B")

        # A sends a message
        payload = b"test message payload"
        await ta.send(payload)

        # B receives — should get the original payload (framing handled internally)
        received = await tb.receive()
        check(received == payload, f"Received matches: {len(received)} bytes")
        check(ta.bytes_sent == len(payload), f"A bytes_sent: {ta.bytes_sent}")
        check(tb.bytes_received == len(payload), f"B bytes_received: {tb.bytes_received}")

        await ta.close()
        await tb.close()

    run(_run())


# ── 3. BluetoothTransport round-trip ─────────────────────────────

def test_transport_roundtrip():
    print("\n3. BluetoothTransport round-trip...")

    async def _run():
        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")
        tb = BluetoothTransport(sock_b, name="bt-B")

        # Bidirectional: A sends, B receives, B sends, A receives
        await ta.send(b"from A to B")
        data_at_b = await tb.receive()
        check(data_at_b == b"from A to B", "A->B OK")

        await tb.send(b"from B to A")
        data_at_a = await ta.receive()
        check(data_at_a == b"from B to A", "B->A OK")

        await ta.close()
        await tb.close()

    run(_run())


# ── 4. Multiple messages ─────────────────────────────────────────

def test_multiple_messages():
    print("\n4. Multiple messages...")

    async def _run():
        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")
        tb = BluetoothTransport(sock_b, name="bt-B")

        messages = [f"message {i}: {'x' * (i * 100)}".encode() for i in range(5)]

        # Send all from A
        for msg in messages:
            await ta.send(msg)

        # Receive all at B
        for i, expected in enumerate(messages):
            received = await tb.receive()
            check(received == expected, f"Message {i}: {len(received)} bytes")

        await ta.close()
        await tb.close()

    run(_run())


# ── 5. Large message ─────────────────────────────────────────────

def test_large_message():
    print("\n5. Large message (1MB)...")

    async def _run():
        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")
        tb = BluetoothTransport(sock_b, name="bt-B")

        # 1MB payload
        payload = os.urandom(1024 * 1024)
        await ta.send(payload)
        received = await tb.receive()

        check(len(received) == len(payload), f"Size: {len(received)}")
        check(received == payload, "Content matches")

        await ta.close()
        await tb.close()

    run(_run())


# ── 6. Full sync over Bluetooth transport ─────────────────────────

def test_full_sync():
    print("\n6. Full sync over Bluetooth transport...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="bt_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="bt_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Node A has chunks 1, 2
        c1 = KnowledgeChunk(content="BT sync chunk alpha from node A", origin_node_id="bt-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="BT sync chunk beta from node A", origin_node_id="bt-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        # Node B has chunks 3, 4
        c3 = KnowledgeChunk(content="BT sync chunk gamma from node B", origin_node_id="bt-B")
        c3.make_public().sign(priv_b)
        c4 = KnowledgeChunk(content="BT sync chunk delta from node B", origin_node_id="bt-B")
        c4.make_public().sign(priv_b)
        queue_b.add_to_queue(c3)
        queue_b.add_to_queue(c4)

        check(queue_a.stats()["total"] == 2, "A starts with 2")
        check(queue_b.stats()["total"] == 2, "B starts with 2")

        # Create mock Bluetooth socket pair
        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")
        tb = BluetoothTransport(sock_b, name="bt-B")

        # Run sync on both sides concurrently
        result_a, result_b = await asyncio.gather(
            run_sync(queue_a, "bt-A", priv_a, b"", ta),
            run_sync(queue_b, "bt-B", priv_b, b"", tb),
        )

        check(result_a.success, f"A sync success (errors: {result_a.errors})")
        check(result_b.success, f"B sync success (errors: {result_b.errors})")
        check(result_a.chunks_sent == 2, f"A sent 2: {result_a.chunks_sent}")
        check(result_a.chunks_received == 2, f"A received 2: {result_a.chunks_received}")
        check(result_b.chunks_sent == 2, f"B sent 2: {result_b.chunks_sent}")
        check(result_b.chunks_received == 2, f"B received 2: {result_b.chunks_received}")

        check(queue_a.stats()["total"] == 4, f"A has 4: {queue_a.stats()['total']}")
        check(queue_b.stats()["total"] == 4, f"B has 4: {queue_b.stats()['total']}")

        # Verify specific content
        check(queue_a.get_chunk(c3.content_hash) is not None, "A has B's chunk gamma")
        check(queue_b.get_chunk(c1.content_hash) is not None, "B has A's chunk alpha")

        await ta.close()
        await tb.close()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 7. Sync with empty queues ────────────────────────────────────

def test_sync_empty():
    print("\n7. Sync with empty queues (noop)...")

    async def _run():
        priv_a, _ = generate_node_keypair()
        priv_b, _ = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="bt_empty_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="bt_empty_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")
        tb = BluetoothTransport(sock_b, name="bt-B")

        result_a, result_b = await asyncio.gather(
            run_sync(queue_a, "bt-A", priv_a, b"", ta),
            run_sync(queue_b, "bt-B", priv_b, b"", tb),
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


# ── 8. StaticBluetoothPeers ──────────────────────────────────────

def test_static_peers():
    print("\n8. StaticBluetoothPeers...")

    # (address, channel) tuples
    sp = StaticBluetoothPeers([("AA:BB:CC:DD:EE:FF", 1), ("11:22:33:44:55:66", 3)])
    check(len(sp.peers) == 2, f"2 peers: {len(sp.peers)}")
    check(sp.peers[0].address == "AA:BB:CC:DD:EE:FF", f"Addr: {sp.peers[0].address}")
    check(sp.peers[0].channel == 1, f"Channel: {sp.peers[0].channel}")
    check(sp.peers[1].channel == 3, f"Channel: {sp.peers[1].channel}")

    # (address, name, channel) tuples
    sp2 = StaticBluetoothPeers([("AA:BB:CC:DD:EE:FF", "Phone", 2)])
    check(sp2.peers[0].name == "Phone", f"Name: {sp2.peers[0].name}")

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


# ── 9. BluetoothSyncManager with mock ────────────────────────────

def test_bluetooth_sync_manager():
    print("\n9. BluetoothSyncManager integration (mock connect)...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="bt_mgr_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="bt_mgr_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Add chunks to A
        c1 = KnowledgeChunk(content="BT manager chunk one", origin_node_id="bt-mgr-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="BT manager chunk two", origin_node_id="bt-mgr-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        # We need to simulate a connection. The manager calls connect_to_device
        # which creates a real BT socket. Instead, we'll use the mock_connect
        # parameter to inject a mock transport creator.

        # Create a mock "server" that runs run_sync on A's side
        # when B connects. We simulate this by having both sides
        # use mock sockets connected together.

        sock_a = None
        sock_b = None

        async def mock_connect(address, channel, name=""):
            nonlocal sock_a, sock_b
            sock_a_local, sock_b_local = create_mock_bluetooth_pair("server", "client")
            sock_a = sock_a_local  # server side
            sock_b = sock_b_local  # client side

            # Start server-side run_sync as background task
            ta = BluetoothTransport(sock_a, name="server")
            asyncio.create_task(
                run_sync(queue_a, "bt-mgr-A", priv_a, b"", ta)
            )

            # Return client-side transport
            return BluetoothTransport(sock_b, name="client")

        mgr_b = BluetoothSyncManager(
            sync_queue=queue_b,
            node_id="bt-mgr-B",
            private_key=priv_b,
            public_key=pub_b,
            static_peers=[("AA:BB:CC:DD:EE:FF", "bt-mgr-A", 1)],
            mock_connect=mock_connect,
        )
        await mgr_b.start()

        # Trigger sync
        results = await mgr_b.sync_all()

        check("AA:BB:CC:DD:EE:FF" in results, "Synced with peer")
        result = results.get("AA:BB:CC:DD:EE:FF")
        check(result is not None, "Result not None")
        if result:
            check(result.success, f"Sync success (errors: {result.errors})")
            check(result.chunks_received == 2, f"Received 2: {result.chunks_received}")

        check(queue_b.stats()["total"] == 2, f"B has 2 chunks: {queue_b.stats()['total']}")
        check(queue_b.get_chunk(c1.content_hash) is not None, "B has chunk 1")

        # Check stats
        stats = mgr_b.stats
        check(stats["node_id"] == "bt-mgr-B", "Stats node_id")
        check(stats["running"], "Stats running")
        check("AA:BB:CC:DD:EE:FF" in stats["sync_records"], "Stats has peer record")
        if "AA:BB:CC:DD:EE:FF" in stats["sync_records"]:
            rec = stats["sync_records"]["AA:BB:CC:DD:EE:FF"]
            check(rec["sync_count"] == 1, f"Sync count: {rec['sync_count']}")
            check(rec["total_received"] == 2, f"Total received: {rec['total_received']}")

        await mgr_b.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 10. Transport close behavior ─────────────────────────────────

def test_transport_close():
    print("\n10. Transport close behavior...")

    async def _run():
        sock_a, sock_b = create_mock_bluetooth_pair()
        ta = BluetoothTransport(sock_a, name="bt-A")

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

        sock_b.close()

    run(_run())


# ── 11. Rate limiting ────────────────────────────────────────────

def test_rate_limiting():
    print("\n11. Rate limiting...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="bt_rate_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="bt_rate_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        c = make_chunk("Rate limit test")
        queue_a.add_to_queue(c)

        async def mock_connect(address, channel, name=""):
            sa, sb = create_mock_bluetooth_pair()
            ta = BluetoothTransport(sa, name="server")
            asyncio.create_task(run_sync(queue_a, "rate-A", priv_a, b"", ta))
            return BluetoothTransport(sb, name="client")

        mgr = BluetoothSyncManager(
            sync_queue=queue_b, node_id="rate-B",
            private_key=priv_b, public_key=pub_b,
            static_peers=[("FF:EE:DD:CC:BB:AA", 1)],
            mock_connect=mock_connect,
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


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Bluetooth Sync Tests")
    print("=" * 60)

    test_mock_socket_pair()
    test_transport_framing()
    test_transport_roundtrip()
    test_multiple_messages()
    test_large_message()
    test_full_sync()
    test_sync_empty()
    test_static_peers()
    test_bluetooth_sync_manager()
    test_transport_close()
    test_rate_limiting()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
