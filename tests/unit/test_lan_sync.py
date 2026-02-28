"""SuperBrain LAN Sync Test Suite

Tests mDNS discovery (static fallback), WebSocket transport over HTTP,
HTTP sync server, LANSyncManager, and full LAN sync flows.
All tests use localhost (127.0.0.1) with ephemeral ports.
"""
import sys
import os
import asyncio
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    generate_node_keypair,
)
from sync.protocol.delta_sync import run_sync, PROTOCOL_VERSION
from sync.queue.sync_queue import SyncQueue
from sync.lan.discovery import PeerInfo, StaticPeerList
from sync.lan.http_transport import WebSocketTransport, connect_to_peer
from sync.lan.http_server import HttpSyncServer
from sync.lan.lan_sync import LANSyncManager

import aiohttp

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
    """Run an async coroutine."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── 1. StaticPeerList ─────────────────────────────────────────────

def test_static_peer_list():
    print("\n1. StaticPeerList...")

    # (host, port) tuples
    spl = StaticPeerList([("192.168.1.10", 8384), ("192.168.1.20", 8384)])
    check(len(spl.peers) == 2, f"2 peers: {len(spl.peers)}")
    check(spl.peers[0].host == "192.168.1.10", f"Host: {spl.peers[0].host}")
    check(spl.peers[0].port == 8384, f"Port: {spl.peers[0].port}")
    check(spl.peers[0].node_id.startswith("static-"), f"Auto node_id: {spl.peers[0].node_id}")

    # (node_id, host, port) tuples
    spl2 = StaticPeerList([("node-X", "10.0.0.1", 9000)])
    check(spl2.peers[0].node_id == "node-X", "Custom node_id")
    check(spl2.peers[0].host == "10.0.0.1", "Custom host")

    # start/stop are no-ops
    async def _noop():
        await spl.start()
        await spl.stop()
    run(_noop())
    check(True, "start/stop no-ops")


# ── 2. WebSocket transport send/receive ───────────────────────────

def test_websocket_transport():
    print("\n2. WebSocket transport send/receive...")

    async def _run():
        priv_a, _ = generate_node_keypair()
        _, path = tempfile.mkstemp(suffix=".db", prefix="ws_test_")
        queue = SyncQueue(db_path=path)

        # Start server on ephemeral port
        server = HttpSyncServer(
            sync_queue=queue, node_id="server-node",
            private_key=priv_a, host="127.0.0.1", port=0,
        )
        await server.start()
        port = server.port

        # Connect client
        session = aiohttp.ClientSession()
        ws = await session.ws_connect(f"http://127.0.0.1:{port}/sync/ws")
        client_transport = WebSocketTransport(ws, name="test-client")

        # Client sends, server receives (via the sync handler)
        # Instead of testing raw send/receive (which would require
        # a custom handler), we verify the transport wraps correctly.
        check(isinstance(client_transport, WebSocketTransport), "Transport created")
        check(not client_transport.closed, "Not closed initially")
        check(client_transport.bytes_sent == 0, "0 bytes sent initially")

        await client_transport.close()
        check(client_transport.closed, "Closed after close()")

        await session.close()
        await server.stop()
        queue.close()
        os.unlink(path)

    run(_run())


# ── 3. Full sync over WebSocket ───────────────────────────────────

def test_full_sync_over_websocket():
    print("\n3. Full sync over WebSocket (two nodes, different chunks)...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="lan_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="lan_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Node A has chunks 1, 2
        c1 = KnowledgeChunk(content="LAN sync chunk alpha from node A", origin_node_id="node-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="LAN sync chunk beta from node A", origin_node_id="node-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        # Node B has chunks 3, 4
        c3 = KnowledgeChunk(content="LAN sync chunk gamma from node B", origin_node_id="node-B")
        c3.make_public().sign(priv_b)
        c4 = KnowledgeChunk(content="LAN sync chunk delta from node B", origin_node_id="node-B")
        c4.make_public().sign(priv_b)
        queue_b.add_to_queue(c3)
        queue_b.add_to_queue(c4)

        check(queue_a.stats()["total"] == 2, "A starts with 2 chunks")
        check(queue_b.stats()["total"] == 2, "B starts with 2 chunks")

        # Start server for node A
        server_a = HttpSyncServer(
            sync_queue=queue_a, node_id="node-A",
            private_key=priv_a, host="127.0.0.1", port=0,
        )
        await server_a.start()
        port_a = server_a.port

        # Node B connects to A and syncs
        session = aiohttp.ClientSession()
        transport = await connect_to_peer("127.0.0.1", port_a, session=session, name="node-B")

        result_b = await run_sync(
            queue_b, "node-B", priv_b, b"", transport,
        )

        check(result_b.success, f"B sync success (errors: {result_b.errors})")
        check(result_b.chunks_sent == 2, f"B sent 2: {result_b.chunks_sent}")
        check(result_b.chunks_received == 2, f"B received 2: {result_b.chunks_received}")

        # Wait briefly for server-side ingestion to complete
        await asyncio.sleep(0.1)

        check(queue_a.stats()["total"] == 4, f"A has 4 chunks: {queue_a.stats()['total']}")
        check(queue_b.stats()["total"] == 4, f"B has 4 chunks: {queue_b.stats()['total']}")

        # Verify specific content
        check(queue_a.get_chunk(c3.content_hash) is not None, "A has B's chunk gamma")
        check(queue_b.get_chunk(c1.content_hash) is not None, "B has A's chunk alpha")

        await transport.close()
        await session.close()
        await server_a.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 4. Sync with empty queues ─────────────────────────────────────

def test_sync_empty():
    print("\n4. Sync with empty queues (noop)...")

    async def _run():
        priv_a, _ = generate_node_keypair()
        priv_b, _ = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="empty_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="empty_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        server_a = HttpSyncServer(
            sync_queue=queue_a, node_id="node-A",
            private_key=priv_a, host="127.0.0.1", port=0,
        )
        await server_a.start()

        session = aiohttp.ClientSession()
        transport = await connect_to_peer("127.0.0.1", server_a.port, session=session)
        result = await run_sync(queue_b, "node-B", priv_b, b"", transport)

        check(result.success, "Empty sync success")
        check(result.chunks_sent == 0, f"Sent 0: {result.chunks_sent}")
        check(result.chunks_received == 0, f"Received 0: {result.chunks_received}")

        await transport.close()
        await session.close()
        await server_a.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 5. Sync with identical content ───────────────────────────────

def test_sync_identical():
    print("\n5. Sync with identical content (noop)...")

    async def _run():
        priv_a, _ = generate_node_keypair()
        priv_b, _ = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="ident_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="ident_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Same chunk in both
        chunk = KnowledgeChunk(content="Identical on both nodes", origin_node_id="shared")
        chunk.make_public().sign(priv_a)
        queue_a.add_to_queue(chunk)

        chunk_copy = KnowledgeChunk(
            content="Identical on both nodes",
            content_hash=chunk.content_hash,
            origin_node_id="shared",
            timestamp=chunk.timestamp,
            signature=chunk.signature,
            pool_visibility="public",
            shared_at=chunk.shared_at,
        )
        queue_b.add_to_queue(chunk_copy)

        server_a = HttpSyncServer(
            sync_queue=queue_a, node_id="node-A",
            private_key=priv_a, host="127.0.0.1", port=0,
        )
        await server_a.start()

        session = aiohttp.ClientSession()
        transport = await connect_to_peer("127.0.0.1", server_a.port, session=session)
        result = await run_sync(queue_b, "node-B", priv_b, b"", transport)

        check(result.success, "Identical sync success")
        check(result.chunks_sent == 0, f"Sent 0: {result.chunks_sent}")
        check(result.chunks_received == 0, f"Received 0: {result.chunks_received}")
        check(queue_a.stats()["total"] == 1, "A still 1")
        check(queue_b.stats()["total"] == 1, "B still 1")

        await transport.close()
        await session.close()
        await server_a.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 6. Concurrent sync sessions ──────────────────────────────────

def test_concurrent_syncs():
    print("\n6. Concurrent sync sessions...")

    async def _run():
        priv_s, _ = generate_node_keypair()
        _, path_s = tempfile.mkstemp(suffix=".db", prefix="conc_s_")
        queue_s = SyncQueue(db_path=path_s)

        # Server has 2 chunks
        c1 = make_chunk("Concurrent chunk one")
        c2 = make_chunk("Concurrent chunk two")
        queue_s.add_to_queue(c1)
        queue_s.add_to_queue(c2)

        server = HttpSyncServer(
            sync_queue=queue_s, node_id="server",
            private_key=priv_s, host="127.0.0.1", port=0,
        )
        await server.start()
        port = server.port

        # 3 clients sync concurrently
        client_paths = []
        client_queues = []
        results = []

        async def client_sync(name):
            priv, _ = generate_node_keypair()
            _, path = tempfile.mkstemp(suffix=".db", prefix=f"conc_{name}_")
            client_paths.append(path)
            q = SyncQueue(db_path=path)
            client_queues.append(q)

            session = aiohttp.ClientSession()
            transport = await connect_to_peer("127.0.0.1", port, session=session, name=name)
            result = await run_sync(q, name, priv, b"", transport)
            await transport.close()
            await session.close()
            return result

        r1, r2, r3 = await asyncio.gather(
            client_sync("client-1"),
            client_sync("client-2"),
            client_sync("client-3"),
        )

        check(r1.success, f"Client 1 success (errors: {r1.errors})")
        check(r2.success, f"Client 2 success (errors: {r2.errors})")
        check(r3.success, f"Client 3 success (errors: {r3.errors})")
        check(r1.chunks_received == 2, f"Client 1 received 2: {r1.chunks_received}")
        check(r2.chunks_received == 2, f"Client 2 received 2: {r2.chunks_received}")
        check(r3.chunks_received == 2, f"Client 3 received 2: {r3.chunks_received}")

        await server.stop()
        queue_s.close()
        for q in client_queues:
            q.close()
        os.unlink(path_s)
        for p in client_paths:
            os.unlink(p)

    run(_run())


# ── 7. Health endpoint ────────────────────────────────────────────

def test_health_endpoint():
    print("\n7. Health endpoint...")

    async def _run():
        priv, _ = generate_node_keypair()
        _, path = tempfile.mkstemp(suffix=".db", prefix="health_")
        queue = SyncQueue(db_path=path)

        server = HttpSyncServer(
            sync_queue=queue, node_id="health-node",
            private_key=priv, host="127.0.0.1", port=0,
        )
        await server.start()
        port = server.port

        session = aiohttp.ClientSession()
        resp = await session.get(f"http://127.0.0.1:{port}/health")
        data = await resp.json()

        check(resp.status == 200, f"Status 200: {resp.status}")
        check(data["status"] == "ok", "Status ok")
        check(data["node_id"] == "health-node", f"Node ID: {data['node_id']}")
        check("queue_stats" in data, "Has queue_stats")

        await session.close()
        await server.stop()
        queue.close()
        os.unlink(path)

    run(_run())


# ── 8. LANSyncManager integration ────────────────────────────────

def test_lan_sync_manager():
    print("\n8. LANSyncManager integration (two managers, static peers)...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="mgr_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="mgr_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Add chunks to A
        c1 = KnowledgeChunk(content="Manager test chunk one", origin_node_id="mgr-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="Manager test chunk two", origin_node_id="mgr-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        # Start manager A on ephemeral port (no static peers — it's the server)
        mgr_a = LANSyncManager(
            sync_queue=queue_a, node_id="mgr-A",
            private_key=priv_a, public_key=pub_a,
            port=0, sync_interval=999,  # disable auto-sync
            static_peers=[],
        )
        await mgr_a.start()
        port_a = mgr_a.server_port

        # Manager B points at A
        mgr_b = LANSyncManager(
            sync_queue=queue_b, node_id="mgr-B",
            private_key=priv_b, public_key=pub_b,
            port=0, sync_interval=999,
            static_peers=[("mgr-A", "127.0.0.1", port_a)],
        )
        await mgr_b.start()

        # Trigger sync from B to A
        results = await mgr_b.sync_all()

        check("mgr-A" in results, "Synced with mgr-A")
        result = results.get("mgr-A")
        check(result is not None, "Result not None")
        if result:
            check(result.success, f"Sync success (errors: {result.errors})")
            check(result.chunks_received == 2, f"B received 2: {result.chunks_received}")

        check(queue_b.stats()["total"] == 2, f"B has 2 chunks: {queue_b.stats()['total']}")
        check(queue_b.get_chunk(c1.content_hash) is not None, "B has chunk 1")

        # Check stats
        stats = mgr_b.stats
        check(stats["node_id"] == "mgr-B", "Stats node_id")
        check(stats["running"], "Stats running")
        check("mgr-A" in stats["sync_records"], "Stats has mgr-A record")
        if "mgr-A" in stats["sync_records"]:
            rec = stats["sync_records"]["mgr-A"]
            check(rec["sync_count"] == 1, f"Sync count: {rec['sync_count']}")
            check(rec["total_received"] == 2, f"Total received: {rec['total_received']}")

        await mgr_b.stop()
        await mgr_a.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── 9. Connection failure ─────────────────────────────────────────

def test_connection_failure():
    print("\n9. Connection failure (unreachable peer)...")

    async def _run():
        priv, pub = generate_node_keypair()
        _, path = tempfile.mkstemp(suffix=".db", prefix="fail_")
        queue = SyncQueue(db_path=path)

        mgr = LANSyncManager(
            sync_queue=queue, node_id="fail-node",
            private_key=priv, public_key=pub,
            port=0, sync_interval=999,
            static_peers=[("dead-peer", "127.0.0.1", 19999)],
        )
        await mgr.start()

        # Should fail gracefully (return None, not raise)
        result = await mgr.sync_with_peer(PeerInfo(
            node_id="dead-peer", host="127.0.0.1", port=19999,
        ))
        check(result is None, "Failed sync returns None")

        # Error recorded in stats
        stats = mgr.stats
        check("dead-peer" in stats["sync_records"], "Error recorded in stats")
        if "dead-peer" in stats["sync_records"]:
            check(stats["sync_records"]["dead-peer"]["last_error"] != "", "Has error message")

        await mgr.stop()
        queue.close()
        os.unlink(path)

    run(_run())


# ── 10. Manager rate limiting ─────────────────────────────────────

def test_rate_limiting():
    print("\n10. Manager rate limiting...")

    async def _run():
        priv_a, pub_a = generate_node_keypair()
        priv_b, pub_b = generate_node_keypair()

        _, path_a = tempfile.mkstemp(suffix=".db", prefix="rate_a_")
        _, path_b = tempfile.mkstemp(suffix=".db", prefix="rate_b_")

        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # A has 1 chunk
        c = make_chunk("Rate limit test chunk")
        queue_a.add_to_queue(c)

        mgr_a = LANSyncManager(
            sync_queue=queue_a, node_id="rate-A",
            private_key=priv_a, public_key=pub_a,
            port=0, sync_interval=999, static_peers=[],
        )
        await mgr_a.start()
        port_a = mgr_a.server_port

        mgr_b = LANSyncManager(
            sync_queue=queue_b, node_id="rate-B",
            private_key=priv_b, public_key=pub_b,
            port=0, sync_interval=999,
            static_peers=[("rate-A", "127.0.0.1", port_a)],
        )
        await mgr_b.start()

        # First sync should work
        peer = mgr_b._discovery.peers[0]
        r1 = await mgr_b.sync_with_peer(peer)
        check(r1 is not None, "First sync works")

        # Immediate second sync should be rate-limited (returns None)
        r2 = await mgr_b.sync_with_peer(peer)
        check(r2 is None, "Second sync rate-limited (None)")

        await mgr_b.stop()
        await mgr_a.stop()
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)

    run(_run())


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain LAN Sync Tests")
    print("=" * 60)

    test_static_peer_list()
    test_websocket_transport()
    test_full_sync_over_websocket()
    test_sync_empty()
    test_sync_identical()
    test_concurrent_syncs()
    test_health_endpoint()
    test_lan_sync_manager()
    test_connection_failure()
    test_rate_limiting()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
