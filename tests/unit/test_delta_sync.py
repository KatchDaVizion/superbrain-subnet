"""SuperBrain Delta Sync Protocol Test Suite

Tests the transport-agnostic sync protocol: handshake, manifest exchange,
chunk transfer, full sync, and error handling.
Uses MockTransport for in-memory testing.
"""
import sys
import os
import asyncio
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    SyncManifest,
    ManifestEntry,
    generate_node_keypair,
    compute_content_hash,
)
from sync.protocol.delta_sync import (
    TransportLayer,
    HandshakeResult,
    TransferResult,
    handshake,
    exchange_manifests,
    send_chunks,
    receive_chunks,
    run_sync,
    _serialize_message,
    _deserialize_message,
    _serialize_chunk,
    _deserialize_chunk,
    PROTOCOL_VERSION,
)
from sync.queue.sync_queue import SyncQueue
from tests.mock.mock_transport import MockTransport, create_mock_pair

passed = failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


def make_chunk(content, node_id="test-node", public=True):
    """Helper to create a signed public chunk."""
    priv, _ = generate_node_keypair()
    chunk = KnowledgeChunk(content=content, origin_node_id=node_id)
    if public:
        chunk.make_public()
    chunk.sign(priv)
    return chunk


def run(coro):
    """Run an async coroutine."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── 1. Wire format helpers ──────────────────────────────────────────

def test_wire_format():
    print("\n1. Wire format helpers...")
    # Message round-trip
    msg = {"type": "test", "data": "hello", "number": 42}
    serialized = _serialize_message(msg)
    check(isinstance(serialized, bytes), "Serialized is bytes")
    check(len(serialized) > 0, "Non-empty")

    # Skip the 4-byte frame header for deserialization
    deserialized = _deserialize_message(serialized[4:])
    check(deserialized == msg, "Message round-trip OK")

    # Chunk round-trip
    chunk = make_chunk("Wire format test content")
    chunk_bytes = _serialize_chunk(chunk)
    check(isinstance(chunk_bytes, bytes), "Chunk serialized")

    restored = _deserialize_chunk(chunk_bytes[4:])
    check(restored.content == chunk.content, "Chunk content round-trip")
    check(restored.content_hash == chunk.content_hash, "Chunk hash round-trip")


# ── 2. Handshake ────────────────────────────────────────────────────

def test_handshake():
    print("\n2. Handshake...")

    async def _run():
        ta, tb = create_mock_pair()

        # Run both sides concurrently
        result_a, result_b = await asyncio.gather(
            handshake("node-A", ta),
            handshake("node-B", tb),
        )
        return result_a, result_b

    ra, rb = run(_run())
    check(ra.success, "Side A handshake success")
    check(rb.success, "Side B handshake success")
    check(ra.local_node_id == "node-A", f"A local: {ra.local_node_id}")
    check(ra.remote_node_id == "node-B", f"A sees remote: {ra.remote_node_id}")
    check(rb.remote_node_id == "node-A", f"B sees remote: {rb.remote_node_id}")
    check(ra.protocol_version == PROTOCOL_VERSION, f"Version: {ra.protocol_version}")


# ── 3. Manifest exchange ───────────────────────────────────────────

def test_manifest_exchange():
    print("\n3. Manifest exchange...")

    priv_a, _ = generate_node_keypair()
    priv_b, _ = generate_node_keypair()

    manifest_a = SyncManifest(
        node_id="node-A",
        chunks=[ManifestEntry(hash="aaa", timestamp=1.0, size=100)],
    ).sign(priv_a)

    manifest_b = SyncManifest(
        node_id="node-B",
        chunks=[ManifestEntry(hash="bbb", timestamp=2.0, size=200)],
    ).sign(priv_b)

    async def _run():
        ta, tb = create_mock_pair()
        remote_b, remote_a = await asyncio.gather(
            exchange_manifests(manifest_a, ta),
            exchange_manifests(manifest_b, tb),
        )
        return remote_b, remote_a

    got_b, got_a = run(_run())
    check(got_b.node_id == "node-B", f"A received B's manifest: {got_b.node_id}")
    check(got_a.node_id == "node-A", f"B received A's manifest: {got_a.node_id}")
    check("bbb" in got_b.chunk_hashes, "A sees B's chunk hash")
    check("aaa" in got_a.chunk_hashes, "B sees A's chunk hash")


# ── 4. Chunk transfer ──────────────────────────────────────────────

def test_chunk_transfer():
    print("\n4. Chunk transfer...")

    chunks = [
        make_chunk("Transfer chunk one"),
        make_chunk("Transfer chunk two"),
        make_chunk("Transfer chunk three"),
    ]

    async def _run():
        ta, tb = create_mock_pair()
        sent_count, (received, errors) = await asyncio.gather(
            send_chunks(chunks, ta),
            receive_chunks(tb),
        )
        return sent_count, received, errors

    sent, received, errors = run(_run())
    check(sent == 3, f"Sent 3 chunks: {sent}")
    check(len(received) == 3, f"Received 3 chunks: {len(received)}")
    check(len(errors) == 0, f"No errors: {errors}")

    # Verify content integrity
    for orig, recv in zip(chunks, received):
        check(recv.content == orig.content, f"Content match: {orig.content[:20]}...")
        check(recv.content_hash == orig.content_hash, "Hash match")


def test_empty_transfer():
    print("\n5. Empty transfer...")

    async def _run():
        ta, tb = create_mock_pair()
        sent, (received, errors) = await asyncio.gather(
            send_chunks([], ta),
            receive_chunks(tb),
        )
        return sent, received, errors

    sent, received, errors = run(_run())
    check(sent == 0, "Sent 0")
    check(len(received) == 0, "Received 0")
    check(len(errors) == 0, "No errors")


# ── 6. Hash integrity check ────────────────────────────────────────

def test_hash_integrity():
    print("\n6. Hash integrity on receive...")

    chunk = make_chunk("Integrity test")
    # Tamper with the hash after serialization
    tampered = KnowledgeChunk(
        content=chunk.content,
        content_hash="tampered_hash_value_that_wont_match",
        origin_node_id=chunk.origin_node_id,
        timestamp=chunk.timestamp,
        signature=chunk.signature,
        pool_visibility="public",
        shared_at=chunk.shared_at,
    )

    async def _run():
        ta, tb = create_mock_pair()
        await send_chunks([tampered], ta)
        received, errors = await receive_chunks(tb)
        return received, errors

    received, errors = run(_run())
    check(len(received) == 0, "Tampered chunk rejected")
    check(len(errors) == 1, f"1 error reported: {errors[0][:50] if errors else 'none'}...")


# ── 7. Full run_sync between two nodes ─────────────────────────────

def test_full_sync():
    print("\n7. Full run_sync (two nodes, different chunks)...")

    priv_a, pub_a = generate_node_keypair()
    priv_b, pub_b = generate_node_keypair()

    # Create temp DBs
    _, path_a = tempfile.mkstemp(suffix=".db", prefix="sync_a_")
    _, path_b = tempfile.mkstemp(suffix=".db", prefix="sync_b_")

    try:
        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Node A has chunks 1, 2
        c1 = KnowledgeChunk(content="Shared by node A: chunk one", origin_node_id="node-A")
        c1.make_public().sign(priv_a)
        c2 = KnowledgeChunk(content="Shared by node A: chunk two", origin_node_id="node-A")
        c2.make_public().sign(priv_a)
        queue_a.add_to_queue(c1)
        queue_a.add_to_queue(c2)

        # Node B has chunks 3, 4
        c3 = KnowledgeChunk(content="Shared by node B: chunk three", origin_node_id="node-B")
        c3.make_public().sign(priv_b)
        c4 = KnowledgeChunk(content="Shared by node B: chunk four", origin_node_id="node-B")
        c4.make_public().sign(priv_b)
        queue_b.add_to_queue(c3)
        queue_b.add_to_queue(c4)

        check(queue_a.stats()["total"] == 2, "A starts with 2 chunks")
        check(queue_b.stats()["total"] == 2, "B starts with 2 chunks")

        async def _run():
            ta, tb = create_mock_pair()
            result_a, result_b = await asyncio.gather(
                run_sync(queue_a, "node-A", priv_a, pub_b, ta),
                run_sync(queue_b, "node-B", priv_b, pub_a, tb),
            )
            return result_a, result_b

        result_a, result_b = run(_run())

        check(result_a.success, f"A sync success (errors: {result_a.errors})")
        check(result_b.success, f"B sync success (errors: {result_b.errors})")
        check(result_a.chunks_sent == 2, f"A sent 2: {result_a.chunks_sent}")
        check(result_a.chunks_received == 2, f"A received 2: {result_a.chunks_received}")
        check(result_b.chunks_sent == 2, f"B sent 2: {result_b.chunks_sent}")
        check(result_b.chunks_received == 2, f"B received 2: {result_b.chunks_received}")

        # Both nodes should now have 4 chunks
        check(queue_a.stats()["total"] == 4, f"A has 4 chunks: {queue_a.stats()['total']}")
        check(queue_b.stats()["total"] == 4, f"B has 4 chunks: {queue_b.stats()['total']}")

        # Verify specific content
        a_has_c3 = queue_a.get_chunk(c3.content_hash) is not None
        b_has_c1 = queue_b.get_chunk(c1.content_hash) is not None
        check(a_has_c3, "A now has B's chunk 3")
        check(b_has_c1, "B now has A's chunk 1")

    finally:
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)


# ── 8. Sync with identical content (no transfer) ───────────────────

def test_sync_identical():
    print("\n8. Sync with identical content (noop)...")

    priv_a, pub_a = generate_node_keypair()
    priv_b, pub_b = generate_node_keypair()

    _, path_a = tempfile.mkstemp(suffix=".db", prefix="sync_id_a_")
    _, path_b = tempfile.mkstemp(suffix=".db", prefix="sync_id_b_")

    try:
        queue_a = SyncQueue(db_path=path_a)
        queue_b = SyncQueue(db_path=path_b)

        # Same chunk in both
        chunk = KnowledgeChunk(content="Identical content on both nodes", origin_node_id="shared")
        chunk.make_public().sign(priv_a)
        queue_a.add_to_queue(chunk)

        # Need separate instance for queue_b (same content, different insertion)
        chunk_b = KnowledgeChunk(
            content="Identical content on both nodes",
            content_hash=chunk.content_hash,
            origin_node_id="shared",
            timestamp=chunk.timestamp,
            signature=chunk.signature,
            pool_visibility="public",
            shared_at=chunk.shared_at,
        )
        queue_b.add_to_queue(chunk_b)

        async def _run():
            ta, tb = create_mock_pair()
            ra, rb = await asyncio.gather(
                run_sync(queue_a, "node-A", priv_a, pub_b, ta),
                run_sync(queue_b, "node-B", priv_b, pub_a, tb),
            )
            return ra, rb

        ra, rb = run(_run())
        check(ra.success, "Sync success")
        check(ra.chunks_sent == 0, f"A sent 0: {ra.chunks_sent}")
        check(ra.chunks_received == 0, f"A received 0: {ra.chunks_received}")
        check(queue_a.stats()["total"] == 1, "A still has 1")
        check(queue_b.stats()["total"] == 1, "B still has 1")

    finally:
        queue_a.close()
        queue_b.close()
        os.unlink(path_a)
        os.unlink(path_b)


# ── 9. TransferResult properties ────────────────────────────────────

def test_transfer_result():
    print("\n9. TransferResult properties...")
    r1 = TransferResult(chunks_sent=5, chunks_received=3, bytes_transferred=1024)
    check(r1.success, "No errors = success")

    r2 = TransferResult(errors=["something went wrong"])
    check(not r2.success, "With errors = not success")


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Delta Sync Protocol Tests")
    print("=" * 60)

    test_wire_format()
    test_handshake()
    test_manifest_exchange()
    test_chunk_transfer()
    test_empty_transfer()
    test_hash_integrity()
    test_full_sync()
    test_sync_identical()
    test_transfer_result()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
