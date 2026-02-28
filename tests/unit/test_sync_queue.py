"""SuperBrain Sync Queue Test Suite

Tests SyncQueue SQLite operations: add, get pending, mark synced,
get manifest, remove, stats, and persistence.
Uses temp file for DB, cleaned up after tests.
"""
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    ManifestEntry,
    SyncManifest,
    generate_node_keypair,
)
from sync.queue.sync_queue import SyncQueue

passed = failed = 0
NODE_ID = "test-node-001"


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


def make_chunk(content: str, public: bool = True, **meta) -> KnowledgeChunk:
    """Helper to create a signed public chunk."""
    priv, _ = generate_node_keypair()
    chunk = KnowledgeChunk(
        content=content,
        origin_node_id=NODE_ID,
        metadata=meta,
    )
    if public:
        chunk.make_public()
    chunk.sign(priv)
    return chunk


def new_queue() -> tuple:
    """Create a fresh SyncQueue with a temp file. Returns (queue, db_path)."""
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="sb_sync_test_")
    os.close(fd)
    return SyncQueue(db_path=db_path), db_path


# ── 1. Basic add and retrieve ───────────────────────────────────────

def test_add_to_queue():
    print("\n1. Add to queue...")
    q, path = new_queue()
    try:
        chunk = make_chunk("Bittensor is a decentralized AI network.")
        result = q.add_to_queue(chunk)
        check(result is True, "First add returns True")

        # Duplicate
        result2 = q.add_to_queue(chunk)
        check(result2 is False, "Duplicate returns False")

        stats = q.stats()
        check(stats["total"] == 1, f"Total = 1: {stats['total']}")
    finally:
        q.close()
        os.unlink(path)


def test_private_chunk_rejected():
    print("\n2. Private chunk rejected...")
    q, path = new_queue()
    try:
        chunk = make_chunk("Private content", public=False)
        try:
            q.add_to_queue(chunk)
            check(False, "Should have raised ValueError")
        except ValueError:
            check(True, "ValueError raised for private chunk")
    finally:
        q.close()
        os.unlink(path)


# ── 3. Get pending ──────────────────────────────────────────────────

def test_get_pending():
    print("\n3. Get pending...")
    q, path = new_queue()
    try:
        c1 = make_chunk("Chunk one")
        c2 = make_chunk("Chunk two")
        c3 = make_chunk("Chunk three")
        q.add_to_queue(c1)
        q.add_to_queue(c2)
        q.add_to_queue(c3)

        pending = q.get_pending()
        check(len(pending) == 3, f"3 pending: {len(pending)}")
        check(all(c.pool_visibility == "public" for c in pending), "All public")

        # With limit
        limited = q.get_pending(limit=2)
        check(len(limited) == 2, f"Limit=2 returns 2: {len(limited)}")
    finally:
        q.close()
        os.unlink(path)


# ── 4. Mark synced ──────────────────────────────────────────────────

def test_mark_synced():
    print("\n4. Mark synced...")
    q, path = new_queue()
    try:
        c1 = make_chunk("Sync me")
        c2 = make_chunk("Keep pending")
        q.add_to_queue(c1)
        q.add_to_queue(c2)

        q.mark_synced(c1.content_hash, "peer-node-42")

        pending = q.get_pending()
        check(len(pending) == 1, f"1 pending after sync: {len(pending)}")
        check(pending[0].content_hash == c2.content_hash, "Correct chunk still pending")

        stats = q.stats()
        check(stats["synced"] == 1, f"1 synced: {stats['synced']}")
        check(stats["pending"] == 1, f"1 pending: {stats['pending']}")
        check(stats["sync_events"] == 1, f"1 sync event: {stats['sync_events']}")
    finally:
        q.close()
        os.unlink(path)


def test_mark_synced_multiple_peers():
    print("\n5. Mark synced to multiple peers...")
    q, path = new_queue()
    try:
        chunk = make_chunk("Multi-peer sync")
        q.add_to_queue(chunk)

        q.mark_synced(chunk.content_hash, "peer-1")
        q.mark_synced(chunk.content_hash, "peer-2")
        q.mark_synced(chunk.content_hash, "peer-3")

        stats = q.stats()
        check(stats["sync_events"] == 3, f"3 sync events: {stats['sync_events']}")
        check(stats["synced"] == 1, f"1 synced chunk: {stats['synced']}")
    finally:
        q.close()
        os.unlink(path)


# ── 6. Get manifest ────────────────────────────────────────────────

def test_get_manifest():
    print("\n6. Get manifest...")
    q, path = new_queue()
    try:
        c1 = make_chunk("Manifest chunk one")
        c2 = make_chunk("Manifest chunk two")
        q.add_to_queue(c1)
        q.add_to_queue(c2)

        manifest = q.get_manifest(NODE_ID)
        check(manifest.node_id == NODE_ID, f"Node ID: {manifest.node_id}")
        check(len(manifest.chunks) == 2, f"2 entries: {len(manifest.chunks)}")
        check(all(isinstance(e, ManifestEntry) for e in manifest.chunks), "All ManifestEntry")

        hashes = manifest.chunk_hashes
        check(c1.content_hash in hashes, "Chunk 1 in manifest")
        check(c2.content_hash in hashes, "Chunk 2 in manifest")

        # Sizes match content length
        for entry in manifest.chunks:
            chunk = q.get_chunk(entry.hash)
            check(entry.size == len(chunk.content), f"Size matches: {entry.size}")
    finally:
        q.close()
        os.unlink(path)


# ── 7. Get chunk by hash ───────────────────────────────────────────

def test_get_chunk():
    print("\n7. Get chunk by hash...")
    q, path = new_queue()
    try:
        chunk = make_chunk("Retrievable content", source="test.pdf", page=5)
        q.add_to_queue(chunk)

        retrieved = q.get_chunk(chunk.content_hash)
        check(retrieved is not None, "Chunk found")
        check(retrieved.content == "Retrievable content", "Content matches")
        check(retrieved.metadata.get("source") == "test.pdf", "Metadata preserved")
        check(retrieved.pool_visibility == "public", "Visibility is public")

        # Non-existent hash
        missing = q.get_chunk("nonexistent_hash")
        check(missing is None, "Missing hash returns None")
    finally:
        q.close()
        os.unlink(path)


# ── 8. Get chunks by hashes (batch) ────────────────────────────────

def test_get_chunks_by_hashes():
    print("\n8. Get chunks by hashes (batch)...")
    q, path = new_queue()
    try:
        c1 = make_chunk("Batch one")
        c2 = make_chunk("Batch two")
        c3 = make_chunk("Batch three")
        q.add_to_queue(c1)
        q.add_to_queue(c2)
        q.add_to_queue(c3)

        results = q.get_chunks_by_hashes([c1.content_hash, c3.content_hash])
        check(len(results) == 2, f"Got 2 chunks: {len(results)}")

        # Empty list
        empty = q.get_chunks_by_hashes([])
        check(len(empty) == 0, "Empty list returns empty")
    finally:
        q.close()
        os.unlink(path)


# ── 9. Remove from queue ───────────────────────────────────────────

def test_remove_from_queue():
    print("\n9. Remove from queue (revoke sharing)...")
    q, path = new_queue()
    try:
        chunk = make_chunk("Revocable chunk")
        q.add_to_queue(chunk)

        check(q.stats()["total"] == 1, "1 in queue")

        removed = q.remove_from_queue(chunk.content_hash)
        check(removed is True, "Remove returns True")
        check(q.stats()["total"] == 0, "0 in queue after remove")

        # Remove non-existent
        removed2 = q.remove_from_queue("nonexistent")
        check(removed2 is False, "Remove non-existent returns False")
    finally:
        q.close()
        os.unlink(path)


# ── 10. Stats ───────────────────────────────────────────────────────

def test_stats():
    print("\n10. Stats...")
    q, path = new_queue()
    try:
        # Empty queue
        stats = q.stats()
        check(stats["total"] == 0, "Empty: total=0")
        check(stats["pending"] == 0, "Empty: pending=0")
        check(stats["synced"] == 0, "Empty: synced=0")
        check(stats["sync_events"] == 0, "Empty: events=0")

        # Add 3, sync 1
        for i in range(3):
            q.add_to_queue(make_chunk(f"Stats chunk {i}"))
        q.mark_synced(q.get_pending(1)[0].content_hash, "peer-x")

        stats = q.stats()
        check(stats["total"] == 3, f"Total=3: {stats['total']}")
        check(stats["pending"] == 2, f"Pending=2: {stats['pending']}")
        check(stats["synced"] == 1, f"Synced=1: {stats['synced']}")
    finally:
        q.close()
        os.unlink(path)


# ── 11. Persistence ────────────────────────────────────────────────

def test_persistence():
    print("\n11. Persistence across close/reopen...")
    _, path = tempfile.mkstemp(suffix=".db", prefix="sb_persist_")
    try:
        # Write data
        q1 = SyncQueue(db_path=path)
        q1.add_to_queue(make_chunk("Persistent chunk"))
        q1.close()

        # Reopen and verify
        q2 = SyncQueue(db_path=path)
        stats = q2.stats()
        check(stats["total"] == 1, f"Data survives restart: {stats['total']}")

        pending = q2.get_pending()
        check(len(pending) == 1, "Chunk retrievable after restart")
        check(pending[0].content == "Persistent chunk", "Content intact")
        q2.close()
    finally:
        os.unlink(path)


# ── 12. Manifest last_sync ─────────────────────────────────────────

def test_manifest_last_sync():
    print("\n12. Manifest last_sync tracking...")
    q, path = new_queue()
    try:
        c1 = make_chunk("Sync time tracking")
        q.add_to_queue(c1)

        # Before any sync
        m1 = q.get_manifest(NODE_ID)
        check(m1.last_sync == 0.0, f"No sync yet: last_sync={m1.last_sync}")

        # After sync
        q.mark_synced(c1.content_hash, "peer-1")
        m2 = q.get_manifest(NODE_ID)
        check(m2.last_sync > 0.0, f"After sync: last_sync={m2.last_sync:.1f}")
    finally:
        q.close()
        os.unlink(path)


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Sync Queue Tests")
    print("=" * 60)

    test_add_to_queue()
    test_private_chunk_rejected()
    test_get_pending()
    test_mark_synced()
    test_mark_synced_multiple_peers()
    test_get_manifest()
    test_get_chunk()
    test_get_chunks_by_hashes()
    test_remove_from_queue()
    test_stats()
    test_persistence()
    test_manifest_last_sync()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
