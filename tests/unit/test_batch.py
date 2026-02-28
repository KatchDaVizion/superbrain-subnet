"""SuperBrain Store-and-Forward Batch Test Suite

Tests SyncBatch: create, extract, to_bytes/from_bytes round-trip,
compression, signature verification, max size, and idempotency.
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    generate_node_keypair,
    compute_content_hash,
)
from sync.protocol.batch import SyncBatch, MAX_BATCH_SIZE
from sync.queue.sync_queue import SyncQueue

passed = failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


def make_chunks(count=3):
    """Create a list of signed public chunks."""
    priv, _ = generate_node_keypair()
    chunks = []
    for i in range(count):
        c = KnowledgeChunk(
            content=f"Batch test chunk number {i}: knowledge about topic {i}",
            origin_node_id="batch-node",
            metadata={"index": i, "source": "test"},
        )
        c.make_public().sign(priv)
        chunks.append(c)
    return chunks


# ── 1. Create + extract round-trip ──────────────────────────────────

def test_create_extract():
    print("\n1. Create + extract round-trip...")
    priv, pub = generate_node_keypair()
    chunks = make_chunks(5)

    batch = SyncBatch.create(chunks, "batch-node", priv)
    check(len(batch.batch_id) == 36, f"Batch ID is UUID: {batch.batch_id}")
    check(batch.node_id == "batch-node", "Node ID set")
    check(len(batch.chunk_hashes) == 5, f"5 hashes: {len(batch.chunk_hashes)}")
    check(len(batch.compressed_data) > 0, "Has compressed data")
    check(batch.signature != "", "Has signature")
    check(batch.created_at > 0, "Has timestamp")

    # Extract
    extracted = SyncBatch.extract(batch, pub)
    check(len(extracted) == 5, f"Extracted 5 chunks: {len(extracted)}")
    for i, (orig, ext) in enumerate(zip(chunks, extracted)):
        check(ext.content == orig.content, f"Chunk {i} content matches")
        check(ext.content_hash == orig.content_hash, f"Chunk {i} hash matches")


# ── 2. Compression effectiveness ───────────────────────────────────

def test_compression():
    print("\n2. Compression effectiveness...")
    priv, _ = generate_node_keypair()
    # Create chunks with repetitive content (compresses well)
    chunks = []
    for i in range(10):
        c = KnowledgeChunk(
            content=f"Bittensor is a decentralized AI network. " * 20,
            origin_node_id="compress-node",
        )
        c.make_public().sign(priv)
        chunks.append(c)

    batch = SyncBatch.create(chunks, "compress-node", priv)
    total_raw = sum(len(c.content) for c in chunks)
    compressed_size = len(batch.compressed_data)
    ratio = compressed_size / total_raw

    check(compressed_size < total_raw, f"Compressed ({compressed_size}) < Raw ({total_raw})")
    check(ratio < 0.5, f"Compression ratio: {ratio:.2%}")
    print(f"    Raw: {total_raw} bytes → Compressed: {compressed_size} bytes ({ratio:.1%})")


# ── 3. Signature verification ──────────────────────────────────────

def test_signature_verification():
    print("\n3. Signature verification...")
    priv, pub = generate_node_keypair()
    priv_wrong, pub_wrong = generate_node_keypair()
    chunks = make_chunks(3)

    batch = SyncBatch.create(chunks, "sig-node", priv)

    # Valid signature
    extracted = SyncBatch.extract(batch, pub)
    check(len(extracted) == 3, "Valid signature: extraction OK")

    # Wrong key
    try:
        SyncBatch.extract(batch, pub_wrong)
        check(False, "Should have raised ValueError")
    except ValueError as e:
        check("signature" in str(e).lower(), f"Wrong key rejected: {e}")


# ── 4. to_bytes / from_bytes round-trip ─────────────────────────────

def test_bytes_roundtrip():
    print("\n4. to_bytes / from_bytes round-trip...")
    priv, pub = generate_node_keypair()
    chunks = make_chunks(4)

    batch = SyncBatch.create(chunks, "bytes-node", priv)
    wire = batch.to_bytes()
    check(isinstance(wire, bytes), "to_bytes returns bytes")
    check(len(wire) > 0, "Non-empty wire format")

    restored = SyncBatch.from_bytes(wire)
    check(restored.batch_id == batch.batch_id, "Batch ID preserved")
    check(restored.node_id == batch.node_id, "Node ID preserved")
    check(restored.chunk_hashes == batch.chunk_hashes, "Chunk hashes preserved")
    check(restored.signature == batch.signature, "Signature preserved")
    check(restored.compressed_data == batch.compressed_data, "Data preserved")

    # Extract from restored batch
    extracted = SyncBatch.extract(restored, pub)
    check(len(extracted) == 4, f"Extract from restored: {len(extracted)}")


# ── 5. Idempotent ingestion ─────────────────────────────────────────

def test_idempotent_ingestion():
    print("\n5. Idempotent ingestion...")
    priv, pub = generate_node_keypair()
    chunks = make_chunks(3)
    batch = SyncBatch.create(chunks, "idem-node", priv)

    _, path = tempfile.mkstemp(suffix=".db", prefix="batch_idem_")
    try:
        queue = SyncQueue(db_path=path)

        # Ingest first time
        extracted1 = SyncBatch.extract(batch, pub)
        for c in extracted1:
            c.pool_visibility = "public"
            if c.shared_at is None:
                c.shared_at = c.timestamp
            queue.add_to_queue(c)

        check(queue.stats()["total"] == 3, f"First ingest: 3 chunks")

        # Ingest same batch again (should be noop due to dedup)
        extracted2 = SyncBatch.extract(batch, pub)
        added = 0
        for c in extracted2:
            c.pool_visibility = "public"
            if c.shared_at is None:
                c.shared_at = c.timestamp
            if queue.add_to_queue(c):
                added += 1

        check(added == 0, f"Second ingest: 0 new ({added} added)")
        check(queue.stats()["total"] == 3, "Still 3 chunks total")

        queue.close()
    finally:
        os.unlink(path)


# ── 6. Empty batch ──────────────────────────────────────────────────

def test_empty_batch():
    print("\n6. Empty batch...")
    priv, pub = generate_node_keypair()

    batch = SyncBatch.create([], "empty-node", priv)
    check(len(batch.chunk_hashes) == 0, "0 hashes")
    check(len(batch.compressed_data) > 0, "Has compressed data (empty JSON array)")

    extracted = SyncBatch.extract(batch, pub)
    check(len(extracted) == 0, "Extracted 0 chunks")

    # Wire round-trip
    restored = SyncBatch.from_bytes(batch.to_bytes())
    check(restored.batch_id == batch.batch_id, "Empty batch survives wire round-trip")


# ── 7. Content hash integrity ──────────────────────────────────────

def test_hash_integrity():
    print("\n7. Content hash integrity...")
    priv, pub = generate_node_keypair()

    chunk = KnowledgeChunk(content="Integrity test", origin_node_id="hash-node")
    chunk.make_public().sign(priv)

    # Tamper with content after signing
    chunk_dict = chunk.model_dump()
    chunk_dict["content"] = "Tampered content"
    # Don't update hash — it should mismatch

    import json, zlib
    raw = json.dumps([chunk_dict], sort_keys=True).encode()
    compressed = zlib.compress(raw)
    from sync.protocol.pool_model import sign_data
    sig = sign_data(compressed, priv)

    tampered_batch = SyncBatch(
        node_id="tamper-node",
        chunk_hashes=[chunk.content_hash],
        compressed_data=compressed,
        signature=sig,
    )

    try:
        SyncBatch.extract(tampered_batch, pub)
        check(False, "Should have raised ValueError for hash mismatch")
    except ValueError as e:
        check("mismatch" in str(e).lower(), f"Hash mismatch detected: {e}")


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Store-and-Forward Batch Tests")
    print("=" * 60)

    test_create_extract()
    test_compression()
    test_signature_verification()
    test_bytes_roundtrip()
    test_idempotent_ingestion()
    test_empty_batch()
    test_hash_integrity()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
