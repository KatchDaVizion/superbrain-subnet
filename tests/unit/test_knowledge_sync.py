#!/usr/bin/env python3
"""
Tests for KnowledgeSyncSynapse — synapse, scoring, miner handler, validator flow, end-to-end.
Uses custom runner (check/run helpers, no pytest).
"""

import asyncio
import base64
import json
import math
import os
import sys
import tempfile
import time
import uuid
import zlib

# Add project root AND subnet/ to path
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "subnet"))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    SyncManifest,
    ManifestEntry,
    compute_content_hash,
    generate_node_keypair,
)
from sync.queue.sync_queue import SyncQueue

# Import scoring functions
from subnet.superbrain.validator.sync_reward import (
    W_VALIDITY,
    W_FRESHNESS,
    W_QUANTITY,
    W_LATENCY,
    FRESHNESS_HALF_LIFE,
    _decode_batch_chunks,
    score_validity,
    score_freshness,
    score_quantity,
    score_sync_latency,
    sync_reward,
    get_sync_rewards,
)

# Import sync forward
from subnet.superbrain.validator.sync_forward import (
    _decode_and_validate_batch,
    SYNC_INTERVAL_STEPS,
    MAX_CHUNKS_PER_SYNC,
)

# Import protocol
from subnet.superbrain.protocol import KnowledgeSyncSynapse

# ── Test infrastructure ──────────────────────────────────────────

passed = 0
failed = 0


def check(condition, name):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_chunk(content="Test content", ts=None, public=True):
    """Helper to create a valid KnowledgeChunk."""
    chunk = KnowledgeChunk(
        content=content,
        content_hash=compute_content_hash(content),
        origin_node_id=str(uuid.uuid4()),
        timestamp=ts or time.time(),
        signature="test_sig",
        pool_visibility="public" if public else "private",
        shared_at=time.time() if public else None,
    )
    return chunk


def make_batch_data(chunks):
    """Create base64-encoded batch data from a list of KnowledgeChunks."""
    chunk_dicts = [c.model_dump() for c in chunks]
    raw_json = json.dumps(chunk_dicts, sort_keys=True).encode("utf-8")
    compressed = zlib.compress(raw_json)
    return base64.b64encode(compressed).decode("ascii")


def make_temp_queue():
    """Create a SyncQueue with a temp DB file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return SyncQueue(db_path=path), path


# ── 1. KnowledgeSyncSynapse Tests ────────────────────────────────

def test_synapse_defaults():
    print("\n1. KnowledgeSyncSynapse — Defaults")
    s = KnowledgeSyncSynapse()
    check(s.known_hashes == [], "known_hashes defaults to empty list")
    check(s.max_chunks == 50, "max_chunks defaults to 50")
    check(s.node_id == "", "node_id defaults to empty string")
    check(s.batch_data is None, "batch_data defaults to None")
    check(s.chunk_count is None, "chunk_count defaults to None")
    check(s.batch_id is None, "batch_id defaults to None")


def test_synapse_with_values():
    print("\n2. KnowledgeSyncSynapse — With values")
    hashes = ["abc123", "def456"]
    s = KnowledgeSyncSynapse(
        known_hashes=hashes,
        max_chunks=100,
        node_id="validator_1",
    )
    check(s.known_hashes == hashes, "known_hashes set correctly")
    check(s.max_chunks == 100, "max_chunks set correctly")
    check(s.node_id == "validator_1", "node_id set correctly")


def test_synapse_deserialize():
    print("\n3. KnowledgeSyncSynapse — deserialize()")
    s = KnowledgeSyncSynapse()
    s.batch_data = "test_batch"
    s.chunk_count = 5
    s.batch_id = "batch_123"
    d = s.deserialize()
    check(d["batch_data"] == "test_batch", "deserialize batch_data")
    check(d["chunk_count"] == 5, "deserialize chunk_count")
    check(d["batch_id"] == "batch_123", "deserialize batch_id")


def test_synapse_deserialize_none():
    print("\n4. KnowledgeSyncSynapse — deserialize() with None fields")
    s = KnowledgeSyncSynapse()
    d = s.deserialize()
    check(d["batch_data"] is None, "batch_data is None")
    check(d["chunk_count"] is None, "chunk_count is None")
    check(d["batch_id"] is None, "batch_id is None")


def test_synapse_miner_fills():
    print("\n5. KnowledgeSyncSynapse — Miner fills response fields")
    s = KnowledgeSyncSynapse(known_hashes=["h1", "h2"], max_chunks=10, node_id="val")
    # Simulate miner filling
    chunks = [make_chunk(f"chunk_{i}") for i in range(3)]
    batch_b64 = make_batch_data(chunks)
    s.batch_data = batch_b64
    s.chunk_count = 3
    s.batch_id = "batch_abc"
    d = s.deserialize()
    check(d["batch_data"] == batch_b64, "batch_data round-trips")
    check(d["chunk_count"] == 3, "chunk_count round-trips")
    check(d["batch_id"] == "batch_abc", "batch_id round-trips")


# ── 2. Batch Decode Tests ────────────────────────────────────────

def test_decode_batch_valid():
    print("\n6. _decode_batch_chunks — Valid batch")
    chunks = [make_chunk(f"chunk_{i}") for i in range(5)]
    batch_b64 = make_batch_data(chunks)
    decoded = _decode_batch_chunks(batch_b64)
    check(len(decoded) == 5, "decoded 5 chunks")
    check(decoded[0].content == "chunk_0", "first chunk content correct")
    check(decoded[4].content == "chunk_4", "last chunk content correct")


def test_decode_batch_invalid_base64():
    print("\n7. _decode_batch_chunks — Invalid base64")
    decoded = _decode_batch_chunks("not_valid_base64!!!")
    check(decoded == [], "returns empty list for invalid base64")


def test_decode_batch_invalid_json():
    print("\n8. _decode_batch_chunks — Invalid JSON (valid base64)")
    compressed = zlib.compress(b"not json")
    b64 = base64.b64encode(compressed).decode("ascii")
    decoded = _decode_batch_chunks(b64)
    check(decoded == [], "returns empty list for invalid JSON")


def test_decode_batch_empty():
    print("\n9. _decode_batch_chunks — None input")
    # _decode_batch_chunks expects a string, None should fail gracefully
    try:
        decoded = _decode_batch_chunks(None)
        check(decoded == [], "returns empty for None")
    except Exception:
        check(True, "raises exception for None (acceptable)")


def test_decode_batch_empty_list():
    print("\n10. _decode_batch_chunks — Empty chunk list")
    b64 = make_batch_data([])
    decoded = _decode_batch_chunks(b64)
    check(decoded == [], "returns empty list for empty batch")


# ── 3. Scoring: Validity ─────────────────────────────────────────

def test_score_validity_all_valid():
    print("\n11. score_validity — All valid chunks")
    chunks = [make_chunk(f"valid_chunk_{i}") for i in range(5)]
    score = score_validity(chunks)
    check(abs(score - 1.0) < 0.001, f"score = {score:.3f} (expected 1.0)")


def test_score_validity_some_invalid():
    print("\n12. score_validity — Some invalid hashes")
    chunks = [make_chunk(f"chunk_{i}") for i in range(4)]
    # Corrupt hash on last chunk
    chunks[3].content_hash = "bad_hash"
    score = score_validity(chunks)
    check(abs(score - 0.75) < 0.001, f"score = {score:.3f} (expected 0.75)")


def test_score_validity_all_invalid():
    print("\n13. score_validity — All invalid hashes")
    chunks = [make_chunk(f"chunk_{i}") for i in range(3)]
    for c in chunks:
        c.content_hash = "bad"
    score = score_validity(chunks)
    check(abs(score - 0.0) < 0.001, f"score = {score:.3f} (expected 0.0)")


def test_score_validity_empty():
    print("\n14. score_validity — Empty list")
    check(score_validity([]) == 0.0, "empty list → 0.0")


def test_score_validity_empty_content():
    print("\n15. score_validity — Empty content")
    chunk = make_chunk("real content")
    chunk.content = ""
    score = score_validity([chunk])
    check(score == 0.0, "empty content → 0.0")


def test_score_validity_mixed():
    print("\n16. score_validity — Mixed valid/invalid/empty")
    c1 = make_chunk("valid_1")
    c2 = make_chunk("valid_2")
    c3 = make_chunk("invalid")
    c3.content_hash = "wrong"
    c4 = make_chunk("empty")
    c4.content = "   "
    score = score_validity([c1, c2, c3, c4])
    check(abs(score - 0.5) < 0.001, f"score = {score:.3f} (expected 0.5, 2/4)")


# ── 4. Scoring: Freshness ────────────────────────────────────────

def test_score_freshness_recent():
    print("\n17. score_freshness — All very recent chunks")
    now = time.time()
    chunks = [make_chunk(f"fresh_{i}", ts=now - 60) for i in range(3)]  # 1 min ago
    score = score_freshness(chunks, now=now)
    check(score > 0.99, f"score = {score:.3f} (expected ~1.0)")


def test_score_freshness_old():
    print("\n18. score_freshness — 7-day old chunks (half-life)")
    now = time.time()
    chunks = [make_chunk(f"old_{i}", ts=now - 7 * 24 * 3600) for i in range(3)]
    score = score_freshness(chunks, now=now)
    check(abs(score - 0.5) < 0.01, f"score = {score:.3f} (expected ~0.5)")


def test_score_freshness_very_old():
    print("\n19. score_freshness — 14-day old chunks (two half-lives)")
    now = time.time()
    chunks = [make_chunk(f"vold_{i}", ts=now - 14 * 24 * 3600) for i in range(3)]
    score = score_freshness(chunks, now=now)
    check(abs(score - 0.25) < 0.01, f"score = {score:.3f} (expected ~0.25)")


def test_score_freshness_mixed_ages():
    print("\n20. score_freshness — Mixed ages")
    now = time.time()
    c1 = make_chunk("new", ts=now)  # brand new → ~1.0
    c2 = make_chunk("week_old", ts=now - 7 * 24 * 3600)  # 7 days → ~0.5
    score = score_freshness([c1, c2], now=now)
    check(0.7 < score < 0.8, f"score = {score:.3f} (expected ~0.75)")


def test_score_freshness_empty():
    print("\n21. score_freshness — Empty list")
    check(score_freshness([]) == 0.0, "empty list → 0.0")


def test_score_freshness_future_timestamp():
    print("\n22. score_freshness — Future timestamp (clamped)")
    now = time.time()
    chunk = make_chunk("future", ts=now + 3600)
    score = score_freshness([chunk], now=now)
    # age = max(0, now - (now+3600)) = 0 → score = 1.0
    check(abs(score - 1.0) < 0.001, f"score = {score:.3f} (future clamped to 1.0)")


# ── 5. Scoring: Quantity ─────────────────────────────────────────

def test_score_quantity_full():
    print("\n23. score_quantity — Full batch")
    score = score_quantity(n_valid=50, n_unique=50, max_chunks=50)
    check(abs(score - 1.0) < 0.001, f"score = {score:.3f}")


def test_score_quantity_partial():
    print("\n24. score_quantity — Partial batch (logarithmic diminishing returns)")
    score = score_quantity(n_valid=25, n_unique=25, max_chunks=50)
    # Logarithmic: log(1+25)/log(1+50) ≈ 0.829 (rewards first chunks more)
    check(abs(score - 0.829) < 0.01, f"score = {score:.3f}")


def test_score_quantity_zero():
    print("\n25. score_quantity — Zero chunks")
    score = score_quantity(n_valid=0, n_unique=0, max_chunks=50)
    check(score == 0.0, "zero chunks → 0.0")


def test_score_quantity_unique_less_than_valid():
    print("\n26. score_quantity — Some known (unique < valid, logarithmic)")
    score = score_quantity(n_valid=50, n_unique=10, max_chunks=50)
    # Logarithmic: log(1+10)/log(1+50) ≈ 0.610
    check(abs(score - 0.610) < 0.01, f"score = {score:.3f} (10/50)")


def test_score_quantity_overflow():
    print("\n27. score_quantity — More chunks than max (capped at 1.0)")
    score = score_quantity(n_valid=100, n_unique=100, max_chunks=50)
    check(abs(score - 1.0) < 0.001, f"score = {score:.3f} (capped at 1.0)")


def test_score_quantity_max_zero():
    print("\n28. score_quantity — max_chunks = 0")
    score = score_quantity(n_valid=5, n_unique=5, max_chunks=0)
    check(score == 0.0, "max_chunks=0 → 0.0")


# ── 6. Scoring: Latency ──────────────────────────────────────────

def test_score_sync_latency_fast():
    print("\n29. score_sync_latency — Fast response (1s)")
    score = score_sync_latency(1.0, max_time=60.0)
    expected = 1.0 - (1.0 / 60.0)
    check(abs(score - expected) < 0.001, f"score = {score:.3f}")


def test_score_sync_latency_medium():
    print("\n30. score_sync_latency — Medium response (30s)")
    score = score_sync_latency(30.0, max_time=60.0)
    check(abs(score - 0.5) < 0.001, f"score = {score:.3f}")


def test_score_sync_latency_timeout():
    print("\n31. score_sync_latency — Timeout (>60s)")
    score = score_sync_latency(61.0, max_time=60.0)
    check(score == 0.0, "timeout → 0.0")


def test_score_sync_latency_zero():
    print("\n32. score_sync_latency — Zero response time")
    score = score_sync_latency(0.0)
    check(score == 0.0, "zero time → 0.0")


def test_score_sync_latency_negative():
    print("\n33. score_sync_latency — Negative time")
    score = score_sync_latency(-1.0)
    check(score == 0.0, "negative time → 0.0")


# ── 7. sync_reward — Combined Scoring ────────────────────────────

def test_sync_reward_valid_batch():
    print("\n34. sync_reward — Valid batch with unique chunks")
    chunks = [make_chunk(f"reward_chunk_{i}") for i in range(10)]
    batch_b64 = make_batch_data(chunks)
    score = sync_reward(batch_b64, known_hashes=[], max_chunks=50, response_time=5.0)
    check(score > 0.0, f"score = {score:.3f} (> 0)")
    check(score <= 1.0, f"score = {score:.3f} (<= 1)")


def test_sync_reward_no_batch():
    print("\n35. sync_reward — No batch data")
    score = sync_reward(None, known_hashes=[], max_chunks=50)
    check(score == 0.0, "no batch → 0.0")


def test_sync_reward_empty_batch():
    print("\n36. sync_reward — Empty string batch")
    score = sync_reward("", known_hashes=[], max_chunks=50)
    check(score == 0.0, "empty string → 0.0")


def test_sync_reward_all_known():
    print("\n37. sync_reward — All chunks already known")
    chunks = [make_chunk(f"known_{i}") for i in range(5)]
    known = [c.content_hash for c in chunks]
    batch_b64 = make_batch_data(chunks)
    score_known = sync_reward(batch_b64, known_hashes=known, max_chunks=50, response_time=5.0)
    score_new = sync_reward(batch_b64, known_hashes=[], max_chunks=50, response_time=5.0)
    # All-known should score lower than all-new (quantity=0 vs quantity>0)
    check(score_known < score_new, f"known ({score_known:.3f}) < new ({score_new:.3f})")


def test_sync_reward_corrupt_batch():
    print("\n38. sync_reward — Corrupt batch data")
    score = sync_reward("bm90X3ZhbGlk", known_hashes=[], max_chunks=50)
    check(score == 0.0, "corrupt batch → 0.0")


def test_sync_reward_invalid_hashes():
    print("\n39. sync_reward — Chunks with bad hashes")
    chunks = [make_chunk(f"bad_{i}") for i in range(5)]
    for c in chunks:
        c.content_hash = "corrupted"
    batch_b64 = make_batch_data(chunks)
    score = sync_reward(batch_b64, known_hashes=[], max_chunks=50)
    check(score == 0.0, "all bad hashes → 0.0")


def test_sync_reward_max_chunks_matters():
    print("\n40. sync_reward — More chunks = higher quantity score")
    small_chunks = [make_chunk(f"small_{i}") for i in range(2)]
    big_chunks = [make_chunk(f"big_{i}") for i in range(20)]
    s1 = sync_reward(make_batch_data(small_chunks), [], max_chunks=50, response_time=5.0)
    s2 = sync_reward(make_batch_data(big_chunks), [], max_chunks=50, response_time=5.0)
    check(s2 > s1, f"20 chunks ({s2:.3f}) > 2 chunks ({s1:.3f})")


def test_sync_reward_latency_matters():
    print("\n41. sync_reward — Faster response = higher score")
    chunks = [make_chunk(f"lat_{i}") for i in range(10)]
    batch_b64 = make_batch_data(chunks)
    fast = sync_reward(batch_b64, [], max_chunks=50, response_time=1.0)
    slow = sync_reward(batch_b64, [], max_chunks=50, response_time=50.0)
    check(fast > slow, f"fast ({fast:.3f}) > slow ({slow:.3f})")


# ── 8. get_sync_rewards — Batch Scoring ──────────────────────────

def test_get_sync_rewards_multiple_miners():
    print("\n42. get_sync_rewards — Multiple miners")
    c1 = [make_chunk(f"m1_{i}") for i in range(5)]
    c2 = [make_chunk(f"m2_{i}") for i in range(10)]
    responses = [
        {"batch_data": make_batch_data(c1), "chunk_count": 5, "batch_id": "b1"},
        {"batch_data": make_batch_data(c2), "chunk_count": 10, "batch_id": "b2"},
    ]
    rewards = get_sync_rewards(None, responses, [], 50, [5.0, 5.0])
    check(len(rewards) == 2, "2 rewards")
    check(rewards[1] > rewards[0], f"miner2 ({rewards[1]:.3f}) > miner1 ({rewards[0]:.3f})")


def test_get_sync_rewards_empty_response():
    print("\n43. get_sync_rewards — Empty response")
    responses = [{"batch_data": None, "chunk_count": 0, "batch_id": None}]
    rewards = get_sync_rewards(None, responses, [], 50, [5.0])
    check(len(rewards) == 1, "1 reward")
    check(rewards[0] == 0.0, "empty response → 0.0")


def test_get_sync_rewards_mixed():
    print("\n44. get_sync_rewards — Mixed valid and empty")
    chunks = [make_chunk(f"mix_{i}") for i in range(5)]
    responses = [
        {"batch_data": make_batch_data(chunks), "chunk_count": 5, "batch_id": "b1"},
        {"batch_data": None, "chunk_count": 0, "batch_id": None},
        {"batch_data": make_batch_data(chunks), "chunk_count": 5, "batch_id": "b3"},
    ]
    rewards = get_sync_rewards(None, responses, [], 50, [5.0, 5.0, 5.0])
    check(len(rewards) == 3, "3 rewards")
    check(rewards[0] > 0, "first miner scored")
    check(rewards[1] == 0.0, "second miner = 0 (no data)")
    check(rewards[2] > 0, "third miner scored")


def test_get_sync_rewards_non_dict():
    print("\n45. get_sync_rewards — Non-dict response")
    responses = ["not a dict", None, 42]
    rewards = get_sync_rewards(None, responses, [], 50, [5.0, 5.0, 5.0])
    check(len(rewards) == 3, "3 rewards")
    check(all(r == 0.0 for r in rewards), "all non-dict → 0.0")


# ── 9. _decode_and_validate_batch ─────────────────────────────────

def test_validate_batch_valid():
    print("\n46. _decode_and_validate_batch — Valid batch")
    chunks = [make_chunk(f"vb_{i}") for i in range(5)]
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, [])
    check(len(valid) == 5, f"5 valid chunks (got {len(valid)})")


def test_validate_batch_with_known():
    print("\n47. _decode_and_validate_batch — Filters known hashes")
    chunks = [make_chunk(f"filt_{i}") for i in range(5)]
    known = [chunks[0].content_hash, chunks[2].content_hash]
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, known)
    check(len(valid) == 3, f"3 valid (5 minus 2 known), got {len(valid)}")


def test_validate_batch_bad_hashes():
    print("\n48. _decode_and_validate_batch — Bad hashes filtered out")
    chunks = [make_chunk(f"bh_{i}") for i in range(4)]
    chunks[1].content_hash = "wrong"
    chunks[3].content_hash = "also_wrong"
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, [])
    check(len(valid) == 2, f"2 valid (2 bad filtered), got {len(valid)}")


def test_validate_batch_empty_content():
    print("\n49. _decode_and_validate_batch — Empty content filtered")
    chunks = [make_chunk(f"ec_{i}") for i in range(3)]
    chunks[1].content = ""
    chunks[1].content_hash = compute_content_hash("")
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, [])
    check(len(valid) == 2, f"2 valid (1 empty filtered), got {len(valid)}")


def test_validate_batch_corrupt():
    print("\n50. _decode_and_validate_batch — Corrupt batch")
    valid = _decode_and_validate_batch("bm90X3ZhbGlk", [])
    check(valid == [], "corrupt batch → empty list")


def test_validate_batch_none():
    print("\n51. _decode_and_validate_batch — None input")
    valid = _decode_and_validate_batch(None, [])
    check(valid == [], "None → empty list")


def test_validate_batch_dedup_within():
    print("\n52. _decode_and_validate_batch — Dedup within batch")
    chunk = make_chunk("duplicate_content")
    # Same chunk appears twice
    chunks = [chunk, chunk]
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, [])
    check(len(valid) == 1, f"deduped to 1 (got {len(valid)})")


# ── 10. SyncQueue Integration ────────────────────────────────────

def test_queue_add_and_retrieve():
    print("\n53. SyncQueue — Add and retrieve chunks")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"q_chunk_{i}") for i in range(5)]
        for c in chunks:
            queue.add_to_queue(c)
        pending = queue.get_pending(limit=100)
        check(len(pending) == 5, f"5 chunks in queue (got {len(pending)})")
    finally:
        queue.close()
        os.unlink(path)


def test_queue_manifest():
    print("\n54. SyncQueue — Manifest generation")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"mf_{i}") for i in range(3)]
        for c in chunks:
            queue.add_to_queue(c)
        manifest = queue.get_manifest(node_id="test")
        check(len(manifest.chunks) == 3, "manifest has 3 entries")
        hashes = manifest.chunk_hashes
        for c in chunks:
            check(c.content_hash in hashes, f"hash {c.content_hash[:8]} in manifest")
    finally:
        queue.close()
        os.unlink(path)


def test_queue_dedup():
    print("\n55. SyncQueue — Duplicate rejection")
    queue, path = make_temp_queue()
    try:
        chunk = make_chunk("dedup_test")
        check(queue.add_to_queue(chunk) == True, "first add succeeds")
        check(queue.add_to_queue(chunk) == False, "duplicate rejected")
        check(len(queue.get_pending()) == 1, "still 1 chunk")
    finally:
        queue.close()
        os.unlink(path)


def test_queue_get_by_hashes():
    print("\n56. SyncQueue — Get chunks by hashes")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"byh_{i}") for i in range(5)]
        for c in chunks:
            queue.add_to_queue(c)
        target = [chunks[1].content_hash, chunks[3].content_hash]
        result = queue.get_chunks_by_hashes(target)
        check(len(result) == 2, f"got 2 chunks (expected 2, got {len(result)})")
    finally:
        queue.close()
        os.unlink(path)


def test_queue_stats():
    print("\n57. SyncQueue — Stats")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"st_{i}") for i in range(4)]
        for c in chunks:
            queue.add_to_queue(c)
        queue.mark_synced(chunks[0].content_hash, "peer1")
        stats = queue.stats()
        check(stats["total"] == 4, "total = 4")
        check(stats["pending"] == 3, "pending = 3")
        check(stats["synced"] == 1, "synced = 1")
        check(stats["sync_events"] == 1, "sync_events = 1")
    finally:
        queue.close()
        os.unlink(path)


def test_queue_private_rejected():
    print("\n58. SyncQueue — Private chunks rejected")
    queue, path = make_temp_queue()
    try:
        chunk = make_chunk("private_test", public=False)
        try:
            queue.add_to_queue(chunk)
            check(False, "should have raised ValueError")
        except ValueError:
            check(True, "ValueError raised for private chunk")
    finally:
        queue.close()
        os.unlink(path)


# ── 11. Miner forward_sync Logic ─────────────────────────────────

def test_miner_batch_creation():
    print("\n59. Miner — Batch creation from queue")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"miner_{i}") for i in range(5)]
        for c in chunks:
            queue.add_to_queue(c)

        # Simulate what miner forward_sync does
        known_set = set()
        all_pending = queue.get_pending(limit=1000)
        new_chunks = [c for c in all_pending if c.content_hash not in known_set]
        new_chunks = new_chunks[:50]

        batch_b64 = make_batch_data(new_chunks)
        decoded = _decode_batch_chunks(batch_b64)
        check(len(decoded) == 5, f"batch has 5 chunks (got {len(decoded)})")
    finally:
        queue.close()
        os.unlink(path)


def test_miner_known_filter():
    print("\n60. Miner — Known hashes filtered")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"mkf_{i}") for i in range(5)]
        for c in chunks:
            queue.add_to_queue(c)

        known = {chunks[0].content_hash, chunks[2].content_hash, chunks[4].content_hash}
        all_pending = queue.get_pending(limit=1000)
        new_chunks = [c for c in all_pending if c.content_hash not in known]

        check(len(new_chunks) == 2, f"2 new chunks after filter (got {len(new_chunks)})")
    finally:
        queue.close()
        os.unlink(path)


def test_miner_max_chunks_limit():
    print("\n61. Miner — max_chunks limit respected")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"mcl_{i}") for i in range(20)]
        for c in chunks:
            queue.add_to_queue(c)

        all_pending = queue.get_pending(limit=1000)
        limited = all_pending[:5]
        check(len(limited) == 5, f"limited to 5 chunks (got {len(limited)})")
    finally:
        queue.close()
        os.unlink(path)


def test_miner_empty_queue():
    print("\n62. Miner — Empty queue returns no batch")
    queue, path = make_temp_queue()
    try:
        all_pending = queue.get_pending(limit=1000)
        check(len(all_pending) == 0, "empty queue → 0 chunks")
    finally:
        queue.close()
        os.unlink(path)


def test_miner_all_known():
    print("\n63. Miner — All chunks known → empty result")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"ak_{i}") for i in range(3)]
        for c in chunks:
            queue.add_to_queue(c)

        known = {c.content_hash for c in chunks}
        all_pending = queue.get_pending(limit=1000)
        new_chunks = [c for c in all_pending if c.content_hash not in known]
        check(len(new_chunks) == 0, "all known → empty")
    finally:
        queue.close()
        os.unlink(path)


# ── 12. Validator Ingestion Flow ──────────────────────────────────

def test_validator_ingest_new():
    print("\n64. Validator — Ingest new chunks from batch")
    queue, path = make_temp_queue()
    try:
        chunks = [make_chunk(f"vin_{i}") for i in range(5)]
        batch_b64 = make_batch_data(chunks)
        valid = _decode_and_validate_batch(batch_b64, [])

        for chunk in valid:
            chunk.pool_visibility = "public"
            if chunk.shared_at is None:
                chunk.shared_at = time.time()
            queue.add_to_queue(chunk)

        stats = queue.stats()
        check(stats["total"] == 5, f"ingested 5 chunks (got {stats['total']})")
    finally:
        queue.close()
        os.unlink(path)


def test_validator_ingest_dedup():
    print("\n65. Validator — Dedup across multiple miners")
    queue, path = make_temp_queue()
    try:
        # Miner 1 and Miner 2 send overlapping chunks
        c_shared = make_chunk("shared_content")
        c1_only = make_chunk("miner1_only")
        c2_only = make_chunk("miner2_only")

        batch1 = make_batch_data([c_shared, c1_only])
        batch2 = make_batch_data([c_shared, c2_only])

        known_hashes = []

        # Process miner 1
        valid1 = _decode_and_validate_batch(batch1, known_hashes)
        for c in valid1:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            if queue.add_to_queue(c):
                known_hashes.append(c.content_hash)

        # Process miner 2
        valid2 = _decode_and_validate_batch(batch2, known_hashes)
        for c in valid2:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            queue.add_to_queue(c)

        stats = queue.stats()
        check(stats["total"] == 3, f"3 unique chunks (got {stats['total']})")
    finally:
        queue.close()
        os.unlink(path)


def test_validator_ingest_corrupt_skipped():
    print("\n66. Validator — Corrupt chunks skipped during ingestion")
    queue, path = make_temp_queue()
    try:
        c_valid = make_chunk("valid_content")
        c_bad = make_chunk("bad_content")
        c_bad.content_hash = "corrupted_hash"

        batch_b64 = make_batch_data([c_valid, c_bad])
        valid = _decode_and_validate_batch(batch_b64, [])

        for c in valid:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            queue.add_to_queue(c)

        stats = queue.stats()
        check(stats["total"] == 1, f"only valid chunk ingested (got {stats['total']})")
    finally:
        queue.close()
        os.unlink(path)


# ── 13. End-to-End Tests ──────────────────────────────────────────

def test_e2e_miner_to_validator():
    print("\n67. E2E — Miner queue → batch → validator queue")
    miner_queue, mpath = make_temp_queue()
    val_queue, vpath = make_temp_queue()
    try:
        # Miner has chunks
        chunks = [make_chunk(f"e2e_{i}") for i in range(8)]
        for c in chunks:
            miner_queue.add_to_queue(c)

        # Miner creates batch (simulating forward_sync)
        known_hashes = []
        pending = miner_queue.get_pending(limit=1000)
        new = [c for c in pending if c.content_hash not in set(known_hashes)]
        new = new[:50]
        batch_b64 = make_batch_data(new)

        # Validator receives and validates
        valid = _decode_and_validate_batch(batch_b64, known_hashes)
        for c in valid:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            val_queue.add_to_queue(c)

        m_stats = miner_queue.stats()
        v_stats = val_queue.stats()
        check(m_stats["total"] == 8, f"miner has 8 chunks")
        check(v_stats["total"] == 8, f"validator got 8 chunks (got {v_stats['total']})")
    finally:
        miner_queue.close()
        val_queue.close()
        os.unlink(mpath)
        os.unlink(vpath)


def test_e2e_known_hashes_filtering():
    print("\n68. E2E — Known hashes prevent re-sending")
    miner_queue, mpath = make_temp_queue()
    val_queue, vpath = make_temp_queue()
    try:
        chunks = [make_chunk(f"khf_{i}") for i in range(5)]
        for c in chunks:
            miner_queue.add_to_queue(c)

        # Validator already has first 3
        for c in chunks[:3]:
            c_copy = make_chunk(c.content)
            c_copy.content_hash = c.content_hash
            c_copy.origin_node_id = c.origin_node_id
            c_copy.timestamp = c.timestamp
            c_copy.signature = c.signature
            val_queue.add_to_queue(c_copy)

        known = list(val_queue.get_manifest("val").chunk_hashes)

        # Miner filters
        pending = miner_queue.get_pending(limit=1000)
        new = [c for c in pending if c.content_hash not in set(known)]
        batch_b64 = make_batch_data(new)

        # Validator validates
        valid = _decode_and_validate_batch(batch_b64, known)
        for c in valid:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            val_queue.add_to_queue(c)

        v_stats = val_queue.stats()
        check(v_stats["total"] == 5, f"validator has all 5 (got {v_stats['total']})")
        check(len(new) == 2, f"miner only sent 2 new")
    finally:
        miner_queue.close()
        val_queue.close()
        os.unlink(mpath)
        os.unlink(vpath)


def test_e2e_multiple_miners():
    print("\n69. E2E — Multiple miners with overlapping content")
    val_queue, vpath = make_temp_queue()
    try:
        # Miner 1: chunks A, B, C
        m1_chunks = [make_chunk(f"m1_{i}") for i in range(3)]
        # Miner 2: chunks B, D, E (B is same content as m1_chunks[1])
        m2_chunks = [
            make_chunk(m1_chunks[1].content),  # same as B
            make_chunk("m2_unique_1"),
            make_chunk("m2_unique_2"),
        ]
        # Rebuild m2_chunks[0] so hash matches
        m2_chunks[0].content_hash = m1_chunks[1].content_hash

        batch1 = make_batch_data(m1_chunks)
        batch2 = make_batch_data(m2_chunks)

        known = []

        # Process miner 1
        valid1 = _decode_and_validate_batch(batch1, known)
        for c in valid1:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            if val_queue.add_to_queue(c):
                known.append(c.content_hash)

        # Process miner 2
        valid2 = _decode_and_validate_batch(batch2, known)
        for c in valid2:
            c.pool_visibility = "public"
            c.shared_at = c.shared_at or time.time()
            val_queue.add_to_queue(c)

        v_stats = val_queue.stats()
        # A, B, C from m1; D, E from m2 (B deduped)
        check(v_stats["total"] == 5, f"5 unique chunks (got {v_stats['total']})")
    finally:
        val_queue.close()
        os.unlink(vpath)


def test_e2e_empty_miner():
    print("\n70. E2E — Miner with no chunks → score 0")
    score = sync_reward(None, known_hashes=[], max_chunks=50, response_time=5.0)
    check(score == 0.0, "no batch → score 0")


def test_e2e_scoring_pipeline():
    print("\n71. E2E — Full scoring pipeline")
    # Multiple miners with varying quality
    good_chunks = [make_chunk(f"good_{i}") for i in range(20)]
    bad_chunks = [make_chunk(f"bad_{i}") for i in range(5)]
    for c in bad_chunks:
        c.content_hash = "corrupted"

    responses = [
        {"batch_data": make_batch_data(good_chunks), "chunk_count": 20, "batch_id": "g1"},
        {"batch_data": make_batch_data(bad_chunks), "chunk_count": 5, "batch_id": "b1"},
        {"batch_data": None, "chunk_count": 0, "batch_id": None},
    ]
    rewards = get_sync_rewards(None, responses, [], 50, [5.0, 5.0, 5.0])

    check(rewards[0] > rewards[1], f"good ({rewards[0]:.3f}) > bad ({rewards[1]:.3f})")
    check(rewards[2] == 0.0, "empty miner → 0.0")
    check(rewards[0] > 0.3, f"good miner score ({rewards[0]:.3f}) is substantial")


def test_e2e_round_trip_batch():
    print("\n72. E2E — Round-trip: create → encode → decode → validate")
    chunks = [make_chunk(f"rt_{i}") for i in range(10)]
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, [])
    check(len(valid) == 10, f"all 10 round-tripped (got {len(valid)})")
    for i, v in enumerate(valid):
        check(v.content == f"rt_{i}", f"chunk {i} content preserved")


# ── 14. Weight Constants ──────────────────────────────────────────

def test_weights_sum_to_one():
    print("\n73. Weight constants — Sum to 1.0")
    total = W_VALIDITY + W_FRESHNESS + W_QUANTITY + W_LATENCY
    check(abs(total - 1.0) < 0.001, f"weights sum = {total:.3f}")


def test_sync_interval():
    print("\n74. Constants — SYNC_INTERVAL_STEPS")
    check(SYNC_INTERVAL_STEPS == 10, f"interval = {SYNC_INTERVAL_STEPS}")


def test_max_chunks_per_sync():
    print("\n75. Constants — MAX_CHUNKS_PER_SYNC")
    check(MAX_CHUNKS_PER_SYNC == 50, f"max = {MAX_CHUNKS_PER_SYNC}")


def test_freshness_half_life():
    print("\n76. Constants — FRESHNESS_HALF_LIFE = 7 days")
    check(FRESHNESS_HALF_LIFE == 7 * 24 * 3600, f"half_life = {FRESHNESS_HALF_LIFE}")


# ── 15. Edge Cases ────────────────────────────────────────────────

def test_edge_single_chunk():
    print("\n77. Edge — Single chunk batch")
    chunk = make_chunk("solo")
    batch_b64 = make_batch_data([chunk])
    score = sync_reward(batch_b64, [], max_chunks=50, response_time=5.0)
    check(score > 0, f"single chunk scores ({score:.3f})")


def test_edge_large_batch():
    print("\n78. Edge — Large batch (100 chunks)")
    chunks = [make_chunk(f"large_{i}") for i in range(100)]
    batch_b64 = make_batch_data(chunks)
    valid = _decode_and_validate_batch(batch_b64, [])
    check(len(valid) == 100, f"100 chunks validated (got {len(valid)})")
    score = sync_reward(batch_b64, [], max_chunks=50, response_time=5.0)
    check(score > 0.5, f"large batch score ({score:.3f})")


def test_edge_whitespace_content():
    print("\n79. Edge — Whitespace-only content")
    chunk = KnowledgeChunk(
        content="   \n\t  ",
        content_hash=compute_content_hash("   \n\t  "),
        origin_node_id="test",
        timestamp=time.time(),
        signature="sig",
        pool_visibility="public",
        shared_at=time.time(),
    )
    score = score_validity([chunk])
    check(score == 0.0, "whitespace-only → invalid")


def test_edge_very_old_chunks():
    print("\n80. Edge — Very old chunks (365 days)")
    now = time.time()
    chunk = make_chunk("ancient", ts=now - 365 * 24 * 3600)
    score = score_freshness([chunk], now=now)
    check(score < 0.001, f"365-day old chunk freshness = {score:.6f}")


def test_edge_batch_data_empty_string():
    print("\n81. Edge — Batch data is empty string")
    valid = _decode_and_validate_batch("", [])
    check(valid == [], "empty string batch → empty")


# ── 16. Synapse Type Distinction ──────────────────────────────────

def test_synapse_has_known_hashes():
    print("\n82. Synapse type — has known_hashes attribute")
    s = KnowledgeSyncSynapse()
    check(hasattr(s, 'known_hashes'), "KnowledgeSyncSynapse has known_hashes")
    check(hasattr(s, 'max_chunks'), "KnowledgeSyncSynapse has max_chunks")
    check(not hasattr(s, 'query'), "KnowledgeSyncSynapse does NOT have query")
    check(not hasattr(s, 'context_chunks'), "KnowledgeSyncSynapse does NOT have context_chunks")


def test_rag_synapse_distinct():
    print("\n83. Synapse type — RAGSynapse is distinct")
    from subnet.superbrain.protocol import RAGSynapse
    r = RAGSynapse()
    check(hasattr(r, 'query'), "RAGSynapse has query")
    check(hasattr(r, 'context_chunks'), "RAGSynapse has context_chunks")
    check(not hasattr(r, 'known_hashes'), "RAGSynapse does NOT have known_hashes")


# ── 17. Score Boundaries ──────────────────────────────────────────

def test_score_bounds_validity():
    print("\n84. Bounds — score_validity always in [0, 1]")
    for n in [0, 1, 5, 10, 50]:
        chunks = [make_chunk(f"bound_v_{i}") for i in range(n)]
        s = score_validity(chunks)
        check(0.0 <= s <= 1.0, f"n={n}: {s:.3f} in [0,1]")


def test_score_bounds_freshness():
    print("\n85. Bounds — score_freshness always in [0, 1]")
    now = time.time()
    for age_days in [0, 1, 7, 30, 365]:
        chunk = make_chunk(f"bound_f_{age_days}", ts=now - age_days * 86400)
        s = score_freshness([chunk], now=now)
        check(0.0 <= s <= 1.0, f"age={age_days}d: {s:.3f} in [0,1]")


def test_score_bounds_quantity():
    print("\n86. Bounds — score_quantity always in [0, 1]")
    for v, u, m in [(0, 0, 50), (10, 10, 50), (50, 50, 50), (100, 100, 50)]:
        s = score_quantity(v, u, m)
        check(0.0 <= s <= 1.0, f"v={v},u={u},m={m}: {s:.3f} in [0,1]")


def test_score_bounds_latency():
    print("\n87. Bounds — score_sync_latency always in [0, 1]")
    for t in [-1, 0, 0.1, 30, 59.9, 60, 61, 100]:
        s = score_sync_latency(t)
        check(0.0 <= s <= 1.0, f"t={t}: {s:.3f} in [0,1]")


def test_score_bounds_sync_reward():
    print("\n88. Bounds — sync_reward always in [0, 1]")
    chunks = [make_chunk(f"bound_sr_{i}") for i in range(10)]
    batch_b64 = make_batch_data(chunks)
    for rt in [0.1, 5, 30, 59]:
        s = sync_reward(batch_b64, [], max_chunks=50, response_time=rt)
        check(0.0 <= s <= 1.0, f"rt={rt}: {s:.3f} in [0,1]")


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("KnowledgeSyncSynapse Test Suite")
    print("=" * 60)

    # 1. Synapse tests
    test_synapse_defaults()
    test_synapse_with_values()
    test_synapse_deserialize()
    test_synapse_deserialize_none()
    test_synapse_miner_fills()

    # 2. Batch decode
    test_decode_batch_valid()
    test_decode_batch_invalid_base64()
    test_decode_batch_invalid_json()
    test_decode_batch_empty()
    test_decode_batch_empty_list()

    # 3. Validity scoring
    test_score_validity_all_valid()
    test_score_validity_some_invalid()
    test_score_validity_all_invalid()
    test_score_validity_empty()
    test_score_validity_empty_content()
    test_score_validity_mixed()

    # 4. Freshness scoring
    test_score_freshness_recent()
    test_score_freshness_old()
    test_score_freshness_very_old()
    test_score_freshness_mixed_ages()
    test_score_freshness_empty()
    test_score_freshness_future_timestamp()

    # 5. Quantity scoring
    test_score_quantity_full()
    test_score_quantity_partial()
    test_score_quantity_zero()
    test_score_quantity_unique_less_than_valid()
    test_score_quantity_overflow()
    test_score_quantity_max_zero()

    # 6. Latency scoring
    test_score_sync_latency_fast()
    test_score_sync_latency_medium()
    test_score_sync_latency_timeout()
    test_score_sync_latency_zero()
    test_score_sync_latency_negative()

    # 7. Combined scoring
    test_sync_reward_valid_batch()
    test_sync_reward_no_batch()
    test_sync_reward_empty_batch()
    test_sync_reward_all_known()
    test_sync_reward_corrupt_batch()
    test_sync_reward_invalid_hashes()
    test_sync_reward_max_chunks_matters()
    test_sync_reward_latency_matters()

    # 8. Batch scoring
    test_get_sync_rewards_multiple_miners()
    test_get_sync_rewards_empty_response()
    test_get_sync_rewards_mixed()
    test_get_sync_rewards_non_dict()

    # 9. Batch validation
    test_validate_batch_valid()
    test_validate_batch_with_known()
    test_validate_batch_bad_hashes()
    test_validate_batch_empty_content()
    test_validate_batch_corrupt()
    test_validate_batch_none()
    test_validate_batch_dedup_within()

    # 10. Queue integration
    test_queue_add_and_retrieve()
    test_queue_manifest()
    test_queue_dedup()
    test_queue_get_by_hashes()
    test_queue_stats()
    test_queue_private_rejected()

    # 11. Miner logic
    test_miner_batch_creation()
    test_miner_known_filter()
    test_miner_max_chunks_limit()
    test_miner_empty_queue()
    test_miner_all_known()

    # 12. Validator ingestion
    test_validator_ingest_new()
    test_validator_ingest_dedup()
    test_validator_ingest_corrupt_skipped()

    # 13. End-to-end
    test_e2e_miner_to_validator()
    test_e2e_known_hashes_filtering()
    test_e2e_multiple_miners()
    test_e2e_empty_miner()
    test_e2e_scoring_pipeline()
    test_e2e_round_trip_batch()

    # 14. Constants
    test_weights_sum_to_one()
    test_sync_interval()
    test_max_chunks_per_sync()
    test_freshness_half_life()

    # 15. Edge cases
    test_edge_single_chunk()
    test_edge_large_batch()
    test_edge_whitespace_content()
    test_edge_very_old_chunks()
    test_edge_batch_data_empty_string()

    # 16. Type distinction
    test_synapse_has_known_hashes()
    test_rag_synapse_distinct()

    # 17. Score boundaries
    test_score_bounds_validity()
    test_score_bounds_freshness()
    test_score_bounds_quantity()
    test_score_bounds_latency()
    test_score_bounds_sync_reward()

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 60)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failed else 0)
