# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Sync Reward Function — 4-Factor Scoring for Knowledge Sync
#
# Final Score = (0.35 x Validity) + (0.25 x Freshness) + (0.25 x Quantity) + (0.15 x Latency)
#
# Miners are rewarded for contributing valid, fresh, unique knowledge chunks
# via the KnowledgeSyncSynapse protocol.

import base64
import json
import math
import time
import zlib

import numpy as np

from typing import List, Optional

try:
    import bittensor as bt
    _has_bt = True
except ImportError:
    _has_bt = False

try:
    from sync.protocol.pool_model import KnowledgeChunk, compute_content_hash
except ImportError:
    # sync package not on path — define minimal stubs for import safety
    KnowledgeChunk = None
    compute_content_hash = None

# Weight Constants
W_VALIDITY = 0.35
W_FRESHNESS = 0.25
W_QUANTITY = 0.25
W_LATENCY = 0.15

# Freshness half-life: 7 days in seconds
FRESHNESS_HALF_LIFE = 7 * 24 * 3600


def _decode_batch_chunks(batch_data_b64: str) -> List[KnowledgeChunk]:
    """
    Decode base64 SyncBatch data into KnowledgeChunks.
    Returns empty list on any decode/decompress error.
    """
    try:
        compressed = base64.b64decode(batch_data_b64)
        raw_json = zlib.decompress(compressed)
        chunk_dicts = json.loads(raw_json)
        return [KnowledgeChunk.model_validate(d) for d in chunk_dicts]
    except Exception:
        return []


def score_validity(chunks: List[KnowledgeChunk]) -> float:
    """
    35% — Are all chunks well-formed with correct SHA-256 hashes?
    Returns ratio of valid chunks to total chunks.
    """
    if not chunks:
        return 0.0

    valid = 0
    for chunk in chunks:
        if not chunk.content or not chunk.content.strip():
            continue
        if not chunk.content_hash:
            continue
        computed = compute_content_hash(chunk.content)
        if computed == chunk.content_hash:
            valid += 1

    return valid / len(chunks)


def score_freshness(chunks: List[KnowledgeChunk], now: Optional[float] = None) -> float:
    """
    25% — How recent are the chunks? Exponential decay with 7-day half-life.
    Returns average freshness score across all chunks.
    """
    if not chunks:
        return 0.0

    if now is None:
        now = time.time()

    scores = []
    for chunk in chunks:
        age = max(0, now - chunk.timestamp)
        # Exponential decay: score = 2^(-age / half_life)
        score = math.pow(2, -age / FRESHNESS_HALF_LIFE)
        scores.append(score)

    return sum(scores) / len(scores)


def score_quantity(n_valid: int, n_unique: int, max_chunks: int) -> float:
    """
    25% — How many valid, unique chunks did the miner provide?
    Uses logarithmic diminishing returns to discourage chunk spam.
    First batch of chunks scores at full weight; additional chunks score less.
    """
    if max_chunks <= 0:
        return 0.0
    effective = min(n_valid, n_unique)
    if effective <= 0:
        return 0.0
    # Logarithmic diminishing returns: log(1 + effective) / log(1 + max_chunks)
    return min(1.0, math.log(1 + effective) / math.log(1 + max_chunks))


def score_sync_latency(response_time: float, max_time: float = 60.0) -> float:
    """
    15% — How fast did the miner respond? Faster = higher score.
    """
    if response_time <= 0 or response_time > max_time:
        return 0.0
    return 1.0 - (response_time / max_time)


def sync_reward(
    batch_data: Optional[str],
    known_hashes: List[str],
    max_chunks: int,
    response_time: float = 5.0,
) -> float:
    """
    Score a single miner's sync response. Returns float in [0.0, 1.0].
    """
    if not batch_data:
        return 0.0

    chunks = _decode_batch_chunks(batch_data)
    if not chunks:
        return 0.0

    # Count valid chunks (correct hashes)
    valid_chunks = []
    for chunk in chunks:
        if (chunk.content and chunk.content.strip() and
                chunk.content_hash and
                compute_content_hash(chunk.content) == chunk.content_hash):
            valid_chunks.append(chunk)

    n_valid = len(valid_chunks)
    if n_valid == 0:
        return 0.0

    # Count unique chunks (not in known_hashes)
    known_set = set(known_hashes)
    unique_chunks = [c for c in valid_chunks if c.content_hash not in known_set]
    n_unique = len(unique_chunks)

    v = score_validity(chunks)
    f = score_freshness(valid_chunks)
    q = score_quantity(n_valid, n_unique, max_chunks)
    l = score_sync_latency(response_time)

    final = (W_VALIDITY * v) + (W_FRESHNESS * f) + (W_QUANTITY * q) + (W_LATENCY * l)

    if _has_bt:
        bt.logging.debug(
            f"Sync scores: V={v:.3f} F={f:.3f} Q={q:.3f} L={l:.3f} -> Final={final:.3f} "
            f"[{n_valid} valid, {n_unique} unique of {len(chunks)} chunks]"
        )

    return final


def get_sync_rewards(
    self,
    responses: List[dict],
    known_hashes: List[str],
    max_chunks: int,
    response_times: List[float],
) -> np.ndarray:
    """
    Returns an array of rewards for all miner sync responses in one epoch.

    Args:
        self: The validator instance.
        responses: List of deserialized response dicts from miners.
        known_hashes: Chunk hashes the validator already has.
        max_chunks: Maximum chunks requested.
        response_times: Response times per miner.

    Returns:
        np.ndarray of shape (N,) with scores in [0.0, 1.0].
    """
    rewards = []

    for resp, rtime in zip(responses, response_times):
        batch_data = resp.get("batch_data") if isinstance(resp, dict) else None

        if not batch_data:
            rewards.append(0.0)
            continue

        score = sync_reward(
            batch_data=batch_data,
            known_hashes=known_hashes,
            max_chunks=max_chunks,
            response_time=rtime,
        )
        rewards.append(score)

    return np.array(rewards, dtype=np.float32)
