# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Reward Function — 4-Factor Scoring (V2: Embedding-Enhanced)
#
# Final Score = (0.40 x Supportedness) + (0.25 x Relevance) + (0.20 x Novelty) + (0.15 x Latency)
# V2: When semantic embeddings are available, uses cosine similarity instead of word overlap.
#     Adds length penalty (multiplier) and citation quality (sub-factor of supportedness).
#     When no embedding backend is available, behavior is identical to V1.

import hashlib
import numpy as np
from typing import List, Optional

try:
    import bittensor as bt
    _has_bt = True
except ImportError:
    _has_bt = False

from superbrain.validator.embeddings import (
    cosine_similarity as _embed_similarity,
    batch_cosine_similarity as _batch_embed_similarity,
    is_semantic as _is_semantic,
    get_backend as _get_embedding_backend,
)

# Weight Constants — FINAL, DO NOT CHANGE
W_SUPPORTEDNESS = 0.40
W_RELEVANCE = 0.25
W_NOVELTY = 0.20
W_LATENCY = 0.15


def _word_set(text: str) -> set:
    """Lowercase word set for overlap scoring."""
    return set(text.lower().split())


# Individual Scoring Functions

def score_supportedness(response: str, context_chunks: List[str], citations: List[int]) -> float:
    """
    40% — Does the answer cite actual source chunks? Are claims backed by documents?
    V2: Uses embedding similarity when available, with citation quality sub-factor.
    V1 fallback: Word overlap with 30% hallucination tolerance (exact backward compat).
    """
    if not response or not response.strip():
        return 0.0
    if not citations:
        return 0.0

    valid = [i for i in citations if 0 <= i < len(context_chunks)]
    if not valid:
        return 0.0

    cited_text = " ".join([context_chunks[i] for i in valid])

    if _is_semantic():
        # V2 path: embedding-based supportedness + citation quality blend
        similarity = _embed_similarity(response, cited_text)
        hallucination_penalty = max(0, (1.0 - similarity) - 0.3)
        base_score = max(0.0, min(1.0, similarity - hallucination_penalty))
        cq = score_citation_quality(valid, context_chunks, response)
        return 0.7 * base_score + 0.3 * cq
    else:
        # V1 path: exact backward-compatible word overlap
        response_words = _word_set(response)
        cited_words = _word_set(cited_text)
        if not response_words:
            return 0.0
        overlap = len(response_words & cited_words) / len(response_words)
        uncited_ratio = 1.0 - overlap
        hallucination_penalty = max(0, uncited_ratio - 0.3)
        return max(0.0, min(1.0, overlap - hallucination_penalty))


def score_relevance(query: str, context_chunks: List[str], citations: List[int]) -> float:
    """
    25% — Are the retrieved/cited chunks actually relevant to the query?
    V2: Mean embedding similarity between query and each cited chunk.
    V1 fallback: Word overlap between query and joined cited text.
    """
    if not query or not query.strip():
        return 0.0
    if not citations:
        return 0.0

    valid = [i for i in citations if 0 <= i < len(context_chunks)]
    if not valid:
        return 0.0

    if _is_semantic():
        # V2 path: mean embedding similarity per cited chunk
        cited_chunks = [context_chunks[i] for i in valid]
        sims = _batch_embed_similarity(query, cited_chunks)
        return float(max(0.0, min(1.0, sum(sims) / len(sims)))) if sims else 0.0
    else:
        # V1 path: exact backward-compatible word overlap
        cited_text = " ".join([context_chunks[i] for i in valid])
        query_words = _word_set(query)
        cited_words = _word_set(cited_text)
        if not query_words:
            return 0.0
        return len(query_words & cited_words) / len(query_words)


def score_novelty(response: str, previous_responses: List[str]) -> float:
    """
    20% — Is this miner's response unique compared to other miners' responses?
    Uses SHA-256 exact duplicate detection + word overlap for paraphrase detection.
    """
    if not response or not response.strip():
        return 0.0
    if not previous_responses:
        return 1.0  # First response is always novel

    # Exact duplicate detection via SHA-256
    response_hash = hashlib.sha256(response.strip().encode()).hexdigest()
    for prev in previous_responses:
        if hashlib.sha256(prev.strip().encode()).hexdigest() == response_hash:
            return 0.0  # Exact duplicate

    # Word-overlap paraphrase detection
    response_words = _word_set(response)
    all_prev_words = set()
    for prev in previous_responses:
        all_prev_words.update(_word_set(prev))

    if not response_words:
        return 0.0

    unique_ratio = len(response_words - all_prev_words) / len(response_words)
    return unique_ratio


def score_latency(response_time: float, max_time: float = 30.0) -> float:
    """
    15% — How fast did the miner respond? Faster = higher score.
    """
    if response_time <= 0 or response_time > max_time:
        return 0.0
    return 1.0 - (response_time / max_time)


def score_citation_quality(valid_citations: List[int], context_chunks: List[str], response: str) -> float:
    """
    V2 — Bonus for citing multiple relevant chunks. Returns 0.0-1.0.
    Used as a 30% sub-factor of supportedness when semantic backend is active.
    """
    n = len(valid_citations)
    if n == 0:
        return 0.0

    # Base score from citation count (diminishing returns)
    if n == 1:
        count_score = 0.3
    elif n == 2:
        count_score = 0.6
    else:
        count_score = min(1.0, 0.8 + 0.05 * (n - 3))

    # When semantic embeddings available, weight by relevance of each cited chunk
    if _is_semantic():
        cited_chunks = [context_chunks[i] for i in valid_citations]
        sims = _batch_embed_similarity(response, cited_chunks)
        if sims:
            avg_relevance = sum(sims) / len(sims)
            return count_score * avg_relevance

    return count_score


def score_length_penalty(response: str, min_chars: int = 50, max_chars: int = 2000) -> float:
    """
    V2 — Penalize responses that are too short or too long.
    Returns a multiplier in [0.0, 1.0].
    Applied as post-hoc multiplier on final score (only when semantic backend active).
    """
    if not response:
        return 0.0
    length = len(response.strip())
    if length == 0:
        return 0.0
    if length < min_chars:
        return length / min_chars
    if length <= max_chars:
        return 1.0
    # Gradual decay above max_chars
    excess = length - max_chars
    return max(0.3, 1.0 - (excess / (2 * max_chars)))


# Single Response Reward

def reward(
    query: str,
    context_chunks: List[str],
    response: dict,
    previous_responses: List[str],
    response_time: float = 5.0,
) -> float:
    """
    Score a single miner response. Returns a float in [0.0, 1.0].

    Args:
        query: The question sent to the miner.
        context_chunks: The document chunks sent as context.
        response: Deserialized response dict with 'response', 'citations', 'confidence_score'.
        previous_responses: Responses from other miners (for novelty).
        response_time: How long the miner took.
    """
    resp_text = response.get("response", "") or ""
    citations = response.get("citations", []) or []

    if not resp_text.strip():
        return 0.0

    s = score_supportedness(resp_text, context_chunks, citations)
    r = score_relevance(query, context_chunks, citations)
    n = score_novelty(resp_text, previous_responses)
    l = score_latency(response_time)

    final = (W_SUPPORTEDNESS * s) + (W_RELEVANCE * r) + (W_NOVELTY * n) + (W_LATENCY * l)

    # V2: Apply length penalty as multiplier (only when semantic backend active)
    if _is_semantic():
        lp = score_length_penalty(resp_text)
        final *= lp

    if _has_bt:
        backend = _get_embedding_backend()
        extra = f" LP={lp:.3f}" if _is_semantic() else ""
        bt.logging.debug(
            f"Scores: S={s:.3f} R={r:.3f} N={n:.3f} L={l:.3f}{extra} -> Final={final:.3f} [{backend}]"
        )

    return final


# Batch Reward (called by forward)

def get_rewards(
    self,
    query: str,
    context_chunks: List[str],
    responses: List[dict],
    response_times: List[float],
) -> np.ndarray:
    """
    Returns an array of rewards for all miner responses in one epoch.

    Args:
        self: The validator instance (for accessing config if needed).
        query: The question sent to miners.
        context_chunks: Document chunks sent as context.
        responses: List of deserialized response dicts from miners.
        response_times: List of response times (seconds) per miner.

    Returns:
        np.ndarray of shape (N,) with scores in [0.0, 1.0].
    """
    previous_responses = []
    rewards = []

    for i, (resp, rtime) in enumerate(zip(responses, response_times)):
        resp_text = resp.get("response", "") if isinstance(resp, dict) else ""

        if not resp_text or not resp_text.strip():
            rewards.append(0.0)
            continue

        score = reward(
            query=query,
            context_chunks=context_chunks,
            response=resp,
            previous_responses=previous_responses,
            response_time=rtime,
        )
        rewards.append(score)

        # Track for novelty scoring of subsequent miners
        if resp_text.strip():
            previous_responses.append(resp_text)

    return np.array(rewards, dtype=np.float32)
