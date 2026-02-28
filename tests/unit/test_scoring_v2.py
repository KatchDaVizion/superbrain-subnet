"""SuperBrain Scoring V2 Test Suite

Tests embedding layer, length penalty, citation quality, backend fallback,
and backward compatibility with V1 scoring.
Works regardless of which embedding backend is available.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from superbrain.validator.embeddings import (
    cosine_similarity,
    batch_cosine_similarity,
    get_backend,
    is_semantic,
    word_overlap,
    BACKEND_WORDOVERLAP,
    BACKEND_TFIDF,
    BACKEND_OLLAMA,
)
from superbrain.validator.reward import (
    score_supportedness,
    score_relevance,
    score_novelty,
    score_latency,
    score_length_penalty,
    score_citation_quality,
    reward,
    get_rewards,
    _word_set,
    W_SUPPORTEDNESS,
    W_RELEVANCE,
    W_NOVELTY,
    W_LATENCY,
)

passed = failed = 0

QUERY = "What is Bittensor and how does it work?"
CHUNKS = [
    "Bittensor is a decentralized machine learning network that creates an open marketplace for AI models. It uses a blockchain-based incentive mechanism to reward miners who contribute useful intelligence.",
    "The Bittensor network operates through subnets, each focused on a specific task like text generation, image recognition, or data storage. Validators evaluate miner outputs and set weights.",
    "TAO is the native cryptocurrency of the Bittensor network. It is used to incentivize miners, stake on validators, and govern subnet creation.",
]


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


# ── 1. Embedding backend detection ──────────────────────────────────

def test_backend_detection():
    print("\n1. Embedding backend detection...")
    backend = get_backend()
    check(backend in (BACKEND_OLLAMA, BACKEND_TFIDF, BACKEND_WORDOVERLAP),
          f"Backend is valid: {backend}")
    check(isinstance(is_semantic(), bool), "is_semantic returns bool")
    if backend == BACKEND_WORDOVERLAP:
        check(not is_semantic(), "wordoverlap is not semantic")
    else:
        check(is_semantic(), f"{backend} is semantic")
    print(f"    Active backend: {backend}")


# ── 2. Word overlap baseline ────────────────────────────────────────

def test_word_overlap():
    print("\n2. Word overlap baseline...")
    # Matches V1 _word_set behavior
    a = "Bittensor is a decentralized network"
    b = "Bittensor network uses blockchain"
    sim = word_overlap(a, b)
    check(sim > 0.0, f"Overlap > 0: {sim:.3f}")
    check(sim <= 1.0, "Overlap <= 1.0")

    # Self-overlap
    check(word_overlap(a, a) == 1.0, "Self-overlap = 1.0")
    # Empty
    check(word_overlap("", "something") == 0.0, "Empty a = 0.0")
    check(word_overlap("something", "") == 0.0, "Empty b = 0.0")

    # Match V1 _word_set exactly
    v1_words_a = _word_set(a)
    v1_words_b = _word_set(b)
    v1_overlap = len(v1_words_a & v1_words_b) / len(v1_words_a)
    check(abs(sim - v1_overlap) < 1e-10, f"Matches V1 _word_set: {sim} == {v1_overlap}")


# ── 3. cosine_similarity basic ──────────────────────────────────────

def test_cosine_similarity():
    print("\n3. cosine_similarity basic...")
    # Empty strings
    check(cosine_similarity("", "test") == 0.0, "Empty a = 0.0")
    check(cosine_similarity("test", "") == 0.0, "Empty b = 0.0")
    check(cosine_similarity("", "") == 0.0, "Both empty = 0.0")

    # Identical
    sim = cosine_similarity("Bittensor is great", "Bittensor is great")
    check(sim >= 0.99, f"Identical texts ~ 1.0: {sim:.3f}")

    # Related
    sim_related = cosine_similarity(
        "Bittensor is a decentralized AI network",
        "Bittensor network for machine learning"
    )
    check(sim_related > 0.0, f"Related > 0: {sim_related:.3f}")

    # Unrelated
    sim_unrelated = cosine_similarity(
        "Bittensor is a blockchain network",
        "The cat sat on the mat"
    )
    check(sim_related > sim_unrelated, f"Related ({sim_related:.3f}) > Unrelated ({sim_unrelated:.3f})")

    # Returns [0, 1]
    check(0.0 <= sim_related <= 1.0, "In range [0, 1]")


# ── 4. batch_cosine_similarity ──────────────────────────────────────

def test_batch_cosine_similarity():
    print("\n4. batch_cosine_similarity...")
    sims = batch_cosine_similarity(QUERY, CHUNKS)
    check(len(sims) == 3, f"Length matches chunks: {len(sims)}")
    check(all(0.0 <= s <= 1.0 for s in sims), "All in [0, 1]")
    check(any(s > 0.0 for s in sims), "At least one > 0")

    # Empty query
    empty_sims = batch_cosine_similarity("", CHUNKS)
    check(all(s == 0.0 for s in empty_sims), "Empty query = all zeros")

    # Empty chunks list
    no_sims = batch_cosine_similarity(QUERY, [])
    check(len(no_sims) == 0, "Empty chunks = empty list")


# ── 5. score_length_penalty ─────────────────────────────────────────

def test_length_penalty():
    print("\n5. score_length_penalty...")
    # Empty
    check(score_length_penalty("") == 0.0, "Empty = 0.0")
    check(score_length_penalty("   ") == 0.0, "Whitespace = 0.0")

    # Too short (< 50 chars)
    short_10 = score_length_penalty("a" * 10)
    check(abs(short_10 - 0.2) < 0.01, f"10 chars = {short_10:.2f} (~0.2)")

    short_25 = score_length_penalty("a" * 25)
    check(abs(short_25 - 0.5) < 0.01, f"25 chars = {short_25:.2f} (~0.5)")

    # Good length (50-2000)
    check(score_length_penalty("a" * 50) == 1.0, "50 chars = 1.0")
    check(score_length_penalty("a" * 500) == 1.0, "500 chars = 1.0")
    check(score_length_penalty("a" * 2000) == 1.0, "2000 chars = 1.0")

    # Too long (> 2000)
    long_3000 = score_length_penalty("a" * 3000)
    check(0.5 < long_3000 < 1.0, f"3000 chars = {long_3000:.2f} (between 0.5 and 1.0)")

    long_5000 = score_length_penalty("a" * 5000)
    check(0.0 < long_5000 < long_3000, f"5000 chars = {long_5000:.2f} (less than 3000)")

    # Monotonically decreasing above 2000
    check(long_3000 > long_5000, "Longer = lower penalty")


# ── 6. score_citation_quality ────────────────────────────────────────

def test_citation_quality():
    print("\n6. score_citation_quality...")
    response = "Bittensor is a decentralized network using subnets and TAO."

    # No citations
    check(score_citation_quality([], CHUNKS, response) == 0.0, "0 citations = 0.0")

    # 1 citation
    cq1 = score_citation_quality([0], CHUNKS, response)
    check(cq1 > 0.0, f"1 citation > 0: {cq1:.3f}")

    # 2 citations
    cq2 = score_citation_quality([0, 1], CHUNKS, response)
    check(cq2 > cq1, f"2 citations ({cq2:.3f}) > 1 citation ({cq1:.3f})")

    # 3 citations
    cq3 = score_citation_quality([0, 1, 2], CHUNKS, response)
    check(cq3 > cq2, f"3 citations ({cq3:.3f}) > 2 citations ({cq2:.3f})")

    # All in range
    check(0.0 <= cq3 <= 1.0, f"In range [0, 1]: {cq3:.3f}")


# ── 7. V1 backward compatibility ────────────────────────────────────

def test_v1_backward_compat():
    print("\n7. V1 backward compatibility...")
    backend = get_backend()

    if backend == BACKEND_WORDOVERLAP:
        # These must match EXACTLY the V1 baseline output
        good = "Bittensor is a decentralized machine learning network [1] that uses subnets [2]. TAO is the native token [3]."
        s = score_supportedness(good, CHUNKS, [0, 1, 2])
        check(abs(s - 0.706) < 0.001, f"Supportedness = {s:.3f} (V1: 0.706)")

        r = score_relevance(QUERY, CHUNKS, [0, 1])
        check(abs(r - 0.500) < 0.001, f"Relevance = {r:.3f} (V1: 0.500)")

        # Combined score — V1 uses default response_time=5.0
        resp = {"response": "Bittensor is a decentralized network [1] using subnets [2] with TAO [3].",
                "citations": [0, 1, 2], "confidence_score": 0.9}
        score = reward(QUERY, CHUNKS, resp, [])  # default response_time=5.0
        check(abs(score - 0.5700) < 0.001, f"Combined = {score:.4f} (V1: 0.5700)")

        print("    (Backend is wordoverlap — exact V1 match verified)")
    else:
        print(f"    (Backend is {backend} — exact V1 match not applicable)")
        # Just verify directional correctness
        good = "Bittensor is a decentralized machine learning network [1] that uses subnets [2]."
        s = score_supportedness(good, CHUNKS, [0, 1, 2])
        check(s > 0.0, f"Supportedness > 0 with semantic backend: {s:.3f}")


# ── 8. V2 directional ordering ──────────────────────────────────────

def test_v2_directional():
    print("\n8. V2 directional ordering...")
    good_resp = {
        "response": "Bittensor is a decentralized network [1] using subnets [2] with TAO [3].",
        "citations": [0, 1, 2],
        "confidence_score": 0.9,
    }
    bad_resp = {
        "response": "I don't know the answer.",
        "citations": [],
        "confidence_score": 0.1,
    }
    empty_resp = {
        "response": "",
        "citations": [],
        "confidence_score": 0.0,
    }
    halluc_resp = {
        "response": "Bitcoin was created by Satoshi Nakamoto in 2009 as a peer-to-peer payment system.",
        "citations": [0],
        "confidence_score": 0.5,
    }

    good_score = reward(QUERY, CHUNKS, good_resp, [], 2.0)
    bad_score = reward(QUERY, CHUNKS, bad_resp, [], 5.0)
    empty_score = reward(QUERY, CHUNKS, empty_resp, [], 30.0)
    halluc_score = reward(QUERY, CHUNKS, halluc_resp, [], 3.0)

    check(good_score > bad_score, f"Good ({good_score:.4f}) > Bad ({bad_score:.4f})")
    check(bad_score > empty_score, f"Bad ({bad_score:.4f}) > Empty ({empty_score:.4f})")
    check(empty_score == 0.0, f"Empty = 0.0: {empty_score:.4f}")
    check(good_score > halluc_score, f"Good ({good_score:.4f}) > Hallucinated ({halluc_score:.4f})")


# ── 9. Batch integration ────────────────────────────────────────────

def test_batch_integration():
    print("\n9. Batch integration...")
    resps = [
        {"response": "Bittensor is decentralized [1] with subnets [2] and TAO [3].", "citations": [0, 1, 2]},
        {"response": "I don't know.", "citations": [0]},
        {},
    ]
    rewards = get_rewards(None, QUERY, CHUNKS, resps, [2.0, 5.0, 30.0])
    check(len(rewards) == 3, f"Length = {len(rewards)}")
    check(rewards[0] > rewards[1], f"Best > worst: {rewards[0]:.4f} > {rewards[1]:.4f}")
    check(rewards[2] == 0.0, f"Empty = 0.0: {rewards[2]:.4f}")
    check(all(0.0 <= r <= 1.0 for r in rewards), "All in [0, 1]")


# ── 10. Anti-gaming still works ─────────────────────────────────────

def test_anti_gaming():
    print("\n10. Anti-gaming (novelty unchanged)...")
    dup = "Bittensor is a decentralized network for AI with TAO tokens."

    # Novelty is always word-based (not affected by embedding backend)
    check(score_novelty(dup, []) == 1.0, "First = 1.0")
    check(score_novelty(dup, [dup]) == 0.0, "Exact dup = 0.0")

    # Different response has high novelty
    diff = "The ocean contains many marine species and diverse ecosystems."
    diff_novelty = score_novelty(diff, [dup])
    check(diff_novelty > 0.5, f"Different = {diff_novelty:.3f} > 0.5")

    # Batch: duplicates penalized
    resps = [{"response": dup, "citations": [0]}] * 3
    r = get_rewards(None, QUERY, CHUNKS, resps, [2.0] * 3)
    check(r[0] >= r[1], f"First ({r[0]:.4f}) >= dup ({r[1]:.4f})")


# ── 11. Weight constants frozen ─────────────────────────────────────

def test_weights_frozen():
    print("\n11. Weight constants frozen...")
    total = W_SUPPORTEDNESS + W_RELEVANCE + W_NOVELTY + W_LATENCY
    check(abs(total - 1.0) < 0.001, f"Sum = {total:.2f}")
    check(W_SUPPORTEDNESS == 0.40, f"S = {W_SUPPORTEDNESS}")
    check(W_RELEVANCE == 0.25, f"R = {W_RELEVANCE}")
    check(W_NOVELTY == 0.20, f"N = {W_NOVELTY}")
    check(W_LATENCY == 0.15, f"L = {W_LATENCY}")


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Scoring V2 Tests")
    print(f"  Backend: {get_backend()}")
    print("=" * 60)

    test_backend_detection()
    test_word_overlap()
    test_cosine_similarity()
    test_batch_cosine_similarity()
    test_length_penalty()
    test_citation_quality()
    test_v1_backward_compat()
    test_v2_directional()
    test_batch_integration()
    test_anti_gaming()
    test_weights_frozen()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
