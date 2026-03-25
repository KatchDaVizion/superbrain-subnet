"""
Tests for V3 anti-gaming scoring rules in reward.py (Mission 2).

Covers:
  Rule 1 — Relevance gate  (< 15% vocab overlap → score 0.0)
  Rule 2 — Word-length penalty (< 10 or > 2000 words → 0.7× multiplier)
  Rule 3 — Citation validity (out-of-range indices stripped → supportedness 0)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from superbrain.validator.reward import (
    check_relevance_gate,
    apply_word_length_penalty,
    check_citation_validity,
    reward,
    RELEVANCE_GATE_THRESHOLD,
    WORD_LENGTH_MIN,
    WORD_LENGTH_MAX,
    WORD_LENGTH_PENALTY,
)

CONTEXT = [
    "Bittensor is a decentralized machine learning network that rewards miners with TAO tokens.",
    "Validators evaluate miner outputs and set weights to determine token rewards.",
    "The Yuma Consensus aggregates validator weights using stake-weighted averaging.",
]
QUERY = "What is Bittensor and how does it reward miners?"

# A response with clear vocabulary overlap with the query+context
RELEVANT_RESPONSE = (
    "Bittensor is a decentralized network where miners contribute intelligence and earn TAO. "
    "Validators set weights to determine reward distribution."
)

# A response with no overlap whatsoever
IRRELEVANT_RESPONSE = (
    "The recipe calls for two cups of flour, one egg, and a pinch of salt. "
    "Mix well and bake at 350 degrees for 30 minutes until golden."
)

# ---------------------------------------------------------------------------
# Rule 1 — Relevance gate
# ---------------------------------------------------------------------------

def test_relevance_gate_passes_on_relevant_response():
    assert check_relevance_gate(RELEVANT_RESPONSE, QUERY, CONTEXT) is True

def test_relevance_gate_fails_on_irrelevant_response():
    assert check_relevance_gate(IRRELEVANT_RESPONSE, QUERY, CONTEXT) is False

def test_relevance_gate_fails_on_empty_response():
    assert check_relevance_gate("", QUERY, CONTEXT) is False

def test_relevance_gate_fails_on_whitespace_only():
    assert check_relevance_gate("   ", QUERY, CONTEXT) is False

def test_relevance_gate_threshold_boundary():
    """A response that exactly matches the threshold word fraction should pass."""
    # Build a response that shares exactly 15% of its words with context
    context_words = list(set(QUERY.lower().split() + " ".join(CONTEXT).lower().split()))
    total_words = 20
    overlap_needed = max(1, int(total_words * RELEVANCE_GATE_THRESHOLD))
    response_words = context_words[:overlap_needed] + ["zzz_unique"] * (total_words - overlap_needed)
    response = " ".join(response_words)
    # Should pass since overlap / total == RELEVANCE_GATE_THRESHOLD
    assert check_relevance_gate(response, QUERY, CONTEXT) is True

def test_relevance_gate_forces_zero_score():
    """When gate fails, full reward() must return 0.0."""
    response = {"response": IRRELEVANT_RESPONSE, "citations": [0], "confidence_score": 0.9}
    score = reward(QUERY, CONTEXT, response, [], response_time=1.0)
    assert score == 0.0, f"Expected 0.0, got {score}"

# ---------------------------------------------------------------------------
# Rule 2 — Word-length sanity penalty
# ---------------------------------------------------------------------------

def test_word_length_penalty_normal_response():
    """Responses in the 10–2000 word range get no penalty (multiplier = 1.0)."""
    normal = " ".join(["word"] * 50)
    assert apply_word_length_penalty(normal) == 1.0

def test_word_length_penalty_too_short():
    """Fewer than 10 words → WORD_LENGTH_PENALTY multiplier."""
    short = "Too short response here."  # 4 words
    assert apply_word_length_penalty(short) == WORD_LENGTH_PENALTY

def test_word_length_penalty_exactly_min():
    """Exactly WORD_LENGTH_MIN words → no penalty."""
    at_min = " ".join(["word"] * WORD_LENGTH_MIN)
    assert apply_word_length_penalty(at_min) == 1.0

def test_word_length_penalty_too_long():
    """More than 2000 words → WORD_LENGTH_PENALTY multiplier."""
    long_resp = " ".join(["word"] * (WORD_LENGTH_MAX + 1))
    assert apply_word_length_penalty(long_resp) == WORD_LENGTH_PENALTY

def test_word_length_penalty_exactly_max():
    """Exactly WORD_LENGTH_MAX words → no penalty."""
    at_max = " ".join(["word"] * WORD_LENGTH_MAX)
    assert apply_word_length_penalty(at_max) == 1.0

def test_word_length_penalty_empty():
    """Empty string → penalty (no words = below minimum)."""
    assert apply_word_length_penalty("") == WORD_LENGTH_PENALTY

def test_word_length_penalty_applied_in_reward():
    """Short response must score lower than an equivalent normal-length response."""
    short_text = "Bittensor miners earn TAO."  # 5 words — triggers penalty
    # Make it long enough to pass relevance gate but short word count
    short_resp = {"response": short_text, "citations": [0], "confidence_score": 0.5}

    normal_text = (
        "Bittensor is a decentralized network where miners contribute intelligence and earn TAO tokens. "
        "Validators evaluate miner outputs and set weights to determine reward distribution via Yuma Consensus. "
        "The protocol aggregates validator weights through stake-weighted averaging each tempo period."
    )  # > 10 words
    normal_resp = {"response": normal_text, "citations": [0], "confidence_score": 0.5}

    short_score = reward(QUERY, CONTEXT, short_resp, [], response_time=1.0)
    normal_score = reward(QUERY, CONTEXT, normal_resp, [], response_time=1.0)

    assert short_score <= normal_score, (
        f"Short response ({short_score:.3f}) should score ≤ normal ({normal_score:.3f})"
    )

# ---------------------------------------------------------------------------
# Rule 3 — Citation validity
# ---------------------------------------------------------------------------

def test_citation_validity_all_valid():
    assert check_citation_validity([0, 1, 2], n_chunks=3) is True

def test_citation_validity_empty():
    """No citations = vacuously valid."""
    assert check_citation_validity([], n_chunks=3) is True

def test_citation_validity_out_of_range_high():
    """Index >= n_chunks is invalid."""
    assert check_citation_validity([0, 1, 99], n_chunks=3) is False

def test_citation_validity_negative_index():
    """Negative index is invalid."""
    assert check_citation_validity([-1, 0], n_chunks=3) is False

def test_citation_validity_boundary():
    """Index == n_chunks (off-by-one) is invalid."""
    assert check_citation_validity([3], n_chunks=3) is False
    assert check_citation_validity([2], n_chunks=3) is True

def test_citation_validity_strips_bad_indices_in_reward():
    """
    When citations include out-of-range indices, reward() must strip them.
    A response with only invalid citations → supportedness = 0 → lower score
    than an identical response with valid citations.
    """
    response_text = RELEVANT_RESPONSE

    valid_resp = {"response": response_text, "citations": [0, 1], "confidence_score": 0.8}
    invalid_resp = {"response": response_text, "citations": [99, 100], "confidence_score": 0.8}

    valid_score = reward(QUERY, CONTEXT, valid_resp, [], response_time=1.0)
    invalid_score = reward(QUERY, CONTEXT, invalid_resp, [], response_time=1.0)

    assert valid_score > invalid_score, (
        f"Valid citations ({valid_score:.3f}) should score higher than "
        f"invented citations ({invalid_score:.3f})"
    )

def test_citation_mixed_valid_and_invalid():
    """Mixing valid and invented indices: invented ones are stripped, valid ones kept."""
    response_text = RELEVANT_RESPONSE
    mixed_resp = {"response": response_text, "citations": [0, 99], "confidence_score": 0.8}
    pure_resp = {"response": response_text, "citations": [0], "confidence_score": 0.8}

    mixed_score = reward(QUERY, CONTEXT, mixed_resp, [], response_time=1.0)
    pure_score = reward(QUERY, CONTEXT, pure_resp, [], response_time=1.0)

    # After stripping [99], both effectively cite [0] → scores should be equal
    assert abs(mixed_score - pure_score) < 1e-6, (
        f"Mixed ({mixed_score:.4f}) != pure ({pure_score:.4f}) after stripping invalid citation"
    )

# ---------------------------------------------------------------------------
# Interaction: all three rules together
# ---------------------------------------------------------------------------

def test_irrelevant_short_response_scores_zero():
    """Off-topic response fails Rule 1 gate before any other rule applies."""
    short_irrelevant = {"response": "flour eggs butter", "citations": [0], "confidence_score": 0.9}
    score = reward(QUERY, CONTEXT, short_irrelevant, [], response_time=1.0)
    assert score == 0.0

def test_empty_response_scores_zero():
    """Empty response must always score 0."""
    resp = {"response": "", "citations": [], "confidence_score": 0.0}
    assert reward(QUERY, CONTEXT, resp, [], response_time=1.0) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
