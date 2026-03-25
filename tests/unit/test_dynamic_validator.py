"""
Tests for dynamic validator query generation (Mission 1 — anti-gaming).

Verifies:
  - Dynamic queries differ from old hardcoded template strings
  - Consecutive rounds produce different chunk selections (diversity gate)
  - Natural questions are content-specific (not generic templates)
  - Graceful fallback when sync_queue / chunk pool is empty or too small
"""
import hashlib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from superbrain.validator.forward import (
    _build_dynamic_query,
    _extract_question,
    _SEEN_CHUNK_HASHES,
    MIN_CHUNKS_FOR_DYNAMIC,
)
import superbrain.validator.forward as fwd_module

# ---------------------------------------------------------------------------
# Minimal mock KnowledgeChunk (no bittensor dependency)
# ---------------------------------------------------------------------------

class _MockChunk:
    def __init__(self, content: str, source: str = ""):
        self.content = content
        self.metadata = {"source_file": source} if source else {}
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()

# Old generic templates (must NOT appear in dynamic queries)
_OLD_TEMPLATES = {
    "Based on the following information, what are the key concepts discussed?",
    "Summarize the main points from the provided context.",
    "What is the most important information in the given sources?",
    "Explain the concepts described in the provided documents.",
    "What conclusions can be drawn from the provided information?",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int) -> list:
    texts = [
        "Bittensor is a decentralized machine learning network that rewards miners with TAO.",
        "RAG enables LLM outputs by retrieving relevant documents from a vector database.",
        "Delta sync uses a 5-step protocol: Handshake, Manifest, Diff, Transfer, Confirm.",
        "Ed25519 signatures ensure every knowledge chunk is cryptographically signed.",
        "Yuma Consensus aggregates validator weights using stake-weighted averaging.",
        "SuperBrain is a local-first knowledge network on Bittensor, private by default.",
        "Ollama allows users to run language models locally without cloud API costs.",
        "mDNS peer discovery lets sync nodes find each other on the same LAN automatically.",
    ]
    return [_MockChunk(texts[i % len(texts)] + f" (variant {i})", f"doc_{i}.md") for i in range(n)]

# ---------------------------------------------------------------------------
# Test 1: _extract_question produces content-specific questions
# ---------------------------------------------------------------------------

def test_extract_question_is_pattern():
    """Questions must differ from every old generic template."""
    chunks = _make_chunks(6)
    for c in chunks:
        q = _extract_question(c.content)
        assert q not in _OLD_TEMPLATES, (
            f"_extract_question returned a generic template: {q!r}"
        )
        assert "?" in q, f"Question has no '?': {q!r}"
        assert len(q) > 10, f"Question too short: {q!r}"

def test_extract_question_what_is_pattern():
    q = _extract_question("Bittensor is a decentralized ML network.")
    assert q.startswith("What is"), f"Expected 'What is …' but got: {q!r}"
    assert "Bittensor" in q

def test_extract_question_how_does_pattern():
    q = _extract_question("mDNS enables automatic peer discovery on a LAN.")
    assert "mDNS" in q

def test_extract_question_fallback():
    """Short or unpatterned text must still return a valid question."""
    q = _extract_question("short")
    assert "?" in q

# ---------------------------------------------------------------------------
# Test 2: dynamic queries differ from hardcoded templates
# ---------------------------------------------------------------------------

def test_dynamic_query_not_template():
    """_build_dynamic_query must not return a generic template query."""
    chunks = _make_chunks(6)
    result = _build_dynamic_query(chunks)
    assert result is not None
    query, texts, sources = result
    assert query not in _OLD_TEMPLATES, (
        f"Dynamic query is still a generic template: {query!r}"
    )
    assert "?" in query

def test_dynamic_query_returns_3_chunks():
    """Selects up to 3 chunks."""
    chunks = _make_chunks(6)
    result = _build_dynamic_query(chunks)
    assert result is not None
    _, texts, sources = result
    assert len(texts) == 3
    assert len(sources) == 3

def test_dynamic_query_exactly_available_when_fewer_than_3():
    """When fewer than 3 chunks are available, returns all of them."""
    chunks = _make_chunks(2)
    result = _build_dynamic_query(chunks)
    assert result is not None
    _, texts, _ = result
    assert len(texts) == 2

# ---------------------------------------------------------------------------
# Test 3: consecutive rounds use different chunks (diversity gate)
# ---------------------------------------------------------------------------

def test_consecutive_rounds_different_chunks():
    """
    After round 1 records _SEEN_CHUNK_HASHES, round 2 must prefer fresh chunks
    when enough are available — producing a different selection.
    """
    # 8 chunks: rounds should differ because 5 fresh ones remain after first pick
    chunks = _make_chunks(8)

    # Reset diversity state
    fwd_module._SEEN_CHUNK_HASHES = set()

    result1 = _build_dynamic_query(chunks)
    assert result1 is not None
    _, texts1, _ = result1
    hashes1 = set(fwd_module._SEEN_CHUNK_HASHES)

    result2 = _build_dynamic_query(chunks)
    assert result2 is not None
    _, texts2, _ = result2

    # The two selections must not be identical
    assert set(texts1) != set(texts2), (
        "Consecutive rounds returned identical chunk sets — diversity gate not working"
    )

def test_seen_hashes_updated_after_round():
    """_SEEN_CHUNK_HASHES must be populated after a build call.

    We read via _build_dynamic_query.__globals__ (the authoritative module dict)
    rather than via fwd_module to avoid any import-alias reference mismatch.
    """
    chunks = _make_chunks(4)
    # Reset via the function's own globals dict — guaranteed to be the same namespace
    _build_dynamic_query.__globals__["_SEEN_CHUNK_HASHES"] = set()
    _build_dynamic_query(chunks)
    seen = _build_dynamic_query.__globals__["_SEEN_CHUNK_HASHES"]
    assert len(seen) > 0, "_SEEN_CHUNK_HASHES is empty after a successful build call"

def test_diversity_fallback_when_all_seen():
    """If all chunks were seen last round, fall back to full pool (no crash)."""
    chunks = _make_chunks(3)
    # Mark all as seen
    fwd_module._SEEN_CHUNK_HASHES = {c.content_hash for c in chunks}
    result = _build_dynamic_query(chunks)
    # Must still return a result (uses full pool as fallback)
    assert result is not None

# ---------------------------------------------------------------------------
# Test 4: graceful fallback when pool is empty or too small
# ---------------------------------------------------------------------------

def test_empty_pool_returns_none():
    """Empty chunk list must return None (not crash)."""
    assert _build_dynamic_query([]) is None

def test_single_chunk_returns_none():
    """One chunk is below MIN_CHUNKS_FOR_DYNAMIC — must return None."""
    assert _build_dynamic_query(_make_chunks(1)) is None

def test_exactly_min_chunks_returns_result():
    """Exactly MIN_CHUNKS_FOR_DYNAMIC chunks must succeed."""
    chunks = _make_chunks(MIN_CHUNKS_FOR_DYNAMIC)
    result = _build_dynamic_query(chunks)
    assert result is not None
    query, texts, sources = result
    assert len(texts) == MIN_CHUNKS_FOR_DYNAMIC

# ---------------------------------------------------------------------------
# Test 5: source labels
# ---------------------------------------------------------------------------

def test_source_labels_from_metadata():
    chunks = [_MockChunk("Bittensor is a network.", "bittensor.md")]
    chunks += _make_chunks(2)
    result = _build_dynamic_query(chunks)
    assert result is not None
    _, _, sources = result
    assert all(isinstance(s, str) and len(s) > 0 for s in sources)

def test_source_labels_fallback_to_hash():
    """Chunks without metadata source_file get chunk_<hash> labels."""
    chunks = [_MockChunk("Plain text with no source.") for _ in range(3)]
    result = _build_dynamic_query(chunks)
    assert result is not None
    _, _, sources = result
    assert all(s.startswith("chunk_") for s in sources)


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
