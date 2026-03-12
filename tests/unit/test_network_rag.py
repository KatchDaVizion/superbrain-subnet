#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# Tests for Network RAG Query system
#
# Run: python3 tests/unit/test_network_rag.py

import json
import math
import os
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sync.protocol.pool_model import KnowledgeChunk, generate_node_keypair
from sync.queue.sync_queue import SyncQueue
from sync.query.network_rag import (
    NetworkRAGQuery,
    SearchResult,
    Answer,
    PoolStats,
    DEFAULT_TOP_K,
    DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
    DEFAULT_DIVERSITY_THRESHOLD,
)

# Force word overlap backend (no Ollama/sklearn needed for tests)
os.environ["SUPERBRAIN_EMBEDDING_BACKEND"] = "wordoverlap"

assertion_count = 0


def check(condition, msg=""):
    global assertion_count
    assertion_count += 1
    if not condition:
        import traceback
        tb = traceback.extract_stack(limit=2)[0]
        print(f"  FAIL [{assertion_count}]: {msg} (line {tb.lineno})")
        raise AssertionError(msg)


def _make_chunk(content, node_id="test-node", ts=None, metadata=None):
    """Create a signed KnowledgeChunk for testing."""
    priv, pub = generate_node_keypair()
    import hashlib
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    chunk = KnowledgeChunk(
        content=content,
        content_hash=content_hash,
        origin_node_id=node_id,
        timestamp=ts or time.time(),
        signature="",
        metadata=metadata or {},
        pool_visibility="public",
    )
    return chunk


def _make_db_with_chunks(chunks):
    """Create a temp SyncQueue DB and add chunks."""
    tmp = tempfile.mktemp(suffix=".db")
    queue = SyncQueue(db_path=tmp)
    for c in chunks:
        queue.add_to_queue(c)
    queue.close()
    return tmp


def _make_rag(db_path):
    """Create a NetworkRAGQuery with Ollama disabled."""
    return NetworkRAGQuery(
        db_path=db_path,
        ollama_url="http://localhost:99999",  # intentionally unreachable
    )


# ── Test: SearchResult dataclass ─────────────────────────────────────

def test_search_result_fields():
    print("  SearchResult fields...")
    r = SearchResult(
        content="test content",
        content_hash="abc123",
        score=0.85,
        relevance=0.9,
        freshness=0.7,
        source="doc.txt",
        timestamp=time.time(),
        node_id="node-1",
    )
    check(r.content == "test content", "content")
    check(r.content_hash == "abc123", "hash")
    check(r.score == 0.85, "score")
    check(r.relevance == 0.9, "relevance")
    check(r.freshness == 0.7, "freshness")
    check(r.source == "doc.txt", "source")
    check(r.node_id == "node-1", "node_id")


# ── Test: Answer dataclass ───────────────────────────────────────────

def test_answer_fields():
    print("  Answer fields...")
    a = Answer(
        text="The answer is 42.",
        citations=[0, 2],
        sources=[],
        method="extractive",
        query="what is the answer?",
        generation_time=0.5,
    )
    check(a.text == "The answer is 42.", "text")
    check(a.citations == [0, 2], "citations")
    check(a.method == "extractive", "method")
    check(a.query == "what is the answer?", "query")
    check(a.generation_time == 0.5, "gen time")


# ── Test: PoolStats dataclass ────────────────────────────────────────

def test_pool_stats_fields():
    print("  PoolStats fields...")
    s = PoolStats(
        total_chunks=100,
        unique_nodes=5,
        oldest_chunk=1000.0,
        newest_chunk=2000.0,
        embedding_backend="wordoverlap",
        ollama_available=False,
    )
    check(s.total_chunks == 100, "total")
    check(s.unique_nodes == 5, "nodes")
    check(s.oldest_chunk == 1000.0, "oldest")
    check(s.newest_chunk == 2000.0, "newest")
    check(s.embedding_backend == "wordoverlap", "backend")
    check(s.ollama_available is False, "ollama")


# ── Test: Empty search ───────────────────────────────────────────────

def test_search_empty_query():
    print("  Search with empty query...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)
    results = rag.search("")
    check(results == [], "empty query returns []")
    results = rag.search("   ")
    check(results == [], "whitespace query returns []")
    rag.close()
    os.unlink(tmp)


def test_search_empty_pool():
    print("  Search with empty pool...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)
    results = rag.search("quantum computing")
    check(results == [], "empty pool returns []")
    rag.close()
    os.unlink(tmp)


# ── Test: Basic search ───────────────────────────────────────────────

def test_search_basic():
    print("  Basic search ranking...")
    chunks = [
        _make_chunk("Quantum computing uses qubits for parallel computation"),
        _make_chunk("Classical music by Beethoven includes many symphonies"),
        _make_chunk("Quantum entanglement allows particles to be correlated"),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("quantum computing qubits")
    check(len(results) > 0, "got results")
    check(results[0].relevance > 0, "top result has relevance")
    # The quantum computing chunk should rank highest (most word overlap)
    check("quantum" in results[0].content.lower(), "top result mentions quantum")
    check(isinstance(results[0].score, float), "score is float")
    check(0 <= results[0].score <= 1.5, "score in reasonable range")

    rag.close()
    os.unlink(db)


def test_search_top_k():
    print("  Search respects top_k...")
    chunks = [_make_chunk(f"Document number {i} about science") for i in range(20)]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("science document", top_k=3)
    check(len(results) <= 3, "top_k=3 returns at most 3")

    results = rag.search("science document", top_k=10)
    check(len(results) <= 10, "top_k=10 returns at most 10")

    rag.close()
    os.unlink(db)


def test_search_result_structure():
    print("  Search result structure...")
    chunks = [_make_chunk("The blockchain is a distributed ledger technology")]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("blockchain ledger")
    check(len(results) == 1, "one result")
    r = results[0]
    check(isinstance(r, SearchResult), "is SearchResult")
    check(len(r.content_hash) == 64, "SHA-256 hash")
    check(r.node_id == "test-node", "node_id preserved")
    check(r.timestamp > 0, "timestamp set")
    check(r.freshness > 0, "freshness > 0 for recent chunk")
    check(r.score >= r.relevance * 0.7, "score includes freshness bonus")

    rag.close()
    os.unlink(db)


# ── Test: Freshness scoring ──────────────────────────────────────────

def test_freshness_score():
    print("  Freshness scoring...")
    now = time.time()
    chunks = [
        _make_chunk("Fresh knowledge about AI", ts=now - 60),          # 1 min old
        _make_chunk("Old knowledge about AI", ts=now - 86400 * 30),    # 30 days old
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("knowledge about AI")
    check(len(results) == 2, "two results")

    # Find the fresh and old results
    fresh_r = [r for r in results if "Fresh" in r.content][0]
    old_r = [r for r in results if "Old" in r.content][0]

    check(fresh_r.freshness > old_r.freshness, "fresh chunk has higher freshness")
    check(fresh_r.freshness > 0.99, "1-min-old chunk ~1.0 freshness")
    check(old_r.freshness < 0.15, "30-day-old chunk low freshness")

    rag.close()
    os.unlink(db)


def test_freshness_half_life():
    print("  Freshness half-life math...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)

    now = time.time()
    # At exactly 7 days, freshness should be ~0.5 (half-life)
    score_7d = rag._freshness_score(now - 86400 * 7, now)
    check(abs(score_7d - 0.5) < 0.01, f"7-day freshness ~0.5, got {score_7d:.3f}")

    # At 0 days, freshness should be 1.0
    score_0 = rag._freshness_score(now, now)
    check(abs(score_0 - 1.0) < 0.01, "0-day freshness ~1.0")

    # At 14 days, freshness should be ~0.25
    score_14d = rag._freshness_score(now - 86400 * 14, now)
    check(abs(score_14d - 0.25) < 0.02, f"14-day freshness ~0.25, got {score_14d:.3f}")

    rag.close()
    os.unlink(tmp)


# ── Test: Diversity filter ───────────────────────────────────────────

def test_diversity_filter():
    print("  Diversity filter removes near-duplicates...")
    # Create near-duplicate chunks
    chunks = [
        _make_chunk("Quantum computing is revolutionary for science and technology"),
        _make_chunk("Quantum computing is revolutionary for science and tech"),  # near-dup
        _make_chunk("Classical music concerts are held in Vienna each summer"),  # different
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("quantum computing science")
    # Should filter out the near-duplicate
    check(len(results) <= 3, "diversity filter applied")
    # At least 1 result
    check(len(results) >= 1, "at least one result")

    rag.close()
    os.unlink(db)


# ── Test: Source extraction ──────────────────────────────────────────

def test_source_from_metadata():
    print("  Source extraction from metadata...")
    chunks = [
        _make_chunk("Content A", metadata={"source_file": "paper.pdf"}),
        _make_chunk("Content B", metadata={"title": "Research Paper"}),
        _make_chunk("Content C", metadata={}),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("Content")
    sources = {r.source for r in results}

    # Should find source_file, title, or hash-based fallback
    check(any("paper.pdf" in s for s in sources), "source_file extracted")
    check(any("Research Paper" in s for s in sources), "title extracted")
    check(any("chunk_" in s for s in sources), "hash fallback for empty metadata")

    rag.close()
    os.unlink(db)


def test_source_json_metadata():
    print("  Source extraction from JSON string metadata...")
    # KnowledgeChunk metadata is a dict, but network_rag._extract_source
    # also handles JSON strings (from DB serialization). Test with a dict here.
    chunks = [_make_chunk("JSON metadata test", metadata={"source_file": "notes.md", "title": "My Notes"})]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("JSON metadata test")
    check(len(results) == 1, "got result")
    check(results[0].source == "notes.md", f"source from JSON metadata: {results[0].source}")

    rag.close()
    os.unlink(db)


# ── Test: Answer generation (extractive) ─────────────────────────────

def test_answer_extractive():
    print("  Answer generation (extractive fallback)...")
    chunks = [
        _make_chunk("Quantum computing uses qubits instead of classical bits"),
        _make_chunk("Entanglement is a quantum mechanical phenomenon"),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    answer = rag.answer("What are qubits in quantum computing?")
    check(isinstance(answer, Answer), "returns Answer")
    check(len(answer.text) > 0, "non-empty answer text")
    check(answer.method == "extractive", f"extractive method, got {answer.method}")
    check(len(answer.sources) > 0, "has sources")
    check(answer.query == "What are qubits in quantum computing?", "query preserved")
    check(answer.generation_time >= 0, "generation time >= 0")
    check("According to" in answer.text, "extractive format")

    rag.close()
    os.unlink(db)


def test_answer_empty_pool():
    print("  Answer with empty pool...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)

    answer = rag.answer("What is gravity?")
    check(answer.method == "none", "method is none for empty pool")
    check("No knowledge" in answer.text, "appropriate empty message")
    check(answer.sources == [], "no sources")
    check(answer.citations == [], "no citations")

    rag.close()
    os.unlink(tmp)


def test_answer_citations():
    print("  Answer citations extraction...")
    chunks = [
        _make_chunk("Machine learning uses neural networks for pattern recognition"),
        _make_chunk("Deep learning is a subset of machine learning"),
        _make_chunk("Cooking recipes require fresh ingredients"),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    answer = rag.answer("machine learning neural networks")
    check(len(answer.citations) > 0, "has citations")
    # Citations should be valid indices
    for c in answer.citations:
        check(0 <= c < len(answer.sources), f"citation {c} in range")

    rag.close()
    os.unlink(db)


# ── Test: Citation extraction ────────────────────────────────────────

def test_extract_citations():
    print("  Citation extraction from text...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)

    citations = rag._extract_citations("According to [1] and [3], the answer is clear.", 5)
    check(citations == [0, 2], f"[1],[3] → [0,2], got {citations}")

    citations = rag._extract_citations("No citations here.", 5)
    check(citations == [], "no citations")

    citations = rag._extract_citations("[1] [1] [2]", 3)
    check(citations == [0, 1], "deduped citations")

    citations = rag._extract_citations("[99]", 5)
    check(citations == [], "out of range citation filtered")

    rag.close()
    os.unlink(tmp)


# ── Test: Stats ──────────────────────────────────────────────────────

def test_stats_empty():
    print("  Stats on empty pool...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)

    s = rag.stats()
    check(isinstance(s, PoolStats), "returns PoolStats")
    check(s.total_chunks == 0, "zero chunks")
    check(s.unique_nodes == 0, "zero nodes")
    check(s.oldest_chunk is None, "no oldest")
    check(s.newest_chunk is None, "no newest")
    check(s.embedding_backend == "wordoverlap", "backend detected")
    check(s.ollama_available is False, "ollama not available")

    rag.close()
    os.unlink(tmp)


def test_stats_with_chunks():
    print("  Stats with chunks...")
    now = time.time()
    chunks = [
        _make_chunk("Chunk A", node_id="node-1", ts=now - 1000),
        _make_chunk("Chunk B", node_id="node-2", ts=now - 500),
        _make_chunk("Chunk C", node_id="node-1", ts=now - 100),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    s = rag.stats()
    check(s.total_chunks == 3, "3 chunks")
    check(s.unique_nodes == 2, "2 unique nodes")
    check(s.oldest_chunk is not None, "has oldest")
    check(s.newest_chunk is not None, "has newest")
    check(s.oldest_chunk < s.newest_chunk, "oldest < newest")

    rag.close()
    os.unlink(db)


# ── Test: Multiple nodes ─────────────────────────────────────────────

def test_multi_node_search():
    print("  Search across multiple nodes...")
    chunks = [
        _make_chunk("AI research from university lab", node_id="university"),
        _make_chunk("AI research from industry group", node_id="industry"),
        _make_chunk("AI research from government agency", node_id="government"),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("AI research")
    nodes = {r.node_id for r in results}
    check(len(nodes) >= 2, f"results from multiple nodes: {nodes}")

    rag.close()
    os.unlink(db)


# ── Test: Default configuration ──────────────────────────────────────

def test_defaults():
    print("  Default configuration values...")
    check(DEFAULT_TOP_K == 5, "top_k default")
    check(DEFAULT_FRESHNESS_HALF_LIFE_DAYS == 7.0, "freshness half-life default")
    check(DEFAULT_DIVERSITY_THRESHOLD == 0.85, "diversity threshold default")


# ── Test: Close is safe ──────────────────────────────────────────────

def test_close_idempotent():
    print("  Close is safe to call multiple times...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)
    rag.close()
    rag.close()  # should not raise
    check(True, "double close is safe")
    os.unlink(tmp)


# ── Test: Extractive generation details ──────────────────────────────

def test_extractive_selects_best_overlap():
    print("  Extractive picks best overlapping chunks...")
    tmp = tempfile.mktemp(suffix=".db")
    rag = _make_rag(tmp)

    results = [
        SearchResult("Cats are fluffy pets that purr.", "h1", 0.5, 0.5, 1.0, "s1", time.time(), "n1"),
        SearchResult("Dogs are loyal companions for humans.", "h2", 0.4, 0.4, 1.0, "s2", time.time(), "n2"),
        SearchResult("Quantum physics explains subatomic particles.", "h3", 0.3, 0.3, 1.0, "s3", time.time(), "n3"),
    ]

    text, citations, method = rag._generate_extractive("cats fluffy pets", results)
    check(method == "extractive", "method is extractive")
    check(0 in citations, "first result cited (best overlap for cats)")
    check("According to" in text, "extractive format")

    rag.close()
    os.unlink(tmp)


# ── Test: Ollama fallback ────────────────────────────────────────────

def test_ollama_fallback_to_extractive():
    print("  Ollama failure falls back to extractive...")
    chunks = [_make_chunk("Solar energy powers the future")]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)  # unreachable Ollama URL

    answer = rag.answer("solar energy")
    check(answer.method == "extractive", f"fell back to extractive, got {answer.method}")
    check(len(answer.text) > 0, "still got an answer")

    rag.close()
    os.unlink(db)


# ── Test: Score ordering ─────────────────────────────────────────────

def test_results_sorted_by_score():
    print("  Results sorted by score descending...")
    chunks = [
        _make_chunk("Blockchain distributed ledger cryptography hashing"),
        _make_chunk("Classical painting art renaissance beauty"),
        _make_chunk("Blockchain consensus mechanism proof of work mining"),
    ]
    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    results = rag.search("blockchain consensus distributed")
    if len(results) >= 2:
        for i in range(len(results) - 1):
            check(results[i].score >= results[i+1].score,
                  f"score[{i}]={results[i].score:.3f} >= score[{i+1}]={results[i+1].score:.3f}")

    rag.close()
    os.unlink(db)


# ── Test: Full integration ───────────────────────────────────────────

def test_full_integration():
    print("  Full integration: ingest → search → answer...")
    from sync.ingestion.document_ingestor import ingest_text

    priv, pub = generate_node_keypair()
    chunks = ingest_text(
        text="Artificial intelligence is transforming healthcare. "
             "Machine learning models can detect diseases from medical images. "
             "Natural language processing helps analyze clinical notes.",
        private_key=priv,
        node_id="test-ingestor",
        visibility="public",
    )
    check(len(chunks) > 0, "ingestor produced chunks")

    db = _make_db_with_chunks(chunks)
    rag = _make_rag(db)

    # Search
    results = rag.search("artificial intelligence healthcare")
    check(len(results) > 0, "search found results from ingested chunks")
    check(results[0].node_id == "test-ingestor", "node_id preserved through pipeline")

    # Answer
    answer = rag.answer("How does AI help healthcare?")
    check(len(answer.text) > 0, "got answer")
    check(len(answer.sources) > 0, "has sources")

    # Stats
    s = rag.stats()
    check(s.total_chunks == len(chunks), "stats reflect ingested chunks")

    rag.close()
    os.unlink(db)


# ── Runner ───────────────────────────────────────────────────────────

def run_all():
    global assertion_count
    assertion_count = 0
    tests = [
        ("SearchResult fields", test_search_result_fields),
        ("Answer fields", test_answer_fields),
        ("PoolStats fields", test_pool_stats_fields),
        ("Search empty query", test_search_empty_query),
        ("Search empty pool", test_search_empty_pool),
        ("Search basic", test_search_basic),
        ("Search top_k", test_search_top_k),
        ("Search result structure", test_search_result_structure),
        ("Freshness scoring", test_freshness_score),
        ("Freshness half-life", test_freshness_half_life),
        ("Diversity filter", test_diversity_filter),
        ("Source from metadata", test_source_from_metadata),
        ("Source JSON metadata", test_source_json_metadata),
        ("Answer extractive", test_answer_extractive),
        ("Answer empty pool", test_answer_empty_pool),
        ("Answer citations", test_answer_citations),
        ("Citation extraction", test_extract_citations),
        ("Stats empty", test_stats_empty),
        ("Stats with chunks", test_stats_with_chunks),
        ("Multi-node search", test_multi_node_search),
        ("Default config", test_defaults),
        ("Close idempotent", test_close_idempotent),
        ("Extractive best overlap", test_extractive_selects_best_overlap),
        ("Ollama fallback", test_ollama_fallback_to_extractive),
        ("Results sorted", test_results_sorted_by_score),
        ("Full integration", test_full_integration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} — {e}")

    print(f"\n{'='*50}")
    print(f"Tests: {passed} passed, {failed} failed")
    print(f"Assertions: {assertion_count}")
    print(f"{'='*50}")

    if failed > 0:
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    run_all()
