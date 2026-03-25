#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# Tests for Network RAG Query IPC Bridge
#
# Tests the IPC handler script that the desktop app calls via subprocess.
# Validates JSON input/output format, all three modes, and error handling.
#
# Run: python3 tests/unit/test_network_query_ipc.py

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sync.protocol.pool_model import KnowledgeChunk, generate_node_keypair
from sync.queue.sync_queue import SyncQueue

# Force word overlap backend for tests
os.environ["SUPERBRAIN_EMBEDDING_BACKEND"] = "wordoverlap"

assertion_count = 0
IPC_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "network_query_ipc.py")


def check(condition, msg=""):
    global assertion_count
    assertion_count += 1
    if not condition:
        import traceback
        tb = traceback.extract_stack(limit=2)[0]
        print(f"  FAIL [{assertion_count}]: {msg} (line {tb.lineno})")
        raise AssertionError(msg)


def _make_chunk(content, node_id="test-node", ts=None, metadata=None):
    """Create a public KnowledgeChunk."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return KnowledgeChunk(
        content=content,
        content_hash=content_hash,
        origin_node_id=node_id,
        timestamp=ts or time.time(),
        signature="",
        metadata=metadata or {},
        pool_visibility="public",
    )


def _make_db_with_chunks(chunks):
    """Create a temp SyncQueue DB and add chunks."""
    tmp = tempfile.mktemp(suffix=".db")
    queue = SyncQueue(db_path=tmp)
    for c in chunks:
        queue.add_to_queue(c)
    queue.close()
    return tmp


def _run_ipc(args_dict):
    """Run the IPC script as a subprocess (simulating Electron's exec)."""
    result = subprocess.run(
        [sys.executable, IPC_SCRIPT, json.dumps(args_dict)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=PROJECT_ROOT,
        env={**os.environ, "SUPERBRAIN_EMBEDDING_BACKEND": "wordoverlap"},
    )
    return result


# ── Test: Script exists and is importable ────────────────────────────

def test_script_exists():
    print("  Script exists...")
    check(os.path.isfile(IPC_SCRIPT), f"IPC script exists at {IPC_SCRIPT}")


# ── Test: Missing argument ───────────────────────────────────────────

def test_missing_argument():
    print("  Missing argument error...")
    result = subprocess.run(
        [sys.executable, IPC_SCRIPT],
        capture_output=True, text=True, timeout=10,
        cwd=PROJECT_ROOT,
    )
    output = json.loads(result.stdout.strip())
    check("error" in output, "returns error for missing arg")
    check("Missing" in output["error"], f"error mentions missing: {output['error']}")


# ── Test: Invalid JSON argument ──────────────────────────────────────

def test_invalid_json():
    print("  Invalid JSON error...")
    result = subprocess.run(
        [sys.executable, IPC_SCRIPT, "not valid json{{{"],
        capture_output=True, text=True, timeout=10,
        cwd=PROJECT_ROOT,
    )
    output = json.loads(result.stdout.strip())
    check("error" in output, "returns error for bad JSON")
    check("Invalid JSON" in output["error"], f"error mentions invalid JSON")


# ── Test: Stats mode (empty DB) ─────────────────────────────────────

def test_stats_empty():
    print("  Stats mode (empty DB)...")
    tmp = tempfile.mktemp(suffix=".db")
    # Create empty DB
    q = SyncQueue(db_path=tmp)
    q.close()

    result = _run_ipc({"query": "", "db_path": tmp, "top_k": 0, "mode": "stats"})
    check(result.returncode == 0, f"exit code 0, got {result.returncode}")

    output = json.loads(result.stdout.strip())
    check("total_chunks" in output, "has total_chunks")
    check(output["total_chunks"] == 0, "zero chunks")
    check("unique_nodes" in output, "has unique_nodes")
    check(output["unique_nodes"] == 0, "zero nodes")
    check("embedding_backend" in output, "has embedding_backend")
    check("ollama_available" in output, "has ollama_available")
    check(output["oldest_chunk"] is None, "oldest is null")
    check(output["newest_chunk"] is None, "newest is null")

    os.unlink(tmp)


# ── Test: Stats mode (with chunks) ──────────────────────────────────

def test_stats_with_data():
    print("  Stats mode (with chunks)...")
    chunks = [
        _make_chunk("Knowledge about AI", node_id="node-a"),
        _make_chunk("Knowledge about physics", node_id="node-b"),
        _make_chunk("More AI knowledge", node_id="node-a"),
    ]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "", "db_path": db, "top_k": 0, "mode": "stats"})
    output = json.loads(result.stdout.strip())

    check(output["total_chunks"] == 3, f"3 chunks, got {output['total_chunks']}")
    check(output["unique_nodes"] == 2, f"2 nodes, got {output['unique_nodes']}")
    check(output["oldest_chunk"] is not None, "has oldest")
    check(output["newest_chunk"] is not None, "has newest")

    os.unlink(db)


# ── Test: Search mode ────────────────────────────────────────────────

def test_search_mode():
    print("  Search mode...")
    chunks = [
        _make_chunk("Quantum computing uses qubits for parallel computation"),
        _make_chunk("Classical music by Beethoven includes symphonies"),
        _make_chunk("Quantum entanglement allows particles to be correlated"),
    ]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "quantum computing qubits", "db_path": db, "top_k": 5, "mode": "search"})
    check(result.returncode == 0, f"exit code 0, got {result.returncode}")

    output = json.loads(result.stdout.strip())
    check("results" in output, "has results array")
    check(len(output["results"]) > 0, "got results")

    first = output["results"][0]
    check("content" in first, "result has content")
    check("content_hash" in first, "result has content_hash")
    check("score" in first, "result has score")
    check("relevance" in first, "result has relevance")
    check("freshness" in first, "result has freshness")
    check("source" in first, "result has source")
    check("timestamp" in first, "result has timestamp")
    check("node_id" in first, "result has node_id")
    check(isinstance(first["score"], float), "score is float")
    check(first["node_id"] == "test-node", f"node_id preserved: {first['node_id']}")

    os.unlink(db)


# ── Test: Search mode (empty query) ─────────────────────────────────

def test_search_empty_query():
    print("  Search mode (empty query)...")
    db = tempfile.mktemp(suffix=".db")
    SyncQueue(db_path=db).close()

    result = _run_ipc({"query": "", "db_path": db, "top_k": 5, "mode": "search"})
    output = json.loads(result.stdout.strip())
    check("results" in output, "has results")
    check(len(output["results"]) == 0, "empty results for empty query")

    os.unlink(db)


# ── Test: Answer mode ────────────────────────────────────────────────

def test_answer_mode():
    print("  Answer mode (extractive fallback)...")
    chunks = [
        _make_chunk("Machine learning uses neural networks for pattern recognition"),
        _make_chunk("Deep learning is a subset of machine learning"),
    ]
    db = _make_db_with_chunks(chunks)

    result = subprocess.run(
        [sys.executable, IPC_SCRIPT, json.dumps({"query": "machine learning neural networks", "db_path": db, "top_k": 5, "mode": "answer"})],
        capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
        env={**os.environ, "SUPERBRAIN_EMBEDDING_BACKEND": "wordoverlap", "OLLAMA_URL": "http://127.0.0.1:1"},
    )
    check(result.returncode == 0, f"exit code 0, got {result.returncode}")

    output = json.loads(result.stdout.strip())
    check("text" in output, "has text")
    check("citations" in output, "has citations")
    check("sources" in output, "has sources")
    check("method" in output, "has method")
    check("query" in output, "has query")
    check("generation_time" in output, "has generation_time")

    check(len(output["text"]) > 0, "non-empty answer text")
    check(output["method"] == "extractive", f"extractive method, got {output['method']}")
    check(output["query"] == "machine learning neural networks", "query preserved")
    check(isinstance(output["generation_time"], float), "gen time is float")
    check(len(output["sources"]) > 0, "has sources")
    check(isinstance(output["citations"], list), "citations is list")

    # Verify source structure
    src = output["sources"][0]
    check("content" in src, "source has content")
    check("score" in src, "source has score")
    check("node_id" in src, "source has node_id")

    os.unlink(db)


# ── Test: Answer mode (empty pool) ───────────────────────────────────

def test_answer_empty_pool():
    print("  Answer mode (empty pool)...")
    db = tempfile.mktemp(suffix=".db")
    SyncQueue(db_path=db).close()

    result = _run_ipc({"query": "anything", "db_path": db, "top_k": 5, "mode": "answer"})
    output = json.loads(result.stdout.strip())

    check(output["method"] == "none", f"method=none for empty pool, got {output['method']}")
    check("No knowledge" in output["text"], "appropriate empty message")
    check(output["sources"] == [], "no sources")

    os.unlink(db)


# ── Test: JSON output is clean (no logging noise) ────────────────────

def test_clean_json_output():
    print("  Clean JSON output (no stderr noise on stdout)...")
    chunks = [_make_chunk("Test content for clean output")]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "test", "db_path": db, "top_k": 1, "mode": "search"})
    stdout = result.stdout.strip()

    # Should be valid JSON with no extra lines
    lines = stdout.split("\n")
    check(len(lines) == 1, f"single line of JSON, got {len(lines)} lines")

    # Should parse cleanly
    parsed = json.loads(stdout)
    check(isinstance(parsed, dict), "parses as dict")

    os.unlink(db)


# ── Test: Top-K limiting via IPC ─────────────────────────────────────

def test_top_k_via_ipc():
    print("  Top-K limiting via IPC...")
    chunks = [_make_chunk(f"Document {i} about science research") for i in range(15)]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "science research", "db_path": db, "top_k": 3, "mode": "search"})
    output = json.loads(result.stdout.strip())
    check(len(output["results"]) <= 3, f"top_k=3 limits results, got {len(output['results'])}")

    os.unlink(db)


# ── Test: Source metadata preserved through IPC ──────────────────────

def test_metadata_through_ipc():
    print("  Source metadata preserved through IPC...")
    chunks = [_make_chunk("Content with source", metadata={"source_file": "paper.pdf"})]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "Content with source", "db_path": db, "top_k": 5, "mode": "search"})
    output = json.loads(result.stdout.strip())

    check(len(output["results"]) > 0, "got results")
    check(output["results"][0]["source"] == "paper.pdf", f"source metadata: {output['results'][0]['source']}")

    os.unlink(db)


# ── Test: Multi-node sources in answer ───────────────────────────────

def test_multi_node_answer():
    print("  Multi-node answer sources...")
    chunks = [
        _make_chunk("AI helps doctors diagnose diseases", node_id="hospital"),
        _make_chunk("AI assists in medical imaging analysis", node_id="university"),
        _make_chunk("Cooking recipes need ingredients", node_id="kitchen"),
    ]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "AI medical diagnosis", "db_path": db, "top_k": 5, "mode": "answer"})
    output = json.loads(result.stdout.strip())

    source_nodes = {s["node_id"] for s in output["sources"]}
    check(len(source_nodes) >= 1, f"got sources from nodes: {source_nodes}")

    os.unlink(db)


# ── Test: Scores are rounded floats ─────────────────────────────────

def test_scores_rounded():
    print("  Scores are rounded to 4 decimal places...")
    chunks = [_make_chunk("Blockchain distributed ledger")]
    db = _make_db_with_chunks(chunks)

    result = _run_ipc({"query": "blockchain", "db_path": db, "top_k": 1, "mode": "search"})
    output = json.loads(result.stdout.strip())

    if output["results"]:
        score_str = str(output["results"][0]["score"])
        # Should have at most 4 decimal places
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            check(decimals <= 4, f"score rounded to <= 4 decimals, got {decimals}")

    os.unlink(db)


# ── Runner ───────────────────────────────────────────────────────────

def run_all():
    global assertion_count
    assertion_count = 0
    tests = [
        ("Script exists", test_script_exists),
        ("Missing argument", test_missing_argument),
        ("Invalid JSON", test_invalid_json),
        ("Stats empty", test_stats_empty),
        ("Stats with data", test_stats_with_data),
        ("Search mode", test_search_mode),
        ("Search empty query", test_search_empty_query),
        ("Answer mode", test_answer_mode),
        ("Answer empty pool", test_answer_empty_pool),
        ("Clean JSON output", test_clean_json_output),
        ("Top-K via IPC", test_top_k_via_ipc),
        ("Metadata through IPC", test_metadata_through_ipc),
        ("Multi-node answer", test_multi_node_answer),
        ("Scores rounded", test_scores_rounded),
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
