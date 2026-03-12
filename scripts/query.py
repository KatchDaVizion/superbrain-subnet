#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Network RAG Query CLI
#
# Search the collective public knowledge pool and get answers.
#
# Usage:
#   python scripts/query.py "What is quantum entanglement?"
#   python scripts/query.py --search "blockchain consensus"
#   python scripts/query.py --stats
#   python scripts/query.py "dark matter" --db ./miner_sync_queue.db --top-k 10

import argparse
import os
import sys
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sync.query.network_rag import NetworkRAGQuery


def _format_time_ago(timestamp: float) -> str:
    """Human-readable time ago string."""
    age = time.time() - timestamp
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    if age < 86400:
        return f"{int(age / 3600)}h ago"
    return f"{int(age / 86400)}d ago"


def cmd_search(rag: NetworkRAGQuery, query: str, top_k: int):
    """Search-only mode — show ranked results."""
    results = rag.search(query, top_k=top_k)
    if not results:
        print("No results found.")
        return

    print(f"\n=== Search Results ({len(results)} matches) ===\n")
    for i, r in enumerate(results):
        age = _format_time_ago(r.timestamp)
        print(f"[{i+1}] Score: {r.score:.3f} (rel={r.relevance:.3f}, fresh={r.freshness:.3f})")
        print(f"    Source: {r.source} | Node: {r.node_id} | {age}")
        preview = r.content[:120].replace("\n", " ")
        print(f"    {preview}...")
        print()


def cmd_answer(rag: NetworkRAGQuery, query: str, top_k: int):
    """Answer mode — search + generate answer."""
    answer = rag.answer(query, top_k=top_k)

    if not answer.sources:
        print(answer.text)
        return

    print(f"\n=== Answer (via {answer.method}, {answer.generation_time:.1f}s) ===\n")
    print(answer.text)

    if answer.citations:
        print(f"\n=== Sources Cited ===")
        for idx in answer.citations:
            if idx < len(answer.sources):
                r = answer.sources[idx]
                age = _format_time_ago(r.timestamp)
                print(f"  [{idx+1}] {r.source} (score={r.score:.3f}, {age})")

    print(f"\n=== All Sources ({len(answer.sources)}) ===")
    for i, r in enumerate(answer.sources):
        age = _format_time_ago(r.timestamp)
        marker = " *" if i in answer.citations else ""
        print(f"  [{i+1}] {r.source} — score={r.score:.3f}, {age}{marker}")


def cmd_stats(rag: NetworkRAGQuery):
    """Show pool statistics."""
    s = rag.stats()
    print(f"\n=== Knowledge Pool Stats ===\n")
    print(f"Total chunks:      {s.total_chunks}")
    print(f"Unique nodes:      {s.unique_nodes}")
    if s.oldest_chunk:
        oldest = datetime.fromtimestamp(s.oldest_chunk).strftime("%Y-%m-%d %H:%M")
        newest = datetime.fromtimestamp(s.newest_chunk).strftime("%Y-%m-%d %H:%M")
        print(f"Oldest chunk:      {oldest}")
        print(f"Newest chunk:      {newest}")
    print(f"Embedding backend: {s.embedding_backend}")
    print(f"Ollama available:  {'yes' if s.ollama_available else 'no'}")


def main():
    parser = argparse.ArgumentParser(
        description="Search the SuperBrain collective knowledge pool",
    )

    parser.add_argument("query", nargs="?", default="", help="Query to search/answer")
    parser.add_argument("--search", action="store_true",
                        help="Search-only mode (no answer generation)")
    parser.add_argument("--stats", action="store_true",
                        help="Show knowledge pool statistics")
    parser.add_argument("--db", type=str, default="",
                        help="SyncQueue database path (default: ./miner_sync_queue.db)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--model", type=str, default="",
                        help="Ollama model for answer generation")
    parser.add_argument("--ollama-url", type=str, default="",
                        help="Ollama API URL (default: http://localhost:11434)")

    args = parser.parse_args()

    db_path = args.db or "miner_sync_queue.db"

    rag = NetworkRAGQuery(
        db_path=db_path,
        ollama_url=args.ollama_url,
        ollama_model=args.model,
        top_k=args.top_k,
    )

    try:
        if args.stats:
            cmd_stats(rag)
        elif args.query:
            if args.search:
                cmd_search(rag, args.query, args.top_k)
            else:
                cmd_answer(rag, args.query, args.top_k)
        else:
            parser.print_help()
            return 1
    finally:
        rag.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
