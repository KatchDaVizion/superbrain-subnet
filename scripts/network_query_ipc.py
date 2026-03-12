#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Network RAG Query — IPC Bridge
#
# Called by the desktop app's Electron main process via subprocess.
# Accepts a JSON argument, runs the query, outputs JSON to stdout.
#
# Usage (from Electron):
#   python3 scripts/network_query_ipc.py '{"query":"...", "db_path":"...", "top_k":5, "mode":"answer"}'
#
# Modes: "answer" (search + generate), "search" (search only), "stats" (pool stats)

import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Suppress all logging to stderr so only JSON goes to stdout
import logging
logging.disable(logging.CRITICAL)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing JSON argument"}))
        return 1

    try:
        args = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
        return 1

    query = args.get("query", "")
    db_path = args.get("db_path", "miner_sync_queue.db")
    top_k = args.get("top_k", 5)
    mode = args.get("mode", "answer")

    # Force word overlap if no Ollama embeddings available (fast, no dependencies)
    if not os.environ.get("SUPERBRAIN_EMBEDDING_BACKEND"):
        os.environ["SUPERBRAIN_EMBEDDING_BACKEND"] = "auto"

    from sync.query.network_rag import NetworkRAGQuery

    rag = NetworkRAGQuery(db_path=db_path, top_k=top_k)

    try:
        if mode == "stats":
            s = rag.stats()
            result = {
                "total_chunks": s.total_chunks,
                "unique_nodes": s.unique_nodes,
                "oldest_chunk": s.oldest_chunk,
                "newest_chunk": s.newest_chunk,
                "embedding_backend": s.embedding_backend,
                "ollama_available": s.ollama_available,
            }

        elif mode == "search":
            results = rag.search(query, top_k=top_k)
            result = {
                "results": [
                    {
                        "content": r.content,
                        "content_hash": r.content_hash,
                        "score": round(r.score, 4),
                        "relevance": round(r.relevance, 4),
                        "freshness": round(r.freshness, 4),
                        "source": r.source,
                        "timestamp": r.timestamp,
                        "node_id": r.node_id,
                    }
                    for r in results
                ]
            }

        else:  # mode == "answer"
            answer = rag.answer(query, top_k=top_k)
            result = {
                "text": answer.text,
                "citations": answer.citations,
                "sources": [
                    {
                        "content": r.content,
                        "content_hash": r.content_hash,
                        "score": round(r.score, 4),
                        "relevance": round(r.relevance, 4),
                        "freshness": round(r.freshness, 4),
                        "source": r.source,
                        "timestamp": r.timestamp,
                        "node_id": r.node_id,
                    }
                    for r in answer.sources
                ],
                "method": answer.method,
                "query": answer.query,
                "generation_time": round(answer.generation_time, 3),
            }

        print(json.dumps(result))
        return 0

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1
    finally:
        rag.close()


if __name__ == "__main__":
    sys.exit(main())
