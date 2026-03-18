#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Share to Network — IPC Bridge
#
# Called by the desktop app when user toggles "Share to Network" during ingest.
# Takes text or file path, creates Ed25519-signed KnowledgeChunks, adds to SyncQueue.
#
# Usage (from Electron):
#   python3 scripts/share_to_network.py '{"mode":"text","content":"...","db_path":"...","title":"..."}'
#   python3 scripts/share_to_network.py '{"mode":"file","file_path":"/path/to/doc.pdf","db_path":"..."}'

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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

    mode = args.get("mode", "text")
    db_path = args.get("db_path", "miner_sync_queue.db")
    title = args.get("title", "")

    from sync.protocol.pool_model import generate_node_keypair
    from sync.ingestion.document_ingestor import ingest_text, ingest_file, deduplicate_chunks
    from sync.queue.sync_queue import SyncQueue

    # Generate a keypair for signing (ephemeral per session)
    # In production, this would use the miner's persistent keypair
    priv, pub = generate_node_keypair()
    node_id = f"desktop-{os.getpid()}"

    try:
        if mode == "file":
            file_path = args.get("file_path", "")
            if not file_path or not os.path.isfile(file_path):
                print(json.dumps({"error": f"File not found: {file_path}"}))
                return 1
            chunks = ingest_file(
                file_path=file_path,
                private_key=priv,
                node_id=node_id,
                visibility="public",
                title=title,
                public_key=pub,
            )
        else:
            content = args.get("content", "")
            if not content.strip():
                print(json.dumps({"error": "Empty content"}))
                return 1
            chunks = ingest_text(
                text=content,
                private_key=priv,
                node_id=node_id,
                visibility="public",
                source=title or "desktop_share",
                title=title,
                public_key=pub,
            )

        if not chunks:
            print(json.dumps({"error": "No chunks produced from content"}))
            return 1

        # Add to SyncQueue
        q = SyncQueue(db_path)
        existing = {e.content_hash for e in q.get_all_chunks(limit=10000)}
        unique = deduplicate_chunks(chunks, existing)

        added = 0
        for chunk in unique:
            if q.add_to_queue(chunk):
                added += 1

        result = {
            "success": True,
            "total_chunks": len(chunks),
            "new_chunks": added,
            "duplicates": len(chunks) - added,
            "db_path": db_path,
        }
        print(json.dumps(result))
        return 0

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
