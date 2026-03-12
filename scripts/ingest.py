#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Document Ingest CLI
#
# Ingests documents into a miner's SyncQueue as KnowledgeChunks.
#
# Usage:
#   python scripts/ingest.py --file mydoc.pdf --wallet sb_miner --visibility public
#   python scripts/ingest.py --file notes.md --db ./miner_sync_queue.db
#   python scripts/ingest.py --file paper.txt --wallet sb_miner --title "Research Paper"
#   python scripts/ingest.py --text "Inline knowledge to ingest" --wallet sb_miner

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sync.ingestion.document_ingestor import (
    ingest_file,
    ingest_text,
    deduplicate_chunks,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from sync.protocol.pool_model import generate_node_keypair
from sync.queue.sync_queue import SyncQueue


def _load_keypair_from_wallet(wallet_name: str, hotkey: str = "default"):
    """
    Try to load or derive a keypair from a Bittensor wallet.
    Falls back to generating a fresh Ed25519 keypair if bittensor is not available.
    """
    try:
        import bittensor as bt

        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        # Use the hotkey's raw bytes as seed for deterministic Ed25519 keypair
        seed = wallet.hotkey.public_key[:32] if hasattr(wallet.hotkey, "public_key") else None
        if seed and len(seed) >= 32:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                NoEncryption,
                PrivateFormat,
                PublicFormat,
            )

            # Derive Ed25519 key from wallet seed
            priv_key = Ed25519PrivateKey.from_private_bytes(seed[:32])
            priv_bytes = priv_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            pub_bytes = priv_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            return priv_bytes, pub_bytes, wallet.hotkey.ss58_address
    except Exception:
        pass

    # Fallback: generate fresh keypair
    priv, pub = generate_node_keypair()
    return priv, pub, "local-node"


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into SuperBrain SyncQueue as KnowledgeChunks",
    )

    # Input source (one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=str, help="Path to document file (.txt, .md, .pdf)")
    input_group.add_argument("--text", type=str, help="Raw text to ingest directly")

    # Identity
    parser.add_argument("--wallet", type=str, default="", help="Bittensor wallet name (for signing)")
    parser.add_argument("--hotkey", type=str, default="default", help="Wallet hotkey name")
    parser.add_argument("--node-id", type=str, default="", help="Node ID (auto-generated if omitted)")

    # Chunk settings
    parser.add_argument("--visibility", choices=["private", "public"], default="private",
                        help="Chunk visibility (default: private)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--title", type=str, default="", help="Document title for metadata")

    # Database
    parser.add_argument("--db", type=str, default="",
                        help="SyncQueue database path (default: ./miner_sync_queue.db)")

    # Options
    parser.add_argument("--dry-run", action="store_true", help="Show chunks without saving to DB")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate detection")

    args = parser.parse_args()

    # Load keypair
    if args.wallet:
        priv, pub, node_label = _load_keypair_from_wallet(args.wallet, args.hotkey)
        print(f"Wallet: {args.wallet} (node: {node_label})")
    else:
        priv, pub = generate_node_keypair()
        node_label = "local-node"
        print("No wallet specified — using ephemeral keypair")

    node_id = args.node_id or node_label

    # Ingest
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        print(f"Ingesting: {args.file}")
        chunks = ingest_file(
            file_path=args.file,
            private_key=priv,
            node_id=node_id,
            visibility=args.visibility,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            title=args.title,
        )
    else:
        print(f"Ingesting: inline text ({len(args.text)} chars)")
        chunks = ingest_text(
            text=args.text,
            private_key=priv,
            node_id=node_id,
            visibility=args.visibility,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source="cli_input",
            title=args.title or "CLI Input",
        )

    if not chunks:
        print("No chunks produced (empty input?).")
        sys.exit(0)

    print(f"Chunks created: {len(chunks)}")

    # Dedup against existing queue
    if not args.dry_run and not args.no_dedup:
        db_path = args.db or "miner_sync_queue.db"
        if os.path.exists(db_path):
            queue = SyncQueue(db_path=db_path)
            manifest = queue.get_manifest(node_id)
            existing = manifest.chunk_hashes
            before = len(chunks)
            chunks = deduplicate_chunks(chunks, existing)
            dupes = before - len(chunks)
            if dupes > 0:
                print(f"Duplicates skipped: {dupes}")
            queue.close()

    # Show chunks
    for i, chunk in enumerate(chunks):
        vis = "PUBLIC" if chunk.pool_visibility == "public" else "private"
        sig = "signed" if chunk.signature else "unsigned"
        print(f"  [{i+1}] {chunk.content_hash[:16]}... {len(chunk.content):>4} chars [{vis}] [{sig}]")
        if len(chunks) <= 10:
            preview = chunk.content[:80].replace("\n", " ")
            print(f"      {preview}...")

    # Save to DB
    if args.dry_run:
        print("\n(dry run — not saved to database)")
    else:
        db_path = args.db or "miner_sync_queue.db"
        queue = SyncQueue(db_path=db_path)
        added = 0
        for chunk in chunks:
            if chunk.pool_visibility == "public":
                if queue.add_to_queue(chunk):
                    added += 1
            else:
                # Private chunks: store but mark differently
                # SyncQueue only accepts public chunks, so we make them public for storage
                # but they won't be synced until the user explicitly shares them.
                print(f"  Note: chunk {chunk.content_hash[:16]}... is private — not added to sync queue")

        queue.close()
        print(f"\nSaved to: {db_path}")
        print(f"Chunks added: {added}")
        if added < len(chunks):
            private_count = len([c for c in chunks if c.pool_visibility == "private"])
            print(f"Private (not queued): {private_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
