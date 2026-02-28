#!/usr/bin/env python3
"""SuperBrain LAN Sync Node — standalone peer for syncing knowledge chunks.

Run this on a friend's machine (or a second terminal) to sync knowledge
with a SuperBrain miner or another sync node on the same network.

Usage:
  # Seed 10 sample chunks + auto-discover peers via mDNS
  python3 run_sync_node.py --seed

  # Connect to a specific peer (e.g. miner on Tailscale)
  python3 run_sync_node.py --seed --static 100.x.x.x:8384

  # Use the miner's database directly (Way 1 — shared DB)
  python3 run_sync_node.py --seed --db ~/superbrain-dev/subnet/miner_sync_queue.db

  # Add a custom knowledge chunk
  python3 run_sync_node.py --add "Bittensor subnets are specialized task networks."
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sync.protocol.pool_model import KnowledgeChunk, generate_node_keypair
from sync.queue.sync_queue import SyncQueue
from sync.lan.lan_sync import LANSyncManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sb-node")

SAMPLE_CHUNKS = [
    "Bittensor is a decentralized ML network. Miners contribute intelligence and earn TAO tokens.",
    "SuperBrain is a local-first knowledge network on Bittensor. Private by default, public by choice.",
    "RAG enhances LLM outputs by retrieving relevant documents from a vector database before generating.",
    "The Two-Pool Privacy Model ensures all knowledge is private by default. Users explicitly share.",
    "Delta sync uses a 5-step protocol: Handshake, Manifest, Diff, Transfer, Confirm.",
    "Ed25519 signatures ensure every knowledge chunk is cryptographically signed by its originator.",
    "LAN sync uses mDNS to auto-discover peers on the same WiFi. No configuration needed.",
    "Yuma Consensus aggregates validator weights using stake-weighted averaging for TAO distribution.",
    "KnowledgeSyncSynapse lets validators pull knowledge chunks from miners over Bittensor.",
    "Offline-first AI processes data locally. Better privacy, lower latency, zero API costs.",
]


async def main():
    parser = argparse.ArgumentParser(description="SuperBrain LAN Sync Node")
    parser.add_argument("--seed", action="store_true", help="Add 10 sample knowledge chunks")
    parser.add_argument("--add", type=str, help="Add a custom knowledge chunk")
    parser.add_argument("--port", type=int, default=8384, help="Sync server port (default: 8384)")
    parser.add_argument("--static", type=str, help="Static peer IP:PORT (skip mDNS discovery)")
    parser.add_argument("--interval", type=int, default=10, help="Sync interval in seconds (default: 10)")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database (default: temp dir)")
    args = parser.parse_args()

    priv, pub = generate_node_keypair()
    node_id = f"node-{os.getpid()}"

    # Database: use provided path or create temp
    if args.db:
        db_path = os.path.abspath(args.db)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    else:
        db_path = os.path.join(tempfile.mkdtemp(prefix="sb-"), "sync.db")

    queue = SyncQueue(db_path=db_path)
    logger.info(f"Database: {db_path}")

    if args.seed:
        added = 0
        for text in SAMPLE_CHUNKS:
            c = KnowledgeChunk(
                content=text,
                origin_node_id=node_id,
                pool_visibility="public",
                shared_at=time.time(),
            )
            c.sign(priv)
            if queue.add_to_queue(c):
                added += 1
        logger.info(f"Seeded {added} new chunks ({len(SAMPLE_CHUNKS) - added} already existed)")

    if args.add:
        c = KnowledgeChunk(
            content=args.add,
            origin_node_id=node_id,
            pool_visibility="public",
            shared_at=time.time(),
        )
        c.sign(priv)
        if queue.add_to_queue(c):
            logger.info("Added custom chunk")
        else:
            logger.info("Custom chunk already exists in queue")

    # Static peers
    static_peers = None
    if args.static:
        h, p = args.static.rsplit(":", 1)
        static_peers = [(h, int(p))]
        logger.info(f"Static peer: {h}:{p}")

    mgr = LANSyncManager(
        sync_queue=queue,
        node_id=node_id,
        private_key=priv,
        public_key=pub,
        port=args.port,
        sync_interval=args.interval,
        static_peers=static_peers,
    )

    # Graceful shutdown
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await mgr.start()
    m = queue.get_manifest(node_id)
    logger.info(f"NODE STARTED | id={node_id} port={args.port} chunks={len(m.chunk_hashes)}")
    logger.info("Waiting for peers... (Ctrl+C to stop)")

    while not stop.is_set():
        await asyncio.sleep(10)
        m = queue.get_manifest(node_id)
        peers = mgr._discovery.peers
        sent = sum(r.total_sent for r in mgr._sync_records.values())
        recv = sum(r.total_received for r in mgr._sync_records.values())
        logger.info(f"chunks={len(m.chunk_hashes)} peers={len(peers)} sent={sent} recv={recv}")
        for p in peers:
            logger.info(f"  peer: {p.node_id[:12]}... @ {p.host}:{p.port}")

    await mgr.stop()
    m = queue.get_manifest(node_id)
    logger.info(f"STOPPED | final chunks={len(m.chunk_hashes)}")


if __name__ == "__main__":
    asyncio.run(main())
