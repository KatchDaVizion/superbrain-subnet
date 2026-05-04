#!/usr/bin/env python3
"""SuperBrain LAN + I2P Sync Node — standalone peer for syncing knowledge chunks.

Run this on a friend's machine (or a second terminal) to sync knowledge
with a SuperBrain miner or another sync node on the same network.

Usage:
  # Seed 10 sample chunks + auto-discover peers via mDNS
  python3 run_sync_node.py --seed

  # Connect to a specific LAN peer (e.g. miner on Tailscale)
  python3 run_sync_node.py --seed --static 100.x.x.x:8385

  # Join the SuperBrain I2P mesh (both bootstrap nodes built-in)
  python3 run_sync_node.py --seed

  # Add extra I2P peers beyond the built-in bootstrap nodes
  python3 run_sync_node.py --seed --i2p-peers "base64dest1:alias1,base64dest2:alias2"

  # Use the miner's database directly (shared DB)
  python3 run_sync_node.py --seed --db ~/superbrain-dev/subnet/miner_sync_queue.db

  # Add a custom knowledge chunk
  python3 run_sync_node.py --add "Bittensor subnets are specialized task networks."

  # Disable I2P sync (LAN only)
  python3 run_sync_node.py --seed --no-i2p
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

# ── Known I2P bootstrap nodes ──────────────────────────────────────────────
# These are persistent SAM destinations (key file at data/i2p-sync.keys on each server).
# Full base64 private-key-derived public destinations — stable across restarts.
#
# Frankfurt sync-node  46.225.114.202  (persistent key: data/i2p-sync.keys)
FRANKFURT_I2P_DEST = (
    "B9gQNaaB6-uXkWW8FH56EHixv8zXUIdHlTcqUGgDT2lGVatO16KFHf9MoNbnLWIx-2WDvKlnL9G6ltaI"
    "D~e1oE67YAywlVxR-YpJQVuHiMrKJiVQxx4fgd2SaN1YeDfVtH445Ox2bb48wFd4LtIWYaLDgiEayNYVR"
    "~ikWslagxxBDs79~Zv~fEcgFa1cfXItcD0842gIwlxxHHY8sA--0O14iIFXTrIChIVymmEu3HuDBeYk-Rd"
    "oVhUpWsSq9fa~8f1LKm2N9Vma46Ix2~GVei6uQ3SlRg4lyAqNZQ9L~VzWefNucypORi359dhCb-XgyErvZ"
    "JStJgYiiEusHdFBW3joayUMqxbh3ae5Nxi7Zj-A3v6N42aydw9IkQ-Lm2fMYACrTJbjtXZOBEzXl8MLIR"
    "DkahGu1jD5zIPeAmeAhenFY~0Mcw-HZPvJhZ~dgNVIqp6E~7ndObWgpmAaoJPZWJCH0nbe4UAGqvrOkBZa"
    "zc4LZWpu9aaIdMMiYim1qtCuAAAAaDrEck8odK27KWJL45g6QdbckjovzRKX7CFB3NAk2lsTTXPxzdcWl7d"
    "jh5GOvISOwIIyBlCWpdhrMDAK8GqhREuqASuxj5a12aPpD9tXVr~A5gZXtPzB9qSzjtZ~9NUEhvu3p5GcD"
    "WWZsrLVkKLY-lQYcOpa5Yc6XsWQFn-WFD~8ibT3IqMTvs5k6uiZYPCsra1QGETDv8HUeebObHZB2yLPJI"
    "HTbBSuhzfq9aengIItZTYBtxXcsxd9qzvu0ktWjktVbJTntVFev6aCbRxvLXrO7HfU0kAX-ezTEnskSVf1"
    "8eMJhxMVBncjZ1Z-IsnkIj1VYYebblcwmcwx4HuZhF8yJXp2ITpjV2gpmUkuNOy9Tg7q"
)

# Helsinki sync-node   89.167.61.241  (persistent key: data/i2p-sync.keys)
HELSINKI_I2P_DEST = (
    "diGWuHmkh0W56Oauw2n7TAVFZh1AnvVDAo-7iAL6zRcgZTzUdQgSKsfJpwQn7Z8U1HQH4JPR2T-6dbe~E"
    "-k70ik-DHZBWvuJh-9fD5wcGAXyB~7fsfPxNjuAumk1dIKIsfV-pm-PrpyBN2z5tC~X6yXsBhfcpEwC1Rb"
    "awN2i-09puhAhyxOWNMcvOov~XvO-FTpMrX1bZbsgq82v1vDwCLP6mMiwl3-a0SVhscdn4XCEdPBUq8P6A"
    "9RNgw-OZMF52VJRSKGz2frdWJySYjpyw0D8n76siKAZCP8gcXR0P~ia7San-bQ6u49WHXyDmCJRJ5mcz0xU"
    "jGL6bOM9en6xSwCsHKRFKjdF6yvHt3--8LrcFqTWYv~NsjRZnxzusQTwuirw8LpUZUoXdtTvVKpsHopl6N"
    "lK-XPSm3jHpwAV61CKpRLTieXp3g6pBNhU2Z7lK40SU5NegaSc89taMztG1mrOm4eYvf1myw1uRyGy5jV2"
    "rkuc0jvUVuvBTZTw4-fwAAAAa4YE0zLPca118Gh0Lvx96jlhfkraoAseWODKbCfC3A31DHJeIwrGh6kqIns"
    "yFXweh4Dh4TpeNY4iz2ylZSeoESUjdf3pZ60NIV-N6VJb-F2dCKCQtkIHkNGJNLECr-vfqSaFdfOcfeOv8"
    "A2AurMk~~UqDV5-1QN1JmyMbkM74D-Y4f2FoX-OVRiF7tqQxNrGrJVwdv5MI4pYmW1y~DIOCKSFc5~8FA"
    "w71P8-Txc4qR8Ivc04X6mWyxqxEDhIHDx3HrbevIBssxtifpKpcDShg~C~~Oq7tPCB6gZpxAqDCSlJFWPV"
    "l92lUPziTeQ7Qc2zXREmRpkUPOzpcjf0RhqfW5-SePtlaUQqWHsUJOYQ~7iCpP9w"
)

# ── Sample data ───────────────────────────────────────────────────────────
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


def _filter_self(peers, key_path):
    """Remove any peer whose destination matches the local node's own I2P key."""
    if not os.path.exists(key_path):
        return peers
    local_dest = None
    try:
        with open(key_path, 'r') as _f:
            for _line in _f:
                if _line.startswith("DEST="):
                    local_dest = _line.strip()[5:]
                    break
    except Exception:
        return peers
    if not local_dest:
        return peers
    return [(d, n) for (d, n) in peers if d != local_dest]


def _dedup_peers(peers):
    """Remove duplicate destinations, keeping first occurrence."""
    seen = set()
    result = []
    for d, n in peers:
        if d not in seen:
            seen.add(d)
            result.append((d, n))
    return result


def _parse_i2p_peers(raw: str):
    """Parse comma-separated 'dest:alias' or 'dest' strings into [(dest, alias)] list."""
    peers = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry and len(entry.split(":")[0]) > 64:
            # base64 dest contains '~' and '+', not ':' — the ':' here is alias separator
            parts = entry.split(":", 1)
            peers.append((parts[0].strip(), parts[1].strip()))
        else:
            peers.append((entry, ""))
    return peers


async def _start_i2p(queue, node_id, priv, pub, static_peers, key_path):
    """Start I2P sync manager. Returns manager or None if I2P is unavailable."""
    if not static_peers:
        logger.info("I2P: no peers configured — server-only mode (accepting inbound)")
    try:
        from sync.i2p.sync_manager import I2PSyncManager
        i2p_queue = SyncQueue(db_path=queue.db_path)
        mgr = I2PSyncManager(
            sync_queue=i2p_queue,
            node_id=node_id + "-i2p",
            private_key=priv,
            public_key=pub,
            static_peers=static_peers,
            sync_interval=120,
            key_path=key_path,
        )
        await asyncio.wait_for(mgr.start(), timeout=45)
        dest = mgr.local_destination or ""
        logger.info(
            f"I2P sync started — local dest: {dest[:32]}... "
            f"peers={len(static_peers)} key={os.path.basename(key_path) if key_path else 'transient'}"
        )
        logger.info(f"I2P FULL DEST: {dest}")
        return mgr
    except asyncio.TimeoutError:
        logger.warning("I2P sync disabled: SAM startup timed out after 45s (LAN sync unaffected)")
        return None
    except Exception as e:
        logger.warning(f"I2P sync disabled: {e} (LAN sync unaffected)")
        return None


async def main():
    parser = argparse.ArgumentParser(description="SuperBrain LAN + I2P Sync Node")
    parser.add_argument("--seed", action="store_true", help="Add 10 sample knowledge chunks")
    parser.add_argument("--add", type=str, help="Add a custom knowledge chunk")
    parser.add_argument("--port", type=int, default=8385, help="Sync server port (default: 8385)")
    parser.add_argument("--static", type=str, help="Static LAN peer IP:PORT (skip mDNS discovery)")
    parser.add_argument("--interval", type=int, default=10, help="Sync interval in seconds (default: 10)")
    parser.add_argument("--db", type=str, default=None, help="Path to SQLite database (default: temp dir)")
    parser.add_argument("--no-i2p", action="store_true", help="Disable I2P sync (LAN only)")
    parser.add_argument(
        "--i2p-peers", type=str, default="",
        help="Extra I2P peers: comma-separated 'dest64:alias' or 'dest64' (built-in bootstrap nodes always included)",
    )
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

    # LAN static peers
    static_peers = None
    if args.static:
        h, p = args.static.rsplit(":", 1)
        static_peers = [(h, int(p))]
        logger.info(f"Static LAN peer: {h}:{p}")

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

    # I2P sync — optional, starts after LAN is confirmed up
    i2p_mgr = None
    if not args.no_i2p:
        # Build peer list: built-in bootstrap nodes + --i2p-peers arg
        i2p_peers = []
        if FRANKFURT_I2P_DEST:
            i2p_peers.append((FRANKFURT_I2P_DEST, "frankfurt"))
        if HELSINKI_I2P_DEST:
            i2p_peers.append((HELSINKI_I2P_DEST, "helsinki"))
        if args.i2p_peers:
            i2p_peers.extend(_parse_i2p_peers(args.i2p_peers))

        # Persistent key file lives next to the database
        key_path = os.path.join(os.path.dirname(db_path), "i2p-sync.keys")

        # Remove own destination and duplicates (happens when node is in built-in list)
        i2p_peers = _dedup_peers(_filter_self(i2p_peers, key_path))

        i2p_mgr = await _start_i2p(queue, node_id, priv, pub, i2p_peers, key_path)

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
        if i2p_mgr:
            stats = i2p_mgr.stats
            dest = stats.get("local_destination") or ""
            i2p_sent = sum(v["total_sent"] for v in stats["sync_records"].values())
            i2p_recv = sum(v["total_received"] for v in stats["sync_records"].values())
            logger.info(
                f"  i2p: dest={dest[:16]}... peers={stats['peers_known']} "
                f"sent={i2p_sent} recv={i2p_recv}"
            )

    await mgr.stop()
    if i2p_mgr:
        try:
            await asyncio.wait_for(i2p_mgr.stop(), timeout=10)
        except Exception:
            pass
    m = queue.get_manifest(node_id)
    logger.info(f"STOPPED | final chunks={len(m.chunk_hashes)}")


if __name__ == "__main__":
    asyncio.run(main())
