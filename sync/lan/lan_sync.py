# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain LAN Sync Manager
#
# Orchestrates LAN knowledge sync:
#   1. Starts mDNS discovery (or static peer list)
#   2. Starts HTTP/WebSocket sync server
#   3. Periodically connects to discovered peers and syncs
#
# Usage:
#   manager = LANSyncManager(sync_queue, port=8384)
#   await manager.start()
#   # ... runs in background ...
#   await manager.stop()

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp

from sync.protocol.delta_sync import run_sync, TransferResult
from sync.protocol.pool_model import generate_node_keypair
from sync.queue.sync_queue import SyncQueue
from sync.lan.discovery import Discovery, PeerInfo, StaticPeerList, DEFAULT_PORT
from sync.lan.http_server import HttpSyncServer
from sync.lan.http_transport import connect_to_peer

logger = logging.getLogger(__name__)

SYNC_INTERVAL = 30.0
MIN_SYNC_INTERVAL = 5.0


@dataclass
class SyncRecord:
    """Tracks sync history with a specific peer."""
    peer_id: str
    last_sync: float = 0.0
    total_sent: int = 0
    total_received: int = 0
    sync_count: int = 0
    last_error: str = ""


class LANSyncManager:
    """Orchestrates LAN knowledge sync: discovery + server + periodic sync."""

    def __init__(
        self,
        sync_queue: SyncQueue,
        node_id: Optional[str] = None,
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
        port: int = DEFAULT_PORT,
        sync_interval: float = SYNC_INTERVAL,
        static_peers: Optional[List[tuple]] = None,
    ):
        if private_key is None or public_key is None:
            private_key, public_key = generate_node_keypair()
        if node_id is None:
            node_id = str(uuid.uuid4())

        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._public_key = public_key
        self._port = port
        self._sync_interval = sync_interval
        self._sync_records: Dict[str, SyncRecord] = {}
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None

        # Discovery: mDNS or static
        if static_peers is not None:
            self._discovery = StaticPeerList(static_peers)
        else:
            self._discovery = Discovery(
                node_id=node_id,
                port=port,
                on_peer_added=self._on_peer_discovered,
                on_peer_removed=None,
            )

        # Server
        self._server = HttpSyncServer(
            sync_queue=sync_queue,
            node_id=node_id,
            private_key=private_key,
            port=port,
        )

    async def start(self) -> None:
        """Start discovery, server, and sync loop."""
        self._running = True
        await self._server.start()
        await self._discovery.start()
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"LAN sync started: node={self._node_id[:8]}... port={self._server.port}")

    async def stop(self) -> None:
        """Stop everything gracefully."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        await self._discovery.stop()
        await self._server.stop()
        logger.info("LAN sync stopped")

    async def sync_with_peer(self, peer: PeerInfo) -> Optional[TransferResult]:
        """Run a single sync session with a specific peer."""
        record = self._sync_records.get(peer.node_id)
        if record and (time.time() - record.last_sync) < MIN_SYNC_INTERVAL:
            return None

        transport = None
        session = None
        try:
            session = aiohttp.ClientSession()
            transport = await connect_to_peer(
                peer.host, peer.port, session=session, name=peer.node_id,
            )

            result = await run_sync(
                self._queue,
                self._node_id,
                self._private_key,
                b"",  # Peer public key — relaxed for LAN
                transport,
            )

            # Update sync record
            if peer.node_id not in self._sync_records:
                self._sync_records[peer.node_id] = SyncRecord(peer_id=peer.node_id)
            rec = self._sync_records[peer.node_id]
            rec.last_sync = time.time()
            rec.total_sent += result.chunks_sent
            rec.total_received += result.chunks_received
            rec.sync_count += 1
            if result.errors:
                rec.last_error = result.errors[-1]

            return result

        except Exception as e:
            logger.warning(f"Sync failed with {peer.node_id}: {e}")
            if peer.node_id not in self._sync_records:
                self._sync_records[peer.node_id] = SyncRecord(peer_id=peer.node_id)
            self._sync_records[peer.node_id].last_error = str(e)
            return None
        finally:
            if transport and not transport.closed:
                await transport.close()
            if session:
                await session.close()

    async def sync_all(self) -> Dict[str, Optional[TransferResult]]:
        """Sync with all discovered peers."""
        results = {}
        for peer in self._discovery.peers:
            result = await self.sync_with_peer(peer)
            results[peer.node_id] = result
        return results

    async def _sync_loop(self) -> None:
        """Periodic sync loop."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                if not self._running:
                    break
                peers = self._discovery.peers
                if peers:
                    logger.debug(f"Sync round: {len(peers)} peers")
                    await self.sync_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    def _on_peer_discovered(self, peer: PeerInfo) -> None:
        """Callback when a new peer appears on the network."""
        logger.info(f"Discovered peer: {peer.node_id[:8]}... at {peer.host}:{peer.port}")

    @property
    def server_port(self) -> int:
        """Actual bound server port."""
        return self._server.port

    @property
    def stats(self) -> dict:
        """Current manager statistics."""
        return {
            "node_id": self._node_id,
            "running": self._running,
            "port": self._server.port,
            "peers_known": len(self._discovery.peers),
            "sync_records": {
                k: {
                    "last_sync": v.last_sync,
                    "total_sent": v.total_sent,
                    "total_received": v.total_received,
                    "sync_count": v.sync_count,
                    "last_error": v.last_error,
                }
                for k, v in self._sync_records.items()
            },
        }
