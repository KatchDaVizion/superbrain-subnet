# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain I2P Sync Manager
#
# Orchestrates I2P knowledge sync:
#   1. Discovers peers via static list or address book file
#   2. Runs SAM server for incoming connections
#   3. Connects to peers and syncs on demand or periodically
#
# Usage:
#   manager = I2PSyncManager(sync_queue, static_peers=[("dest_base64",)])
#   await manager.start()
#   await manager.sync_all()
#   await manager.stop()

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from sync.protocol.delta_sync import run_sync, TransferResult
from sync.protocol.pool_model import generate_node_keypair
from sync.queue.sync_queue import SyncQueue
from sync.i2p.transport import connect_to_destination
from sync.i2p.server import I2PSyncServer
from sync.i2p.discovery import I2PAddressBook, I2PPeerInfo, StaticI2PPeers

logger = logging.getLogger(__name__)

SYNC_INTERVAL = 120.0      # I2P is slower — longer interval than LAN/BT
MIN_SYNC_INTERVAL = 30.0


@dataclass
class SyncRecord:
    """Tracks sync history with a specific peer."""
    peer_destination: str
    last_sync: float = 0.0
    total_sent: int = 0
    total_received: int = 0
    sync_count: int = 0
    last_error: str = ""


class I2PSyncManager:
    """Orchestrates I2P knowledge sync: discovery + server + sync."""

    def __init__(
        self,
        sync_queue: SyncQueue,
        node_id: Optional[str] = None,
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
        sync_interval: float = SYNC_INTERVAL,
        static_peers: Optional[List[tuple]] = None,
        address_book_path: Optional[str] = None,
        mock_connect=None,
        _server_socket_factory=None,
    ):
        if private_key is None or public_key is None:
            private_key, public_key = generate_node_keypair()
        if node_id is None:
            node_id = str(uuid.uuid4())

        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._public_key = public_key
        self._sync_interval = sync_interval
        self._sync_records: Dict[str, SyncRecord] = {}
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._connect_fn = mock_connect or connect_to_destination

        # Discovery
        if static_peers is not None:
            self._discovery = StaticI2PPeers(static_peers)
        elif address_book_path is not None:
            self._discovery = I2PAddressBook(
                address_book_path,
                on_peer_found=self._on_peer_discovered,
            )
        else:
            self._discovery = StaticI2PPeers([])

        # Server
        self._server = I2PSyncServer(
            sync_queue=sync_queue,
            node_id=node_id,
            private_key=private_key,
            _socket_factory=_server_socket_factory,
        )

    async def start(self) -> None:
        """Start server and discovery."""
        self._running = True
        await self._server.start()
        await self._discovery.start()
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"I2P sync started: node={self._node_id[:8]}...")

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
        logger.info("I2P sync stopped")

    async def sync_with_peer(self, peer: I2PPeerInfo) -> Optional[TransferResult]:
        """Run a single sync session with an I2P peer."""
        record = self._sync_records.get(peer.destination)
        if record and (time.time() - record.last_sync) < MIN_SYNC_INTERVAL:
            return None

        transport = None
        try:
            transport = await self._connect_fn(
                peer.destination, name=peer.name,
            )

            result = await run_sync(
                self._queue,
                self._node_id,
                self._private_key,
                b"",
                transport,
            )

            # Update sync record
            if peer.destination not in self._sync_records:
                self._sync_records[peer.destination] = SyncRecord(
                    peer_destination=peer.destination,
                )
            rec = self._sync_records[peer.destination]
            rec.last_sync = time.time()
            rec.total_sent += result.chunks_sent
            rec.total_received += result.chunks_received
            rec.sync_count += 1
            if result.errors:
                rec.last_error = result.errors[-1]

            return result

        except Exception as e:
            logger.warning(f"I2P sync failed with {peer.destination[:16]}...: {e}")
            if peer.destination not in self._sync_records:
                self._sync_records[peer.destination] = SyncRecord(
                    peer_destination=peer.destination,
                )
            self._sync_records[peer.destination].last_error = str(e)
            return None
        finally:
            if transport and not transport.closed:
                await transport.close()

    async def sync_all(self) -> Dict[str, Optional[TransferResult]]:
        """Sync with all known peers."""
        results = {}
        for peer in self._discovery.peers:
            result = await self.sync_with_peer(peer)
            results[peer.destination] = result
        return results

    async def _sync_loop(self) -> None:
        """Periodic sync loop."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                peers = self._discovery.peers
                if peers:
                    logger.debug(f"I2P sync round: {len(peers)} peers")
                    await self.sync_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"I2P sync loop error: {e}")

    def _on_peer_discovered(self, peer: I2PPeerInfo) -> None:
        """Callback when a new peer is found in the address book."""
        logger.info(f"Discovered I2P peer: {peer.name} ({peer.destination[:16]}...)")

    @property
    def local_destination(self) -> Optional[str]:
        """The server's I2P destination, if started."""
        return self._server.local_destination

    @property
    def stats(self) -> dict:
        """Current manager statistics."""
        return {
            "node_id": self._node_id,
            "running": self._running,
            "local_destination": self._server.local_destination,
            "peers_known": len(self._discovery.peers),
            "active_syncs": self._server.active_syncs,
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
