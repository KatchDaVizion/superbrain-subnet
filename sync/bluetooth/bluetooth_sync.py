# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Bluetooth Sync Manager
#
# Orchestrates Bluetooth knowledge sync:
#   1. Discovers nearby Bluetooth peers (or uses static list)
#   2. Runs RFCOMM server for incoming connections
#   3. Connects to peers and syncs on demand or periodically
#
# Usage:
#   manager = BluetoothSyncManager(sync_queue)
#   await manager.start()
#   await manager.scan_and_sync()
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
from sync.bluetooth.rfcomm_transport import BluetoothTransport, connect_to_device
from sync.bluetooth.rfcomm_server import BluetoothSyncServer
from sync.bluetooth.discovery import (
    BluetoothDiscovery,
    BluetoothPeerInfo,
    StaticBluetoothPeers,
    DEFAULT_CHANNEL,
)

logger = logging.getLogger(__name__)

SYNC_INTERVAL = 60.0       # BT scans are expensive — longer interval
MIN_SYNC_INTERVAL = 10.0   # BT is slower than LAN


@dataclass
class SyncRecord:
    """Tracks sync history with a specific peer."""
    peer_address: str
    last_sync: float = 0.0
    total_sent: int = 0
    total_received: int = 0
    sync_count: int = 0
    last_error: str = ""


class BluetoothSyncManager:
    """Orchestrates Bluetooth knowledge sync: discovery + server + sync."""

    def __init__(
        self,
        sync_queue: SyncQueue,
        node_id: Optional[str] = None,
        private_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
        channel: int = DEFAULT_CHANNEL,
        sync_interval: float = SYNC_INTERVAL,
        static_peers: Optional[List[tuple]] = None,
        mock_connect=None,
    ):
        if private_key is None or public_key is None:
            private_key, public_key = generate_node_keypair()
        if node_id is None:
            node_id = str(uuid.uuid4())

        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._public_key = public_key
        self._channel = channel
        self._sync_interval = sync_interval
        self._sync_records: Dict[str, SyncRecord] = {}
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        # Allow injecting mock connect function for testing
        self._connect_fn = mock_connect or connect_to_device

        # Discovery
        if static_peers is not None:
            self._discovery = StaticBluetoothPeers(static_peers)
        else:
            self._discovery = BluetoothDiscovery(
                on_peer_found=self._on_peer_discovered,
            )

        # Server
        self._server = BluetoothSyncServer(
            sync_queue=sync_queue,
            node_id=node_id,
            private_key=private_key,
            channel=channel,
        )

    async def start(self) -> None:
        """Start server and discovery."""
        self._running = True
        await self._server.start()
        await self._discovery.start()
        logger.info(f"Bluetooth sync started: node={self._node_id[:8]}...")

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
        logger.info("Bluetooth sync stopped")

    async def sync_with_peer(self, peer: BluetoothPeerInfo) -> Optional[TransferResult]:
        """Run a single sync session with a Bluetooth peer."""
        record = self._sync_records.get(peer.address)
        if record and (time.time() - record.last_sync) < MIN_SYNC_INTERVAL:
            return None

        transport = None
        try:
            transport = await self._connect_fn(
                peer.address, peer.channel, name=peer.name,
            )

            result = await run_sync(
                self._queue,
                self._node_id,
                self._private_key,
                b"",  # Peer public key — relaxed for Bluetooth
                transport,
            )

            # Update sync record
            if peer.address not in self._sync_records:
                self._sync_records[peer.address] = SyncRecord(peer_address=peer.address)
            rec = self._sync_records[peer.address]
            rec.last_sync = time.time()
            rec.total_sent += result.chunks_sent
            rec.total_received += result.chunks_received
            rec.sync_count += 1
            if result.errors:
                rec.last_error = result.errors[-1]

            return result

        except Exception as e:
            logger.warning(f"BT sync failed with {peer.address}: {e}")
            if peer.address not in self._sync_records:
                self._sync_records[peer.address] = SyncRecord(peer_address=peer.address)
            self._sync_records[peer.address].last_error = str(e)
            return None
        finally:
            if transport and not transport.closed:
                await transport.close()

    async def sync_all(self) -> Dict[str, Optional[TransferResult]]:
        """Sync with all known peers."""
        results = {}
        for peer in self._discovery.peers:
            result = await self.sync_with_peer(peer)
            results[peer.address] = result
        return results

    async def scan_and_sync(self, scan_duration: int = 8) -> Dict[str, Optional[TransferResult]]:
        """Scan for peers, then sync with all found."""
        await self._discovery.scan(duration=scan_duration)
        return await self.sync_all()

    def _on_peer_discovered(self, peer: BluetoothPeerInfo) -> None:
        """Callback when a new peer is found."""
        logger.info(f"Discovered BT peer: {peer.name} ({peer.address})")

    @property
    def stats(self) -> dict:
        """Current manager statistics."""
        return {
            "node_id": self._node_id,
            "running": self._running,
            "channel": self._channel,
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
