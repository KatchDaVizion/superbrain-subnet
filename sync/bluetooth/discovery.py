# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Bluetooth Device Discovery
#
# Discovers nearby Bluetooth devices running SuperBrain by scanning
# for the SPP service UUID. Falls back gracefully if PyBluez not installed.

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SPP_UUID = "00001101-0000-1000-8000-00805F9B34FB"
DEFAULT_CHANNEL = 1


@dataclass
class BluetoothPeerInfo:
    """A discovered Bluetooth peer."""
    address: str
    name: str = ""
    channel: int = DEFAULT_CHANNEL
    last_seen: float = 0.0


class BluetoothDiscovery:
    """Discover nearby Bluetooth devices running SuperBrain."""

    def __init__(
        self,
        on_peer_found: Optional[Callable[[BluetoothPeerInfo], None]] = None,
    ):
        self._on_peer_found = on_peer_found
        self._peers: Dict[str, BluetoothPeerInfo] = {}
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._has_bluetooth = False

        try:
            import bluetooth
            self._has_bluetooth = True
        except ImportError:
            logger.warning("PyBluez not installed — Bluetooth discovery disabled")

    async def start(self) -> None:
        """Start periodic background scanning."""
        self._running = True

    async def stop(self) -> None:
        """Stop scanning."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

    async def scan(self, duration: int = 8) -> List[BluetoothPeerInfo]:
        """Scan for nearby Bluetooth devices with SuperBrain SPP service.

        Returns list of discovered peers. Runs blocking scan in executor.
        Returns empty list if PyBluez not installed.
        """
        if not self._has_bluetooth:
            return []

        import bluetooth

        loop = asyncio.get_event_loop()

        def _scan():
            found = []
            # Discover nearby devices
            devices = bluetooth.discover_devices(
                duration=duration, lookup_names=True, lookup_class=False,
            )

            for addr, name in devices:
                # Check if device has SuperBrain SPP service
                services = bluetooth.find_service(uuid=SPP_UUID, address=addr)
                if services:
                    channel = services[0].get("port", DEFAULT_CHANNEL)
                    peer = BluetoothPeerInfo(
                        address=addr,
                        name=name or addr,
                        channel=channel,
                        last_seen=time.time(),
                    )
                    found.append(peer)

            return found

        peers = await loop.run_in_executor(None, _scan)

        # Update peer dict
        for peer in peers:
            if peer.address not in self._peers:
                if self._on_peer_found:
                    self._on_peer_found(peer)
                logger.info(f"Discovered BT peer: {peer.name} ({peer.address})")
            self._peers[peer.address] = peer

        return peers

    @property
    def peers(self) -> List[BluetoothPeerInfo]:
        """Currently known peers."""
        return list(self._peers.values())


class StaticBluetoothPeers:
    """Fallback with explicit Bluetooth peer addresses."""

    def __init__(self, peers: List[tuple]):
        """peers: list of (address, channel) or (address, name, channel) tuples."""
        self._peers: List[BluetoothPeerInfo] = []
        for entry in peers:
            if len(entry) == 3:
                address, name, channel = entry
            else:
                address, channel = entry
                name = address
            self._peers.append(BluetoothPeerInfo(
                address=address, name=name, channel=channel,
            ))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def scan(self, duration: int = 8) -> List[BluetoothPeerInfo]:
        return list(self._peers)

    @property
    def peers(self) -> List[BluetoothPeerInfo]:
        return list(self._peers)
