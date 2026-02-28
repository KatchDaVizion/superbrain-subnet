# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain LAN Peer Discovery
#
# Discovers SuperBrain peers on the local network using mDNS (zeroconf).
# Falls back to StaticPeerList when mDNS is unavailable.
#
# Service type: _superbrain._tcp.local.

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_superbrain._tcp.local."
DEFAULT_PORT = 8384


@dataclass
class PeerInfo:
    """A discovered LAN peer."""
    node_id: str
    host: str
    port: int
    properties: Dict[str, str] = field(default_factory=dict)
    last_seen: float = 0.0


class Discovery:
    """mDNS-based peer discovery using zeroconf."""

    def __init__(
        self,
        node_id: str,
        port: int = DEFAULT_PORT,
        on_peer_added: Optional[Callable[[PeerInfo], None]] = None,
        on_peer_removed: Optional[Callable[[PeerInfo], None]] = None,
    ):
        self._node_id = node_id
        self._port = port
        self._on_peer_added = on_peer_added
        self._on_peer_removed = on_peer_removed
        self._peers: Dict[str, PeerInfo] = {}
        self._lock = threading.Lock()
        self._async_zc = None
        self._zeroconf = None
        self._browser = None
        self._info = None

    async def start(self) -> None:
        """Register our service and start browsing for peers."""
        try:
            from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo, ServiceStateChange
            from zeroconf.asyncio import AsyncZeroconf
        except ImportError:
            raise RuntimeError(
                "zeroconf package not installed. Install with: pip install zeroconf\n"
                "Or use StaticPeerList as a fallback."
            )

        # Get local IP
        local_ip = self._get_local_ip()

        # Register our service using async API
        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"superbrain-{self._node_id[:8]}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=self._port,
            properties={"node_id": self._node_id},
        )

        self._async_zc = AsyncZeroconf()
        self._zeroconf = self._async_zc.zeroconf
        await self._async_zc.async_register_service(self._info)

        # Browse for peers
        def on_state_change(zeroconf, service_type, name, state_change):
            if state_change == ServiceStateChange.Added:
                info = zeroconf.get_service_info(service_type, name)
                if info and info.properties:
                    node_id = info.properties.get(b"node_id", b"").decode("utf-8", errors="ignore")
                    if node_id and node_id != self._node_id:
                        addresses = info.parsed_addresses()
                        if addresses:
                            peer = PeerInfo(
                                node_id=node_id,
                                host=addresses[0],
                                port=info.port,
                                properties={
                                    k.decode(): v.decode()
                                    for k, v in info.properties.items()
                                },
                                last_seen=time.time(),
                            )
                            with self._lock:
                                self._peers[node_id] = peer
                            if self._on_peer_added:
                                self._on_peer_added(peer)
                            logger.info(f"Discovered peer: {node_id[:8]}... at {peer.host}:{peer.port}")

            elif state_change == ServiceStateChange.Removed:
                # Try to find which peer was removed by service name
                with self._lock:
                    for nid, peer in list(self._peers.items()):
                        if name.startswith(f"superbrain-{nid[:8]}"):
                            del self._peers[nid]
                            if self._on_peer_removed:
                                self._on_peer_removed(peer)
                            logger.info(f"Peer removed: {nid[:8]}...")
                            break

        self._browser = ServiceBrowser(self._zeroconf, SERVICE_TYPE, handlers=[on_state_change])
        logger.info(f"mDNS discovery started: {self._node_id[:8]}... on {local_ip}:{self._port}")

    async def stop(self) -> None:
        """Unregister and stop browsing."""
        if self._browser:
            self._browser.cancel()
            self._browser = None
        if self._info and self._async_zc:
            await self._async_zc.async_unregister_service(self._info)
        if self._async_zc:
            await self._async_zc.async_close()
            self._async_zc = None
            self._zeroconf = None

    @property
    def peers(self) -> List[PeerInfo]:
        """Currently known peers (excluding self)."""
        with self._lock:
            return list(self._peers.values())

    @staticmethod
    def _get_local_ip() -> str:
        """Get local IP address (best effort)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


class StaticPeerList:
    """Fallback when mDNS is unavailable. Accepts explicit peer addresses."""

    def __init__(self, peers: List[tuple]):
        """peers: list of (host, port) or (node_id, host, port) tuples."""
        self._peers: List[PeerInfo] = []
        for entry in peers:
            if len(entry) == 3:
                node_id, host, port = entry
            else:
                host, port = entry
                node_id = f"static-{host}:{port}"
            self._peers.append(PeerInfo(node_id=node_id, host=host, port=port))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @property
    def peers(self) -> List[PeerInfo]:
        return list(self._peers)
