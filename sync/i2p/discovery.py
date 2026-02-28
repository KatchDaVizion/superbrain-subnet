# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain I2P Peer Discovery
#
# I2P has no broadcast discovery mechanism like mDNS or Bluetooth scanning.
# Discovery is done via:
#   1. StaticI2PPeers — explicit base64 destinations (primary)
#   2. I2PAddressBook — file-based name=destination mappings

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class I2PPeerInfo:
    """A known I2P peer."""
    destination: str        # Base64 I2P destination
    name: str = ""          # Human-readable alias
    last_seen: float = 0.0


class StaticI2PPeers:
    """Fallback with explicit I2P peer destinations."""

    def __init__(self, peers: List[tuple]):
        """peers: list of (destination,) or (destination, name) tuples."""
        self._peers: List[I2PPeerInfo] = []
        for entry in peers:
            if len(entry) >= 2:
                self._peers.append(I2PPeerInfo(
                    destination=entry[0], name=entry[1],
                ))
            else:
                self._peers.append(I2PPeerInfo(destination=entry[0]))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def scan(self, duration: int = 0) -> List[I2PPeerInfo]:
        return list(self._peers)

    @property
    def peers(self) -> List[I2PPeerInfo]:
        return list(self._peers)


class I2PAddressBook:
    """File-based peer discovery using an I2P address book.

    Reads destinations from a simple text file:
      name=base64destination
    One per line, # comments allowed.
    """

    def __init__(
        self,
        path: str,
        on_peer_found: Optional[Callable[[I2PPeerInfo], None]] = None,
    ):
        self._path = path
        self._on_peer_found = on_peer_found
        self._peers: Dict[str, I2PPeerInfo] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        await self.scan()

    async def stop(self) -> None:
        self._running = False

    async def scan(self, duration: int = 0) -> List[I2PPeerInfo]:
        try:
            with open(self._path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        name, dest = line.split('=', 1)
                        name = name.strip()
                        dest = dest.strip()
                    else:
                        name = ""
                        dest = line
                    if dest not in self._peers:
                        peer = I2PPeerInfo(
                            destination=dest,
                            name=name,
                            last_seen=time.time(),
                        )
                        self._peers[dest] = peer
                        if self._on_peer_found:
                            self._on_peer_found(peer)
                        logger.info(f"I2P peer from address book: {name} ({dest[:16]}...)")
        except FileNotFoundError:
            logger.warning(f"I2P address book not found: {self._path}")
        except Exception as e:
            logger.warning(f"Error reading I2P address book: {e}")
        return list(self._peers.values())

    @property
    def peers(self) -> List[I2PPeerInfo]:
        return list(self._peers.values())
