# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Bluetooth RFCOMM Sync Server
#
# Listens for incoming Bluetooth RFCOMM connections and runs the
# delta sync protocol on each client. Advertises an SDP service
# so peers can discover us via Bluetooth service discovery.
#
# Requires PyBluez for real Bluetooth hardware.
# Falls back gracefully if not installed.

import asyncio
import logging
from typing import Optional

from sync.protocol.delta_sync import run_sync
from sync.queue.sync_queue import SyncQueue
from sync.bluetooth.rfcomm_transport import BluetoothTransport

logger = logging.getLogger(__name__)

SPP_UUID = "00001101-0000-1000-8000-00805F9B34FB"
SERVICE_NAME = "SuperBrain Sync"
DEFAULT_CHANNEL = 1


class BluetoothSyncServer:
    """RFCOMM server that accepts Bluetooth sync connections."""

    def __init__(
        self,
        sync_queue: SyncQueue,
        node_id: str,
        private_key: bytes,
        channel: int = DEFAULT_CHANNEL,
    ):
        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._channel = channel
        self._running = False
        self._server_sock = None
        self._accept_task: Optional[asyncio.Task] = None
        self._active_syncs = 0
        self._has_bluetooth = False

    async def start(self) -> None:
        """Start the RFCOMM server. No-op if PyBluez not installed."""
        try:
            import bluetooth
            self._has_bluetooth = True
        except ImportError:
            logger.warning("PyBluez not installed — Bluetooth server disabled")
            return

        loop = asyncio.get_event_loop()

        def _bind():
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.bind(("", self._channel))
            sock.listen(5)
            # Advertise SDP service
            bluetooth.advertise_service(
                sock, SERVICE_NAME,
                service_id=SPP_UUID,
                service_classes=[SPP_UUID, bluetooth.SERIAL_PORT_CLASS],
                profiles=[bluetooth.SERIAL_PORT_PROFILE],
            )
            return sock

        self._server_sock = await loop.run_in_executor(None, _bind)
        self._running = True
        self._accept_task = asyncio.create_task(self._accept_loop())
        logger.info(f"Bluetooth sync server started on channel {self._channel}")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._accept_task:
            self._accept_task.cancel()
            try:
                await self._accept_task
            except asyncio.CancelledError:
                pass
        if self._server_sock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._server_sock.close)
            self._server_sock = None

    async def _accept_loop(self) -> None:
        """Accept incoming connections in a thread executor."""
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                client_sock, client_addr = await loop.run_in_executor(
                    None, self._server_sock.accept,
                )
                asyncio.create_task(self._handle_client(client_sock, client_addr))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.error(f"Bluetooth accept error: {e}")

    async def _handle_client(self, sock, addr) -> None:
        """Handle a single Bluetooth sync connection."""
        self._active_syncs += 1
        bt_addr = addr[0] if isinstance(addr, tuple) else str(addr)
        logger.info(f"Bluetooth sync from {bt_addr} (active: {self._active_syncs})")

        transport = BluetoothTransport(sock, name=f"bt-server<-{bt_addr}")

        try:
            result = await run_sync(
                self._queue,
                self._node_id,
                self._private_key,
                b"",  # Peer public key — relaxed for Bluetooth
                transport,
            )
            logger.info(
                f"BT sync with {bt_addr}: sent={result.chunks_sent}, "
                f"received={result.chunks_received}, errors={len(result.errors)}"
            )
        except Exception as e:
            logger.error(f"BT sync error with {bt_addr}: {e}")
        finally:
            self._active_syncs -= 1
            if not transport.closed:
                await transport.close()

    @property
    def active_syncs(self) -> int:
        return self._active_syncs
