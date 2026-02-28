# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain I2P SAM Sync Server
#
# Accepts incoming I2P connections via the SAM bridge's STREAM ACCEPT
# command and runs the delta sync protocol on each client.
#
# The server creates one SAM session on start(), then repeatedly issues
# STREAM ACCEPT on new TCP connections (reusing the session ID).
# Each accepted connection becomes a raw byte stream to the remote peer.

import asyncio
import logging
import uuid
from typing import Optional

from sync.protocol.delta_sync import run_sync
from sync.queue.sync_queue import SyncQueue
from sync.i2p.transport import (
    I2PTransport, SAMError, _recv_line,
    _sam_handshake, _sam_create_session,
    DEFAULT_SAM_HOST, DEFAULT_SAM_PORT,
)

logger = logging.getLogger(__name__)


class I2PSyncServer:
    """SAM-based server that accepts I2P sync connections."""

    def __init__(
        self,
        sync_queue: SyncQueue,
        node_id: str,
        private_key: bytes,
        sam_host: str = DEFAULT_SAM_HOST,
        sam_port: int = DEFAULT_SAM_PORT,
        _socket_factory=None,
    ):
        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._sam_host = sam_host
        self._sam_port = sam_port
        self._socket_factory = _socket_factory
        self._running = False
        self._session_id: Optional[str] = None
        self._local_destination: Optional[str] = None
        self._control_sock = None
        self._accept_task: Optional[asyncio.Task] = None
        self._active_syncs = 0

    async def start(self) -> None:
        """Start the I2P server: create SAM session, begin accepting."""
        import socket as socket_mod

        loop = asyncio.get_event_loop()

        # Open control connection to SAM bridge
        if self._socket_factory:
            self._control_sock = self._socket_factory()
        else:
            def _open():
                s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
                s.connect((self._sam_host, self._sam_port))
                return s
            self._control_sock = await loop.run_in_executor(None, _open)

        # SAM handshake + session creation
        self._session_id = f"superbrain-server-{uuid.uuid4().hex[:8]}"
        await _sam_handshake(self._control_sock)
        self._local_destination = await _sam_create_session(
            self._control_sock, self._session_id,
        )

        self._running = True
        self._accept_task = asyncio.create_task(self._accept_loop())
        logger.info(f"I2P sync server started: {self._local_destination[:32]}...")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._accept_task:
            self._accept_task.cancel()
            try:
                await self._accept_task
            except asyncio.CancelledError:
                pass
        if self._control_sock:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._control_sock.close)
            except Exception:
                pass
            self._control_sock = None

    @property
    def local_destination(self) -> Optional[str]:
        """The server's I2P destination (base64), available after start()."""
        return self._local_destination

    @property
    def active_syncs(self) -> int:
        return self._active_syncs

    async def _accept_loop(self) -> None:
        """Accept incoming SAM streams in a loop."""
        import socket as socket_mod

        while self._running:
            try:
                loop = asyncio.get_event_loop()

                # Each STREAM ACCEPT needs a new TCP connection to SAM bridge
                if self._socket_factory:
                    accept_sock = self._socket_factory()
                else:
                    def _open():
                        s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
                        s.connect((self._sam_host, self._sam_port))
                        return s
                    accept_sock = await loop.run_in_executor(None, _open)

                # Handshake on the new connection
                await _sam_handshake(accept_sock)

                # Issue STREAM ACCEPT (blocks until a peer connects)
                def _do_accept():
                    cmd = f"STREAM ACCEPT ID={self._session_id}\n"
                    accept_sock.send(cmd.encode('ascii'))
                    return _recv_line(accept_sock)

                response = await loop.run_in_executor(None, _do_accept)

                if "RESULT=OK" not in response:
                    logger.warning(f"SAM STREAM ACCEPT failed: {response}")
                    try:
                        await loop.run_in_executor(None, accept_sock.close)
                    except Exception:
                        pass
                    continue

                # Socket is now a raw stream to the remote peer
                asyncio.create_task(self._handle_client(accept_sock))

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.error(f"I2P accept error: {e}")
                    await asyncio.sleep(1)

    async def _handle_client(self, sock) -> None:
        """Handle a single I2P sync connection."""
        self._active_syncs += 1
        transport = None
        try:
            logger.info(f"I2P sync connection (active: {self._active_syncs})")
            transport = I2PTransport(sock, name="i2p-server<-peer")
            result = await run_sync(
                self._queue,
                self._node_id,
                self._private_key,
                b"",
                transport,
            )
            logger.info(
                f"I2P sync: sent={result.chunks_sent}, "
                f"received={result.chunks_received}, "
                f"errors={len(result.errors)}"
            )
        except Exception as e:
            logger.error(f"I2P sync error: {e}")
        finally:
            self._active_syncs -= 1
            if transport and not transport.closed:
                await transport.close()
