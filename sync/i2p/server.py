# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain I2P SAM Sync Server
#
# Accepts incoming I2P connections via the SAM bridge's STREAM ACCEPT
# command and runs the delta sync protocol on each client.
#
# The server creates one SAM session on start(), then repeatedly issues
# STREAM ACCEPT on new TCP connections (reusing the session ID).
# Each accepted connection becomes a raw byte stream to the remote peer.
#
# Session recovery: i2pd drops the SAM session when its control socket
# goes idle (no keepalive, firewall TCP-idle timeout, or i2pd restart).
# _restart_session() recreates the session in-place so the accept loop
# and outbound syncs automatically pick up the new session ID.

import asyncio
import logging
import os
import uuid
from typing import Optional

from sync.protocol.delta_sync import run_sync
from sync.queue.sync_queue import SyncQueue
from sync.i2p.transport import (
    I2PTransport, SAMError, _recv_line,
    _sam_handshake, _sam_create_session, _sam_dest_generate,
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
        key_path: Optional[str] = None,
    ):
        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._sam_host = sam_host
        self._sam_port = sam_port
        self._socket_factory = _socket_factory
        self._key_path = key_path
        self._running = False
        self._session_id: Optional[str] = None
        self._local_destination: Optional[str] = None
        self._control_sock = None
        self._accept_task: Optional[asyncio.Task] = None
        self._active_syncs = 0
        self._restart_lock = asyncio.Lock()

    # ── Session lifecycle ────────────────────────────────────────────

    async def _create_session(self) -> None:
        """Open (or reopen) a SAM control socket and create a session.

        Sets self._session_id and self._local_destination.
        Closes any existing control socket first.
        """
        import socket as socket_mod

        loop = asyncio.get_event_loop()

        # Close stale control socket; i2pd drops the session when it closes
        if self._control_sock:
            try:
                await loop.run_in_executor(None, self._control_sock.close)
            except Exception:
                pass
            self._control_sock = None

        # New control connection with TCP keepalive so the session survives
        # firewall idle-timeout between sync rounds
        if self._socket_factory:
            self._control_sock = self._socket_factory()
        else:
            def _open():
                s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
                s.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_KEEPALIVE, 1)
                s.connect((self._sam_host, self._sam_port))
                return s
            self._control_sock = await loop.run_in_executor(None, _open)

        self._session_id = f"superbrain-server-{uuid.uuid4().hex[:8]}"
        await _sam_handshake(self._control_sock)

        # Persistent I2P identity: load key file or generate + save
        destination = "TRANSIENT"
        if self._key_path:
            if os.path.exists(self._key_path):
                with open(self._key_path, 'r') as _f:
                    for _line in _f:
                        if _line.startswith("PRIV="):
                            destination = _line.strip()[5:]
                            break
                logger.debug(f"Loaded persistent I2P key from {self._key_path}")
            else:
                priv, pub = await _sam_dest_generate(self._control_sock)
                os.makedirs(os.path.dirname(os.path.abspath(self._key_path)), exist_ok=True)
                with open(self._key_path, 'w') as _f:
                    _f.write(f"PRIV={priv}\n")
                destination = priv
                logger.info(f"Generated new persistent I2P key → {self._key_path}")

        self._local_destination = await _sam_create_session(
            self._control_sock, self._session_id, destination=destination,
        )

        # Save public destination alongside private key on first generation
        if self._key_path and destination != "TRANSIENT":
            lines = []
            if os.path.exists(self._key_path):
                with open(self._key_path, 'r') as _f:
                    lines = _f.readlines()
            if not any(l.startswith("DEST=") for l in lines):
                with open(self._key_path, 'a') as _f:
                    _f.write(f"DEST={self._local_destination}\n")

    async def _restart_session(self) -> None:
        """Recreate the SAM session after i2pd has dropped it.

        Serialised by _restart_lock so concurrent INVALID_ID detections
        (from the accept loop and outbound sync) don't double-recreate.
        If a restart is already in progress the second caller simply waits
        for it to finish and returns — the session will already be fresh.
        """
        if self._restart_lock.locked():
            # Another coroutine is already restarting; wait for it
            async with self._restart_lock:
                return
        async with self._restart_lock:
            logger.warning("SAM session dropped by i2pd — recreating session")
            try:
                await self._create_session()
                logger.info(
                    f"SAM session recreated: sid={self._session_id} "
                    f"dest={self._local_destination[:32]}..."
                )
            except Exception as e:
                logger.error(f"SAM session recreate failed: {e}")
                raise

    # ── Public API ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the I2P server: create SAM session, begin accepting."""
        await self._create_session()
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
    def session_id(self) -> Optional[str]:
        """The SAM session ID, available after start()."""
        return self._session_id

    @property
    def active_syncs(self) -> int:
        return self._active_syncs

    # ── Accept loop ──────────────────────────────────────────────────

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

                # Issue STREAM ACCEPT (blocks until a peer connects).
                # After RESULT=OK, i2pd sends a second line: the peer's b64
                # destination.  Read and discard it so the socket is a clean
                # binary stream before passing it to I2PTransport.
                def _do_accept():
                    cmd = f"STREAM ACCEPT ID={self._session_id}\n"
                    accept_sock.send(cmd.encode('ascii'))
                    status = _recv_line(accept_sock)
                    if "RESULT=OK" in status:
                        try:
                            _recv_line(accept_sock)  # peer dest b64 — discard
                        except Exception:
                            pass
                    return status

                response = await loop.run_in_executor(None, _do_accept)

                if "RESULT=OK" not in response:
                    try:
                        await loop.run_in_executor(None, accept_sock.close)
                    except Exception:
                        pass
                    if "Already accepting" in response:
                        # Another accept is pending — back off
                        await asyncio.sleep(5)
                    elif "INVALID_ID" in response:
                        # Session was dropped by i2pd — recreate it
                        try:
                            await self._restart_session()
                        except Exception:
                            await asyncio.sleep(5)
                    else:
                        logger.warning(f"SAM STREAM ACCEPT failed: {response}")
                        await asyncio.sleep(2)
                    continue

                # Back off if too many in-flight syncs (stale connection burst protection)
                if self._active_syncs >= 3:
                    try:
                        await loop.run_in_executor(None, accept_sock.close)
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                    continue

                # Socket is now a raw stream to the remote peer
                asyncio.create_task(self._handle_client(accept_sock))
                # Yield to event loop so tasks can start before next accept
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning(f"I2P accept error: {e}")
                    await asyncio.sleep(2)

    async def _handle_client(self, sock) -> None:
        """Handle a single I2P sync connection."""
        self._active_syncs += 1
        transport = None
        try:
            logger.debug(f"I2P sync connection (active: {self._active_syncs})")
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
            err = str(e).lower()
            if "closed" in err or "recv" in err or "eof" in err or "reset" in err:
                logger.debug(f"I2P connection closed: {e}")
            else:
                logger.warning(f"I2P sync error: {e}")
        finally:
            self._active_syncs -= 1
            if transport and not transport.closed:
                await transport.close()
