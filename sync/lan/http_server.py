# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain HTTP/WebSocket Sync Server
#
# aiohttp-based server that accepts WebSocket sync connections.
# When a peer connects to /sync/ws, the server wraps the WebSocket
# in a WebSocketTransport and runs run_sync() as the server side.

import logging
from typing import Optional

from aiohttp import web

from sync.protocol.delta_sync import run_sync
from sync.queue.sync_queue import SyncQueue
from sync.lan.http_transport import WebSocketTransport, SYNC_WS_PATH

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8384


class HttpSyncServer:
    """HTTP server that accepts WebSocket sync connections."""

    def __init__(
        self,
        sync_queue: SyncQueue,
        node_id: str,
        private_key: bytes,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
    ):
        self._queue = sync_queue
        self._node_id = node_id
        self._private_key = private_key
        self._host = host
        self._port = port
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._active_syncs = 0

    async def start(self) -> None:
        """Start the HTTP server."""
        self._app = web.Application()
        self._app.router.add_get(SYNC_WS_PATH, self._handle_sync)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(f"Sync server listening on {self._host}:{self.port}")

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    @property
    def port(self) -> int:
        """Actual bound port (useful when port=0 for ephemeral)."""
        if self._site and self._site._server:
            sockets = self._site._server.sockets
            if sockets:
                return sockets[0].getsockname()[1]
        return self._port

    async def _handle_sync(self, request: web.Request) -> web.WebSocketResponse:
        """Handle incoming WebSocket sync connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._active_syncs += 1
        peer_addr = request.remote
        logger.info(f"Sync connection from {peer_addr} (active: {self._active_syncs})")

        transport = WebSocketTransport(ws, name=f"server<-{peer_addr}")

        try:
            result = await run_sync(
                self._queue,
                self._node_id,
                self._private_key,
                b"",  # Peer public key — relaxed for LAN (skip sig verification)
                transport,
            )
            logger.info(
                f"Sync with {peer_addr}: sent={result.chunks_sent}, "
                f"received={result.chunks_received}, errors={len(result.errors)}"
            )
        except Exception as e:
            logger.error(f"Sync error with {peer_addr}: {e}")
        finally:
            self._active_syncs -= 1
            if not transport.closed:
                await transport.close()

        return ws

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "node_id": self._node_id,
            "active_syncs": self._active_syncs,
            "queue_stats": self._queue.stats(),
        })
