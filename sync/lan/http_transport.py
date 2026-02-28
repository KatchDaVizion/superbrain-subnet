# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain WebSocket Transport Layer
#
# Implements the TransportLayer ABC over aiohttp WebSocket.
# Works for both client and server sides — aiohttp's
# ClientWebSocketResponse and web.WebSocketResponse both
# support send_bytes() and receive() with compatible signatures.

import logging
from typing import Optional

import aiohttp
from aiohttp import WSMsgType

from sync.protocol.delta_sync import TransportLayer

logger = logging.getLogger(__name__)

SYNC_WS_PATH = "/sync/ws"


class WebSocketTransport(TransportLayer):
    """TransportLayer over an aiohttp WebSocket connection."""

    def __init__(self, ws, name: str = ""):
        self._ws = ws
        self.name = name
        self.closed = False
        self.bytes_sent = 0
        self.bytes_received = 0

    async def send(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionError(f"WebSocket transport {self.name} is closed")
        await self._ws.send_bytes(data)
        self.bytes_sent += len(data)

    async def receive(self) -> bytes:
        if self.closed:
            raise ConnectionError(f"WebSocket transport {self.name} is closed")
        msg = await self._ws.receive()
        if msg.type == WSMsgType.BINARY:
            self.bytes_received += len(msg.data)
            return msg.data
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            self.closed = True
            raise ConnectionError(f"WebSocket closed by remote: {msg.data}")
        elif msg.type == WSMsgType.ERROR:
            self.closed = True
            raise ConnectionError(f"WebSocket error: {self._ws.exception()}")
        else:
            raise ValueError(f"Unexpected WebSocket message type: {msg.type}")

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                await self._ws.close()
            except Exception:
                pass


async def connect_to_peer(
    host: str,
    port: int,
    session: Optional[aiohttp.ClientSession] = None,
    name: str = "",
) -> WebSocketTransport:
    """
    Open a WebSocket connection to a peer's sync server.
    Returns a WebSocketTransport ready for use with run_sync.
    """
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    url = f"http://{host}:{port}{SYNC_WS_PATH}"
    ws = await session.ws_connect(url)
    transport = WebSocketTransport(ws, name=name or f"client->{host}:{port}")
    # Track session for cleanup if we created it
    transport._session = session if own_session else None
    return transport
