# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain I2P SAM Transport Layer
#
# Implements TransportLayer over I2P SAM v3.1 streams.
#
# SAM (Simple Anonymous Messaging) provides a TCP bridge to the I2P network.
# After the SAM handshake and STREAM CONNECT/ACCEPT, the TCP socket becomes
# a raw byte stream to the remote peer — just like Bluetooth RFCOMM.
#
# This transport adds 4-byte big-endian length-prefix framing (same as
# BluetoothTransport) since SAM streams are byte-stream-oriented.
#
# All socket operations are blocking, wrapped with run_in_executor().

import asyncio
import logging
import struct
import uuid
from typing import Optional

from sync.protocol.delta_sync import TransportLayer

logger = logging.getLogger(__name__)

FRAME_HEADER_SIZE = 4  # 4-byte big-endian length prefix
DEFAULT_SAM_HOST = "127.0.0.1"
DEFAULT_SAM_PORT = 7656


class SAMError(Exception):
    """Error communicating with the SAM bridge."""
    pass


class I2PTransport(TransportLayer):
    """TransportLayer over an I2P SAM stream connection.

    Adds length-prefix framing on top of the raw byte stream:
      send: [4-byte length][payload]
      receive: read 4-byte length, then read exactly that many bytes
    """

    def __init__(self, sock, name: str = "i2p"):
        self._sock = sock
        self.name = name
        self.closed = False
        self.bytes_sent = 0
        self.bytes_received = 0

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the socket (blocking)."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(f"I2P connection {self.name} closed during recv")
            buf.extend(chunk)
        return bytes(buf)

    def _send_all(self, data: bytes) -> None:
        """Send all bytes (blocking), handling partial sends."""
        total_sent = 0
        while total_sent < len(data):
            sent = self._sock.send(data[total_sent:])
            if sent == 0:
                raise ConnectionError(f"I2P connection {self.name} broken during send")
            total_sent += sent

    async def send(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionError(f"I2P transport {self.name} is closed")

        frame = struct.pack(">I", len(data)) + data
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_all, frame)
        self.bytes_sent += len(data)

    async def receive(self) -> bytes:
        if self.closed:
            raise ConnectionError(f"I2P transport {self.name} is closed")

        loop = asyncio.get_event_loop()

        header = await loop.run_in_executor(None, self._recv_exact, FRAME_HEADER_SIZE)
        length = struct.unpack(">I", header)[0]

        if length == 0:
            return b""

        payload = await loop.run_in_executor(None, self._recv_exact, length)
        self.bytes_received += len(payload)
        return payload

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._sock.close)
            except Exception:
                pass


# ── SAM protocol helpers ────────────────────────────────────────

def _recv_line(sock) -> str:
    """Read a newline-terminated SAM response (blocking)."""
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise SAMError("SAM bridge closed connection")
        buf += chunk
    return buf.decode('ascii').strip()


async def _sam_handshake(sock) -> None:
    """Perform SAM HELLO handshake.

    Sends:   HELLO VERSION MIN=3.1 MAX=3.1
    Expects: HELLO REPLY RESULT=OK VERSION=3.1
    """
    loop = asyncio.get_event_loop()

    def _do():
        sock.send(b"HELLO VERSION MIN=3.1 MAX=3.1\n")
        return _recv_line(sock)

    response = await loop.run_in_executor(None, _do)
    if "RESULT=OK" not in response:
        raise SAMError(f"SAM HELLO failed: {response}")


async def _sam_create_session(sock, session_id: str) -> str:
    """Create a SAM stream session, return the local destination (base64).

    Sends:   SESSION CREATE STYLE=STREAM ID=<id> DESTINATION=TRANSIENT
    Expects: SESSION STATUS RESULT=OK DESTINATION=<base64>
    """
    loop = asyncio.get_event_loop()

    def _do():
        cmd = f"SESSION CREATE STYLE=STREAM ID={session_id} DESTINATION=TRANSIENT\n"
        sock.send(cmd.encode('ascii'))
        return _recv_line(sock)

    response = await loop.run_in_executor(None, _do)
    if "RESULT=OK" not in response:
        raise SAMError(f"SAM SESSION CREATE failed: {response}")

    for part in response.split():
        if part.startswith("DESTINATION="):
            return part[len("DESTINATION="):]
    raise SAMError(f"No DESTINATION in SESSION CREATE response: {response}")


async def _sam_stream_connect(sock, session_id: str, destination: str) -> None:
    """Connect a SAM stream to a remote I2P destination.

    After success, the socket becomes a raw byte stream to the remote peer.

    Sends:   STREAM CONNECT ID=<id> DESTINATION=<dest>
    Expects: STREAM STATUS RESULT=OK
    """
    loop = asyncio.get_event_loop()

    def _do():
        cmd = f"STREAM CONNECT ID={session_id} DESTINATION={destination}\n"
        sock.send(cmd.encode('ascii'))
        return _recv_line(sock)

    response = await loop.run_in_executor(None, _do)
    if "RESULT=OK" not in response:
        raise SAMError(f"SAM STREAM CONNECT failed: {response}")


async def connect_to_destination(
    destination: str,
    sam_host: str = DEFAULT_SAM_HOST,
    sam_port: int = DEFAULT_SAM_PORT,
    name: str = "",
    _socket_factory=None,
) -> I2PTransport:
    """Connect to an I2P peer via the SAM bridge.

    1. Opens TCP connection to SAM bridge
    2. HELLO handshake
    3. Creates transient session
    4. STREAM CONNECT to destination
    5. Returns I2PTransport wrapping the now-raw-stream socket
    """
    import socket as socket_mod

    loop = asyncio.get_event_loop()

    if _socket_factory:
        sock = _socket_factory()
    else:
        def _open():
            s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
            s.connect((sam_host, sam_port))
            return s
        sock = await loop.run_in_executor(None, _open)

    session_id = f"superbrain-{uuid.uuid4().hex[:8]}"

    await _sam_handshake(sock)
    await _sam_create_session(sock, session_id)
    await _sam_stream_connect(sock, session_id, destination)

    return I2PTransport(sock, name=name or f"i2p->{destination[:16]}...")
