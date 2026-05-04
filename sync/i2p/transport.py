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

    def __init__(self, sock, name: str = "i2p", _session_sock=None):
        self._sock = sock
        self._session_sock = _session_sock  # kept alive to hold the SAM session
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
            if self._session_sock:
                try:
                    await loop.run_in_executor(None, self._session_sock.close)
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


async def _sam_dest_generate(sock) -> tuple:
    """Generate a persistent I2P keypair via SAM. Returns (priv_b64, pub_b64).

    Sends:   DEST GENERATE SIGNATURE_TYPE=ED_DSA_SHA512_ED25519
    Expects: DEST REPLY PUB=<pub> PRIV=<priv>
    """
    loop = asyncio.get_event_loop()

    def _do():
        sock.send(b"DEST GENERATE SIGNATURE_TYPE=ED_DSA_SHA512_ED25519\n")
        return _recv_line(sock)

    response = await loop.run_in_executor(None, _do)
    pub = priv = None
    for part in response.split():
        if part.startswith("PUB="):
            pub = part[4:]
        elif part.startswith("PRIV="):
            priv = part[5:]
    if not priv or not pub:
        raise SAMError(f"SAM DEST GENERATE failed: {response}")
    return priv, pub


async def _sam_create_session(sock, session_id: str, destination: str = "TRANSIENT") -> str:
    """Create a SAM stream session, return the local destination (base64).

    Sends:   SESSION CREATE STYLE=STREAM ID=<id> DESTINATION=<key|TRANSIENT>
    Expects: SESSION STATUS RESULT=OK DESTINATION=<base64>
    """
    loop = asyncio.get_event_loop()

    def _do():
        cmd = (
            f"SESSION CREATE STYLE=STREAM ID={session_id} DESTINATION={destination}"
            " inbound.length=1 outbound.length=1"
            " inbound.quantity=5 outbound.quantity=5\n"
        )
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
    session_id: Optional[str] = None,
) -> I2PTransport:
    """Connect to an I2P peer via the SAM bridge.

    SAM requires STREAM CONNECT on a fresh socket (type=Unknown), not on the
    SESSION CREATE socket (type=Session).  Two modes:

    session_id provided (preferred): reuse an existing session.
      1. Fresh socket → HELLO → STREAM CONNECT ID=session_id → raw stream.

    session_id=None (standalone): create a new transient session.
      1. Socket C1 → HELLO → SESSION CREATE TRANSIENT → session owned by C1.
      2. Socket C2 → HELLO → STREAM CONNECT ID=new_id → raw stream.
      C1 is kept alive inside the transport so the session stays valid.
    """
    import socket as socket_mod

    loop = asyncio.get_event_loop()

    def _open_sock():
        s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        s.connect((sam_host, sam_port))
        return s

    def _make_sock():
        if _socket_factory:
            return _socket_factory()
        return _open_sock()

    if session_id is not None:
        # Reuse existing session: fresh socket → HELLO → STREAM CONNECT
        stream_sock = await loop.run_in_executor(None, _make_sock)
        await _sam_handshake(stream_sock)
        await _sam_stream_connect(stream_sock, session_id, destination)
        return I2PTransport(stream_sock, name=name or f"i2p->{destination[:16]}...")

    # Standalone: C1 owns the session, C2 carries the stream
    session_sock = await loop.run_in_executor(None, _make_sock)
    new_sid = f"superbrain-{uuid.uuid4().hex[:8]}"
    await _sam_handshake(session_sock)
    await _sam_create_session(session_sock, new_sid)

    stream_sock = await loop.run_in_executor(None, _make_sock)
    await _sam_handshake(stream_sock)
    await _sam_stream_connect(stream_sock, new_sid, destination)
    return I2PTransport(
        stream_sock,
        name=name or f"i2p->{destination[:16]}...",
        _session_sock=session_sock,
    )
