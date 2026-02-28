# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Bluetooth RFCOMM Transport Layer
#
# Implements TransportLayer over Bluetooth RFCOMM (Serial Port Profile).
#
# Key difference from WebSocket: RFCOMM is a byte stream, not message-oriented.
# This transport adds 4-byte big-endian length-prefix framing so the receiver
# knows when a complete message has arrived.
#
# All socket operations are blocking, so they are wrapped with
# asyncio.run_in_executor() to avoid blocking the event loop.

import asyncio
import logging
import struct
from typing import Optional

from sync.protocol.delta_sync import TransportLayer

logger = logging.getLogger(__name__)

FRAME_HEADER_SIZE = 4  # 4-byte big-endian length prefix


class BluetoothTransport(TransportLayer):
    """TransportLayer over a Bluetooth RFCOMM socket (or any socket-like object).

    Adds length-prefix framing on top of the raw byte stream:
      send: [4-byte length][payload]
      receive: read 4-byte length, then read exactly that many bytes
    """

    def __init__(self, sock, name: str = ""):
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
                raise ConnectionError(f"Bluetooth connection {self.name} closed during recv")
            buf.extend(chunk)
        return bytes(buf)

    def _send_all(self, data: bytes) -> None:
        """Send all bytes (blocking), handling partial sends."""
        total_sent = 0
        while total_sent < len(data):
            sent = self._sock.send(data[total_sent:])
            if sent == 0:
                raise ConnectionError(f"Bluetooth connection {self.name} broken during send")
            total_sent += sent

    async def send(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionError(f"Bluetooth transport {self.name} is closed")

        # Frame: 4-byte length header + payload
        frame = struct.pack(">I", len(data)) + data
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_all, frame)
        self.bytes_sent += len(data)

    async def receive(self) -> bytes:
        if self.closed:
            raise ConnectionError(f"Bluetooth transport {self.name} is closed")

        loop = asyncio.get_event_loop()

        # Read 4-byte length header
        header = await loop.run_in_executor(None, self._recv_exact, FRAME_HEADER_SIZE)
        length = struct.unpack(">I", header)[0]

        if length == 0:
            return b""

        # Read exactly `length` bytes of payload
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


async def connect_to_device(
    address: str,
    channel: int = 1,
    name: str = "",
) -> BluetoothTransport:
    """
    Connect to a Bluetooth device via RFCOMM.
    Requires PyBluez (import bluetooth).
    Returns a BluetoothTransport ready for run_sync.
    """
    try:
        import bluetooth
    except ImportError:
        raise RuntimeError(
            "PyBluez not installed. Install with: pip install pybluez\n"
            "Or use mock sockets for testing."
        )

    loop = asyncio.get_event_loop()

    def _connect():
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((address, channel))
        return sock

    sock = await loop.run_in_executor(None, _connect)
    return BluetoothTransport(sock, name=name or f"bt->{address}")
