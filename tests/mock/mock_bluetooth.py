# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# Mock Bluetooth Socket — In-memory socket pair for testing RFCOMM transport
#
# Simulates a Bluetooth RFCOMM byte-stream connection using shared
# threading.Condition-guarded byte buffers. Unlike MockTransport
# (which is message-oriented via asyncio.Queue), this simulates
# raw byte-stream behavior where partial reads/writes are possible.

import threading
from typing import Tuple


class MockBluetoothSocket:
    """In-memory socket that behaves like a Bluetooth RFCOMM socket.

    Uses a shared byte buffer with a threading.Condition for blocking
    recv() until data is available (matches real socket behavior).
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._send_buf = None  # Set by create_mock_bluetooth_pair
        self._recv_buf = None  # Set by create_mock_bluetooth_pair
        self._recv_cond = None
        self._send_cond = None
        self._closed = False
        self._connected = False

    def connect(self, addr_channel) -> None:
        """Simulate connecting to a remote device."""
        self._connected = True

    def bind(self, addr_channel) -> None:
        """Simulate binding to a local address."""
        pass

    def listen(self, backlog: int) -> None:
        """Simulate listening for connections."""
        pass

    def accept(self):
        """Not implemented for mock pairs (use create_mock_bluetooth_pair)."""
        raise NotImplementedError("Use create_mock_bluetooth_pair for testing")

    def send(self, data: bytes) -> int:
        """Write bytes to the outgoing buffer (peer reads from this)."""
        if self._closed:
            raise ConnectionError(f"MockBluetoothSocket {self.name} is closed")
        with self._send_cond:
            self._send_buf.extend(data)
            self._send_cond.notify_all()
        return len(data)

    def recv(self, bufsize: int) -> bytes:
        """Read up to bufsize bytes from the incoming buffer (blocking)."""
        if self._closed:
            raise ConnectionError(f"MockBluetoothSocket {self.name} is closed")
        with self._recv_cond:
            # Wait until data is available or peer closes
            while len(self._recv_buf) == 0:
                if self._closed:
                    return b""
                self._recv_cond.wait(timeout=5.0)
                if self._closed:
                    return b""
                if len(self._recv_buf) == 0:
                    # Timeout — avoid infinite blocking in tests
                    continue

            # Read up to bufsize bytes
            count = min(bufsize, len(self._recv_buf))
            data = bytes(self._recv_buf[:count])
            del self._recv_buf[:count]
            return data

    def close(self) -> None:
        """Close the socket."""
        self._closed = True
        # Wake up any blocked recv()
        if self._recv_cond:
            with self._recv_cond:
                self._recv_cond.notify_all()
        if self._send_cond:
            with self._send_cond:
                self._send_cond.notify_all()


def create_mock_bluetooth_pair(
    name_a: str = "bt-A",
    name_b: str = "bt-B",
) -> Tuple[MockBluetoothSocket, MockBluetoothSocket]:
    """Create a pair of connected MockBluetoothSocket instances.

    What A sends, B receives, and vice versa (byte-stream semantics).
    """
    buf_a_to_b = bytearray()
    buf_b_to_a = bytearray()

    cond_a_to_b = threading.Condition()
    cond_b_to_a = threading.Condition()

    sock_a = MockBluetoothSocket(name=name_a)
    sock_a._send_buf = buf_a_to_b
    sock_a._recv_buf = buf_b_to_a
    sock_a._send_cond = cond_a_to_b
    sock_a._recv_cond = cond_b_to_a
    sock_a._connected = True

    sock_b = MockBluetoothSocket(name=name_b)
    sock_b._send_buf = buf_b_to_a
    sock_b._recv_buf = buf_a_to_b
    sock_b._send_cond = cond_b_to_a
    sock_b._recv_cond = cond_a_to_b
    sock_b._connected = True

    return sock_a, sock_b
