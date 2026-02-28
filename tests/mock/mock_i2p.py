# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# Mock I2P SAM Socket — In-memory socket pair for testing I2P transport
#
# Two levels of mocking:
#   1. MockSAMSocket + create_mock_i2p_pair() — raw byte-stream pair
#      (use after SAM handshake, for direct I2PTransport testing)
#   2. MockSAMBridge + MockSAMBridgeSocket — full SAM protocol state machine
#      (handles HELLO, SESSION CREATE, STREAM CONNECT/ACCEPT, NAMING LOOKUP)
#
# No real I2P router or network access needed.

import base64
import os
import threading
from typing import Dict, List, Optional, Tuple


class MockSAMSocket:
    """In-memory socket that behaves like a post-SAM-handshake byte stream.

    Uses shared byte buffers with threading.Condition for blocking
    recv() until data is available (matches real socket behavior).
    Same pattern as MockBluetoothSocket.
    """

    def __init__(self, name: str = "mock-sam"):
        self.name = name
        self._send_buf: Optional[bytearray] = None
        self._recv_buf: Optional[bytearray] = None
        self._recv_cond: Optional[threading.Condition] = None
        self._send_cond: Optional[threading.Condition] = None
        self._closed = False
        self._connected = False

    def connect(self, addr_tuple) -> None:
        """Simulate connecting to SAM bridge."""
        self._connected = True

    def send(self, data: bytes) -> int:
        """Write bytes to the outgoing buffer (peer reads from this)."""
        if self._closed:
            raise ConnectionError(f"MockSAMSocket {self.name} is closed")
        with self._send_cond:
            self._send_buf.extend(data)
            self._send_cond.notify_all()
        return len(data)

    def recv(self, bufsize: int) -> bytes:
        """Read up to bufsize bytes from the incoming buffer (blocking)."""
        if self._closed:
            return b""
        with self._recv_cond:
            while len(self._recv_buf) == 0:
                if self._closed:
                    return b""
                self._recv_cond.wait(timeout=5.0)
                if self._closed:
                    return b""
                if len(self._recv_buf) == 0:
                    continue

            count = min(bufsize, len(self._recv_buf))
            data = bytes(self._recv_buf[:count])
            del self._recv_buf[:count]
            return data

    def close(self) -> None:
        """Close the socket."""
        self._closed = True
        if self._recv_cond:
            with self._recv_cond:
                self._recv_cond.notify_all()
        if self._send_cond:
            with self._send_cond:
                self._send_cond.notify_all()


def create_mock_i2p_pair(
    name_a: str = "i2p-A",
    name_b: str = "i2p-B",
) -> Tuple[MockSAMSocket, MockSAMSocket]:
    """Create a pair of connected MockSAMSocket instances.

    What A sends, B receives, and vice versa (byte-stream semantics).
    Use these for direct I2PTransport testing (post-SAM-handshake).
    """
    buf_a_to_b = bytearray()
    buf_b_to_a = bytearray()

    cond_a_to_b = threading.Condition()
    cond_b_to_a = threading.Condition()

    sock_a = MockSAMSocket(name=name_a)
    sock_a._send_buf = buf_a_to_b
    sock_a._recv_buf = buf_b_to_a
    sock_a._send_cond = cond_a_to_b
    sock_a._recv_cond = cond_b_to_a
    sock_a._connected = True

    sock_b = MockSAMSocket(name=name_b)
    sock_b._send_buf = buf_b_to_a
    sock_b._recv_buf = buf_a_to_b
    sock_b._send_cond = cond_b_to_a
    sock_b._recv_cond = cond_a_to_b
    sock_b._connected = True

    return sock_a, sock_b


# ── Full SAM Bridge Mock ─────────────────────────────────────────


class MockSAMBridge:
    """Simulates the I2P SAM bridge for testing.

    Handles SAM protocol commands:
      HELLO VERSION -> HELLO REPLY RESULT=OK VERSION=3.1
      SESSION CREATE -> SESSION STATUS RESULT=OK DESTINATION=<transient>
      STREAM CONNECT -> STREAM STATUS RESULT=OK (transitions to data mode)
      STREAM ACCEPT -> STREAM STATUS RESULT=OK (transitions to data mode)
      NAMING LOOKUP -> NAMING REPLY RESULT=OK VALUE=<dest>

    Usage:
        bridge = MockSAMBridge()
        sock = bridge.create_socket()
        # sock behaves like a TCP connection to a real SAM bridge
    """

    def __init__(self):
        self._sessions: Dict[str, str] = {}        # session_id -> destination
        self._names: Dict[str, str] = {}            # name -> destination
        self._pending_accepts: Dict[str, List] = {} # session_id -> [waiting sockets]
        self._lock = threading.Lock()

    def add_name(self, name: str, destination: str) -> None:
        """Add a name -> destination mapping for NAMING LOOKUP."""
        self._names[name] = destination

    def _generate_destination(self) -> str:
        """Generate a random transient I2P destination (base64)."""
        return base64.b64encode(os.urandom(387)).decode('ascii')

    def create_socket(self) -> 'MockSAMBridgeSocket':
        """Create a socket that talks SAM protocol to this bridge."""
        return MockSAMBridgeSocket(self)


class MockSAMBridgeSocket:
    """Socket connected to a MockSAMBridge.

    Dual-phase:
      1. Command mode — parses SAM protocol text commands
      2. Data mode — raw byte stream (after STREAM CONNECT/ACCEPT)
    """

    def __init__(self, bridge: MockSAMBridge):
        self._bridge = bridge
        self._state = "init"
        self._session_id: Optional[str] = None
        self._destination: Optional[str] = None
        self._closed = False

        # Command-phase: bridge responses queued here
        self._response_buf = bytearray()
        self._response_cond = threading.Condition()

        # Data-phase: raw stream buffers (crossed with peer)
        self._send_buf: Optional[bytearray] = None
        self._recv_buf: Optional[bytearray] = None
        self._send_cond: Optional[threading.Condition] = None
        self._recv_cond: Optional[threading.Condition] = None
        self._data_mode = False

        self._peer: Optional['MockSAMBridgeSocket'] = None

    def connect(self, addr_tuple) -> None:
        """Simulate connecting to SAM bridge."""
        pass

    def send(self, data: bytes) -> int:
        """Send data — SAM commands or raw stream data."""
        if self._closed:
            raise ConnectionError("MockSAMBridgeSocket is closed")

        if self._data_mode and self._send_buf is not None:
            with self._send_cond:
                self._send_buf.extend(data)
                self._send_cond.notify_all()
            return len(data)

        # Command mode: parse SAM protocol
        text = data.decode('ascii').strip()
        self._handle_command(text)
        return len(data)

    def recv(self, bufsize: int) -> bytes:
        """Receive data — SAM responses or raw stream data."""
        if self._closed:
            raise ConnectionError("MockSAMBridgeSocket is closed")

        # Always drain pending SAM responses first (e.g. STREAM STATUS
        # queued by CONNECT/ACCEPT before data mode was set)
        with self._response_cond:
            if len(self._response_buf) > 0:
                count = min(bufsize, len(self._response_buf))
                result = bytes(self._response_buf[:count])
                del self._response_buf[:count]
                return result

        if self._data_mode and self._recv_buf is not None:
            with self._recv_cond:
                while len(self._recv_buf) == 0:
                    if self._closed:
                        return b""
                    self._recv_cond.wait(timeout=5.0)
                    if self._closed:
                        return b""
                    if len(self._recv_buf) == 0:
                        continue
                count = min(bufsize, len(self._recv_buf))
                result = bytes(self._recv_buf[:count])
                del self._recv_buf[:count]
                return result

        # Command mode: read bridge responses
        with self._response_cond:
            while len(self._response_buf) == 0:
                if self._closed:
                    return b""
                self._response_cond.wait(timeout=5.0)
                if self._closed:
                    return b""
                if len(self._response_buf) == 0:
                    continue
            count = min(bufsize, len(self._response_buf))
            result = bytes(self._response_buf[:count])
            del self._response_buf[:count]
            return result

    def close(self) -> None:
        """Close the socket."""
        self._closed = True
        with self._response_cond:
            self._response_cond.notify_all()
        if self._recv_cond:
            with self._recv_cond:
                self._recv_cond.notify_all()
        if self._send_cond:
            with self._send_cond:
                self._send_cond.notify_all()

    def _queue_response(self, text: str) -> None:
        """Queue a SAM protocol response for recv()."""
        with self._response_cond:
            self._response_buf.extend((text + "\n").encode('ascii'))
            self._response_cond.notify_all()

    def _handle_command(self, text: str) -> None:
        """Parse and handle a SAM protocol command."""
        parts = text.split()
        if not parts:
            return

        if parts[0] == "HELLO" and len(parts) >= 2 and parts[1] == "VERSION":
            self._state = "hello_done"
            self._queue_response("HELLO REPLY RESULT=OK VERSION=3.1")

        elif parts[0] == "SESSION" and len(parts) >= 2 and parts[1] == "CREATE":
            params = self._parse_params(parts[2:])
            session_id = params.get("ID", "default")
            self._session_id = session_id
            self._destination = self._bridge._generate_destination()
            with self._bridge._lock:
                self._bridge._sessions[session_id] = self._destination
            self._state = "session_created"
            self._queue_response(
                f"SESSION STATUS RESULT=OK DESTINATION={self._destination}"
            )

        elif parts[0] == "STREAM" and len(parts) >= 2 and parts[1] == "CONNECT":
            params = self._parse_params(parts[2:])
            # Look for a pending ACCEPT to wire up
            peer_sock = None
            with self._bridge._lock:
                for sid, accept_list in self._bridge._pending_accepts.items():
                    if accept_list:
                        peer_sock = accept_list.pop(0)
                        break

            if peer_sock:
                self._setup_data_channel(peer_sock)
                self._queue_response("STREAM STATUS RESULT=OK")
                # Also notify the acceptor
                peer_sock._queue_response("STREAM STATUS RESULT=OK")
            else:
                # No pending accept — go to data mode anyway for simpler testing
                self._queue_response("STREAM STATUS RESULT=OK")
                self._data_mode = True

        elif parts[0] == "STREAM" and len(parts) >= 2 and parts[1] == "ACCEPT":
            params = self._parse_params(parts[2:])
            session_id = params.get("ID", self._session_id)

            # Register as waiting for a connection
            with self._bridge._lock:
                if session_id not in self._bridge._pending_accepts:
                    self._bridge._pending_accepts[session_id] = []
                self._bridge._pending_accepts[session_id].append(self)

            # Don't send response yet — will be sent when a CONNECT matches us
            # But for simple testing where no CONNECT will come, we set a flag
            # Tests using the bridge directly will issue CONNECT after ACCEPT

        elif parts[0] == "NAMING" and len(parts) >= 2 and parts[1] == "LOOKUP":
            params = self._parse_params(parts[2:])
            name = params.get("NAME", "")
            dest = self._bridge._names.get(name)
            if dest:
                self._queue_response(
                    f"NAMING REPLY RESULT=OK NAME={name} VALUE={dest}"
                )
            else:
                self._queue_response(
                    f"NAMING REPLY RESULT=KEY_NOT_FOUND NAME={name}"
                )

    def _setup_data_channel(self, peer: 'MockSAMBridgeSocket') -> None:
        """Wire up shared buffers between two sockets for data mode."""
        buf_a_to_b = bytearray()
        buf_b_to_a = bytearray()
        cond_a_to_b = threading.Condition()
        cond_b_to_a = threading.Condition()

        self._send_buf = buf_a_to_b
        self._recv_buf = buf_b_to_a
        self._send_cond = cond_a_to_b
        self._recv_cond = cond_b_to_a
        self._data_mode = True
        self._peer = peer

        peer._send_buf = buf_b_to_a
        peer._recv_buf = buf_a_to_b
        peer._send_cond = cond_b_to_a
        peer._recv_cond = cond_a_to_b
        peer._data_mode = True
        peer._peer = self

    @staticmethod
    def _parse_params(parts: list) -> dict:
        """Parse KEY=VALUE pairs from SAM command parts."""
        params = {}
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                params[key] = value
        return params
