# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# Mock Transport Layer — In-memory transport for testing delta sync protocol
#
# Creates a pair of connected MockTransport instances that communicate
# via shared asyncio.Queue objects. No real network involved.

import asyncio
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.delta_sync import TransportLayer


class MockTransport(TransportLayer):
    """
    In-memory transport that sends/receives via asyncio.Queue.
    Create a pair with create_mock_pair().
    """

    def __init__(self, send_queue: asyncio.Queue, recv_queue: asyncio.Queue, name: str = ""):
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.name = name
        self.closed = False
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0

    async def send(self, data: bytes) -> None:
        if self.closed:
            raise ConnectionError(f"Transport {self.name} is closed")
        await self.send_queue.put(data)
        self.bytes_sent += len(data)
        self.messages_sent += 1

    async def receive(self) -> bytes:
        if self.closed:
            raise ConnectionError(f"Transport {self.name} is closed")
        data = await self.recv_queue.get()
        self.bytes_received += len(data)
        self.messages_received += 1
        return data

    async def close(self) -> None:
        self.closed = True


def create_mock_pair(name_a: str = "node-A", name_b: str = "node-B") -> Tuple[MockTransport, MockTransport]:
    """
    Create a pair of connected MockTransport instances.
    What A sends, B receives, and vice versa.
    """
    q_a_to_b = asyncio.Queue()
    q_b_to_a = asyncio.Queue()

    transport_a = MockTransport(send_queue=q_a_to_b, recv_queue=q_b_to_a, name=name_a)
    transport_b = MockTransport(send_queue=q_b_to_a, recv_queue=q_a_to_b, name=name_b)

    return transport_a, transport_b
