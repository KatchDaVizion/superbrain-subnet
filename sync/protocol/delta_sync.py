# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Delta Sync Protocol — Transport-Agnostic
#
# The same protocol runs over any transport:
#   - HTTP (LAN sync)
#   - Bluetooth SPP (offline sync)
#   - I2P SAM (anonymous sync)
#   - Mock (testing)
#
# Protocol flow:
#   1. Handshake — exchange node IDs, verify protocol version
#   2. Manifest exchange — send/receive public chunk manifests
#   3. Diff — compute what each side is missing
#   4. Transfer — send/receive missing chunks (compressed, signed, integrity-checked)
#   5. Confirm — acknowledge receipt
#
# Each chunk in transfer is:
#   - Serialized to JSON
#   - Compressed with zlib
#   - Framed with 4-byte big-endian length header
#   - Integrity-checked (SHA-256 matches manifest)

import json
import logging
import struct
import zlib
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from sync.protocol.pool_model import (
    KnowledgeChunk,
    SyncManifest,
    SyncDiff,
    compute_content_hash,
    compute_diff,
    sign_data,
    verify_signature,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per compressed chunk
FRAME_HEADER_SIZE = 4  # 4-byte big-endian length prefix


# ── Transport Abstraction ───────────────────────────────────────────

class TransportLayer(ABC):
    """Abstract transport layer — implemented by LAN, Bluetooth, I2P, Mock."""

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """Send raw bytes to the remote peer."""

    @abstractmethod
    async def receive(self) -> bytes:
        """Receive raw bytes from the remote peer."""

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""


# ── Protocol Messages ───────────────────────────────────────────────

class HandshakeResult(BaseModel):
    local_node_id: str
    remote_node_id: str
    success: bool
    protocol_version: str = PROTOCOL_VERSION


class TransferResult(BaseModel):
    chunks_sent: int = 0
    chunks_received: int = 0
    bytes_transferred: int = 0
    errors: List[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ── Wire Format Helpers ─────────────────────────────────────────────

def _frame(data: bytes) -> bytes:
    """Add 4-byte big-endian length prefix to data."""
    return struct.pack(">I", len(data)) + data


def _serialize_message(msg: dict) -> bytes:
    """Serialize a message dict to framed compressed bytes.
    Returns: 4-byte length header + compressed data.
    For transport send, use the raw compressed part (no frame header).
    """
    raw = json.dumps(msg, sort_keys=True).encode("utf-8")
    compressed = zlib.compress(raw)
    return _frame(compressed)


def _deserialize_message(data: bytes) -> dict:
    """Decompress and deserialize a message from bytes (no frame header)."""
    decompressed = zlib.decompress(data)
    return json.loads(decompressed)


def _serialize_chunk(chunk: KnowledgeChunk) -> bytes:
    """Serialize a single chunk to framed compressed bytes.
    Returns: 4-byte length header + compressed data.
    """
    raw = chunk.model_dump_json().encode("utf-8")
    compressed = zlib.compress(raw)
    return _frame(compressed)


def _deserialize_chunk(data: bytes) -> KnowledgeChunk:
    """Decompress and deserialize a chunk from bytes (no frame header)."""
    decompressed = zlib.decompress(data)
    return KnowledgeChunk.model_validate_json(decompressed)


def _compress_message(msg: dict) -> bytes:
    """Serialize + compress a message dict (no frame header)."""
    raw = json.dumps(msg, sort_keys=True).encode("utf-8")
    return zlib.compress(raw)


def _compress_chunk(chunk: KnowledgeChunk) -> bytes:
    """Serialize + compress a chunk (no frame header)."""
    raw = chunk.model_dump_json().encode("utf-8")
    return zlib.compress(raw)


# ── Protocol Functions ──────────────────────────────────────────────

async def handshake(
    local_node_id: str,
    transport: TransportLayer,
) -> HandshakeResult:
    """
    Step 1: Exchange node IDs and verify protocol version.
    Both sides send their handshake, both receive the other's.
    """
    msg = {"type": "handshake", "node_id": local_node_id, "version": PROTOCOL_VERSION}
    await transport.send(_compress_message(msg))

    remote_data = await transport.receive()
    remote_msg = _deserialize_message(remote_data)

    remote_node_id = remote_msg.get("node_id", "")
    remote_version = remote_msg.get("version", "")

    success = (
        remote_msg.get("type") == "handshake"
        and remote_version == PROTOCOL_VERSION
        and bool(remote_node_id)
    )

    if not success:
        logger.warning(f"Handshake failed: version={remote_version}, node={remote_node_id}")

    return HandshakeResult(
        local_node_id=local_node_id,
        remote_node_id=remote_node_id,
        success=success,
        protocol_version=remote_version or PROTOCOL_VERSION,
    )


async def exchange_manifests(
    local_manifest: SyncManifest,
    transport: TransportLayer,
) -> SyncManifest:
    """
    Step 2: Send local manifest, receive remote manifest.
    Returns the remote peer's manifest.
    """
    msg = {"type": "manifest", "manifest": local_manifest.model_dump()}
    await transport.send(_compress_message(msg))

    remote_data = await transport.receive()
    remote_msg = _deserialize_message(remote_data)

    if remote_msg.get("type") != "manifest":
        raise ValueError(f"Expected manifest message, got {remote_msg.get('type')}")

    return SyncManifest.model_validate(remote_msg["manifest"])


async def send_chunks(
    chunks: List[KnowledgeChunk],
    transport: TransportLayer,
) -> int:
    """
    Step 3a: Send chunks to remote peer.
    Each chunk is compressed and framed. Ends with a sentinel message.
    Returns count of chunks sent.
    """
    count = 0
    for chunk in chunks:
        data = _compress_chunk(chunk)
        if len(data) > MAX_CHUNK_SIZE:
            logger.warning(f"Chunk {chunk.content_hash[:16]} exceeds max size, skipping")
            continue
        await transport.send(data)
        count += 1

    # Send end-of-transfer sentinel
    sentinel = _compress_message({"type": "end_transfer", "count": count})
    await transport.send(sentinel)

    return count


async def receive_chunks(
    transport: TransportLayer,
    expected_hashes: Optional[set] = None,
) -> Tuple[List[KnowledgeChunk], List[str]]:
    """
    Step 3b: Receive chunks from remote peer until sentinel.
    Verifies content hash integrity.
    Returns (valid_chunks, errors).
    """
    chunks = []
    errors = []

    while True:
        data = await transport.receive()

        # Try to parse as sentinel first
        try:
            msg = _deserialize_message(data)
            if isinstance(msg, dict) and msg.get("type") == "end_transfer":
                break
        except Exception:
            pass

        # Parse as chunk
        try:
            chunk = _deserialize_chunk(data)

            # Verify content hash
            computed = compute_content_hash(chunk.content)
            if computed != chunk.content_hash:
                errors.append(f"Hash mismatch for {chunk.content_hash[:16]}: expected {chunk.content_hash[:16]}, got {computed[:16]}")
                continue

            # If we have expected hashes, verify this chunk was requested
            if expected_hashes and chunk.content_hash not in expected_hashes:
                errors.append(f"Unexpected chunk {chunk.content_hash[:16]}")
                continue

            chunks.append(chunk)
        except Exception as e:
            errors.append(f"Failed to deserialize chunk: {e}")

    return chunks, errors


async def run_sync(
    local_queue,  # SyncQueue
    local_node_id: str,
    private_key: bytes,
    peer_public_key: bytes,
    transport: TransportLayer,
) -> TransferResult:
    """
    Full sync orchestration: handshake → manifests → diff → transfer → confirm.

    This is the main entry point for a sync session between two nodes.
    Both sides should call run_sync simultaneously (one via each end of the transport).
    """
    result = TransferResult()

    # Step 1: Handshake
    hs = await handshake(local_node_id, transport)
    if not hs.success:
        result.errors.append(f"Handshake failed with {hs.remote_node_id}")
        return result

    logger.info(f"Handshake OK: {local_node_id} <-> {hs.remote_node_id}")

    # Step 2: Exchange manifests
    local_manifest = local_queue.get_manifest(local_node_id)
    local_manifest.sign(private_key)
    remote_manifest = await exchange_manifests(local_manifest, transport)

    # Step 3: Compute diff
    diff = compute_diff(local_manifest, remote_manifest)
    logger.info(f"Diff: need {len(diff.missing_local)} chunks, sending {len(diff.missing_remote)}")

    # Step 4a: Send chunks they need
    if diff.missing_remote:
        chunks_to_send = local_queue.get_chunks_by_hashes(diff.missing_remote)
        sent = await send_chunks(chunks_to_send, transport)
        result.chunks_sent = sent
        result.bytes_transferred += sum(len(c.content) for c in chunks_to_send)
    else:
        # Still send empty sentinel
        await send_chunks([], transport)

    # Step 4b: Receive chunks we need
    expected = set(diff.missing_local)
    received_chunks, errors = await receive_chunks(transport, expected)
    result.chunks_received = len(received_chunks)
    result.errors.extend(errors)
    result.bytes_transferred += sum(len(c.content) for c in received_chunks)

    # Step 5: Ingest received chunks into local queue
    for chunk in received_chunks:
        # Verify signature if we have peer's public key
        if peer_public_key and chunk.signature:
            if not chunk.verify(peer_public_key):
                result.errors.append(f"Signature verification failed for {chunk.content_hash[:16]}")
                continue

        # Ensure chunk is marked as public for queue insertion
        chunk.pool_visibility = "public"
        if chunk.shared_at is None:
            chunk.shared_at = chunk.timestamp

        local_queue.add_to_queue(chunk)
        local_queue.mark_synced(chunk.content_hash, hs.remote_node_id)

    logger.info(
        f"Sync complete: sent={result.chunks_sent}, received={result.chunks_received}, "
        f"errors={len(result.errors)}"
    )

    return result
