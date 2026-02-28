# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Store-and-Forward Batch Format
#
# Batch format for offline-to-online sync:
#   - Node collects public chunks offline
#   - Serializes + compresses + signs into a SyncBatch
#   - Batch is transported as opaque bytes (USB, email, any channel)
#   - Receiving node decompresses + verifies + ingests
#
# Batches are idempotent — re-sending the same batch is safe (dedup by hash).
# Maximum batch size: 10MB compressed.

import json
import uuid
import time
import zlib
from typing import List

from pydantic import BaseModel, Field

from sync.protocol.pool_model import (
    KnowledgeChunk,
    compute_content_hash,
    sign_data,
    verify_signature,
)

MAX_BATCH_SIZE = 10 * 1024 * 1024  # 10MB compressed


class SyncBatch(BaseModel):
    """Store-and-forward batch of public knowledge chunks."""

    node_id: str
    batch_id: str = ""
    created_at: float = 0.0
    chunk_hashes: List[str] = Field(default_factory=list)
    compressed_data: bytes = b""
    signature: str = ""

    def model_post_init(self, __context) -> None:
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if self.created_at == 0.0:
            self.created_at = time.time()

    @classmethod
    def create(
        cls,
        chunks: List[KnowledgeChunk],
        node_id: str,
        private_key: bytes,
    ) -> "SyncBatch":
        """
        Create a batch from a list of public chunks.
        Serializes, compresses, and signs the batch data.
        """
        # Serialize chunks to JSON array
        chunk_dicts = [chunk.model_dump() for chunk in chunks]
        raw_json = json.dumps(chunk_dicts, sort_keys=True).encode("utf-8")
        compressed = zlib.compress(raw_json)

        if len(compressed) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch exceeds max size: {len(compressed)} bytes > {MAX_BATCH_SIZE} bytes. "
                f"Split into smaller batches."
            )

        # Sign the compressed data
        signature = sign_data(compressed, private_key)

        return cls(
            node_id=node_id,
            chunk_hashes=[c.content_hash for c in chunks],
            compressed_data=compressed,
            signature=signature,
        )

    @classmethod
    def extract(
        cls,
        batch: "SyncBatch",
        public_key: bytes,
    ) -> List[KnowledgeChunk]:
        """
        Extract and verify chunks from a batch.
        Verifies signature, decompresses, deserializes, and checks integrity.
        Returns list of valid KnowledgeChunks.
        """
        # Verify signature
        if not verify_signature(batch.compressed_data, batch.signature, public_key):
            raise ValueError("Batch signature verification failed")

        # Decompress and deserialize
        raw_json = zlib.decompress(batch.compressed_data)
        chunk_dicts = json.loads(raw_json)

        chunks = []
        for d in chunk_dicts:
            chunk = KnowledgeChunk.model_validate(d)

            # Verify content hash integrity
            computed = compute_content_hash(chunk.content)
            if computed != chunk.content_hash:
                raise ValueError(
                    f"Hash mismatch for chunk: expected {chunk.content_hash[:16]}, "
                    f"got {computed[:16]}"
                )

            chunks.append(chunk)

        return chunks

    def to_bytes(self) -> bytes:
        """
        Serialize the entire batch to bytes for transport.
        Format: JSON envelope with base64-encoded compressed_data.
        """
        import base64
        envelope = {
            "node_id": self.node_id,
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "chunk_hashes": self.chunk_hashes,
            "compressed_data": base64.b64encode(self.compressed_data).decode("ascii"),
            "signature": self.signature,
        }
        return json.dumps(envelope, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "SyncBatch":
        """
        Deserialize a batch from bytes.
        """
        import base64
        envelope = json.loads(data)
        return cls(
            node_id=envelope["node_id"],
            batch_id=envelope["batch_id"],
            created_at=envelope["created_at"],
            chunk_hashes=envelope["chunk_hashes"],
            compressed_data=base64.b64decode(envelope["compressed_data"]),
            signature=envelope["signature"],
        )
