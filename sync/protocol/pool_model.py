# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Two-Pool Privacy Model
#
# "Private by default. Public by choice."
#
# Data structures for the two-pool knowledge architecture:
#   - KnowledgeChunk: a single piece of knowledge with pool visibility
#   - SyncManifest: signed list of public chunk hashes for sync negotiation
#   - SyncDiff: computed difference between two manifests
#
# Crypto: Ed25519 when `cryptography` is installed, HMAC-SHA256 fallback for dev.

import hashlib
import hmac
import json
import time
import uuid
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

# ── Crypto backend detection ────────────────────────────────────────

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    _HAS_ED25519 = True
except ImportError:
    _HAS_ED25519 = False


# ── Data Models ─────────────────────────────────────────────────────

class KnowledgeChunk(BaseModel):
    """A single piece of knowledge with pool visibility control."""

    content: str
    content_hash: str = ""  # SHA-256, computed if empty
    origin_node_id: str = ""  # UUID of originating node
    timestamp: float = 0.0  # epoch seconds
    signature: str = ""  # Ed25519 or HMAC signature
    public_key: str = ""  # hex-encoded signer public key (for self-certifying verification)
    pool_visibility: Literal["private", "public"] = "private"
    shared_at: Optional[float] = None  # when marked public
    metadata: Dict = Field(default_factory=dict)  # source_file, page, collection, tags

    def model_post_init(self, __context) -> None:
        if not self.content_hash and self.content:
            self.content_hash = compute_content_hash(self.content)
        if not self.origin_node_id:
            self.origin_node_id = str(uuid.uuid4())
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def make_public(self) -> "KnowledgeChunk":
        """Mark this chunk for the public pool. Returns self for chaining."""
        self.pool_visibility = "public"
        self.shared_at = time.time()
        return self

    def make_private(self) -> "KnowledgeChunk":
        """Revoke public sharing. Returns self for chaining."""
        self.pool_visibility = "private"
        self.shared_at = None
        return self

    @property
    def is_public(self) -> bool:
        return self.pool_visibility == "public"

    @property
    def signable_data(self) -> bytes:
        """The canonical bytes to sign: hash + node_id + timestamp."""
        return f"{self.content_hash}:{self.origin_node_id}:{self.timestamp}".encode()

    def sign(self, private_key: bytes, signer_public_key: bytes = b"") -> "KnowledgeChunk":
        """Sign this chunk and embed the signer's public key. Returns self."""
        self.signature = sign_data(self.signable_data, private_key)
        if signer_public_key:
            self.public_key = signer_public_key.hex()
        elif _HAS_ED25519 and len(private_key) == 32:
            # Derive public key from private key
            try:
                key = Ed25519PrivateKey.from_private_bytes(private_key)
                self.public_key = key.public_key().public_bytes(
                    Encoding.Raw, PublicFormat.Raw
                ).hex()
            except Exception:
                pass
        return self

    def verify(self, public_key: bytes = b"") -> bool:
        """
        Verify this chunk's signature.
        Uses the provided public_key, or falls back to the embedded self.public_key.
        Returns True if valid, False if invalid or unsigned.
        """
        if not self.signature:
            return False
        key = public_key or (bytes.fromhex(self.public_key) if self.public_key else b"")
        if not key:
            return False
        return verify_signature(self.signable_data, self.signature, key)

    def verify_self(self) -> bool:
        """
        Verify this chunk's signature against its own embedded public key.
        This is the self-certifying verification model: the chunk carries
        its own public key, and we verify the signature matches.
        Returns True if signature is valid, False otherwise.
        """
        if not self.signature or not self.public_key:
            return False
        try:
            pub_bytes = bytes.fromhex(self.public_key)
        except ValueError:
            return False
        return verify_signature(self.signable_data, self.signature, pub_bytes)


class ManifestEntry(BaseModel):
    """A single entry in a sync manifest."""
    hash: str
    timestamp: float
    size: int


class SyncManifest(BaseModel):
    """Signed list of public chunk hashes for sync negotiation."""

    node_id: str
    chunks: List[ManifestEntry] = Field(default_factory=list)
    last_sync: float = 0.0
    signature: str = ""

    @property
    def chunk_hashes(self) -> set:
        """Set of all chunk hashes in this manifest."""
        return {entry.hash for entry in self.chunks}

    @property
    def signable_data(self) -> bytes:
        """Canonical bytes for signing: sorted chunk list JSON."""
        sorted_chunks = sorted(
            [{"hash": e.hash, "timestamp": e.timestamp, "size": e.size} for e in self.chunks],
            key=lambda x: x["hash"],
        )
        return json.dumps({"node_id": self.node_id, "chunks": sorted_chunks}, sort_keys=True).encode()

    def sign(self, private_key: bytes) -> "SyncManifest":
        """Sign this manifest. Returns self."""
        self.signature = sign_data(self.signable_data, private_key)
        return self

    def verify(self, public_key: bytes) -> bool:
        """Verify this manifest's signature."""
        if not self.signature:
            return False
        return verify_signature(self.signable_data, self.signature, public_key)


class SyncDiff(BaseModel):
    """Computed difference between two manifests."""
    missing_local: List[str] = Field(default_factory=list)   # hashes we need from remote
    missing_remote: List[str] = Field(default_factory=list)  # hashes they need from us


# ── Crypto Utilities ────────────────────────────────────────────────

def compute_content_hash(content: str) -> str:
    """SHA-256 hash of content. Matches desktop app's documentLoader hash field."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_node_keypair() -> Tuple[bytes, bytes]:
    """
    Generate an Ed25519 keypair for node identity.
    Returns (private_key_bytes, public_key_bytes).
    Falls back to random bytes for HMAC if cryptography not installed.
    """
    if _HAS_ED25519:
        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        public_bytes = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        return private_bytes, public_bytes
    else:
        # HMAC fallback: 32-byte random key used as both private and "public" (symmetric)
        import os
        key = os.urandom(32)
        return key, key


def sign_data(data: bytes, private_key: bytes) -> str:
    """
    Sign data with Ed25519 private key.
    Falls back to HMAC-SHA256 if cryptography not installed.
    Returns hex-encoded signature.
    """
    if _HAS_ED25519:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            key = Ed25519PrivateKey.from_private_bytes(private_key)
            return key.sign(data).hex()
        except Exception:
            pass
    # HMAC fallback
    return hmac.new(private_key, data, hashlib.sha256).hexdigest()


def verify_signature(data: bytes, signature: str, public_key: bytes) -> bool:
    """
    Verify an Ed25519 signature.
    Falls back to HMAC-SHA256 verification if cryptography not installed.
    Returns True if valid.
    """
    if _HAS_ED25519:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            key = Ed25519PublicKey.from_public_bytes(public_key)
            key.verify(bytes.fromhex(signature), data)
            return True
        except Exception:
            return False
    # HMAC fallback
    expected = hmac.new(public_key, data, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Node Identity Registry (TOFU — Trust On First Use) ─────────────

class NodeIdentityRegistry:
    """
    Tracks node_id → public_key bindings.
    First time a node_id is seen with a public_key, the binding is stored.
    Subsequent chunks from the same node_id must use the same public_key.
    This prevents a malicious node from impersonating another.
    """

    def __init__(self):
        self._bindings: Dict[str, str] = {}  # node_id → public_key_hex

    def check_and_register(self, node_id: str, public_key_hex: str) -> bool:
        """
        Check if this node_id/public_key pair is trusted.
        Returns True if trusted (first use or matches existing binding).
        Returns False if node_id is already bound to a DIFFERENT public_key.
        """
        if not node_id or not public_key_hex:
            return True  # Can't enforce without both values

        existing = self._bindings.get(node_id)
        if existing is None:
            # First use — trust and register
            self._bindings[node_id] = public_key_hex
            return True
        return existing == public_key_hex

    def get_public_key(self, node_id: str) -> Optional[str]:
        """Get the registered public key for a node_id, or None."""
        return self._bindings.get(node_id)

    @property
    def registered_count(self) -> int:
        return len(self._bindings)


# ── Diff Computation ────────────────────────────────────────────────

def compute_diff(local: SyncManifest, remote: SyncManifest) -> SyncDiff:
    """
    Compute the difference between local and remote manifests.

    Returns SyncDiff with:
      - missing_local: hashes in remote but not local (we need these)
      - missing_remote: hashes in local but not remote (they need these)
    """
    local_hashes = local.chunk_hashes
    remote_hashes = remote.chunk_hashes

    return SyncDiff(
        missing_local=sorted(remote_hashes - local_hashes),
        missing_remote=sorted(local_hashes - remote_hashes),
    )
