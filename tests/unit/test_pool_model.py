"""SuperBrain Two-Pool Model Test Suite

Tests KnowledgeChunk, SyncManifest, ManifestEntry, SyncDiff,
and all crypto utility functions.
"""
import sys
import os
import hashlib
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sync.protocol.pool_model import (
    KnowledgeChunk,
    ManifestEntry,
    SyncManifest,
    SyncDiff,
    compute_content_hash,
    generate_node_keypair,
    sign_data,
    verify_signature,
    compute_diff,
    _HAS_ED25519,
)

passed = failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


# ── 1. KnowledgeChunk basics ────────────────────────────────────────

def test_chunk_creation():
    print("\n1. KnowledgeChunk creation...")
    chunk = KnowledgeChunk(content="Bittensor is a decentralized AI network.")
    check(chunk.content == "Bittensor is a decentralized AI network.", "Content stored")
    check(chunk.pool_visibility == "private", "Default visibility is private")
    check(chunk.shared_at is None, "shared_at is None when private")
    check(len(chunk.content_hash) == 64, f"Hash is 64 hex chars: {len(chunk.content_hash)}")
    check(len(chunk.origin_node_id) == 36, f"Node ID is UUID: {chunk.origin_node_id}")
    check(chunk.timestamp > 0, f"Timestamp set: {chunk.timestamp:.0f}")
    check(chunk.metadata == {}, "Default metadata is empty dict")
    check(not chunk.is_public, "is_public is False")


def test_chunk_hash_matches_sha256():
    print("\n2. Content hash matches SHA-256...")
    content = "TAO is the native token of Bittensor."
    chunk = KnowledgeChunk(content=content)
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
    check(chunk.content_hash == expected, f"Hash matches: {chunk.content_hash[:16]}...")
    check(chunk.content_hash == compute_content_hash(content), "compute_content_hash matches")


def test_chunk_public_private():
    print("\n3. Public/private transitions...")
    chunk = KnowledgeChunk(content="Test content")
    check(not chunk.is_public, "Starts private")

    chunk.make_public()
    check(chunk.is_public, "Now public")
    check(chunk.pool_visibility == "public", "Visibility is 'public'")
    check(chunk.shared_at is not None, "shared_at set")
    check(chunk.shared_at > 0, f"shared_at is epoch: {chunk.shared_at:.0f}")

    chunk.make_private()
    check(not chunk.is_public, "Back to private")
    check(chunk.shared_at is None, "shared_at cleared")


def test_chunk_metadata():
    print("\n4. Chunk metadata...")
    meta = {
        "source_file": "docs/intro.pdf",
        "page": 3,
        "collection": "sb_docs_v1_ollama",
        "tags": ["bittensor", "subnet"],
    }
    chunk = KnowledgeChunk(content="Content with metadata", metadata=meta)
    check(chunk.metadata["source_file"] == "docs/intro.pdf", "source_file preserved")
    check(chunk.metadata["page"] == 3, "page preserved")
    check(chunk.metadata["tags"] == ["bittensor", "subnet"], "tags preserved")


def test_chunk_serialization():
    print("\n5. Chunk serialization...")
    chunk = KnowledgeChunk(
        content="Serialization test",
        metadata={"key": "value"},
    )
    chunk.make_public()

    # Pydantic model_dump / model_validate round-trip
    data = chunk.model_dump()
    check(isinstance(data, dict), "model_dump returns dict")
    check(data["pool_visibility"] == "public", "Visibility in dump")

    restored = KnowledgeChunk.model_validate(data)
    check(restored.content == chunk.content, "Content survives round-trip")
    check(restored.content_hash == chunk.content_hash, "Hash survives round-trip")
    check(restored.pool_visibility == "public", "Visibility survives round-trip")


# ── 6. Crypto functions ─────────────────────────────────────────────

def test_keypair_generation():
    print("\n6. Keypair generation...")
    priv, pub = generate_node_keypair()
    check(isinstance(priv, bytes), "Private key is bytes")
    check(isinstance(pub, bytes), "Public key is bytes")
    check(len(priv) == 32, f"Private key is 32 bytes: {len(priv)}")
    check(len(pub) == 32, f"Public key is 32 bytes: {len(pub)}")

    # Two keypairs should be different
    priv2, pub2 = generate_node_keypair()
    check(priv != priv2, "Keypairs are unique")


def test_sign_verify_roundtrip():
    print("\n7. Sign/verify round-trip...")
    priv, pub = generate_node_keypair()
    data = b"test data to sign"

    sig = sign_data(data, priv)
    check(isinstance(sig, str), "Signature is hex string")
    check(len(sig) > 0, "Signature is non-empty")

    valid = verify_signature(data, sig, pub)
    check(valid, "Signature verifies correctly")

    # Tampered data should fail
    invalid = verify_signature(b"tampered data", sig, pub)
    check(not invalid, "Tampered data fails verification")

    # Wrong key should fail
    priv2, pub2 = generate_node_keypair()
    wrong_key = verify_signature(data, sig, pub2)
    check(not wrong_key, "Wrong key fails verification")

    backend = "Ed25519" if _HAS_ED25519 else "HMAC-SHA256"
    print(f"    (Crypto backend: {backend})")


def test_chunk_sign_verify():
    print("\n8. Chunk sign/verify...")
    priv, pub = generate_node_keypair()
    chunk = KnowledgeChunk(content="Signed knowledge chunk")
    chunk.sign(priv)

    check(chunk.signature != "", "Chunk is signed")
    check(chunk.verify(pub), "Chunk signature verifies")

    # Tamper with content hash
    original_hash = chunk.content_hash
    chunk.content_hash = "tampered_hash"
    check(not chunk.verify(pub), "Tampered hash fails verify")
    chunk.content_hash = original_hash  # restore


# ── 9. SyncManifest ─────────────────────────────────────────────────

def test_manifest_creation():
    print("\n9. SyncManifest creation...")
    entries = [
        ManifestEntry(hash="abc123", timestamp=1000.0, size=500),
        ManifestEntry(hash="def456", timestamp=1001.0, size=300),
        ManifestEntry(hash="ghi789", timestamp=1002.0, size=700),
    ]
    manifest = SyncManifest(node_id="node-1", chunks=entries, last_sync=999.0)

    check(manifest.node_id == "node-1", "Node ID set")
    check(len(manifest.chunks) == 3, f"Has 3 entries: {len(manifest.chunks)}")
    check(manifest.last_sync == 999.0, "Last sync set")
    check(manifest.chunk_hashes == {"abc123", "def456", "ghi789"}, "chunk_hashes property")


def test_manifest_sign_verify():
    print("\n10. Manifest sign/verify...")
    priv, pub = generate_node_keypair()
    entries = [
        ManifestEntry(hash="abc123", timestamp=1000.0, size=500),
        ManifestEntry(hash="def456", timestamp=1001.0, size=300),
    ]
    manifest = SyncManifest(node_id="node-1", chunks=entries)
    manifest.sign(priv)

    check(manifest.signature != "", "Manifest is signed")
    check(manifest.verify(pub), "Manifest signature verifies")


def test_empty_manifest():
    print("\n11. Empty manifest...")
    manifest = SyncManifest(node_id="node-empty")
    check(len(manifest.chunks) == 0, "No chunks")
    check(manifest.chunk_hashes == set(), "Empty hash set")
    check(manifest.last_sync == 0.0, "Default last_sync")


# ── 12. SyncDiff ────────────────────────────────────────────────────

def test_compute_diff():
    print("\n12. compute_diff...")
    local = SyncManifest(
        node_id="local",
        chunks=[
            ManifestEntry(hash="aaa", timestamp=1.0, size=100),
            ManifestEntry(hash="bbb", timestamp=2.0, size=200),
            ManifestEntry(hash="ccc", timestamp=3.0, size=300),
        ],
    )
    remote = SyncManifest(
        node_id="remote",
        chunks=[
            ManifestEntry(hash="bbb", timestamp=2.0, size=200),
            ManifestEntry(hash="ccc", timestamp=3.0, size=300),
            ManifestEntry(hash="ddd", timestamp=4.0, size=400),
        ],
    )

    diff = compute_diff(local, remote)
    check(diff.missing_local == ["ddd"], f"Missing local: {diff.missing_local}")
    check(diff.missing_remote == ["aaa"], f"Missing remote: {diff.missing_remote}")


def test_diff_identical():
    print("\n13. Diff with identical manifests...")
    entries = [ManifestEntry(hash="aaa", timestamp=1.0, size=100)]
    local = SyncManifest(node_id="a", chunks=entries)
    remote = SyncManifest(node_id="b", chunks=entries)

    diff = compute_diff(local, remote)
    check(diff.missing_local == [], "Nothing missing locally")
    check(diff.missing_remote == [], "Nothing missing remotely")


def test_diff_disjoint():
    print("\n14. Diff with disjoint manifests...")
    local = SyncManifest(
        node_id="local",
        chunks=[ManifestEntry(hash="aaa", timestamp=1.0, size=100)],
    )
    remote = SyncManifest(
        node_id="remote",
        chunks=[ManifestEntry(hash="zzz", timestamp=9.0, size=900)],
    )

    diff = compute_diff(local, remote)
    check(diff.missing_local == ["zzz"], f"Need zzz from remote: {diff.missing_local}")
    check(diff.missing_remote == ["aaa"], f"They need aaa: {diff.missing_remote}")


def test_diff_empty_manifests():
    print("\n15. Diff with empty manifests...")
    local = SyncManifest(node_id="a")
    remote = SyncManifest(node_id="b")
    diff = compute_diff(local, remote)
    check(diff.missing_local == [], "Empty local diff")
    check(diff.missing_remote == [], "Empty remote diff")


# ── 16. Edge cases ──────────────────────────────────────────────────

def test_edge_cases():
    print("\n16. Edge cases...")
    # Empty content
    chunk = KnowledgeChunk(content="")
    check(chunk.content_hash == "", "Empty content = empty hash")

    # Unsigned chunk verify fails
    chunk2 = KnowledgeChunk(content="unsigned")
    _, pub = generate_node_keypair()
    check(not chunk2.verify(pub), "Unsigned chunk fails verify")

    # Duplicate hash detection (same content = same hash)
    c1 = KnowledgeChunk(content="duplicate content")
    c2 = KnowledgeChunk(content="duplicate content")
    check(c1.content_hash == c2.content_hash, "Same content = same hash")


# ── Run all ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Two-Pool Model Tests")
    crypto = "Ed25519" if _HAS_ED25519 else "HMAC-SHA256 (fallback)"
    print(f"  Crypto backend: {crypto}")
    print("=" * 60)

    test_chunk_creation()
    test_chunk_hash_matches_sha256()
    test_chunk_public_private()
    test_chunk_metadata()
    test_chunk_serialization()
    test_keypair_generation()
    test_sign_verify_roundtrip()
    test_chunk_sign_verify()
    test_manifest_creation()
    test_manifest_sign_verify()
    test_empty_manifest()
    test_compute_diff()
    test_diff_identical()
    test_diff_disjoint()
    test_diff_empty_manifests()
    test_edge_cases()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
