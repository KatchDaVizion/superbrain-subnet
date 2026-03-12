"""
SuperBrain Test Suite — Document Ingestor

Tests the document ingestion pipeline:
  1. Text splitting (900 chars, 150 overlap matching desktop app)
  2. KnowledgeChunk creation (hash, signature, metadata)
  3. PDF loading fallback
  4. Empty/edge case handling
  5. Duplicate detection
  6. SyncQueue compatibility
"""
import os
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from sync.protocol.pool_model import (
    KnowledgeChunk,
    compute_content_hash,
    generate_node_keypair,
)
from sync.queue.sync_queue import SyncQueue
from sync.ingestion.document_ingestor import (
    _split_text,
    ingest_file,
    ingest_text,
    deduplicate_chunks,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    _load_file,
)

# ═══════════════════════════════════════════════════════════════════
#  Test framework
# ═══════════════════════════════════════════════════════════════════

_passed = 0
_failed = 0


def check(condition, msg):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  OK {msg}")
    else:
        _failed += 1
        print(f"  FAIL {msg}")


# ═══════════════════════════════════════════════════════════════════
#  Test data
# ═══════════════════════════════════════════════════════════════════

SHORT_TEXT = "This is a short document that fits in a single chunk."

# ~2500 characters — should produce 3-4 chunks at 900/150
MEDIUM_TEXT = """Bittensor is a decentralized machine learning network that creates an open marketplace for artificial intelligence. The network uses a blockchain-based incentive mechanism to reward participants who contribute useful intelligence and computation to the ecosystem.

The architecture is built around subnets, which are specialized networks focused on particular tasks such as text generation, image recognition, data storage, and knowledge retrieval. Each subnet operates independently but is connected through the main Bittensor blockchain for consensus and token distribution.

Validators play a critical role in the Bittensor ecosystem. They evaluate the quality of miner outputs by sending queries, scoring responses, and setting on-chain weights that determine how TAO token rewards are distributed. Validators with more stake have greater influence on the consensus weights through the Yuma Consensus mechanism.

Miners are the workhorses of the network. They receive queries from validators and must produce high-quality responses to earn rewards. In a RAG (Retrieval-Augmented Generation) subnet like SuperBrain, miners maintain knowledge bases and generate cited answers from their stored documents. The quality of their responses directly impacts their earnings.

The TAO token is the native cryptocurrency of the Bittensor network. It has a capped supply of 21 million tokens, similar to Bitcoin. TAO is used for staking, governance, subnet creation, and rewarding network participants. The emission schedule follows a halving pattern to ensure long-term sustainability.

Knowledge synchronization is a key innovation in the SuperBrain subnet. Miners can share knowledge chunks with each other using a peer-to-peer protocol that supports multiple transport layers including LAN (via mDNS and WebSocket), Bluetooth (via RFCOMM), and I2P (for anonymous communication). This enables a truly decentralized knowledge network where information flows freely between participants.

The two-pool privacy model ensures user control over their data. All knowledge chunks are private by default and only become public when the user explicitly chooses to share them. This design respects user privacy while enabling collaborative knowledge building across the network."""

# Generate a very long text for stress testing
LONG_TEXT = "Word " * 5000  # ~25000 chars


# ═══════════════════════════════════════════════════════════════════
#  1. Text splitting tests
# ═══════════════════════════════════════════════════════════════════

def test_split_defaults():
    """Test that default chunk size and overlap match desktop app."""
    print("\n1. Default parameters...")
    check(DEFAULT_CHUNK_SIZE == 900, f"Default chunk size is 900 (got {DEFAULT_CHUNK_SIZE})")
    check(DEFAULT_CHUNK_OVERLAP == 150, f"Default chunk overlap is 150 (got {DEFAULT_CHUNK_OVERLAP})")


def test_split_short_text():
    """Short text that fits in one chunk should produce exactly one chunk."""
    print("\n2. Short text splitting...")
    pieces = _split_text(SHORT_TEXT)
    check(len(pieces) == 1, f"Short text produces 1 chunk (got {len(pieces)})")
    check(pieces[0] == SHORT_TEXT, "Chunk content matches input")


def test_split_medium_text():
    """Medium text should produce multiple chunks with proper sizes."""
    print("\n3. Medium text splitting...")
    pieces = _split_text(MEDIUM_TEXT)
    check(len(pieces) >= 3, f"Medium text produces >= 3 chunks (got {len(pieces)})")

    # All chunks should be <= chunk_size (with some tolerance for overlap mechanics)
    for i, piece in enumerate(pieces):
        check(len(piece) <= DEFAULT_CHUNK_SIZE + 50,
              f"Chunk {i+1} size {len(piece)} <= {DEFAULT_CHUNK_SIZE + 50}")

    # No empty chunks
    for i, piece in enumerate(pieces):
        check(len(piece.strip()) > 0, f"Chunk {i+1} is not empty")

    # Verify content is preserved — all words from original should appear somewhere
    all_chunk_text = " ".join(pieces)
    original_words = set(MEDIUM_TEXT.lower().split())
    chunk_words = set(all_chunk_text.lower().split())
    missing = original_words - chunk_words
    check(len(missing) == 0, f"No words lost in splitting (missing: {len(missing)})")


def test_split_preserves_paragraphs():
    """Splitter should prefer paragraph boundaries over mid-sentence splits."""
    print("\n4. Paragraph boundary preservation...")
    text = "First paragraph with some content.\n\nSecond paragraph with more content."
    pieces = _split_text(text, chunk_size=100, chunk_overlap=0)
    # Should split on \n\n, not mid-sentence
    check(any("First paragraph" in p for p in pieces), "First paragraph preserved")
    check(any("Second paragraph" in p for p in pieces), "Second paragraph preserved")


def test_split_empty_text():
    """Empty or whitespace-only text produces no chunks."""
    print("\n5. Empty text handling...")
    check(_split_text("") == [], "Empty string produces empty list")
    check(_split_text("   ") == [], "Whitespace-only produces empty list")
    check(_split_text("\n\n\n") == [], "Newlines-only produces empty list")


def test_split_long_text():
    """Very long text should produce many chunks without errors."""
    print("\n6. Long text stress test...")
    pieces = _split_text(LONG_TEXT)
    check(len(pieces) > 20, f"Long text produces many chunks (got {len(pieces)})")

    # All chunks should be reasonable size
    for piece in pieces:
        check(len(piece) <= DEFAULT_CHUNK_SIZE + 50,
              f"Long text chunk size {len(piece)} is within bounds")
        if len(piece) > DEFAULT_CHUNK_SIZE + 50:
            break  # Don't flood output


def test_split_custom_params():
    """Custom chunk size and overlap should be respected."""
    print("\n7. Custom chunk parameters...")
    pieces_small = _split_text(MEDIUM_TEXT, chunk_size=200, chunk_overlap=30)
    pieces_large = _split_text(MEDIUM_TEXT, chunk_size=2000, chunk_overlap=300)
    check(len(pieces_small) > len(pieces_large),
          f"Smaller chunks produce more pieces ({len(pieces_small)} > {len(pieces_large)})")


# ═══════════════════════════════════════════════════════════════════
#  2. Chunk creation tests
# ═══════════════════════════════════════════════════════════════════

def test_ingest_text_basic():
    """Ingest raw text and verify KnowledgeChunk fields."""
    print("\n8. Text ingestion — basic fields...")
    priv, pub = generate_node_keypair()

    chunks = ingest_text(
        text=MEDIUM_TEXT,
        private_key=priv,
        node_id="test-miner",
        visibility="public",
        source="test_doc.txt",
        title="Test Document",
    )

    check(len(chunks) >= 3, f"Produced >= 3 chunks (got {len(chunks)})")

    for i, chunk in enumerate(chunks):
        check(isinstance(chunk, KnowledgeChunk), f"Chunk {i+1} is KnowledgeChunk")
        check(bool(chunk.content), f"Chunk {i+1} has content")
        check(bool(chunk.content_hash), f"Chunk {i+1} has content_hash")
        check(chunk.origin_node_id == "test-miner", f"Chunk {i+1} has correct node_id")
        check(chunk.pool_visibility == "public", f"Chunk {i+1} is public")
        check(chunk.shared_at is not None, f"Chunk {i+1} has shared_at timestamp")
        if i == 0:
            break  # Spot check first chunk only


def test_ingest_text_hashing():
    """Content hash should be SHA-256 of chunk content."""
    print("\n9. Content hashing...")
    priv, pub = generate_node_keypair()
    chunks = ingest_text(text=SHORT_TEXT, private_key=priv)

    for chunk in chunks:
        expected_hash = compute_content_hash(chunk.content)
        check(chunk.content_hash == expected_hash,
              f"Hash matches: {chunk.content_hash[:16]}...")


def test_ingest_text_signing():
    """All chunks should be Ed25519 signed."""
    print("\n10. Ed25519 signing...")
    priv, pub = generate_node_keypair()
    chunks = ingest_text(text=MEDIUM_TEXT, private_key=priv)

    for i, chunk in enumerate(chunks):
        check(bool(chunk.signature), f"Chunk {i+1} has signature")
        # Verify signature is valid hex
        try:
            bytes.fromhex(chunk.signature)
            valid_hex = True
        except ValueError:
            valid_hex = False
        check(valid_hex, f"Chunk {i+1} signature is valid hex")

        # Verify signature using public key
        verified = chunk.verify(pub)
        check(verified, f"Chunk {i+1} signature verifies with public key")
        if i >= 2:
            break  # Spot check first 3


def test_ingest_text_metadata():
    """Metadata should contain source_file, title, doc_id, date_ingested."""
    print("\n11. Metadata fields...")
    priv, pub = generate_node_keypair()
    chunks = ingest_text(
        text=MEDIUM_TEXT,
        private_key=priv,
        source="my_notes.txt",
        title="My Notes",
    )

    chunk = chunks[0]
    meta = chunk.metadata

    check(meta.get("source_file") == "my_notes.txt", f"source_file: {meta.get('source_file')}")
    check(meta.get("title") == "My Notes", f"title: {meta.get('title')}")
    check(meta.get("file_type") == "text", f"file_type: {meta.get('file_type')}")
    check("doc_id" in meta, "doc_id present")
    check("date_ingested" in meta, "date_ingested present")
    check("chunk_index" in meta, "chunk_index present")
    check("total_chunks" in meta, "total_chunks present")
    check(meta["chunk_index"] == 0, "First chunk has index 0")
    check(meta["total_chunks"] == len(chunks), f"total_chunks matches ({meta['total_chunks']})")

    # All chunks in same document share doc_id
    doc_ids = {c.metadata["doc_id"] for c in chunks}
    check(len(doc_ids) == 1, "All chunks share same doc_id")


def test_ingest_text_private():
    """Private visibility should set shared_at to None."""
    print("\n12. Private visibility...")
    priv, pub = generate_node_keypair()
    chunks = ingest_text(text=SHORT_TEXT, private_key=priv, visibility="private")

    check(chunks[0].pool_visibility == "private", "Chunk is private")
    check(chunks[0].shared_at is None, "shared_at is None for private")


# ═══════════════════════════════════════════════════════════════════
#  3. File ingestion tests
# ═══════════════════════════════════════════════════════════════════

def test_ingest_text_file():
    """Ingest a .txt file from disk."""
    print("\n13. Text file ingestion...")
    priv, pub = generate_node_keypair()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(MEDIUM_TEXT)
        f.flush()
        tmp_path = f.name

    try:
        chunks = ingest_file(
            file_path=tmp_path,
            private_key=priv,
            visibility="public",
        )
        check(len(chunks) >= 3, f"Text file produced >= 3 chunks (got {len(chunks)})")
        check(chunks[0].metadata["file_type"] == "text", "file_type is 'text'")
        check(chunks[0].metadata["source_file"] == os.path.basename(tmp_path),
              f"source_file: {chunks[0].metadata['source_file']}")
    finally:
        os.unlink(tmp_path)


def test_ingest_markdown_file():
    """Ingest a .md file from disk."""
    print("\n14. Markdown file ingestion...")
    priv, pub = generate_node_keypair()

    md_content = "# Heading\n\nParagraph one with content.\n\n## Subheading\n\nParagraph two."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md_content)
        f.flush()
        tmp_path = f.name

    try:
        chunks = ingest_file(file_path=tmp_path, private_key=priv)
        check(len(chunks) >= 1, f"Markdown file produced chunks (got {len(chunks)})")
        check(chunks[0].metadata["file_type"] == "markdown", "file_type is 'markdown'")
    finally:
        os.unlink(tmp_path)


def test_ingest_file_not_found():
    """Missing file should raise FileNotFoundError."""
    print("\n15. File not found handling...")
    priv, pub = generate_node_keypair()
    try:
        ingest_file(file_path="/nonexistent/file.txt", private_key=priv)
        check(False, "Should have raised FileNotFoundError")
    except FileNotFoundError:
        check(True, "FileNotFoundError raised for missing file")


def test_ingest_empty_file():
    """Empty file should produce no chunks."""
    print("\n16. Empty file handling...")
    priv, pub = generate_node_keypair()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        f.flush()
        tmp_path = f.name

    try:
        chunks = ingest_file(file_path=tmp_path, private_key=priv)
        check(len(chunks) == 0, "Empty file produces 0 chunks")
    finally:
        os.unlink(tmp_path)


def test_ingest_pdf_fallback():
    """PDF fallback should handle non-PDF binary gracefully."""
    print("\n17. PDF fallback (binary file)...")
    priv, pub = generate_node_keypair()

    # Create a fake .pdf that is actually just text
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
        f.write("This is plain text pretending to be a PDF.\nSecond line of content.")
        f.flush()
        tmp_path = f.name

    try:
        text, file_type = _load_file(tmp_path)
        check(file_type == "pdf", "Detected as pdf type")
        # Should extract something via fallback
        check(len(text) > 0, f"Extracted {len(text)} chars from fake PDF")
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════
#  4. Duplicate detection tests
# ═══════════════════════════════════════════════════════════════════

def test_deduplicate_chunks():
    """Duplicate detection should filter known hashes."""
    print("\n18. Duplicate detection...")
    priv, pub = generate_node_keypair()
    chunks = ingest_text(text=MEDIUM_TEXT, private_key=priv, visibility="public")

    # No existing hashes — all should pass through
    unique = deduplicate_chunks(chunks, set())
    check(len(unique) == len(chunks), f"No dupes: all {len(unique)} pass")

    # Mark first two as existing
    existing = {chunks[0].content_hash, chunks[1].content_hash}
    unique = deduplicate_chunks(chunks, existing)
    check(len(unique) == len(chunks) - 2, f"2 dupes filtered: {len(unique)} remain")

    # All existing — none should pass
    all_hashes = {c.content_hash for c in chunks}
    unique = deduplicate_chunks(chunks, all_hashes)
    check(len(unique) == 0, "All dupes: 0 remain")


def test_deduplicate_within_batch():
    """Dedup should also handle duplicates within the same batch."""
    print("\n19. Within-batch dedup...")
    priv, pub = generate_node_keypair()

    chunk = KnowledgeChunk(
        content="Duplicate content",
        origin_node_id="test",
        pool_visibility="public",
        shared_at=time.time(),
    )
    chunk.sign(priv)

    # Same chunk twice in batch
    batch = [chunk, chunk]
    unique = deduplicate_chunks(batch, set())
    check(len(unique) == 1, "Within-batch dedup keeps only 1")


# ═══════════════════════════════════════════════════════════════════
#  5. SyncQueue compatibility tests
# ═══════════════════════════════════════════════════════════════════

def test_syncqueue_compatibility():
    """Ingested chunks must be directly addable to SyncQueue."""
    print("\n20. SyncQueue compatibility...")
    priv, pub = generate_node_keypair()

    tmp = tempfile.mkdtemp(prefix="sb-test-")
    db_path = os.path.join(tmp, "test.db")
    queue = SyncQueue(db_path=db_path)

    chunks = ingest_text(
        text=MEDIUM_TEXT,
        private_key=priv,
        visibility="public",
    )

    added = 0
    for chunk in chunks:
        if queue.add_to_queue(chunk):
            added += 1

    check(added == len(chunks), f"All {added} chunks added to SyncQueue")
    check(queue.chunk_count() == len(chunks), f"Queue count matches: {queue.chunk_count()}")

    # Retrieve and verify
    for chunk in chunks:
        retrieved = queue.get_chunk(chunk.content_hash)
        check(retrieved is not None, f"Chunk {chunk.content_hash[:12]}... retrievable")
        if retrieved:
            check(retrieved.content == chunk.content, "Content preserved in queue")
            break  # Spot check first

    # Verify no duplicates on re-add
    for chunk in chunks:
        result = queue.add_to_queue(chunk)
        check(result is False, "Re-add returns False (duplicate)")
        break  # Spot check first

    queue.close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_syncqueue_manifest_after_ingest():
    """Ingested chunks should appear in SyncQueue manifest."""
    print("\n21. Manifest after ingest...")
    priv, pub = generate_node_keypair()

    tmp = tempfile.mkdtemp(prefix="sb-test-")
    db_path = os.path.join(tmp, "test.db")
    queue = SyncQueue(db_path=db_path)

    chunks = ingest_text(text=MEDIUM_TEXT, private_key=priv, visibility="public")
    for chunk in chunks:
        queue.add_to_queue(chunk)

    manifest = queue.get_manifest("miner")
    check(len(manifest.chunk_hashes) == len(chunks),
          f"Manifest has {len(manifest.chunk_hashes)} entries")

    # All chunk hashes should be in manifest
    for chunk in chunks:
        check(chunk.content_hash in manifest.chunk_hashes,
              f"Chunk {chunk.content_hash[:12]}... in manifest")
        break  # Spot check

    queue.close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_ingest_deterministic_hash():
    """Same content should always produce the same hash."""
    print("\n22. Deterministic hashing...")
    priv, pub = generate_node_keypair()

    chunks1 = ingest_text(text=SHORT_TEXT, private_key=priv, node_id="node1")
    chunks2 = ingest_text(text=SHORT_TEXT, private_key=priv, node_id="node2")

    # Same content → same hash regardless of node_id
    check(chunks1[0].content_hash == chunks2[0].content_hash,
          "Same content produces same hash across different nodes")


# ═══════════════════════════════════════════════════════════════════
#  6. Chunk index and overlap tests
# ═══════════════════════════════════════════════════════════════════

def test_chunk_index_ordering():
    """Chunk indices should be sequential starting from 0."""
    print("\n23. Chunk index ordering...")
    priv, pub = generate_node_keypair()
    chunks = ingest_text(text=MEDIUM_TEXT, private_key=priv)

    for i, chunk in enumerate(chunks):
        check(chunk.metadata["chunk_index"] == i,
              f"Chunk {i} has index {chunk.metadata['chunk_index']}")


def test_overlap_between_chunks():
    """Adjacent chunks should share overlapping text."""
    print("\n24. Chunk overlap verification...")
    # Use a simple repeating text to make overlap verification clear
    text = "Sentence number one is here. " * 100  # ~2900 chars
    pieces = _split_text(text, chunk_size=900, chunk_overlap=150)

    check(len(pieces) >= 3, f"Enough chunks to test overlap (got {len(pieces)})")

    if len(pieces) >= 2:
        # End of chunk N should overlap with start of chunk N+1
        end_of_first = pieces[0][-100:]  # last 100 chars of chunk 1
        start_of_second = pieces[1][:200]  # first 200 chars of chunk 2
        # There should be some shared text
        shared_words = set(end_of_first.lower().split()) & set(start_of_second.lower().split())
        check(len(shared_words) > 0, f"Adjacent chunks share {len(shared_words)} words")


# ═══════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Document Ingestor Tests")
    print("=" * 60)

    test_split_defaults()
    test_split_short_text()
    test_split_medium_text()
    test_split_preserves_paragraphs()
    test_split_empty_text()
    test_split_long_text()
    test_split_custom_params()
    test_ingest_text_basic()
    test_ingest_text_hashing()
    test_ingest_text_signing()
    test_ingest_text_metadata()
    test_ingest_text_private()
    test_ingest_text_file()
    test_ingest_markdown_file()
    test_ingest_file_not_found()
    test_ingest_empty_file()
    test_ingest_pdf_fallback()
    test_deduplicate_chunks()
    test_deduplicate_within_batch()
    test_syncqueue_compatibility()
    test_syncqueue_manifest_after_ingest()
    test_ingest_deterministic_hash()
    test_chunk_index_ordering()
    test_overlap_between_chunks()

    print(f"\n{'=' * 60}")
    if _failed == 0:
        print(f"  ALL {_passed} TESTS PASSED!")
    else:
        print(f"  {_passed} passed, {_failed} FAILED")
    print("=" * 60)
    sys.exit(1 if _failed else 0)
