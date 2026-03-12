# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Document Ingestor
#
# Converts plain text, markdown, and PDF files into KnowledgeChunks
# that are fully compatible with SyncQueue and the Bittensor sync protocol.
#
# Chunking parameters match the desktop app's documentLoader.ts:
#   - Chunk size: 900 characters
#   - Chunk overlap: 150 characters
#   - Splitter: recursive character splitting (paragraph → sentence → word → char)

import os
import time
import uuid
from typing import List, Literal, Optional, Tuple

from sync.protocol.pool_model import (
    KnowledgeChunk,
    compute_content_hash,
    generate_node_keypair,
)

# ── Chunking constants (match desktop app documentLoader.ts) ────────

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150

# Recursive split separators — same strategy as LangChain's
# RecursiveCharacterTextSplitter: try paragraph, then sentence,
# then word, then character.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


# ── PDF loading ─────────────────────────────────────────────────────

def _load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    Uses pypdf if installed, falls back to reading raw bytes as text.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Fallback: try PyPDF2 (older name)
    try:
        from PyPDF2 import PdfReader as PdfReader2

        reader = PdfReader2(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Last resort: read raw and extract printable text
    with open(file_path, "rb") as f:
        raw = f.read()
    # Decode with errors replaced, strip non-printable
    text = raw.decode("utf-8", errors="replace")
    # Filter to printable characters
    return "".join(c for c in text if c.isprintable() or c in "\n\t ")


def _load_file(file_path: str) -> Tuple[str, str]:
    """
    Load a file and return (text, file_type).
    Supports: .pdf, .md, .txt, and any other text file.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _load_pdf(file_path), "pdf"

    # All other formats: read as text
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if ext == ".md":
        return text, "markdown"
    return text, "text"


# ── Text splitting ──────────────────────────────────────────────────

def _split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into chunks using recursive character splitting.

    Matches LangChain RecursiveCharacterTextSplitter behavior:
    tries to split on paragraph boundaries first, then sentences,
    then words, then characters. Each chunk has `chunk_overlap`
    characters of overlap with the previous chunk.
    """
    if not text or not text.strip():
        return []

    return _recursive_split(text.strip(), chunk_size, chunk_overlap, _SEPARATORS)


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str],
) -> List[str]:
    """Recursively split text using the first separator that produces valid splits."""
    # Base case: text fits in one chunk
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Find the best separator (first one that actually appears in text)
    sep = ""
    remaining_seps = []
    for i, s in enumerate(separators):
        if s == "":
            sep = s
            remaining_seps = []
            break
        if s in text:
            sep = s
            remaining_seps = separators[i + 1 :]
            break

    # Split on chosen separator
    if sep:
        pieces = text.split(sep)
    else:
        # Character-level split (last resort)
        pieces = list(text)

    # Merge pieces into chunks respecting chunk_size
    chunks = []
    current = ""

    for piece in pieces:
        # What this chunk would look like if we added this piece
        candidate = current + sep + piece if current else piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Current chunk is full — save it
            if current.strip():
                # If current chunk is still too big, recurse with finer separators
                if len(current) > chunk_size and remaining_seps:
                    chunks.extend(
                        _recursive_split(current, chunk_size, chunk_overlap, remaining_seps)
                    )
                else:
                    chunks.append(current.strip())

                # Start next chunk with overlap from end of previous
                if chunk_overlap > 0 and current:
                    overlap_text = current[-chunk_overlap:]
                    current = overlap_text + sep + piece
                else:
                    current = piece
            else:
                # current is empty, piece alone is too big — recurse
                if remaining_seps:
                    chunks.extend(
                        _recursive_split(piece, chunk_size, chunk_overlap, remaining_seps)
                    )
                    current = ""
                else:
                    # Hard split at chunk_size
                    for start in range(0, len(piece), chunk_size - chunk_overlap):
                        part = piece[start : start + chunk_size].strip()
                        if part:
                            chunks.append(part)
                    current = ""

    # Don't forget the last chunk
    if current.strip():
        if len(current) > chunk_size and remaining_seps:
            chunks.extend(
                _recursive_split(current, chunk_size, chunk_overlap, remaining_seps)
            )
        else:
            chunks.append(current.strip())

    return chunks


# ── Main ingestor ───────────────────────────────────────────────────

def ingest_file(
    file_path: str,
    private_key: bytes,
    node_id: str = "",
    visibility: Literal["private", "public"] = "private",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    title: str = "",
) -> List[KnowledgeChunk]:
    """
    Ingest a document file into KnowledgeChunks.

    Args:
        file_path: Path to the document (.txt, .md, .pdf).
        private_key: Ed25519 private key bytes for signing chunks.
        node_id: Originating node ID. Auto-generated if empty.
        visibility: "private" or "public". Default "private".
        chunk_size: Max characters per chunk (default 900).
        chunk_overlap: Overlap between consecutive chunks (default 150).
        title: Optional document title for metadata.

    Returns:
        List of signed KnowledgeChunks ready for SyncQueue.add_to_queue().
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not node_id:
        node_id = str(uuid.uuid4())

    text, file_type = _load_file(file_path)
    return _ingest_text(
        text=text,
        private_key=private_key,
        node_id=node_id,
        visibility=visibility,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_file=os.path.basename(file_path),
        file_type=file_type,
        title=title or os.path.splitext(os.path.basename(file_path))[0],
    )


def ingest_text(
    text: str,
    private_key: bytes,
    node_id: str = "",
    visibility: Literal["private", "public"] = "private",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source: str = "direct_input",
    title: str = "",
) -> List[KnowledgeChunk]:
    """
    Ingest raw text into KnowledgeChunks.

    Args:
        text: The text content to ingest.
        private_key: Ed25519 private key bytes for signing chunks.
        node_id: Originating node ID. Auto-generated if empty.
        visibility: "private" or "public". Default "private".
        chunk_size: Max characters per chunk (default 900).
        chunk_overlap: Overlap between consecutive chunks (default 150).
        source: Source label for metadata.
        title: Optional title for metadata.

    Returns:
        List of signed KnowledgeChunks ready for SyncQueue.add_to_queue().
    """
    if not node_id:
        node_id = str(uuid.uuid4())

    return _ingest_text(
        text=text,
        private_key=private_key,
        node_id=node_id,
        visibility=visibility,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source_file=source,
        file_type="text",
        title=title or source,
    )


def _ingest_text(
    text: str,
    private_key: bytes,
    node_id: str,
    visibility: str,
    chunk_size: int,
    chunk_overlap: int,
    source_file: str,
    file_type: str,
    title: str,
) -> List[KnowledgeChunk]:
    """Internal: split text and create signed KnowledgeChunks."""
    if not text or not text.strip():
        return []

    pieces = _split_text(text, chunk_size, chunk_overlap)
    if not pieces:
        return []

    doc_id = str(uuid.uuid4())
    now = time.time()
    chunks = []

    for i, piece in enumerate(pieces):
        chunk = KnowledgeChunk(
            content=piece,
            origin_node_id=node_id,
            timestamp=now,
            pool_visibility=visibility,
            shared_at=now if visibility == "public" else None,
            metadata={
                "source_file": source_file,
                "file_type": file_type,
                "title": title,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(pieces),
                "date_ingested": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            },
        )
        chunk.sign(private_key)
        chunks.append(chunk)

    return chunks


def deduplicate_chunks(
    new_chunks: List[KnowledgeChunk],
    existing_hashes: set,
) -> List[KnowledgeChunk]:
    """
    Filter out chunks whose content_hash already exists.

    Args:
        new_chunks: Freshly ingested chunks.
        existing_hashes: Set of content_hash strings already in the queue.

    Returns:
        Only the chunks that are not duplicates.
    """
    unique = []
    seen = set(existing_hashes)
    for chunk in new_chunks:
        if chunk.content_hash not in seen:
            unique.append(chunk)
            seen.add(chunk.content_hash)
    return unique
