# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Network RAG Query
#
# Searches the collective public knowledge pool (SyncQueue) and generates
# answers using the local Ollama LLM with extractive fallback.
#
# Reuses:
#   - embeddings.py: 3-tier similarity (Ollama → TF-IDF → word overlap)
#   - miner.py patterns: Ollama prompt + extractive fallback
#   - sync_queue.py: get_all_chunks(), chunk_count(), stats()

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependencies
_embeddings_module = None
_SyncQueue = None


def _get_embeddings():
    global _embeddings_module
    if _embeddings_module is None:
        from superbrain.validator import embeddings
        _embeddings_module = embeddings
    return _embeddings_module


def _get_sync_queue_class():
    global _SyncQueue
    if _SyncQueue is None:
        from sync.queue.sync_queue import SyncQueue
        _SyncQueue = SyncQueue
    return _SyncQueue


# ── Configuration ────────────────────────────────────────────────────

DEFAULT_TOP_K = 5
DEFAULT_FRESHNESS_HALF_LIFE_DAYS = 7.0
DEFAULT_DIVERSITY_THRESHOLD = 0.85
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"


@dataclass
class SearchResult:
    """A single search result from the knowledge pool."""
    content: str
    content_hash: str
    score: float  # combined relevance + freshness
    relevance: float  # raw embedding similarity
    freshness: float  # time decay factor
    source: str  # source label from metadata or hash prefix
    timestamp: float  # chunk creation time
    node_id: str  # origin node


@dataclass
class Answer:
    """Generated answer with citations."""
    text: str
    citations: List[int]  # 0-indexed into sources list
    sources: List[SearchResult]
    method: str  # "ollama" or "extractive"
    query: str
    generation_time: float  # seconds


@dataclass
class PoolStats:
    """Statistics about the knowledge pool."""
    total_chunks: int
    unique_nodes: int
    oldest_chunk: Optional[float]
    newest_chunk: Optional[float]
    embedding_backend: str
    ollama_available: bool


class NetworkRAGQuery:
    """
    Search the collective public knowledge pool and generate answers.

    Usage:
        rag = NetworkRAGQuery(db_path="miner_sync_queue.db")
        results = rag.search("quantum computing")
        answer = rag.answer("What is quantum entanglement?")
        stats = rag.stats()
    """

    def __init__(
        self,
        db_path: str = "miner_sync_queue.db",
        ollama_url: str = "",
        ollama_model: str = "",
        top_k: int = DEFAULT_TOP_K,
        freshness_half_life: float = DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
        diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    ):
        self.db_path = db_path
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.ollama_model = ollama_model or os.environ.get("SUPERBRAIN_MODEL", DEFAULT_OLLAMA_MODEL)
        self.top_k = top_k
        self.freshness_half_life = freshness_half_life
        self.diversity_threshold = diversity_threshold

        SyncQueue = _get_sync_queue_class()
        self.sync_queue = SyncQueue(db_path=db_path)

    def close(self):
        """Close the underlying SyncQueue connection."""
        if self.sync_queue:
            self.sync_queue.close()

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 0) -> List[SearchResult]:
        """
        Search the knowledge pool for chunks relevant to query.

        Embedding similarity with freshness bonus and diversity filter.
        Returns up to top_k results sorted by combined score.
        """
        if not query or not query.strip():
            return []

        k = top_k or self.top_k
        embeddings = _get_embeddings()

        # Fetch all chunks (limit 500 — same as get_all_chunks default)
        chunks = self.sync_queue.get_all_chunks(limit=500)
        if not chunks:
            return []

        # Compute similarity scores
        contents = [c.content for c in chunks]
        similarities = embeddings.batch_cosine_similarity(query, contents)

        now = time.time()
        results = []
        for chunk, sim in zip(chunks, similarities):
            freshness = self._freshness_score(chunk.timestamp, now)
            combined = 0.7 * sim + 0.3 * freshness
            source = self._extract_source(chunk)
            results.append(SearchResult(
                content=chunk.content,
                content_hash=chunk.content_hash,
                score=combined,
                relevance=sim,
                freshness=freshness,
                source=source,
                timestamp=chunk.timestamp,
                node_id=chunk.origin_node_id,
            ))

        # Sort by combined score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Diversity filter: remove near-duplicates
        results = self._diversity_filter(results, embeddings)

        return results[:k]

    def _freshness_score(self, timestamp: float, now: float) -> float:
        """Exponential decay with configurable half-life."""
        age_days = (now - timestamp) / 86400.0
        if age_days < 0:
            age_days = 0
        return math.exp(-0.693 * age_days / self.freshness_half_life)

    def _diversity_filter(self, results: List[SearchResult], embeddings) -> List[SearchResult]:
        """Remove results too similar to already-selected results."""
        if len(results) <= 1:
            return results

        selected = [results[0]]
        for candidate in results[1:]:
            is_diverse = True
            for kept in selected:
                sim = embeddings.cosine_similarity(candidate.content, kept.content)
                if sim > self.diversity_threshold:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(candidate)

        return selected

    def _extract_source(self, chunk) -> str:
        """Extract human-readable source label from chunk metadata."""
        if chunk.metadata:
            meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
            if isinstance(chunk.metadata, str):
                try:
                    import json
                    meta = json.loads(chunk.metadata)
                except Exception:
                    meta = {}
            if meta.get("source_file"):
                return meta["source_file"]
            if meta.get("title"):
                return meta["title"]
        return f"chunk_{chunk.content_hash[:8]}"

    # ── Answer Generation ────────────────────────────────────────────

    def answer(self, query: str, top_k: int = 0) -> Answer:
        """
        Search for relevant chunks and generate an answer.

        Uses Ollama LLM if available, falls back to extractive method.
        """
        t0 = time.time()
        results = self.search(query, top_k=top_k)

        if not results:
            return Answer(
                text="No knowledge found in the pool for this query.",
                citations=[],
                sources=[],
                method="none",
                query=query,
                generation_time=time.time() - t0,
            )

        # Try Ollama first, fall back to extractive
        text, citations, method = self._generate_ollama(query, results)
        if method == "ollama_failed":
            text, citations, method = self._generate_extractive(query, results)

        return Answer(
            text=text,
            citations=citations,
            sources=results,
            method=method,
            query=query,
            generation_time=time.time() - t0,
        )

    def _generate_ollama(
        self, query: str, results: List[SearchResult]
    ) -> Tuple[str, List[int], str]:
        """Generate answer using Ollama LLM. Returns (text, citations, method)."""
        try:
            import requests
        except ImportError:
            return "", [], "ollama_failed"

        # Build prompt — same pattern as miner.py _generate_ollama
        chunks_text = ""
        for i, r in enumerate(results):
            chunks_text += f"\n[{i+1}] (Source: {r.source})\n{r.content}\n"

        prompt = f"""You are a knowledgeable assistant. Answer using ONLY the provided sources.
Cite sources with [1], [2] markers. If sources don't contain enough info, say so.

=== SOURCES ===
{chunks_text}
=== END SOURCES ===

Question: {query}

Answer (cite with [1], [2], etc.):"""

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 256, "num_ctx": 1024},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                citations = self._extract_citations(answer, len(results))
                return answer, citations, "ollama"
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")

        return "", [], "ollama_failed"

    def _generate_extractive(
        self, query: str, results: List[SearchResult]
    ) -> Tuple[str, List[int], str]:
        """Keyword-based extractive fallback — same pattern as miner.py."""
        if not results:
            return "No source documents provided.", [], "extractive"

        query_words = set(query.lower().split())
        scored = []
        for i, r in enumerate(results):
            overlap = len(query_words & set(r.content.lower().split()))
            scored.append((overlap, i))
        scored.sort(reverse=True)
        top = [idx for _, idx in scored[:2] if _ > 0] or [0]

        parts = []
        for idx in top:
            source = results[idx].source
            sentences = results[idx].content.split(". ")
            excerpt = ". ".join(sentences[:2])
            if not excerpt.endswith("."):
                excerpt += "."
            parts.append(f"According to the sources [{idx+1}], {excerpt}")

        return " ".join(parts), top, "extractive"

    def _extract_citations(self, text: str, max_results: int) -> List[int]:
        """Extract [1], [2] citation markers — same as miner.py."""
        matches = re.findall(r'\[(\d+)\]', text)
        citations = []
        for m in matches:
            idx = int(m) - 1
            if 0 <= idx < max_results and idx not in citations:
                citations.append(idx)
        return citations

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> PoolStats:
        """Return statistics about the knowledge pool."""
        embeddings = _get_embeddings()

        total = self.sync_queue.chunk_count()
        chunks = self.sync_queue.get_all_chunks(limit=500)

        unique_nodes = set()
        oldest = None
        newest = None
        for c in chunks:
            unique_nodes.add(c.origin_node_id)
            if oldest is None or c.timestamp < oldest:
                oldest = c.timestamp
            if newest is None or c.timestamp > newest:
                newest = c.timestamp

        # Check Ollama availability
        ollama_available = False
        try:
            import requests
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            ollama_available = resp.status_code == 200
        except Exception:
            pass

        return PoolStats(
            total_chunks=total,
            unique_nodes=len(unique_nodes),
            oldest_chunk=oldest,
            newest_chunk=newest,
            embedding_backend=embeddings.get_backend(),
            ollama_available=ollama_available,
        )
