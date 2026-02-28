# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Embedding Layer — V2 Scoring Engine
#
# Three-tier fallback chain: Ollama API → TF-IDF (sklearn) → Word Overlap
# Auto-detects best available backend at import time.
# Override with SUPERBRAIN_EMBEDDING_BACKEND env var (auto/ollama/tfidf/wordoverlap).

import os
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Backend identifiers
BACKEND_OLLAMA = "ollama"
BACKEND_TFIDF = "tfidf"
BACKEND_WORDOVERLAP = "wordoverlap"

_embedding_backend: str = BACKEND_WORDOVERLAP
_ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_ollama_model: str = os.environ.get("SUPERBRAIN_EMBED_MODEL", "nomic-embed-text")


# ── Backend detection ────────────────────────────────────────────────

def _detect_backend() -> str:
    """Auto-detect the best available embedding backend."""
    override = os.environ.get("SUPERBRAIN_EMBEDDING_BACKEND", "auto").lower()
    if override != "auto":
        if override in (BACKEND_OLLAMA, BACKEND_TFIDF, BACKEND_WORDOVERLAP):
            return override
        logger.warning(f"Unknown backend '{override}', falling back to auto-detect")

    # Tier 1: Ollama
    try:
        import requests
        resp = requests.post(
            f"{_ollama_url}/api/embeddings",
            json={"model": _ollama_model, "prompt": "test"},
            timeout=3,
        )
        if resp.status_code == 200 and "embedding" in resp.json():
            logger.info(f"Embedding backend: Ollama ({_ollama_model})")
            return BACKEND_OLLAMA
    except Exception:
        pass

    # Tier 2: sklearn TF-IDF
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.metrics.pairwise import cosine_similarity as _cs  # noqa: F401
        logger.info("Embedding backend: TF-IDF (sklearn)")
        return BACKEND_TFIDF
    except ImportError:
        pass

    # Tier 3: word overlap (always available)
    logger.info("Embedding backend: Word overlap (no extra dependencies)")
    return BACKEND_WORDOVERLAP


def init_backend():
    """Initialize embedding backend. Called at module import."""
    global _embedding_backend
    _embedding_backend = _detect_backend()


def get_backend() -> str:
    """Return current backend name."""
    return _embedding_backend


def is_semantic() -> bool:
    """True if current backend provides semantic embeddings (not word overlap)."""
    return _embedding_backend in (BACKEND_OLLAMA, BACKEND_TFIDF)


# ── Ollama backend ───────────────────────────────────────────────────

def _ollama_embed(text: str) -> np.ndarray:
    """Get embedding vector from Ollama API."""
    import requests
    resp = requests.post(
        f"{_ollama_url}/api/embeddings",
        json={"model": _ollama_model, "prompt": text},
        timeout=10,
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def _numpy_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two numpy vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(np.clip(dot / norm, 0.0, 1.0))


def _ollama_cosine(text_a: str, text_b: str) -> float:
    """Cosine similarity via Ollama embeddings."""
    vec_a = _ollama_embed(text_a)
    vec_b = _ollama_embed(text_b)
    return _numpy_cosine(vec_a, vec_b)


def _ollama_batch(query: str, chunks: List[str]) -> List[float]:
    """Batch cosine similarity via Ollama embeddings."""
    q_vec = _ollama_embed(query)
    results = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            results.append(0.0)
            continue
        c_vec = _ollama_embed(chunk)
        results.append(_numpy_cosine(q_vec, c_vec))
    return results


# ── TF-IDF backend ──────────────────────────────────────────────────

def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Cosine similarity via TF-IDF vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text_a, text_b])
    sim = sklearn_cosine(tfidf[0:1], tfidf[1:2])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def _tfidf_batch(query: str, chunks: List[str]) -> List[float]:
    """Batch cosine similarity via TF-IDF vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    texts = [query] + [c if c and c.strip() else "" for c in chunks]
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        # All texts empty
        return [0.0] * len(chunks)
    sims = sklearn_cosine(tfidf[0:1], tfidf[1:])[0]
    return [float(np.clip(s, 0.0, 1.0)) for s in sims]


# ── Word overlap backend ────────────────────────────────────────────

def word_overlap(text_a: str, text_b: str) -> float:
    """Word overlap ratio: |A ∩ B| / |A|. Matches V1 _word_set behavior exactly."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


# ── Public API ───────────────────────────────────────────────────────

def cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute semantic similarity between two texts.
    Uses the best available backend with automatic fallback.
    Returns float in [0.0, 1.0].
    """
    if not text_a or not text_a.strip() or not text_b or not text_b.strip():
        return 0.0

    if _embedding_backend == BACKEND_OLLAMA:
        try:
            return _ollama_cosine(text_a, text_b)
        except Exception as e:
            logger.warning(f"Ollama failed, falling back: {e}")
            try:
                return _tfidf_cosine(text_a, text_b)
            except Exception:
                return word_overlap(text_a, text_b)

    elif _embedding_backend == BACKEND_TFIDF:
        try:
            return _tfidf_cosine(text_a, text_b)
        except Exception as e:
            logger.warning(f"TF-IDF failed, falling back: {e}")
            return word_overlap(text_a, text_b)

    else:
        return word_overlap(text_a, text_b)


def batch_cosine_similarity(query: str, chunks: List[str]) -> List[float]:
    """
    Compute similarity between a query and each chunk.
    Returns List[float] of same length as chunks, each in [0.0, 1.0].
    """
    if not query or not query.strip():
        return [0.0] * len(chunks)
    if not chunks:
        return []

    if _embedding_backend == BACKEND_OLLAMA:
        try:
            return _ollama_batch(query, chunks)
        except Exception as e:
            logger.warning(f"Ollama batch failed, falling back: {e}")
            try:
                return _tfidf_batch(query, chunks)
            except Exception:
                pass

    if _embedding_backend in (BACKEND_OLLAMA, BACKEND_TFIDF):
        try:
            return _tfidf_batch(query, chunks)
        except Exception:
            pass

    # Word overlap fallback
    return [word_overlap(query, chunk) for chunk in chunks]


# Auto-init on import
init_backend()
