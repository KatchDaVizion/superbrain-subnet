# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Embedding Layer — V2 Scoring Engine
#
# Three-tier fallback chain: Ollama API → TF-IDF (sklearn) → Word Overlap
# Auto-detects best available backend at import time.
# Override with SUPERBRAIN_EMBEDDING_BACKEND env var (auto/ollama/tfidf/wordoverlap).

import os
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Backend identifiers
BACKEND_OLLAMA = "ollama"
BACKEND_TFIDF = "tfidf"
BACKEND_WORDOVERLAP = "wordoverlap"

_embedding_backend: str = BACKEND_WORDOVERLAP
_ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_ollama_model: str = os.environ.get("SUPERBRAIN_EMBED_MODEL", "nomic-embed-text")

# ── FAISS index (optional acceleration layer) ───────────────────────
# FAISS sits *on top of* the embedding backend — it indexes vectors from
# whichever backend is active (Ollama or TF-IDF) and provides O(1)
# approximate nearest-neighbor search instead of O(N) cosine scan.

_faiss_available: bool = False
_faiss_index = None          # faiss.IndexFlatIP instance
_faiss_texts: List[str] = []  # parallel list of indexed texts
_faiss_dim: int = 0

try:
    import faiss as _faiss_lib
    _faiss_available = True
except ImportError:
    _faiss_lib = None


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


# ── FAISS index management ─────────────────────────────────────────

def faiss_available() -> bool:
    """True if FAISS library is importable."""
    return _faiss_available


def faiss_build_index(texts: List[str]) -> bool:
    """
    Build a FAISS inner-product index from a list of texts.
    Uses the current embedding backend (Ollama or TF-IDF) to compute vectors.
    Returns True if the index was built successfully.
    """
    global _faiss_index, _faiss_texts, _faiss_dim

    if not _faiss_available:
        return False
    if not texts:
        _faiss_index = None
        _faiss_texts = []
        return True

    vectors = _embed_texts_for_faiss(texts)
    if vectors is None:
        return False

    _faiss_dim = vectors.shape[1]
    # Normalize for inner product = cosine similarity
    _faiss_lib.normalize_L2(vectors)
    index = _faiss_lib.IndexFlatIP(_faiss_dim)
    index.add(vectors)
    _faiss_index = index
    _faiss_texts = list(texts)
    logger.info(f"FAISS index built: {len(texts)} vectors, dim={_faiss_dim}")
    return True


def faiss_search(query: str, top_k: int = 10) -> Optional[List[tuple]]:
    """
    Search the FAISS index for nearest neighbors.
    Returns list of (index, score) tuples, or None if index not available.
    """
    if not _faiss_available or _faiss_index is None or _faiss_index.ntotal == 0:
        return None

    q_vec = _embed_single_for_faiss(query)
    if q_vec is None:
        return None

    _faiss_lib.normalize_L2(q_vec)
    k = min(top_k, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(q_vec, k)

    results = []
    for i in range(k):
        idx = int(indices[0][i])
        if idx < 0:
            continue
        score = float(np.clip(scores[0][i], 0.0, 1.0))
        results.append((idx, score))
    return results


def faiss_batch_similarity(query: str, chunks: List[str]) -> Optional[List[float]]:
    """
    Compute similarity between query and all indexed chunks via FAISS.
    Returns a list parallel to _faiss_texts (not the input chunks list).
    Returns None if FAISS isn't available or index doesn't match chunks.
    """
    if not _faiss_available or _faiss_index is None:
        return None
    if len(_faiss_texts) != len(chunks) or _faiss_texts != chunks:
        return None

    q_vec = _embed_single_for_faiss(query)
    if q_vec is None:
        return None

    _faiss_lib.normalize_L2(q_vec)
    k = _faiss_index.ntotal
    scores, indices = _faiss_index.search(q_vec, k)

    result = [0.0] * len(chunks)
    for i in range(k):
        idx = int(indices[0][i])
        if 0 <= idx < len(result):
            result[idx] = float(np.clip(scores[0][i], 0.0, 1.0))
    return result


def _embed_texts_for_faiss(texts: List[str]) -> Optional[np.ndarray]:
    """Embed a list of texts into numpy vectors using current backend."""
    if _embedding_backend == BACKEND_OLLAMA:
        try:
            vecs = []
            for t in texts:
                if t and t.strip():
                    vecs.append(_ollama_embed(t))
                else:
                    vecs.append(None)
            # Fill None with zero vectors
            dim = next((v.shape[0] for v in vecs if v is not None), None)
            if dim is None:
                return None
            result = np.zeros((len(texts), dim), dtype=np.float32)
            for i, v in enumerate(vecs):
                if v is not None:
                    result[i] = v
            return result
        except Exception as e:
            logger.warning(f"FAISS Ollama embedding failed: {e}")

    if _embedding_backend in (BACKEND_OLLAMA, BACKEND_TFIDF):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            safe = [t if t and t.strip() else "" for t in texts]
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(safe)
            return np.array(tfidf.todense(), dtype=np.float32)
        except Exception as e:
            logger.warning(f"FAISS TF-IDF embedding failed: {e}")

    return None


def _embed_single_for_faiss(text: str) -> Optional[np.ndarray]:
    """Embed a single text for FAISS query. Returns (1, dim) array."""
    if not text or not text.strip():
        return None

    if _embedding_backend == BACKEND_OLLAMA:
        try:
            vec = _ollama_embed(text)
            return vec.reshape(1, -1)
        except Exception:
            pass

    if _embedding_backend in (BACKEND_OLLAMA, BACKEND_TFIDF):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Need to refit with same vocabulary — use stored texts
            if _faiss_texts:
                vectorizer = TfidfVectorizer()
                all_texts = [t if t and t.strip() else "" for t in _faiss_texts] + [text]
                tfidf = vectorizer.fit_transform(all_texts)
                return np.array(tfidf[-1].todense(), dtype=np.float32)
        except Exception:
            pass

    return None


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
