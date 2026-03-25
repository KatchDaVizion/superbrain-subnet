# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Validator Forward Pass
#
# This follows the exact same pattern as the official template's forward():
#   1. Select miner UIDs to query
#   2. Build RAGSynapse with a query + document chunks
#   3. Send to miners via dendrite
#   4. Score responses with get_rewards()
#   5. Update scores via self.update_scores()
#
# Dynamic Knowledge: Pulls chunks from sync_queue when available.
# Falls back to static KNOWLEDGE_BASE when sync_queue is empty.

import asyncio
import random
import re
import time

import bittensor as bt

from superbrain.protocol import RAGSynapse
from superbrain.validator.reward import get_rewards
from superbrain.utils.uids import get_random_uids


# =====================================================================
# STATIC KNOWLEDGE BASE — Fallback when sync_queue has no chunks yet.
# Once miners sync real knowledge, the validator builds queries from
# synced chunks instead. Static KB also seeds challenge queries.
# =====================================================================

KNOWLEDGE_BASE = [
    {
        "query": "What is Bittensor and how does it work?",
        "chunks": [
            "Bittensor is a decentralized machine learning network that creates an open marketplace for AI models. It uses a blockchain-based incentive mechanism to reward miners who contribute useful intelligence to the network.",
            "The Bittensor network operates through subnets, each focused on a specific task like text generation, image recognition, or data storage. Validators evaluate miner outputs and set weights to determine TAO token rewards.",
            "TAO is the native cryptocurrency of the Bittensor network. It is used to incentivize miners, stake on validators, and govern subnet creation. The total supply is capped at 21 million tokens.",
        ],
        "sources": ["bittensor_docs.md", "bittensor_docs.md", "tokenomics.md"],
    },
    {
        "query": "What is Retrieval-Augmented Generation (RAG)?",
        "chunks": [
            "Retrieval-Augmented Generation (RAG) is a technique that enhances large language model outputs by first retrieving relevant documents from a knowledge base, then using those documents as context for generating responses.",
            "RAG systems typically use vector databases like Qdrant, Pinecone, or ChromaDB to store document embeddings. When a query arrives, the system finds the most similar chunks and passes them to the LLM as context.",
            "The main advantage of RAG over fine-tuning is that the knowledge base can be updated without retraining the model. This makes RAG ideal for applications requiring current information or domain-specific knowledge.",
        ],
        "sources": ["rag_overview.md", "vector_databases.md", "rag_vs_finetuning.md"],
    },
    {
        "query": "How does Yuma Consensus work in Bittensor?",
        "chunks": [
            "Yuma Consensus is Bittensor's mechanism for aggregating validator weights into final miner scores. Each validator sets weights for miners they evaluate, and these weights are combined using a stake-weighted average.",
            "Validators with more TAO staked have proportionally more influence on the final consensus weights. This creates an incentive for validators to honestly evaluate miner quality, as their stake is at risk.",
            "The consensus process runs every tempo (approximately 360 blocks). After consensus, TAO emissions are distributed to miners based on their final consensus weights.",
        ],
        "sources": ["yuma_consensus.md", "staking_docs.md", "emissions.md"],
    },
    {
        "query": "What are the benefits of offline-first AI systems?",
        "chunks": [
            "Offline-first AI systems process data locally without requiring constant internet connectivity. This approach provides better privacy since sensitive data never leaves the user's device.",
            "Local AI inference using models like Ollama, llama.cpp, or ONNX Runtime allows users to run language models on consumer hardware. Models like TinyLlama and Phi-2 can run on devices with as little as 4GB RAM.",
            "Offline-first architectures reduce latency by eliminating network round trips. They also reduce operational costs since there are no API fees for cloud-hosted models.",
        ],
        "sources": ["offline_ai.md", "local_inference.md", "architecture_benefits.md"],
    },
    {
        "query": "How do vector databases store and retrieve document embeddings?",
        "chunks": [
            "Vector databases like Qdrant store documents as high-dimensional numerical vectors called embeddings. These embeddings capture the semantic meaning of text, enabling similarity-based retrieval rather than keyword matching.",
            "When a query is received, it is converted to an embedding using the same model that encoded the documents. The database then performs approximate nearest neighbor search to find the most similar stored vectors.",
            "Common embedding models include sentence-transformers models like all-MiniLM-L6-v2 and nomic-embed-text. The choice of model affects both retrieval quality and computational cost.",
        ],
        "sources": ["vector_db_intro.md", "ann_search.md", "embedding_models.md"],
    },
]

# Challenge queries — known-answer queries for anti-gaming
CHALLENGE_QUERIES = [
    {
        "query": "What is the native token of the Bittensor network?",
        "expected_keywords": ["tao", "token", "21 million", "cryptocurrency"],
        "kb_index": 0,
    },
    {
        "query": "Name one vector database used in RAG systems.",
        "expected_keywords": ["qdrant", "pinecone", "chromadb", "vector"],
        "kb_index": 1,
    },
]

# Minimum chunks needed to build a dynamic query
MIN_CHUNKS_FOR_DYNAMIC = 2

# Module-level set of chunk hashes used in the previous round (diversity tracking).
# Replaced wholesale each round so it never grows unbounded.
_SEEN_CHUNK_HASHES: set = set()


def _extract_question(text: str) -> str:
    """
    Derive a natural, content-specific question from a knowledge chunk.

    Applies a hierarchy of regex heuristics so each chunk yields a question
    tied to its subject matter rather than a generic template.
    """
    sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0].strip()

    # "X is [a/an/the] …" → "What is X?"
    m = re.match(r"^(.+?)\s+is\s+(?:a |an |the )?(.+)", sentence, re.IGNORECASE)
    if m:
        subject = m.group(1).strip().rstrip(".,;:")
        return f"What is {subject}?"

    # "X uses/enables/provides/creates …" → "How does X work?"
    m = re.match(
        r"^(.+?)\s+(uses|enables|allows|provides|creates|supports)\b",
        sentence,
        re.IGNORECASE,
    )
    if m:
        subject = m.group(1).strip().rstrip(".,;:")
        return f"How does {subject} work?"

    # "X lets/helps/makes/ensures …" → "What does X do?"
    m = re.match(r"^(.+?)\s+(lets|helps|makes|ensures)\b", sentence, re.IGNORECASE)
    if m:
        subject = m.group(1).strip().rstrip(".,;:")
        return f"What does {subject} do?"

    # Fallback: use first meaningful words as topic
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "and", "or"}
    words = [w.strip(".,;:") for w in sentence.split() if w.lower() not in stop and len(w) > 3]
    topic = " ".join(words[:3]) if words else sentence[:40]
    return f"What can you tell me about {topic}?"


def _build_dynamic_query(chunks):
    """
    Build a RAG query from synced knowledge chunks.

    V2 behaviour (anti-gaming):
    - Selects exactly 3 chunks per round (or all available if fewer).
    - Diversity gate: prefers chunks not seen in the previous round so
      consecutive validation rounds never reuse the same material.
    - Generates a natural, content-specific question from the primary chunk
      via _extract_question() instead of a generic template string.
    - Updates _SEEN_CHUNK_HASHES for the next round.

    Returns (query, chunk_texts, sources) or None if pool is too small.
    """
    global _SEEN_CHUNK_HASHES

    if len(chunks) < MIN_CHUNKS_FOR_DYNAMIC:
        return None

    # Prefer chunks not seen last round; fall back to full pool if needed
    fresh = [c for c in chunks if c.content_hash not in _SEEN_CHUNK_HASHES]
    pool = fresh if len(fresh) >= MIN_CHUNKS_FOR_DYNAMIC else chunks

    n_select = min(3, len(pool))
    selected = random.sample(pool, n_select)

    # Natural question derived from primary chunk content
    query = _extract_question(selected[0].content)

    chunk_texts = [c.content for c in selected]
    sources = []
    for c in selected:
        src = (c.metadata or {}).get("source_file", "")
        if not src:
            src = f"chunk_{c.content_hash[:8]}"
        sources.append(src)

    # Record this round's hashes so next round avoids them
    _SEEN_CHUNK_HASHES = {c.content_hash for c in selected}

    return query, chunk_texts, sources


async def forward(self):
    """
    Validator forward pass — called every step by the base validator loop.

    Follows the exact template pattern:
      1. Select random miner UIDs
      2. Build query synapse (dynamic from sync_queue, or static fallback)
      3. Query miners via dendrite
      4. Score responses
      5. Update scores
    """
    # 1. Select miner UIDs
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    if len(miner_uids) == 0:
        bt.logging.warning("No available miners to query.")
        return

    # 2. Select query source
    # 20% chance of challenge query for anti-gaming (always from static KB)
    is_challenge = random.random() < 0.2
    use_dynamic = False

    if is_challenge and CHALLENGE_QUERIES:
        challenge = random.choice(CHALLENGE_QUERIES)
        kb = KNOWLEDGE_BASE[challenge["kb_index"]]
        query = challenge["query"]
        chunks = kb["chunks"]
        sources = kb["sources"]
        expected_keywords = challenge["expected_keywords"]
    else:
        expected_keywords = []

        # Try dynamic query from sync_queue first
        if hasattr(self, "sync_queue"):
            try:
                pool_chunks = self.sync_queue.get_random_chunks(limit=10)
                dynamic = _build_dynamic_query(pool_chunks)
                if dynamic is not None:
                    query, chunks, sources = dynamic
                    use_dynamic = True
            except Exception as e:
                bt.logging.debug(f"Dynamic query failed, using static KB: {e}")

        # Fallback to static knowledge base
        if not use_dynamic:
            kb = random.choice(KNOWLEDGE_BASE)
            query = kb["query"]
            chunks = kb["chunks"]
            sources = kb["sources"]

    source_label = "dynamic" if use_dynamic else ("challenge" if is_challenge else "static")
    bt.logging.info(
        f"Querying {len(miner_uids)} miners [{source_label}]: {query[:60]}..."
    )

    # 3. Build and send synapse
    synapse = RAGSynapse(
        query=query,
        context_chunks=chunks,
        chunk_sources=sources,
    )

    start_time = time.time()
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )
    total_time = time.time() - start_time

    bt.logging.info(f"Received {len(responses)} responses in {total_time:.2f}s")

    # 4. Extract response times
    response_times = []
    for resp in responses:
        if isinstance(resp, dict):
            response_times.append(total_time / max(1, len(responses)))
        else:
            response_times.append(30.0)  # Penalty for failed responses

    # Normalize responses to dicts
    normalized = []
    for resp in responses:
        if isinstance(resp, dict):
            normalized.append(resp)
        else:
            normalized.append({"response": "", "citations": [], "confidence_score": 0.0})

    # 5. Score responses
    rewards = get_rewards(
        self,
        query=query,
        context_chunks=chunks,
        responses=normalized,
        response_times=response_times,
    )

    # 6. Challenge query penalty
    if is_challenge and expected_keywords:
        for i, resp in enumerate(normalized):
            resp_text = (resp.get("response", "") or "").lower()
            if resp_text:
                hits = sum(1 for kw in expected_keywords if kw in resp_text)
                if hits < len(expected_keywords) * 0.25:
                    rewards[i] *= 0.5
                    bt.logging.warning(
                        f"UID {miner_uids[i]}: CHALLENGE FAILED — score halved"
                    )

    bt.logging.info(f"Scored responses: {rewards}")

    # 7. Update scores
    self.update_scores(rewards, miner_uids)

    await asyncio.sleep(5)
