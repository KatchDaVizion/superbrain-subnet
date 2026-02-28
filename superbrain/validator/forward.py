# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Validator Forward Pass
#
# This follows the exact same pattern as the official template's forward():
#   1. Select miner UIDs to query
#   2. Build RAGSynapse with a query + document chunks
#   3. Send to miners via dendrite
#   4. Score responses with get_rewards()
#   5. Update scores via self.update_scores()

import time
import random
import bittensor as bt

from superbrain.protocol import RAGSynapse
from superbrain.validator.reward import get_rewards
from superbrain.utils.uids import get_random_uids


# =====================================================================
# PROTOTYPE KNOWLEDGE BASE — Static queries for testnet demonstration.
# Production validators MUST maintain curated, rotating ground-truth
# datasets. Static queries allow answer caching by gaming miners.
# This is acknowledged in the README's Known Limitations section.
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


async def forward(self):
    """
    Validator forward pass — called every step by the base validator loop.

    Follows the exact template pattern:
      1. Select random miner UIDs
      2. Build query synapse
      3. Query miners via dendrite
      4. Score responses
      5. Update scores
    """
    # 1. Select miner UIDs
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    if len(miner_uids) == 0:
        bt.logging.warning("No available miners to query.")
        return

    # 2. Select query from knowledge base
    # 20% chance of challenge query for anti-gaming
    is_challenge = random.random() < 0.2
    if is_challenge and CHALLENGE_QUERIES:
        challenge = random.choice(CHALLENGE_QUERIES)
        kb = KNOWLEDGE_BASE[challenge["kb_index"]]
        query = challenge["query"]
        chunks = kb["chunks"]
        sources = kb["sources"]
        expected_keywords = challenge["expected_keywords"]
    else:
        kb = random.choice(KNOWLEDGE_BASE)
        query = kb["query"]
        chunks = kb["chunks"]
        sources = kb["sources"]
        expected_keywords = []

    bt.logging.info(f"Querying {len(miner_uids)} miners: {query[:60]}...")

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

    time.sleep(5)
