# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Miner — RAG Response Generator + LAN Sync
#
# Follows the official bittensor-subnet-template pattern exactly.
# Inherits from BaseMinerNeuron which handles axon, registration, and lifecycle.
#
# Usage:
#   # Basic miner (RAG + Bittensor sync only)
#   python neurons/miner.py --netuid 442 --subtensor.network test \
#       --wallet.name sb_miner --wallet.hotkey default --logging.debug
#
#   # Miner with embedded LAN sync (discovers peers, shares knowledge)
#   python neurons/miner.py --netuid 442 --subtensor.network test \
#       --wallet.name sb_miner --wallet.hotkey default --logging.debug \
#       --lan-sync --lan-port 8384
#
#   # With seed data for demo
#   python neurons/miner.py --netuid 442 --subtensor.network test \
#       --wallet.name sb_miner --wallet.hotkey default --logging.debug \
#       --lan-sync --lan-port 8384 --seed
#
#   # With static peer (e.g. Tailscale)
#   python neurons/miner.py --netuid 442 --subtensor.network test \
#       --wallet.name sb_miner --wallet.hotkey default --logging.debug \
#       --lan-sync --lan-static 100.x.x.x:8384

import asyncio
import argparse
import base64
import json
import logging
import os
import re
import sys
import threading
import time
import typing
import uuid
import zlib

import bittensor as bt

import superbrain
from superbrain.base.miner import BaseMinerNeuron
from sync.queue.sync_queue import SyncQueue
from sync.protocol.pool_model import KnowledgeChunk, generate_node_keypair
from sync.lan.lan_sync import LANSyncManager

SAMPLE_CHUNKS = [
    "Bittensor is a decentralized ML network. Miners contribute intelligence and earn TAO tokens.",
    "SuperBrain is a local-first knowledge network on Bittensor. Private by default, public by choice.",
    "RAG enhances LLM outputs by retrieving relevant documents from a vector database before generating.",
    "The Two-Pool Privacy Model ensures all knowledge is private by default. Users explicitly share.",
    "Delta sync uses a 5-step protocol: Handshake, Manifest, Diff, Transfer, Confirm.",
    "Ed25519 signatures ensure every knowledge chunk is cryptographically signed by its originator.",
    "LAN sync uses mDNS to auto-discover peers on the same WiFi. No configuration needed.",
    "Yuma Consensus aggregates validator weights using stake-weighted averaging for TAO distribution.",
    "KnowledgeSyncSynapse lets validators pull knowledge chunks from miners over Bittensor.",
    "Offline-first AI processes data locally. Better privacy, lower latency, zero API costs.",
]


def _ollama_available(url="http://localhost:11434"):
    try:
        import requests
        resp = requests.get(f"{url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _detect_model(url="http://localhost:11434"):
    try:
        import requests
        resp = requests.get(f"{url}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            for pref in ["qwen2.5:0.5b", "qwen2.5", "gemma3", "gemma2", "llama3", "mistral", "phi", "tinyllama"]:
                for name in models:
                    if pref in name.lower():
                        return name
            return models[0] if models else None
    except Exception:
        return None


class Miner(BaseMinerNeuron):
    """
    SuperBrain miner. Receives RAG queries from validators, generates
    cited answers using Ollama (or extractive fallback), and returns them.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = _detect_model(self.ollama_url)
        if self.ollama_model:
            bt.logging.info(f"Ollama model: {self.ollama_model}")
        else:
            bt.logging.warning("Ollama unavailable — extractive fallback active")

        self.sentence_offset = 0
        self.paraphrase_mode = False
        self.grounding_threshold = 0.35

        # Node identity keypair (for signing chunks)
        self._node_private_key, self._node_public_key = generate_node_keypair()

        # Knowledge sync queue
        db_dir = self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else "."
        sync_db = os.path.join(db_dir, "miner_sync_queue.db")
        self.sync_queue = SyncQueue(db_path=sync_db)
        bt.logging.info(f"Sync queue initialized: {sync_db}")

        # Attach sync handlers to axon
        self.axon.attach(
            forward_fn=self.forward_sync,
            blacklist_fn=self.blacklist_sync,
            priority_fn=self.priority_sync,
        )

        # LAN Sync (embedded — optional, enabled with --lan-sync)
        self._lan_manager = None
        self._lan_thread = None
        self._lan_loop = None
        lan_sync = getattr(self.config, 'lan_sync', False)
        if lan_sync:
            self._start_lan_sync()

    async def forward(
        self, synapse: superbrain.protocol.RAGSynapse
    ) -> superbrain.protocol.RAGSynapse:
        """Process RAGSynapse: generate a cited answer from query + chunks."""
        bt.logging.info(f"Query: {synapse.query[:80]}...")

        # Extractive runs first — guaranteed citations and high S score baseline.
        # Ollama synthesis runs on top; we keep it only if it produces valid citations.
        ext_response, ext_citations = self._generate_extractive(synapse)

        response, citations = ext_response, ext_citations
        if self.ollama_model:
            oll_response, oll_citations = self._generate_ollama(synapse)
            if oll_citations and self._is_grounded(oll_response, synapse.context_chunks, threshold=self.grounding_threshold):
                response, citations = oll_response, oll_citations

        synapse.response = response
        synapse.citations = citations
        synapse.confidence_score = 0.8 if response else 0.0

        bt.logging.info(f"Response: {len(response or '')} chars, citations={citations}")
        return synapse

    def _generate_ollama(self, synapse):
        """Ollama synthesis: low-temp, verbatim-quote prompt, falls back silently."""
        import requests as req

        chunks_text = ""
        for i, chunk in enumerate(synapse.context_chunks):
            chunks_text += f"\n[{i+1}] {chunk}\n"

        style_rule = (
            "- Rephrase the key information in your own words; avoid repeating exact phrases from the sources."
            if self.paraphrase_mode else
            "- Copy key phrases verbatim from the sources."
        )
        prompt = f"""Answer the question using ONLY the sources below.
Rules:
{style_rule}
- Every sentence must end with a citation marker like [1], [2], or [3].
- Do not add information not present in the sources.

Sources:
{chunks_text}

Question: {synapse.query}

Answer (2-3 sentences, every claim cited with [1] [2] or [3]):"""

        try:
            resp = req.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200, "num_ctx": 768},
                },
                timeout=25,
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                citations = self._extract_citations(answer, len(synapse.context_chunks))
                return answer, citations
        except Exception as e:
            bt.logging.debug(f"Ollama failed: {e}")
        return "", []

    def _generate_extractive(self, synapse):
        """Extractive primary: quotes top sentences from every provided chunk.

        Scores each sentence by query-word overlap, picks the best two per chunk,
        and formats them with citation markers. Guarantees citations from all
        provided chunks and maximises semantic similarity to the source material.
        """
        if not synapse.context_chunks:
            return "No source documents provided.", []

        query_words = set(synapse.query.lower().split())
        parts = []
        citations = []

        for i, chunk in enumerate(synapse.context_chunks):
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if len(s.strip()) > 20]
            if not sentences:
                sentences = [chunk[:400]]

            # Pick top 2 sentences by query-word overlap (offset lets secondary miners pick different sentences)
            scored = sorted(sentences, key=lambda s: len(query_words & set(s.lower().split())), reverse=True)
            start = min(self.sentence_offset, max(0, len(scored) - 1))
            excerpt = " ".join(scored[start:start + 2]) or scored[0]
            if not excerpt.endswith("."):
                excerpt += "."

            parts.append(f"[{i+1}] {excerpt}")
            citations.append(i)

        return " ".join(parts), citations

    def _extract_citations(self, text, max_chunks):
        """Extract [1], [2] citation markers from generated text."""
        matches = re.findall(r'\[(\d+)\]', text)
        citations = []
        for m in matches:
            idx = int(m) - 1
            if 0 <= idx < max_chunks and idx not in citations:
                citations.append(idx)
        return citations

    def _is_grounded(self, response, context_chunks, threshold=0.35):
        """Reject Ollama responses that share < threshold word overlap with context."""
        if not response or not context_chunks:
            return False
        resp_words = set(response.lower().split())
        ctx_words = set(" ".join(context_chunks).lower().split())
        if not resp_words:
            return False
        return len(resp_words & ctx_words) / len(resp_words) >= threshold


    async def forward_sync(
        self, synapse: superbrain.protocol.KnowledgeSyncSynapse
    ) -> superbrain.protocol.KnowledgeSyncSynapse:
        """Process KnowledgeSyncSynapse: return chunks the validator doesn't have."""
        known_set = set(synapse.known_hashes)
        max_chunks = synapse.max_chunks

        # Find which of our chunks the validator doesn't have yet
        manifest = self.sync_queue.get_manifest(node_id="miner")
        new_hashes = [h for h in manifest.chunk_hashes if h not in known_set]

        if not new_hashes:
            bt.logging.info("Sync: no new chunks to send")
            return synapse

        # Batch-fetch candidates in a single SQL query, then sort newest-first
        candidates = self.sync_queue.get_chunks_by_hashes(new_hashes[:max_chunks * 4])
        candidates.sort(key=lambda c: getattr(c, 'timestamp', 0) or 0, reverse=True)
        new_chunks = candidates[:max_chunks]

        if not new_chunks:
            bt.logging.info("Sync: no new chunks to send")
            return synapse

        # Serialize chunks to compressed JSON (same format as SyncBatch)
        chunk_dicts = [c.model_dump() for c in new_chunks]
        raw_json = json.dumps(chunk_dicts, sort_keys=True).encode("utf-8")
        compressed = zlib.compress(raw_json)
        batch_data_b64 = base64.b64encode(compressed).decode("ascii")

        synapse.batch_data = batch_data_b64
        synapse.chunk_count = len(new_chunks)
        synapse.batch_id = str(uuid.uuid4())

        bt.logging.info(f"Sync: sending {len(new_chunks)} chunks (batch {synapse.batch_id[:8]})")
        return synapse

    async def blacklist_sync(
        self, synapse: superbrain.protocol.KnowledgeSyncSynapse
    ) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing dendrite or hotkey"
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            return True, "Unrecognized hotkey"
        if self.config.blacklist.force_validator_permit:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if not self.metagraph.validator_permit[uid]:
                return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority_sync(
        self, synapse: superbrain.protocol.KnowledgeSyncSynapse
    ) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])

    async def blacklist(
        self, synapse: superbrain.protocol.RAGSynapse
    ) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing dendrite or hotkey"
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            return True, "Unrecognized hotkey"
        if self.config.blacklist.force_validator_permit:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            if not self.metagraph.validator_permit[uid]:
                return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority(
        self, synapse: superbrain.protocol.RAGSynapse
    ) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])


    def _start_lan_sync(self):
        """Start embedded LANSyncManager in a background thread."""
        priv, pub = generate_node_keypair()
        node_id = f"miner-{os.getpid()}"

        lan_port = getattr(self.config, 'lan_port', 8384)
        lan_static = getattr(self.config, 'lan_static', None)

        static_peers = None
        if lan_static:
            h, p = lan_static.rsplit(":", 1)
            static_peers = [(h, int(p))]
            bt.logging.info(f"LAN sync static peer: {h}:{p}")

        # Seed sample chunks if requested
        seed = getattr(self.config, 'seed', False)
        if seed:
            added = 0
            for text in SAMPLE_CHUNKS:
                c = KnowledgeChunk(
                    content=text,
                    origin_node_id=node_id,
                    pool_visibility="public",
                    shared_at=time.time(),
                )
                c.sign(priv, signer_public_key=pub)
                if self.sync_queue.add_to_queue(c):
                    added += 1
            bt.logging.info(f"Seeded {added} knowledge chunks into sync queue")

        # Store params for thread-local SyncQueue creation
        self._lan_db_path = self.sync_queue.db_path

        def _run_lan():
            # Create a SEPARATE SyncQueue connection in this thread
            # (SQLite objects can't cross threads; WAL mode handles concurrent access)
            lan_queue = SyncQueue(db_path=self._lan_db_path)
            self._lan_manager = LANSyncManager(
                sync_queue=lan_queue,
                node_id=node_id,
                private_key=priv,
                public_key=pub,
                port=lan_port,
                sync_interval=10,
                static_peers=static_peers,
            )
            self._lan_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._lan_loop)
            self._lan_loop.run_until_complete(self._lan_manager.start())
            bt.logging.info(f"LAN sync started on port {lan_port} (node={node_id})")
            self._lan_loop.run_forever()

        self._lan_thread = threading.Thread(target=_run_lan, daemon=True)
        self._lan_thread.start()
        bt.logging.info("LAN sync thread started")

    def _stop_lan_sync(self):
        """Stop embedded LANSyncManager."""
        if self._lan_manager and self._lan_loop:
            asyncio.run_coroutine_threadsafe(
                self._lan_manager.stop(), self._lan_loop
            ).result(timeout=5)
            self._lan_loop.call_soon_threadsafe(self._lan_loop.stop)
            self._lan_thread.join(timeout=5)
            bt.logging.info("LAN sync stopped")

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_lan_sync()
        super().__exit__(exc_type, exc_value, traceback)


if __name__ == "__main__":
    # Parse LAN sync args before bittensor eats them
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lan-sync", action="store_true", default=False)
    parser.add_argument("--lan-port", type=int, default=8384)
    parser.add_argument("--lan-static", type=str, default=None)
    parser.add_argument("--seed", action="store_true", default=False)
    parser.add_argument("--sentence-offset", type=int, default=0)
    parser.add_argument("--paraphrase-mode", action="store_true", default=False)
    parser.add_argument("--grounding-threshold", type=float, default=0.35)
    lan_args, remaining = parser.parse_known_args()

    # Inject into sys.argv so bittensor only sees its own args
    sys.argv = [sys.argv[0]] + remaining

    # Create miner — config comes from bittensor's argparse
    miner = Miner()
    # Inject LAN config
    miner.config.lan_sync = lan_args.lan_sync
    miner.config.lan_port = lan_args.lan_port
    miner.config.lan_static = lan_args.lan_static
    miner.config.seed = lan_args.seed
    miner.sentence_offset = lan_args.sentence_offset
    miner.paraphrase_mode = lan_args.paraphrase_mode
    miner.grounding_threshold = lan_args.grounding_threshold

    # Start LAN sync if requested (after config is set)
    if lan_args.lan_sync:
        miner._start_lan_sync()

    with miner:
        while True:
            if miner._lan_thread and miner._lan_manager:
                m = miner.sync_queue.get_manifest("miner")
                try:
                    peer_count = len(miner._lan_manager._discovery.peers)
                except Exception:
                    peer_count = 0
                bt.logging.info(
                    f"Miner running... chunks={len(m.chunk_hashes)} "
                    f"lan_peers={peer_count}"
                )
            else:
                bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
