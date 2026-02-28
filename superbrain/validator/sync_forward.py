# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Validator Sync Forward Pass
#
# Periodically queries miners for new knowledge chunks via KnowledgeSyncSynapse.
# Validates returned batches, ingests new chunks, and scores miners.

import base64
import json
import time
import zlib

import bittensor as bt

from superbrain.protocol import KnowledgeSyncSynapse
from superbrain.validator.sync_reward import get_sync_rewards
from superbrain.utils.uids import get_random_uids
try:
    from sync.protocol.pool_model import KnowledgeChunk, compute_content_hash
except ImportError:
    KnowledgeChunk = None
    compute_content_hash = None

# How often to run sync (every N validator steps)
SYNC_INTERVAL_STEPS = 10

# Max chunks to request per sync round
MAX_CHUNKS_PER_SYNC = 50


def _decode_and_validate_batch(batch_data_b64, known_hashes):
    """
    Decode base64 batch data and validate each chunk.
    Returns list of valid, unique KnowledgeChunks.
    """
    if not batch_data_b64:
        return []

    try:
        compressed = base64.b64decode(batch_data_b64)
        raw_json = zlib.decompress(compressed)
        chunk_dicts = json.loads(raw_json)
    except Exception as e:
        bt.logging.warning(f"Failed to decode sync batch: {e}")
        return []

    known_set = set(known_hashes)
    valid_chunks = []

    for d in chunk_dicts:
        try:
            chunk = KnowledgeChunk.model_validate(d)
        except Exception:
            continue

        # Verify content hash
        if not chunk.content or not chunk.content.strip():
            continue
        if not chunk.content_hash:
            continue
        computed = compute_content_hash(chunk.content)
        if computed != chunk.content_hash:
            continue

        # TODO: Enforce Ed25519 signature verification once all miners sign chunks
        # if not chunk.verify(public_key):
        #     continue

        # Skip chunks we already have
        if chunk.content_hash in known_set:
            continue

        valid_chunks.append(chunk)
        # Track so we don't double-count within this batch
        known_set.add(chunk.content_hash)

    return valid_chunks


async def sync_forward(self):
    """
    Validator sync forward pass — queries miners for knowledge chunks.

    Called periodically from the main validator forward() based on sync_step counter.
    """
    # Build known hashes from our SyncQueue
    manifest = self.sync_queue.get_manifest(node_id="validator")
    known_hashes = list(manifest.chunk_hashes)

    # Select miner UIDs
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

    if len(miner_uids) == 0:
        bt.logging.warning("No available miners for sync.")
        return

    bt.logging.info(f"Sync: querying {len(miner_uids)} miners for knowledge chunks")

    # Build and send synapse
    synapse = KnowledgeSyncSynapse(
        known_hashes=known_hashes,
        max_chunks=MAX_CHUNKS_PER_SYNC,
        node_id="validator",
    )

    start_time = time.time()
    responses = await self.dendrite(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )
    total_time = time.time() - start_time

    bt.logging.info(f"Sync: received {len(responses)} responses in {total_time:.2f}s")

    # Extract response times
    response_times = []
    for resp in responses:
        if isinstance(resp, dict) and resp.get("batch_data"):
            response_times.append(total_time / max(1, len(responses)))
        else:
            response_times.append(60.0)

    # Normalize responses
    normalized = []
    for resp in responses:
        if isinstance(resp, dict):
            normalized.append(resp)
        else:
            normalized.append({"batch_data": None, "chunk_count": 0, "batch_id": None})

    # Validate and ingest chunks from each miner
    total_ingested = 0
    for resp in normalized:
        batch_data = resp.get("batch_data")
        if not batch_data:
            continue

        valid_chunks = _decode_and_validate_batch(batch_data, known_hashes)
        for chunk in valid_chunks:
            # Ensure chunk is marked public for queue
            chunk.pool_visibility = "public"
            if chunk.shared_at is None:
                chunk.shared_at = time.time()
            if self.sync_queue.add_to_queue(chunk):
                total_ingested += 1
                # Update known_hashes so we don't re-ingest from another miner
                known_hashes.append(chunk.content_hash)

    if total_ingested > 0:
        bt.logging.info(f"Sync: ingested {total_ingested} new chunks")

    # Score miners
    rewards = get_sync_rewards(
        self,
        responses=normalized,
        known_hashes=known_hashes,
        max_chunks=MAX_CHUNKS_PER_SYNC,
        response_times=response_times,
    )

    bt.logging.info(f"Sync scores: {rewards}")

    # Update scores (blends with RAG scores via EMA)
    self.update_scores(rewards, miner_uids)
