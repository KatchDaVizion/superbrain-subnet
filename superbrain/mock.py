# The MIT License (MIT)
# Copyright 2023 Yuma Rao
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# Adapted for SuperBrain RAGSynapse

import base64
import json
import time
import asyncio
import random
import uuid
import zlib

import bittensor as bt

from typing import List

try:
    from sync.protocol.pool_model import KnowledgeChunk, compute_content_hash
    _has_sync = True
except ImportError:
    _has_sync = False


def _patch_neuron_for_uid_lite(mock_subtensor):
    """Patch bittensor 10.x MockSubtensor bug where neuron_for_uid_lite
    references NeuronInfo fields (rank, trust, pruning_score) that no longer exist."""
    from bittensor.core.chain_data import NeuronInfoLite

    original = mock_subtensor.__class__.neuron_for_uid_lite

    def patched(self, uid, netuid, block=None):
        if uid is None:
            return NeuronInfoLite.get_null_neuron()
        if block is not None and self.block_number < block:
            raise Exception("Cannot query block in the future")
        if block is None:
            block = self.block_number
        if netuid not in self.chain_state["SubtensorModule"]["NetworksAdded"]:
            return None
        neuron_info = self._neuron_subnet_exists(uid, netuid, block)
        if neuron_info is None:
            return None
        return NeuronInfoLite(
            hotkey=neuron_info.hotkey,
            coldkey=neuron_info.coldkey,
            uid=neuron_info.uid,
            netuid=neuron_info.netuid,
            active=getattr(neuron_info, 'active', 0),
            stake=getattr(neuron_info, 'stake', bt.Balance(0)),
            stake_dict=getattr(neuron_info, 'stake_dict', {}),
            total_stake=getattr(neuron_info, 'total_stake', bt.Balance(0)),
            emission=getattr(neuron_info, 'emission', 0.0),
            incentive=getattr(neuron_info, 'incentive', 0.0),
            consensus=getattr(neuron_info, 'consensus', 0.0),
            validator_trust=getattr(neuron_info, 'validator_trust', 0.0),
            dividends=getattr(neuron_info, 'dividends', 0.0),
            last_update=getattr(neuron_info, 'last_update', 0),
            validator_permit=getattr(neuron_info, 'validator_permit', False),
            prometheus_info=getattr(neuron_info, 'prometheus_info', None),
            axon_info=getattr(neuron_info, 'axon_info', None),
            is_null=False,
        )

    mock_subtensor.__class__.neuron_for_uid_lite = patched


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super().__init__(network=network)

        # Patch neuron_for_uid_lite bug in bittensor 10.x
        _patch_neuron_for_uid_lite(self)

        # subnet_exists() returns MagicMock in bt 10.x, so check chain state directly
        if netuid not in self.chain_state["SubtensorModule"]["NetworksAdded"]:
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock miners with generated keypairs
        from bittensor_wallet import Keypair
        for i in range(1, n + 1):
            kp = Keypair.create_from_uri(f"//mock_miner_{i}")
            self.force_register_neuron(
                netuid=netuid,
                hotkey_ss58=kp.ss58_address,
                coldkey_ss58=kp.ss58_address,
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.Metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "127.0.0.0"
            axon.port = 8091

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")


class MockDendrite(bt.Dendrite):
    """
    Replaces a real bittensor network request with a mock request that
    returns RAGSynapse-compatible responses for testing.
    """

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.AxonInfo],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            async def single_axon_response(i, axon):
                start_time = time.time()
                s = synapse.copy()
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                process_time = random.random()
                if process_time < timeout:
                    s.dendrite.process_time = str(time.time() - start_time)
                    # Fill in mock response data based on synapse type
                    if _has_sync and hasattr(s, 'known_hashes') and hasattr(s, 'max_chunks'):
                        # KnowledgeSyncSynapse — return mock knowledge chunks
                        mock_chunks = []
                        for j in range(min(3, s.max_chunks)):
                            content = f"Mock knowledge chunk {i}_{j}: This is test content for sync."
                            chunk = KnowledgeChunk(
                                content=content,
                                content_hash=compute_content_hash(content),
                                origin_node_id=f"mock_miner_{i}",
                                timestamp=time.time() - j * 3600,
                                signature="mock_sig",
                                pool_visibility="public",
                                shared_at=time.time(),
                            )
                            # Only include chunks not in known_hashes
                            if chunk.content_hash not in set(s.known_hashes):
                                mock_chunks.append(chunk)

                        if mock_chunks:
                            chunk_dicts = [c.model_dump() for c in mock_chunks]
                            raw_json = json.dumps(chunk_dicts, sort_keys=True).encode("utf-8")
                            compressed = zlib.compress(raw_json)
                            s.batch_data = base64.b64encode(compressed).decode("ascii")
                            s.chunk_count = len(mock_chunks)
                            s.batch_id = str(uuid.uuid4())
                        else:
                            s.batch_data = None
                            s.chunk_count = 0
                            s.batch_id = None

                    elif hasattr(s, 'query') and hasattr(s, 'context_chunks'):
                        # RAGSynapse — generate a mock response using context chunks
                        if s.context_chunks:
                            chunk_idx = i % len(s.context_chunks)
                            s.response = f"Based on the sources [{chunk_idx + 1}], {s.context_chunks[chunk_idx]}"
                            s.citations = [chunk_idx]
                            s.confidence_score = 0.7 + random.random() * 0.3
                        else:
                            s.response = "No context provided."
                            s.citations = []
                            s.confidence_score = 0.0
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                    synapse.dendrite.process_time = str(process_time)
                else:
                    if hasattr(s, 'response'):
                        s.response = ""
                        s.citations = []
                        s.confidence_score = 0.0
                    if hasattr(s, 'batch_data'):
                        s.batch_data = None
                        s.chunk_count = 0
                        s.batch_id = None
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = str(timeout)

                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        return "MockDendrite({})".format(self.keypair.ss58_address)
