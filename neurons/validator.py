# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Validator — Query Generator + Response Scorer
#
# Follows the official bittensor-subnet-template pattern exactly.
# Inherits from BaseValidatorNeuron which handles weights, metagraph, scoring.
#
# Usage:
#   python neurons/validator.py --netuid YOUR_NETUID --subtensor.network test \
#       --wallet.name validator --wallet.hotkey default --logging.debug

import os
import time

import bittensor as bt

from superbrain.base.validator import BaseValidatorNeuron
from superbrain.validator import forward
from superbrain.validator.sync_forward import sync_forward, SYNC_INTERVAL_STEPS
from sync.queue.sync_queue import SyncQueue


class Validator(BaseValidatorNeuron):
    """
    SuperBrain validator. Sends RAG queries to miners, scores responses
    using the 4-factor mechanism, and sets weights on-chain.

    Inherits from BaseValidatorNeuron which handles:
    - Wallet, subtensor, metagraph, dendrite setup
    - Score tracking with exponential moving average
    - Weight setting via process_weights_for_netuid
    - Metagraph resync and hotkey tracking
    - Main run loop with sync and epoch management
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        # Knowledge sync queue
        db_dir = self.config.neuron.full_path if hasattr(self.config.neuron, 'full_path') else "."
        sync_db = os.path.join(db_dir, "validator_sync_queue.db")
        self.sync_queue = SyncQueue(db_path=sync_db)
        self.sync_step = 0
        bt.logging.info(f"Sync queue initialized: {sync_db}")

        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        Validator forward pass:
        1. Select query from knowledge base
        2. Query miners with RAGSynapse
        3. Score responses (Supportedness/Relevance/Novelty/Latency)
        4. Update scores via EMA
        5. Periodically sync knowledge chunks from miners
        """
        # RAG forward pass
        await forward(self)

        # Periodic sync forward pass
        self.sync_step += 1
        if self.sync_step >= SYNC_INTERVAL_STEPS:
            self.sync_step = 0
            await sync_forward(self)


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
