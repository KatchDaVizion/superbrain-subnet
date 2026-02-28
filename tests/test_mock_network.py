"""SuperBrain Mock Network Test — Full Pipeline Test

Tests the complete validator→miner→score pipeline using mock components.
Works around bittensor 10.1.0 MockSubtensor bug (NeuronInfo missing rank attribute).
"""
import sys
import os
import asyncio
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bittensor as bt
from bittensor.utils.mock.subtensor_mock import MockSubtensor

# ─── Monkey-patch bittensor 10.1.0 MockSubtensor bug ───────────────────
# neuron_for_uid_lite references neuron_info.rank etc. but NeuronInfo
# no longer has those fields. Patch to use getattr with defaults.
_original_neuron_for_uid_lite = MockSubtensor.neuron_for_uid_lite

def _patched_neuron_for_uid_lite(self, uid, netuid, block=None):
    from bittensor.core.chain_data import NeuronInfoLite

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

    # NeuronInfoLite in bt 10.x removed: rank, trust, pruning_score, stake (uses total_stake)
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

MockSubtensor.neuron_for_uid_lite = _patched_neuron_for_uid_lite

# ─── Now import our modules (after patch) ──────────────────────────────
from superbrain.protocol import RAGSynapse
from superbrain.validator.reward import get_rewards, reward
from superbrain.validator.forward import KNOWLEDGE_BASE, CHALLENGE_QUERIES

passed = failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"    OK {name}")
    else:
        failed += 1
        print(f"    FAIL {name}")


def test_mock_subtensor_setup():
    """Test that MockSubtensor + Metagraph work with the patch."""
    print("\n1. MockSubtensor + Metagraph setup...")

    ms = bt.MockSubtensor(network='mock')
    ms.create_subnet(netuid=1)

    # Create validator wallet
    wallet = bt.Wallet(name='sb_mock_test', hotkey='sb_mock_test')
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)

    uid = ms.force_register_neuron(
        netuid=1,
        hotkey_ss58=wallet.hotkey.ss58_address,
        coldkey_ss58=wallet.coldkey.ss58_address,
        balance=100000,
        stake=100000,
    )
    check(uid == 0, f"Validator registered at uid={uid}")

    # Register 4 mock miners
    for i in range(1, 5):
        mw = bt.Wallet(name=f'sb_mock_miner_{i}', hotkey=f'sb_mock_miner_{i}')
        mw.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
        muid = ms.force_register_neuron(
            netuid=1,
            hotkey_ss58=mw.hotkey.ss58_address,
            coldkey_ss58=mw.coldkey.ss58_address,
            balance=100000,
            stake=100000,
        )
        check(muid == i, f"Miner {i} registered at uid={muid}")

    # Sync metagraph
    mg = bt.Metagraph(netuid=1, network='mock', sync=False)
    mg.sync(subtensor=ms)
    check(int(mg.n) == 5, f"Metagraph has {mg.n} neurons")
    check(len(mg.hotkeys) == 5, f"Has {len(mg.hotkeys)} hotkeys")

    return ms, wallet, mg


def test_mock_dendrite_forward():
    """Test MockDendrite generates proper RAGSynapse responses."""
    print("\n2. MockDendrite forward...")

    from superbrain.mock import MockDendrite

    wallet = bt.Wallet(name='sb_mock_test', hotkey='sb_mock_test')
    dendrite = MockDendrite(wallet=wallet)
    check(dendrite is not None, "MockDendrite created")

    kb = KNOWLEDGE_BASE[0]
    synapse = RAGSynapse(
        query=kb["query"],
        context_chunks=kb["chunks"],
        chunk_sources=kb["sources"],
    )

    # Create mock axons
    mock_axons = []
    for i in range(4):
        axon = bt.AxonInfo(
            version=1,
            ip="127.0.0.1",
            port=8091 + i,
            ip_type=4,
            hotkey=f"mock-hotkey-{i}",
            coldkey="mock-coldkey",
        )
        mock_axons.append(axon)

    responses = asyncio.get_event_loop().run_until_complete(
        dendrite.forward(
            axons=mock_axons,
            synapse=synapse,
            timeout=10,
            deserialize=True,
        )
    )

    check(len(responses) == 4, f"Got {len(responses)} responses")

    good_responses = 0
    for i, resp in enumerate(responses):
        if isinstance(resp, dict) and resp.get("response"):
            good_responses += 1
            print(f"      Miner {i}: {resp['response'][:60]}... citations={resp.get('citations', [])}")
    check(good_responses > 0, f"{good_responses}/4 miners returned valid responses")

    return responses


def test_full_scoring_pipeline():
    """Test the complete scoring pipeline with mock responses."""
    print("\n3. Full scoring pipeline...")

    kb = KNOWLEDGE_BASE[0]
    query = kb["query"]
    chunks = kb["chunks"]

    # Simulate varied miner responses
    responses = [
        {
            "response": f"Based on the sources [1], {chunks[0]}",
            "citations": [0],
            "confidence_score": 0.9,
        },
        {
            "response": f"Bittensor is a decentralized network [1] using subnets [2] with TAO tokens [3].",
            "citations": [0, 1, 2],
            "confidence_score": 0.85,
        },
        {
            "response": "I don't know the answer.",
            "citations": [],
            "confidence_score": 0.1,
        },
        {
            "response": "",
            "citations": [],
            "confidence_score": 0.0,
        },
    ]
    response_times = [2.0, 3.0, 5.0, 30.0]

    rewards = get_rewards(
        self=None,
        query=query,
        context_chunks=chunks,
        responses=responses,
        response_times=response_times,
    )

    print(f"    Rewards: {rewards}")
    check(len(rewards) == 4, f"Got {len(rewards)} reward scores")
    check(rewards[0] > 0, f"Good response scored: {rewards[0]:.4f}")
    check(rewards[1] > 0, f"Multi-cite response scored: {rewards[1]:.4f}")
    check(rewards[2] < rewards[0], f"Bad < Good: {rewards[2]:.4f} < {rewards[0]:.4f}")
    check(rewards[3] == 0.0, f"Empty response: {rewards[3]:.4f}")

    # Verbatim single-cite can score higher on supportedness than a short multi-cite summary
    check(rewards[0] >= rewards[1], f"Verbatim single-cite >= short multi-cite")
    check(rewards[0] > rewards[2], f"Good > bad response")

    return rewards


def test_score_update_ema():
    """Test EMA score update mechanism."""
    print("\n4. EMA score updates...")

    n = 8
    scores = np.zeros(n, dtype=np.float32)
    alpha = 0.1

    # Simulate several rounds of scoring
    for round_num in range(5):
        uids = np.array([1, 2, 3, 4])
        rewards = np.array([0.8, 0.6, 0.2, 0.0])

        scattered = np.zeros_like(scores)
        scattered[uids] = rewards
        scores = alpha * scattered + (1 - alpha) * scores

    print(f"    After 5 rounds: {scores}")
    check(scores[1] > scores[2], f"UID 1 ({scores[1]:.4f}) > UID 2 ({scores[2]:.4f})")
    check(scores[2] > scores[3], f"UID 2 ({scores[2]:.4f}) > UID 3 ({scores[3]:.4f})")
    check(scores[3] > scores[4], f"UID 3 ({scores[3]:.4f}) > UID 4 ({scores[4]:.4f})")
    check(scores[0] == 0.0, f"Unqueried UID 0 stays at 0")
    check(scores[5] == 0.0, f"Unqueried UID 5 stays at 0")


def test_challenge_query():
    """Test challenge query scoring with keyword checks."""
    print("\n5. Challenge query anti-gaming...")

    challenge = CHALLENGE_QUERIES[0]
    kb = KNOWLEDGE_BASE[challenge["kb_index"]]
    expected_kw = challenge["expected_keywords"]

    good = "TAO is the native cryptocurrency token of Bittensor with a cap of 21 million."
    bad = "The weather is nice today."

    good_hits = sum(1 for kw in expected_kw if kw in good.lower())
    bad_hits = sum(1 for kw in expected_kw if kw in bad.lower())

    print(f"    Good response keywords: {good_hits}/{len(expected_kw)}")
    print(f"    Bad response keywords: {bad_hits}/{len(expected_kw)}")

    check(good_hits >= len(expected_kw) * 0.25, f"Good passes challenge ({good_hits} hits)")
    check(bad_hits < len(expected_kw) * 0.25, f"Bad fails challenge ({bad_hits} hits)")


def test_knowledge_base():
    """Verify knowledge base structure."""
    print("\n6. Knowledge base integrity...")

    check(len(KNOWLEDGE_BASE) == 5, f"Knowledge base has {len(KNOWLEDGE_BASE)} entries")
    check(len(CHALLENGE_QUERIES) == 2, f"Challenge queries: {len(CHALLENGE_QUERIES)}")

    for i, kb in enumerate(KNOWLEDGE_BASE):
        has_fields = all(k in kb for k in ["query", "chunks", "sources"])
        check(has_fields, f"KB[{i}] has all required fields")
        check(len(kb["chunks"]) == len(kb["sources"]), f"KB[{i}] chunks/sources match")


def test_weight_normalization():
    """Test weight normalization for on-chain setting."""
    print("\n7. Weight normalization...")

    scores = np.array([0.8, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)
    if np.any(norm == 0) or np.isnan(norm).any():
        norm = np.ones_like(norm)
    raw_weights = scores / norm

    print(f"    Raw weights: {raw_weights}")
    check(abs(raw_weights.sum() - 1.0) < 0.001, f"Weights sum to {raw_weights.sum():.4f}")
    check(raw_weights[0] > raw_weights[1], "Highest scorer gets most weight")
    check(raw_weights[3] == 0.0, "Zero scorer gets zero weight")


def test_end_to_end_mock():
    """Full end-to-end mock: MockDendrite → score → EMA update."""
    print("\n8. End-to-end mock pipeline...")

    from superbrain.mock import MockDendrite

    wallet = bt.Wallet(name='sb_mock_test', hotkey='sb_mock_test')
    dendrite = MockDendrite(wallet=wallet)

    n_miners = 8
    scores = np.zeros(n_miners, dtype=np.float32)
    alpha = 0.1

    for step in range(3):
        kb = KNOWLEDGE_BASE[step % len(KNOWLEDGE_BASE)]
        synapse = RAGSynapse(
            query=kb["query"],
            context_chunks=kb["chunks"],
            chunk_sources=kb["sources"],
        )

        mock_axons = [
            bt.AxonInfo(version=1, ip="127.0.0.1", port=8091 + i, ip_type=4,
                        hotkey=f"mock-{i}", coldkey="mock-cold")
            for i in range(4)
        ]

        uids = np.array([1, 3, 5, 7])

        responses = asyncio.get_event_loop().run_until_complete(
            dendrite.forward(axons=mock_axons, synapse=synapse, timeout=10, deserialize=True)
        )

        response_times = [random.uniform(1.0, 5.0) for _ in responses]

        normalized = []
        for resp in responses:
            if isinstance(resp, dict):
                normalized.append(resp)
            else:
                normalized.append({"response": "", "citations": [], "confidence_score": 0.0})

        rewards = get_rewards(None, kb["query"], kb["chunks"], normalized, response_times)

        scattered = np.zeros_like(scores)
        scattered[uids] = rewards
        scores = alpha * scattered + (1 - alpha) * scores

        print(f"    Step {step}: rewards={rewards}, scores={scores}")

    check(np.any(scores > 0), f"Some scores > 0 after 3 rounds")
    check(scores[0] == 0.0, "Unqueried UID 0 stays at 0")
    nonzero = np.count_nonzero(scores)
    check(nonzero > 0, f"{nonzero} UIDs have nonzero scores")
    print(f"    Final scores: {scores}")


if __name__ == "__main__":
    print("=" * 60)
    print("  SuperBrain Mock Network Test")
    print("=" * 60)

    test_mock_subtensor_setup()
    test_mock_dendrite_forward()
    test_full_scoring_pipeline()
    test_score_update_ema()
    test_challenge_query()
    test_knowledge_base()
    test_weight_normalization()
    test_end_to_end_mock()

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED!")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 60)
    sys.exit(1 if failed else 0)
