"""SuperBrain Test Suite — Scoring + Protocol Tests"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superbrain.validator.reward import (
    score_supportedness, score_relevance, score_novelty, score_latency,
    reward, get_rewards, W_SUPPORTEDNESS, W_RELEVANCE, W_NOVELTY, W_LATENCY,
)

QUERY = "What is Bittensor and how does it work?"
CHUNKS = [
    "Bittensor is a decentralized machine learning network that creates an open marketplace for AI models. It uses a blockchain-based incentive mechanism to reward miners who contribute useful intelligence.",
    "The Bittensor network operates through subnets, each focused on a specific task like text generation, image recognition, or data storage. Validators evaluate miner outputs and set weights.",
    "TAO is the native cryptocurrency of the Bittensor network. It is used to incentivize miners, stake on validators, and govern subnet creation.",
]
passed = failed = 0

def check(cond, name):
    global passed, failed
    if cond: passed += 1; print(f"    OK {name}")
    else: failed += 1; print(f"    FAIL {name}")

def test_protocol():
    print("\nProtocol...")
    from superbrain.protocol import RAGSynapse
    s = RAGSynapse(query="Test", context_chunks=["c1","c2"], chunk_sources=["s1","s2"])
    check(s.query == "Test", "query field")
    check(s.response is None, "response None")
    s.response = "Answer [1]"; s.citations = [0]; s.confidence_score = 0.9
    d = s.deserialize()
    check(d["response"] == "Answer [1]", "deserialize")

def test_weights():
    print("\nWeights...")
    total = W_SUPPORTEDNESS + W_RELEVANCE + W_NOVELTY + W_LATENCY
    check(abs(total - 1.0) < 0.001, f"Sum={total:.2f}")

def test_supportedness():
    print("\nSupportedness...")
    good = "Bittensor is a decentralized machine learning network [1] that uses subnets [2]. TAO is the native token [3]."
    s = score_supportedness(good, CHUNKS, [0, 1, 2])
    print(f"    Good: {s:.3f}"); check(s > 0.3, "Good > 0.3")
    check(score_supportedness(good, CHUNKS, []) == 0.0, "No cite=0")
    h = score_supportedness("Bitcoin was created by Satoshi Nakamoto in 2009.", CHUNKS, [0])
    print(f"    Halluc: {h:.3f}"); check(h < 0.3, "Halluc < 0.3")
    check(score_supportedness("", CHUNKS, [0]) == 0.0, "Empty=0")

def test_relevance():
    print("\nRelevance...")
    s = score_relevance(QUERY, CHUNKS, [0, 1])
    print(f"    Score: {s:.3f}"); check(s > 0.0, "Relevant > 0")
    check(score_relevance(QUERY, CHUNKS, []) == 0.0, "No cite=0")

def test_novelty():
    print("\nNovelty...")
    r = "Bittensor is a decentralized AI network that rewards miners with TAO tokens."
    check(score_novelty(r, []) == 1.0, "First=1.0")
    check(score_novelty(r, [r]) == 0.0, "Dup=0.0")
    d = score_novelty("The ocean contains many marine species.", [r])
    print(f"    Diff: {d:.3f}"); check(d > 0.5, "Diff > 0.5")

def test_latency():
    print("\nLatency...")
    check(score_latency(1.0) > 0.9, "1s fast")
    check(0.4 < score_latency(15.0) < 0.6, "15s mid")
    check(score_latency(31.0) == 0.0, "31s timeout")

def test_combined():
    print("\nCombined...")
    good = {"response": "Bittensor is a decentralized network [1] using subnets [2] with TAO [3].", "citations": [0,1,2], "process_time": 2.0}
    bad = {"response": "I don't know.", "citations": [0], "process_time": 5.0}
    empty = {"response": "", "citations": [], "process_time": 30.0}
    gs = reward(QUERY, CHUNKS, good, [])
    bs = reward(QUERY, CHUNKS, bad, [])
    es = reward(QUERY, CHUNKS, empty, [])
    print(f"    Good={gs:.4f} Bad={bs:.4f} Empty={es:.4f}")
    check(gs > bs, "Good > Bad"); check(bs > es, "Bad > Empty"); check(es == 0.0, "Empty=0")

def test_get_rewards():
    print("\nBatch rewards...")
    resps = [
        {"response": "Bittensor is decentralized [1] with subnets [2] and TAO [3].", "citations": [0,1,2]},
        {"response": "I don't know.", "citations": [0]},
        {},
    ]
    r = get_rewards(None, QUERY, CHUNKS, resps, [2.0, 5.0, 30.0])
    print(f"    {r}")
    check(len(r) == 3, "Length=3"); check(r[0] > r[1], "Best > worst"); check(r[2] == 0.0, "Empty=0")

def test_anti_gaming():
    print("\nAnti-gaming...")
    dup = "Bittensor is a decentralized network for AI with TAO tokens."
    resps = [{"response": dup, "citations": [0]}] * 3
    r = get_rewards(None, QUERY, CHUNKS, resps, [2.0]*3)
    print(f"    {r[0]:.4f} {r[1]:.4f} {r[2]:.4f}")
    check(r[0] >= r[1], "First >= dup"); check(r[1] == r[2], "Dups equal")

def test_edges():
    print("\nEdge cases...")
    check(score_supportedness("x", CHUNKS, [99]) == 0.0, "Bad idx=0")
    check(score_supportedness("x", [], [0]) == 0.0, "No chunks=0")
    check(score_relevance("Bittensor", CHUNKS, [0]) > 0.0, "1-word ok")

if __name__ == "__main__":
    print("=" * 50)
    print("  SuperBrain Local Tests")
    print("=" * 50)
    test_protocol(); test_weights(); test_supportedness(); test_relevance()
    test_novelty(); test_latency(); test_combined(); test_get_rewards()
    test_anti_gaming(); test_edges()
    print(f"\n{'=' * 50}")
    if failed == 0: print(f"  ALL {passed} TESTS PASSED!")
    else: print(f"  {passed} passed, {failed} FAILED")
    print("=" * 50)
    sys.exit(1 if failed else 0)
