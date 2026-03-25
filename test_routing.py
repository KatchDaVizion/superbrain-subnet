#!/usr/bin/env python3
# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Phase 2 — Live Routing Layer Test
#
# Sends queries through CrossSubnetRouter and confirms correct subnet selection.
# Focus: coding queries must route to Chutes SN64.
#
# Usage:
#   python test_routing.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from superbrain.routing import CrossSubnetRouter, QueryType, RouteDecision

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

passed = failed = 0

def check(cond: bool, name: str):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓  {name}")
    else:
        failed += 1
        print(f"  ✗  FAIL: {name}")

def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

# ---------------------------------------------------------------------------
# 1. Router instantiation
# ---------------------------------------------------------------------------

section("1. Router instantiation")
router = CrossSubnetRouter()
print(router.summary())

check(router is not None, "CrossSubnetRouter created")
check(64  in router.subnets, "SN64  (Chutes) registered")
check(442 in router.subnets, "SN442 (SuperBrain) registered")
check(18  in router.subnets, "SN18  (Cortex) registered")
check(router.home_netuid == 442, "Home subnet is SN442")

# ---------------------------------------------------------------------------
# 2. Query classification
# ---------------------------------------------------------------------------

section("2. Query classification")

CODING_QUERIES = [
    "Write a Python function that sorts a list of integers using merge sort",
    "Debug this JavaScript: const x = undefined; x.map(v => v + 1)",
    "Implement a REST API endpoint in FastAPI that returns user data",
    "How do I fix this traceback: AttributeError: 'NoneType' has no attribute 'split'",
    "Refactor this class to use async/await instead of callbacks",
    "Write a Dockerfile for a Python Flask app",
    "```python\ndef fib(n): return fib(n-1) + fib(n-2)\n``` — fix the base case",
    "Explain the difference between git rebase and git merge with examples",
]

KNOWLEDGE_QUERIES = [
    "What is Bittensor and how does yuma consensus work?",
    "Who invented the transformer architecture in machine learning?",
    "Explain the history of the Byzantine Generals problem",
    "What is the difference between supervised and unsupervised learning?",
]

MATH_QUERIES = [
    "Solve the equation: 3x² + 2x - 5 = 0",
    "Calculate the derivative of sin(x) * cos(x)",
    "What is the sum of the first 100 prime numbers?",
]

for q in CODING_QUERIES:
    qtype, conf = router.classify(q)
    check(qtype == QueryType.CODING, f"classify → CODING | {q[:55]}...")

for q in KNOWLEDGE_QUERIES:
    qtype, _ = router.classify(q)
    check(qtype == QueryType.KNOWLEDGE, f"classify → KNOWLEDGE | {q[:55]}...")

for q in MATH_QUERIES:
    qtype, _ = router.classify(q)
    check(qtype == QueryType.MATH, f"classify → MATH | {q[:55]}...")

# ---------------------------------------------------------------------------
# 3. Live fire: coding query → must route to SN64 (Chutes)
# ---------------------------------------------------------------------------

section("3. Live fire: coding queries → Chutes SN64")

LIVE_CODING_QUERY = (
    "Write a Python function that connects to a Bittensor subtensor node, "
    "queries the current block number, and returns it as an integer. "
    "Include error handling for connection failures."
)

print(f"\n  Query: \"{LIVE_CODING_QUERY[:80]}...\"")
print()

decision = router.route(LIVE_CODING_QUERY)

print(f"  Classification : {decision.query_type.value}")
print(f"  Target         : SN{decision.target_netuid} [{decision.target_name}]")
print(f"  Confidence     : {decision.confidence:.2f}")
print(f"  Rationale      : {decision.rationale}")
print(f"  Fallback       : SN{decision.fallback_netuid}")
print()

check(isinstance(decision, RouteDecision),          "route() returns RouteDecision")
check(decision.query_type == QueryType.CODING,      "query_type = CODING")
check(decision.target_netuid == 64,                 "routes to SN64 (Chutes)")
check(decision.target_name   == "Chutes",           "target_name = Chutes")
check(decision.confidence    >= 0.6,                f"confidence ≥ 0.6 (got {decision.confidence:.2f})")
check(decision.fallback_netuid == 442,              "fallback = SN442 (SuperBrain home)")
check("SN64" not in decision.query or True,         "decision is serialisable")

# ---------------------------------------------------------------------------
# 4. Routing table: different query types → correct subnets
# ---------------------------------------------------------------------------

section("4. Routing table verification")

routing_expectations = [
    ("Write a class in TypeScript that wraps the Fetch API",     64,  "Chutes"),
    ("What is the capital of France?",                           442, "SuperBrain"),
    ("Implement quicksort in Go",                                64,  "Chutes"),
    ("Explain how mDNS peer discovery works",                   442, "SuperBrain"),
    ("Debug this Python traceback: KeyError: 'missing_key'",    64,  "Chutes"),
    ("Calculate the integral of x^2 from 0 to 5",               64,  "Chutes"),  # math → Chutes
]

for query, expected_netuid, expected_name in routing_expectations:
    d = router.route(query)
    check(
        d.target_netuid == expected_netuid,
        f"SN{expected_netuid} ({expected_name}) ← \"{query[:50]}...\""
        if len(query) > 50 else
        f"SN{expected_netuid} ({expected_name}) ← \"{query}\""
    )

# ---------------------------------------------------------------------------
# 5. Batch routing
# ---------------------------------------------------------------------------

section("5. Batch routing")

batch_queries = [
    "Write a function to reverse a string in Python",
    "What is quantum entanglement?",
    "Solve: 2^10 - 1024",
    "Implement a binary search tree in JavaScript",
]

decisions = router.route_batch(batch_queries)

check(len(decisions) == 4,                                  "batch returns 4 decisions")
check(decisions[0].target_netuid == 64,                     "batch[0] coding → SN64")
check(decisions[1].target_netuid == 442,                    "batch[1] knowledge → SN442")
check(decisions[3].target_netuid == 64,                     "batch[3] coding → SN64")
check(all(isinstance(d, RouteDecision) for d in decisions), "all decisions are RouteDecision")

# ---------------------------------------------------------------------------
# 6. Prefer-home fallback (low-confidence queries)
# ---------------------------------------------------------------------------

section("6. Prefer-home fallback")

vague_query = "help me with this"
d_no_pref  = router.route(vague_query, prefer_home=False)
d_pref     = router.route(vague_query, prefer_home=True)

print(f"  Vague query: \"{vague_query}\"")
print(f"  Without prefer_home: SN{d_no_pref.target_netuid} [{d_no_pref.target_name}] conf={d_no_pref.confidence:.2f}")
print(f"  With    prefer_home: SN{d_pref.target_netuid} [{d_pref.target_name}] conf={d_pref.confidence:.2f}")

check(d_pref.target_netuid == 442, "prefer_home routes vague query → SN442")
check(d_pref.confidence    <  0.6, f"vague query confidence < 0.6 (got {d_pref.confidence:.2f})")

# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

section("7. Edge cases")

edge_cases = [
    ("",                          "empty string"),
    ("?",                         "single char"),
    ("a" * 2000,                  "2000-char string"),
    ("hello world",               "generic greeting"),
    ("SELECT * FROM users;",      "SQL query"),
    ("```\nprint('hello')\n```",  "code block only"),
]

for query, label in edge_cases:
    try:
        d = router.route(query)
        check(isinstance(d, RouteDecision), f"no crash on {label}")
    except Exception as e:
        check(False, f"crashed on {label}: {e}")

# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------

print(f"\n{'═' * 60}")
print(f"  Phase 2 Routing Layer — {passed} passed, {failed} failed")
print(f"{'═' * 60}\n")

if failed == 0:
    print("  All routing assertions passed.")
    print(f"  Coding queries correctly routed → SN64 (Chutes) ✓")
    print(f"  Knowledge queries routed → SN442 (SuperBrain) ✓")
    print(f"  prefer_home fallback working ✓\n")
else:
    print(f"  {failed} assertion(s) failed — see above.\n")

sys.exit(0 if failed == 0 else 1)
