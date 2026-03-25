# Copyright 2026 Lys-David Louis-Charles (KatchDaVizion)
# SuperBrain Phase 2 — Cross-Subnet Router
#
# Routes queries to specialist Bittensor subnets based on query type.
# Decision hierarchy:
#   1. Detect query category (coding, image, text, knowledge, general)
#   2. Select the best subnet for that category
#   3. Return a RouteDecision with subnet, confidence, and rationale
#
# Registered specialist subnets:
#   SN18  — Cortex / general AI inference
#   SN64  — Chutes / code execution & coding assistance  ← default for coding
#   SN442 — SuperBrain / local knowledge retrieval       ← home subnet

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class QueryType(str, Enum):
    CODING      = "coding"
    KNOWLEDGE   = "knowledge"
    IMAGE       = "image"
    MATH        = "math"
    GENERAL     = "general"


@dataclass
class SubnetSpec:
    netuid: int
    name: str
    description: str
    handles: List[QueryType]
    endpoint: str = ""          # optional override; empty = use metagraph


@dataclass
class RouteDecision:
    query: str
    query_type: QueryType
    target_netuid: int
    target_name: str
    confidence: float           # 0.0 – 1.0
    rationale: str
    fallback_netuid: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"RouteDecision(type={self.query_type.value}, "
            f"→ SN{self.target_netuid} [{self.target_name}], "
            f"conf={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Keyword patterns per query type
# ---------------------------------------------------------------------------

_CODING_PATTERNS: List[str] = [
    r"\b(def |class |import |from .+ import|async def|await )\b",
    r"\b(function|const |let |var |return |if \(|for \(|while \()\b",
    r"\b(bug|error|exception|traceback|stacktrace|segfault)\b",
    r"\b(code|script|program|algorithm|implement|refactor|debug|compile|lint)\b",
    r"\b(python|javascript|typescript|golang|rust|java|c\+\+|bash|sql)\b",
    r"\b(git|docker|kubernetes|api|rest|graphql|endpoint|deploy)\b",
    r"```",                     # fenced code block signal
    r"\bwrite (?:a |the )?(?:function|method|class|test|script|code)\b",
]

_MATH_PATTERNS: List[str] = [
    r"\b(solve|equation|integral|derivative|matrix|vector|proof|theorem)\b",
    r"\b(calculate|compute|formula|trigonometry|calculus|algebra|geometry)\b",
    r"[\d]+\s*[\+\-\*\/\^]\s*[\d]+",   # arithmetic expressions
    r"\b(sum|product|limit|factorial|fibonacci|prime)\b",
]

_IMAGE_PATTERNS: List[str] = [
    r"\b(image|photo|picture|diagram|chart|graph|visual|screenshot|render)\b",
    r"\b(draw|generate|create|describe|analyse|analyze)\b.{0,20}\b(image|photo|picture)\b",
    r"\b(stable diffusion|dalle|midjourney|flux|vision|ocr)\b",
]

_KNOWLEDGE_PATTERNS: List[str] = [
    r"\b(what is|who is|who invented|who created|who discovered|when did|where is|how does|explain|define)\b",
    r"\b(history|science|biology|physics|chemistry|geography|literature)\b",
    r"\b(fact|information|knowledge|research|paper|study|source)\b",
]


def _match_score(text: str, patterns: List[str]) -> int:
    """Count how many patterns match in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


class CrossSubnetRouter:
    """
    Routes a natural-language query to the most appropriate Bittensor subnet.

    Usage:
        router = CrossSubnetRouter()
        decision = router.route("Write a Python function to sort a list")
        print(decision)  # → SN64 [Chutes], conf=0.92
    """

    # Default subnet registry
    _DEFAULT_SUBNETS: List[SubnetSpec] = [
        SubnetSpec(
            netuid=64,
            name="Chutes",
            description="Code execution and coding assistance (SN64)",
            handles=[QueryType.CODING, QueryType.MATH],
        ),
        SubnetSpec(
            netuid=442,
            name="SuperBrain",
            description="Local-first anonymous knowledge retrieval (SN442)",
            handles=[QueryType.KNOWLEDGE, QueryType.GENERAL],
        ),
        SubnetSpec(
            netuid=18,
            name="Cortex",
            description="General AI inference fallback (SN18)",
            handles=[QueryType.GENERAL, QueryType.IMAGE],
        ),
    ]

    def __init__(
        self,
        subnets: Optional[List[SubnetSpec]] = None,
        home_netuid: int = 442,
        coding_threshold: int = 1,
    ):
        self.subnets: Dict[int, SubnetSpec] = {}
        for s in (subnets or self._DEFAULT_SUBNETS):
            self.subnets[s.netuid] = s
        self.home_netuid = home_netuid
        self.coding_threshold = coding_threshold

        # Build reverse index: QueryType → list of (priority, SubnetSpec)
        self._type_index: Dict[QueryType, List[SubnetSpec]] = {}
        for spec in self.subnets.values():
            for qt in spec.handles:
                self._type_index.setdefault(qt, []).append(spec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query into a QueryType and return (type, confidence).
        Confidence is the normalised fraction of matching patterns.
        """
        scores: Dict[QueryType, int] = {
            QueryType.CODING:    _match_score(query, _CODING_PATTERNS),
            QueryType.MATH:      _match_score(query, _MATH_PATTERNS),
            QueryType.IMAGE:     _match_score(query, _IMAGE_PATTERNS),
            QueryType.KNOWLEDGE: _match_score(query, _KNOWLEDGE_PATTERNS),
        }

        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]

        if best_score == 0:
            return QueryType.GENERAL, 0.5

        # Normalise: max possible matches per type
        max_possible = {
            QueryType.CODING:    len(_CODING_PATTERNS),
            QueryType.MATH:      len(_MATH_PATTERNS),
            QueryType.IMAGE:     len(_IMAGE_PATTERNS),
            QueryType.KNOWLEDGE: len(_KNOWLEDGE_PATTERNS),
        }
        confidence = min(0.99, best_score / max_possible[best_type] + 0.4)
        return best_type, round(confidence, 2)

    def route(self, query: str, prefer_home: bool = False) -> RouteDecision:
        """
        Classify the query and return a RouteDecision for the best subnet.
        Set prefer_home=True to prefer SN442 when confidence is low.
        """
        query_type, confidence = self.classify(query)

        # Find candidate subnets for this type
        candidates = self._type_index.get(query_type, [])
        if not candidates:
            candidates = self._type_index.get(QueryType.GENERAL, [])

        # If prefer_home and low confidence, force home subnet
        if prefer_home and confidence < 0.6:
            target = self.subnets[self.home_netuid]
            rationale = (
                f"Low confidence ({confidence:.2f}) — defaulting to home subnet SN{self.home_netuid}."
            )
        else:
            # Pick first candidate (highest priority = first registered)
            target = candidates[0]
            rationale = (
                f"Query classified as '{query_type.value}' "
                f"(confidence={confidence:.2f}). "
                f"{target.description}."
            )

        # Fallback: if we're routing away from home, home is the fallback
        fallback = (
            self.home_netuid
            if target.netuid != self.home_netuid
            else None
        )

        return RouteDecision(
            query=query,
            query_type=query_type,
            target_netuid=target.netuid,
            target_name=target.name,
            confidence=confidence,
            rationale=rationale,
            fallback_netuid=fallback,
            metadata={"home_netuid": self.home_netuid},
        )

    def route_batch(self, queries: List[str]) -> List[RouteDecision]:
        """Route multiple queries and return a list of RouteDecisions."""
        return [self.route(q) for q in queries]

    def summary(self) -> str:
        lines = ["CrossSubnetRouter — registered subnets:"]
        for spec in self.subnets.values():
            types = ", ".join(t.value for t in spec.handles)
            lines.append(f"  SN{spec.netuid:>4}  {spec.name:<14} handles=[{types}]")
        return "\n".join(lines)
