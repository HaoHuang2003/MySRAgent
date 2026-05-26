# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Data contracts for the experimental meta SR agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CLAIM_LISTS_TEMPLATE = {
    "claim_lists": [[{"tool_name": str, "claim": str}]]
}
GENERATED_EQUATIONS_TEMPLATE = {
    "equation_list": [str],
    "detail": str,
}
CLAIM_EVALUATION_TEMPLATE = [{"claim_id": str, "score": float, "detail": str}]
EFFECTIVENESS_TEMPLATE = {
    "candidate_scores": [{"equation": str, "score": float, "detail": str}],
}


@dataclass
class Claim:
    id: str
    tool_name: str
    claim: str

    @classmethod
    def from_json(cls, data: dict[str, Any], idx: int) -> "Claim":
        return cls(
            id=f"c{idx}",
            tool_name=str(data["tool_name"]),
            claim=str(data["claim"]).strip(),
        )


@dataclass
class SearchRecord:
    claim_list_score: float
    claim_list: list[Claim]
    generated_equations: dict[str, Any]
    truth_evaluation: dict[str, Any]
    discoverability_evaluation: dict[str, Any]
    effectiveness_evaluation: dict[str, Any]
    meta: dict[str, Any]
