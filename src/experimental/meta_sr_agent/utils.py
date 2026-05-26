# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Scoring and reporting helpers for the experimental meta SR agent."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from experimental.meta_sr_agent.schema import Claim, SearchRecord
from sr_agent.utils import NamedTimer


def normalize_claim_list(raw_claims: list[dict[str, Any]], max_length: int) -> list[Claim]:
    if not isinstance(raw_claims, list):
        raise TypeError(f"claim list must be a list, got {type(raw_claims).__name__}")
    claims = [Claim.from_json(item, idx + 1) for idx, item in enumerate(raw_claims)]
    return [claim for claim in claims if claim.claim][:max_length]


def record_to_json(record: SearchRecord) -> dict[str, Any]:
    return {
        "claim_list_score": record.claim_list_score,
        "claim_list": [asdict(c) for c in record.claim_list],
        "generated_equations": record.generated_equations,
        "truth_evaluation": record.truth_evaluation,
        "discoverability_evaluation": record.discoverability_evaluation,
        "effectiveness_evaluation": record.effectiveness_evaluation,
        "meta": record.meta,
    }


def top_records(archive: list[SearchRecord], k: int) -> list[SearchRecord]:
    return sorted(archive, key=lambda r: r.claim_list_score, reverse=True)[:k]


def best_record(archive: list[SearchRecord]) -> SearchRecord:
    return max(archive, key=lambda r: r.claim_list_score)


def dedupe_records(records: list[SearchRecord]) -> list[SearchRecord]:
    deduped: list[SearchRecord] = []
    seen: set[int] = set()
    for record in records:
        if id(record) not in seen:
            deduped.append(record)
            seen.add(id(record))
    return deduped


def summarize_dataset(X: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> dict[str, Any]:
    summary = {
        "num_samples": int(len(next(iter(y.values())))),
        "features": {},
        "targets": {},
    }
    for group_name, values in (("features", X), ("targets", y)):
        for name, arr in values.items():
            arr = np.asarray(arr)
            summary[group_name][name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return summary


def normalize_unit_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(score):
        return math.nan
    if 1.0 < score <= 10.0:
        score = score / 10.0
    return max(0.0, min(1.0, score))


def failed_claim_evaluation(claim_list: list[Claim], message: str) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim.id,
            "score": math.nan,
            "detail": message,
        }
        for claim in claim_list
    ]


def score_for_ranking(score: Any) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(value) else value


def safe_max_score(scores) -> float:
    values = [score_for_ranking(score) for score in scores]
    return max(values, default=0.0)


def safe_min_score(scores) -> float:
    values = [normalize_unit_score(score) for score in scores]
    values = [score for score in values if not math.isnan(score)]
    return min(values, default=math.nan)


def named_timer_summary(timer: NamedTimer) -> dict[str, dict[str, float]]:
    summary = {}
    for name in timer.names:
        stage, _, bucket = name.partition(".")
        stats = summary.setdefault(
            stage,
            {"count": 0, "total_seconds": 0.0},
        )
        count = timer.get_named_count(name)
        seconds = timer.get_named_time(name)
        stats["count"] += count
        stats["total_seconds"] += seconds
    return summary


def serialize_tool_calls(tool_calls) -> list[dict[str, Any]]:
    results = []
    for call in tool_calls:
        results.append({
            "name": call.name,
            "params": call.params,
            "raw_str": call.raw_str,
            "raw": call.raw,
        })
    return results

def serialize_dataclass(obj: list[Any], keys: list[str]) -> list[dict[str, Any]]:
    return [{key: getattr(item, key, None) for key in keys} for item in obj]


def format_tool_calls(tool_calls: list[Any], max_field_length: int = 240) -> str:
    def truncate(value: Any) -> str:
        text = repr(value)
        if len(text) <= max_field_length:
            return text
        return text[:max_field_length].rstrip() + "..."

    if not tool_calls:
        return ""
    lines = []
    for idx, call in enumerate(tool_calls, 1):
        if isinstance(call.params, dict):
            params = "{" + ", ".join(
                f"{key!r}: {truncate(value)}"
                for key, value in call.params.items()
            ) + "}"
        else:
            params = truncate(call.params)
        lines.append(f"  [gold]#{idx}[reset] [cyan]{call.name}[reset] [red]params[reset]={params}")
    return "\n".join(lines)


def format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def colored_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return f"[gray]{value}[reset]"
    if score >= 0.85:
        color = "brightgreen"
    elif score >= 0.70:
        color = "green"
    elif score >= 0.45:
        color = "gold"
    elif score >= 0.25:
        color = "orange"
    else:
        color = "brightred"
    return f"[{color} bold]{score:.0%}[reset]"


def colored_metadata_line(key: str, value: Any) -> str:
    return f"  [red]{key}[reset]: {value}"


def colored_usage_dict(usage: dict[str, Any] | None) -> str:
    usage = usage or {}
    parts = []
    for bucket in ("token", "price"):
        values = usage.get(bucket) or {}
        if values:
            rendered = ", ".join(
                f"[blue]{name}[reset]=[turquoise]{format_number(value)}[reset]"
                for name, value in values.items()
            )
            parts.append(f"[violet]{bucket}[reset]: {rendered}")
    return "; ".join(parts) if parts else "[gray]N/A[reset]"


def colored_claim_line(claim: dict[str, Any]) -> str:
    return (
        f"  - [cyan bold]{claim.get('id', '?')}[reset] "
        f"[violet]{claim.get('tool_name', '?')}[reset]: "
        f"[brightwhite]{claim.get('claim', '')}[reset]"
    )


def colored_claim_eval_line(claim: dict[str, Any], item: dict[str, Any] | str | None) -> str:
    claim_id = f"[cyan]{claim.get('id', '?')}[reset]"
    if isinstance(item, dict):
        return (
            f"{claim_id}: score={colored_score(item.get('score'))} "
            f"[gray]|[reset] [lightgray]{item.get('detail', '')}[reset]"
        )
    if item:
        return f"{claim_id}: [lightgray]{item}[reset]"
    return f"{claim_id}: [gray]No detail.[reset]"


def extend_colored_evaluation_sections(
    claims: list[dict[str, Any]],
    generated: dict[str, Any],
    truth_evaluation: dict[str, Any],
    discoverability_evaluation: dict[str, Any],
    effectiveness_evaluation: dict[str, Any],
) -> None:
    truth_detail = truth_evaluation["detail"] if truth_evaluation else []
    discoverability_detail = discoverability_evaluation["detail"] if discoverability_evaluation else []
    effectiveness_detail = effectiveness_evaluation["detail"] if effectiveness_evaluation else []
    truth_by_id = {
        item.get("claim_id"): item
        for item in truth_detail
        if isinstance(item, dict)
    }
    discoverability_by_id = {
        item.get("claim_id"): item
        for item in discoverability_detail
        if isinstance(item, dict)
    }

    lines = []
    lines.append("[red bold]Best Claim List[reset]")
    for claim in claims:
        lines.append(
            f"  - [cyan bold]{claim.get('id', '?')}[reset] "
            f"[violet]{claim.get('tool_name', '?')}[reset]: "
            f"[brightwhite]{claim.get('claim', '')}[reset]"
        )
    if not claims:
        lines.append("  - [gray]None[reset]")

    lines.append("[red bold]Generated Equations[reset]")
    if generated.get("detail"):
        lines.append(f"  [red]Detail[reset]: [lightgray]{generated.get('detail')}[reset]")
    for equation in generated.get("equation_list") or []:
        lines.append(f"  - [turquoise]{equation}[reset]")
    if not generated.get("equation_list"):
        lines.append("  - [gray]None[reset]")

    lines.append("[red bold]Truth Evaluator[reset]")
    for idx, claim in enumerate(claims):
        if (claim_id := claim.get("id")) in truth_by_id:
            item = truth_by_id[claim_id]
        elif idx < len(truth_detail):
            item = truth_detail[idx]
        else:
            item = 'No truth detail.'
        lines.append(f"  - {colored_claim_eval_line(claim, item)}")
    if not claims:
        lines.append("  - [gray]None[reset]")

    lines.append("[red bold]Discoverability Evaluator[reset]")
    for idx, claim in enumerate(claims):
        if (claim_id := claim.get("id")) in discoverability_by_id:
            item = discoverability_by_id[claim_id]
        elif idx < len(discoverability_detail):
            item = discoverability_detail[idx]
        else:
            item = 'No discoverability detail.'
        lines.append(f"  - {colored_claim_eval_line(claim, item)}")

    lines.append("[red bold]Effectiveness Evaluator[reset]")
    for item in effectiveness_detail:
        if isinstance(item, dict):
            lines.append(
                f"  - [turquoise]{item.get('equation', '')}[reset]: "
                f"score={colored_score(item.get('score'))} "
                f"[gray]|[reset] [lightgray]{item.get('detail', '')}[reset]"
            )
        else:
            lines.append(f"  - [lightgray]{item}[reset]")
    if not effectiveness_detail:
        lines.append("  - [gray]None[reset]")

    return lines
