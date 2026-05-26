# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Prompt helpers for the experimental meta SR agent."""

from __future__ import annotations


def add_evaluator_tool_instructions(
    messages: list[dict[str, str]],
    max_round: int,
) -> list[dict[str, str]]:
    messages = [message.copy() for message in messages]
    messages[0]["content"] = (
        f"{messages[0]['content']}\n\n"
        f"You may have at most {max_round} conversation rounds for this evaluation. "
        "Plan your tool use within this budget. You may inspect the data with tools "
        "when a tool is useful, but you must submit the final strict JSON response "
        f"no later than round {max_round}. Do not wait for an additional reminder "
        "after the final round."
    )
    return messages


def build_round_status_prompt(round_idx: int, max_round: int) -> dict[str, str]:
    if round_idx == max_round:
        content = (
            f"This is round {round_idx}/{max_round}, the final round. "
            "Do not call any more tools. You must return only the strict JSON object "
            "requested in the original instruction in this response."
        )
    else:
        content = (
            f"This is round {round_idx}/{max_round}. "
            f"You have {max_round - round_idx} round(s) remaining after this response. "
            "Use tools only if they are necessary, and be ready to return the final "
            f"strict JSON response no later than round {max_round}."
        )
    return {"role": "user", "content": content}


def build_final_json_retry_prompt() -> dict[str, str]:
    content = (
        "Your previous response in the final round did not provide parseable JSON. "
        "The evaluation budget is exhausted. Do not call any more tools. "
        "Use only the information already available in this conversation and return "
        "only the strict JSON object requested in the original instruction."
    )
    return {"role": "user", "content": content}


def build_parse_retry_round_prompt(round_idx: int, max_round: int) -> dict[str, str]:
    content = (
        "Your previous response did not provide parseable JSON. "
        f"You are still in round {round_idx}/{max_round}, with "
        f"{max_round - round_idx} round(s) remaining after this. "
        "If you need more evidence, call tools now; otherwise return only the "
        "strict JSON object requested in the original instruction."
    )
    return {"role": "user", "content": content}


def stage_title(stage: str) -> str:
    titles = {
        "context_constructor": "Context Constructor Agent",
        "formula_generator": "Formula Generator Agent",
        "truth_evaluator": "Truth Evaluator Agent",
        "discoverability_evaluator": "Discoverability Evaluator Agent",
        "effectiveness_evaluator": "Effectiveness Evaluator Agent",
    }
    return titles.get(stage, stage.replace("_", " ").title())
