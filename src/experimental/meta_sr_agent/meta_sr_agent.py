# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Experimental meta symbolic-regression agent.

MetaSRAgent searches for behavioral claims that can become useful context for a
symbolic-regression system.  The implementation mirrors ``SRAgent.fit``: the
main workflow is intentionally kept in one place, while small reusable pieces
such as JSON parsing, score normalization, timing, and mock heuristics stay as
helpers.
"""
from __future__ import annotations
import re
import time
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List
from sr_agent.parser import BaseParser
from sr_agent.api.llm_api import LLMAPI
from sr_agent.tools import BaseTool, ToolCallResult
from sr_agent.utils import NamedTimer, ParallelTimer, parse_json_with_template, tag2ansi, render_markdown, log_exception
from .mock_backend import (
    mock_claim_lists,
    mock_equations,
    mock_formula_score,
    mock_truth_detail,
    mock_truth_score,
    mock_discoverability_detail,
    mock_discoverability_score,
)
from .prompts import (
    add_evaluator_tool_instructions, 
    build_final_json_retry_prompt,
    build_parse_retry_round_prompt,
    build_round_status_prompt
)
from .schema import (
    CLAIM_EVALUATION_TEMPLATE,
    CLAIM_LISTS_TEMPLATE,
    EFFECTIVENESS_TEMPLATE,
    GENERATED_EQUATIONS_TEMPLATE,
    SearchRecord,
)
from .utils import (
    best_record,
    colored_metadata_line,
    colored_score,
    colored_usage_dict,
    dedupe_records,
    extend_colored_evaluation_sections,
    failed_claim_evaluation,
    format_number,
    format_tool_calls,
    named_timer_summary,
    normalize_claim_list,
    normalize_unit_score,
    record_to_json,
    safe_max_score,
    safe_min_score,
    score_for_ranking,
    serialize_dataclass,
    summarize_dataset,
    top_records,
)

_logger = logging.getLogger(f"sr_agent.{__name__}")


class MetaSRAgent:
    """Search for claim lists that improve symbolic-regression context."""

    def __init__(
        self,
        llm_provider: str = "mock",
        llm_model: str = "mock",
        local_sample_size: int = 2,
        global_width: int = 1,
        restart_top_k: int = 3,
        max_restart_loop: int = 1,
        max_refinement_depth: int = 3,
        max_claim_list_length: int = 6,
        num_equations_per_claim_list: int = 5,
        max_evaluation_depth: int = 3,
        max_json_retry: int = 3,
        length_penalty_weight: float = 0.02,
        save_path: str | None = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.local_sample_size = local_sample_size
        self.global_width = global_width
        self.restart_top_k = restart_top_k
        self.max_restart_loop = max_restart_loop
        self.max_refinement_depth = max_refinement_depth
        self.max_claim_list_length = max_claim_list_length
        self.num_equations_per_claim_list = num_equations_per_claim_list
        self.max_evaluation_depth = max_evaluation_depth
        self.max_json_retry = max_json_retry
        self.length_penalty_weight = length_penalty_weight
        self.save_path = Path(save_path) if save_path is not None else None

        self.context_constructor = None
        self.formula_generator = None
        self.truth_evaluator = None
        self.discoverability_evaluator = None
        self.effectiveness_evaluator = None
        self.tools: list[BaseTool] = []
        self.tool_parser = None

        self.token_counter = ParallelTimer(unit="token")
        self.money_counter = ParallelTimer(unit="$")
        self.stage_token_counters: dict[str, ParallelTimer] = {}
        self.stage_money_counters: dict[str, ParallelTimer] = {}
        self.named_timer = NamedTimer()
        self.llm_call_details: list[dict[str, Any]] = []
        self.evaluator_transcripts: list[dict[str, Any]] = []
        self.current_target_formula: str | None = None
        self.current_dataset_id: str | None = None
        self.current_progress: str | None = None
        self.current_best_record: SearchRecord | None = None

    def fit(
        self,
        X: dict[str, np.ndarray],
        y: np.ndarray | dict[str, np.ndarray],
        problem_description: str,
        target_formula: str,
        dataset_id: str = "dataset",
    ) -> dict[str, Any]:
        if not isinstance(y, dict):
            y = {"target": y}

        variables = list(X.keys())
        target_name = next(iter(y))
        dataset_summary = summarize_dataset(X, y)
        self.current_target_formula = target_formula
        self.current_dataset_id = dataset_id
        self.current_progress = self.format_progress(0, 0, 0)
        self.current_best_record = None

        ## 实例化工具和 LLM API
        if self.llm_provider != "mock":
            tool_context = {"data": X | y, "target": target_name}
            self.tools = [
                tool_cls(**tool_context)
                for tool_cls in BaseTool.load_tool_classes(["code_executor", "statistics_analysis"])
            ]
            self.tool_parser = BaseParser.create("openai", tool_list=self.tools)
            self.context_constructor = LLMAPI.create(self.llm_provider, self.llm_model)
            self.formula_generator = LLMAPI.create(self.llm_provider, self.llm_model)
            self.truth_evaluator = LLMAPI.create(
                self.llm_provider,
                self.llm_model,
                tool_list=self.tools,
                tool_parser="openai",
            )
            self.discoverability_evaluator = LLMAPI.create(
                self.llm_provider,
                self.llm_model,
                tool_list=self.tools,
                tool_parser="openai",
            )
            self.effectiveness_evaluator = LLMAPI.create(self.llm_provider, self.llm_model)

        ## 开始迭代
        archive: list[SearchRecord] = []
        previous_formula_evaluations: list[dict[str, Any]] = []
        R = C = L = 0

        for R in range(1, self.max_restart_loop + 1):
            _logger.info("Start Meta Restart Loop (R=%s/%s)", R, self.max_restart_loop)
            restart_top_records = top_records(archive, self.restart_top_k)

            for C in range(1, self.global_width + 1):
                _logger.info(
                    "(R=%s/%s) x Meta Branch (C=%s/%s)",
                    R,
                    self.max_restart_loop,
                    C,
                    self.global_width,
                )

                for L in range(1, self.max_refinement_depth + 1):
                    progress = {"R": R, "C": C, "L": L}
                    self.current_progress = self.format_progress(R, L, C)
                    _logger.info(
                        "(R=%s/%s) x (C=%s/%s) x Meta Step (L=%s/%s)",
                        R,
                        self.max_restart_loop,
                        C,
                        self.global_width,
                        L,
                        self.max_refinement_depth,
                    )

                    ## 构建上下文
                    selected_records = []
                    selected_records.extend(restart_top_records)
                    selected_records.extend(top_records(archive, self.restart_top_k))
                    selected_records.extend(sorted(archive, key=lambda r: len(r.claim_list))[: self.restart_top_k])
                    previous_records = dedupe_records(selected_records)[: 3 * self.restart_top_k]

                    if self.llm_provider == "mock":
                        raise DeprecationWarning()
                    else:
                        messages = []
                        messages.append({
                            "role": "system",
                            "content": (
                                "You are ContextConstructor. Generate behavioral claims that are true for "
                                "the target formula, discoverable from feature-target data, and useful for formula "
                                "generation. A claim list must form a progressive discovery path: early "
                                "claims should be simple patterns that can be found directly from raw data; "
                                "later claims may be more specific only when they are naturally motivated by "
                                "the earlier claims. Do not directly restate formula subexpressions, exact "
                                "algebraic rearrangements, exact constants, exact exponents, or identities "
                                "of the form complex_expression = constant. Avoid claims that are only easy "
                                "to check after being handed a target-like expression. "
                                "Return strict JSON: "
                                "{\"claim_lists\": [[{\"tool_name\": \"...\", \"claim\": \"...\"}, ...], ...]}. "
                                "Each claim object must contain tool_name and claim only. Claim ids will be "
                                "assigned by the caller."
                            ),
                        })
                        messages.append({
                            "role": "user",
                            "content": json.dumps({
                                "target_formula": target_formula,
                                "target_name": target_name,
                                "variables": variables,
                                "problem_description": problem_description,
                                "max_claim_list_length": self.max_claim_list_length,
                                "num_claim_lists": self.local_sample_size,
                                "previous_records": [record_to_json(r) for r in previous_records],
                            }, ensure_ascii=False),
                        })
                        data = self.call_llm_for_json(
                            self.context_constructor,
                            messages,
                            fallback={"claim_lists": []},
                            stage="context_constructor",
                            max_round=1,
                            template=CLAIM_LISTS_TEMPLATE,
                            max_retry=self.max_json_retry,
                        )
                        claim_list_proposals = [
                            normalize_claim_list(raw, self.max_claim_list_length)
                            for raw in data["claim_lists"]
                        ][: self.local_sample_size]

                    for claim_list in claim_list_proposals:
                        ## 生成公式
                        if self.llm_provider == "mock":
                            raise DeprecationWarning()
                        else:
                            messages = []
                            messages.append({
                                "role": "system",
                                "content": (
                                    "You are FormulaGenerator. Generate simple candidate formulas from claims only. "
                                    "Return strict JSON: {\"equation_list\": [...], \"detail\": \"...\"}."
                                ),
                            })
                            messages.append({
                                "role": "user",
                                "content": json.dumps({
                                    "variables": variables,
                                    "claim_list": [asdict(c) for c in claim_list],
                                    "num_candidates": self.num_equations_per_claim_list,
                                }, ensure_ascii=False),
                            })
                            generated_equations = self.call_llm_for_json(
                                self.formula_generator,
                                messages,
                                fallback={"equation_list": [], "detail": "Formula generation failed."},
                                stage="formula_generator",
                                max_round=1,
                                template=GENERATED_EQUATIONS_TEMPLATE,
                                max_retry=self.max_json_retry,
                            )

                        ## 评估真实性
                        if self.llm_provider == "mock":
                            raise DeprecationWarning()
                        else:
                            messages = []
                            messages.append({
                                "role": "system",
                                "content": (
                                    "You are TruthEvaluator. Score each claim against the target formula. "
                                    "Use this normalized rubric for each claim: 1.0 = clearly true as stated; "
                                    "0.5 = partly true, ambiguous, missing a condition, or too strong; "
                                    "0.0 = false. Your numeric score must be consistent with your explanation. "
                                    "Return strict JSON: [{\"claim_id\": \"...\", "
                                    "\"score\": number, \"detail\": \"...\"}, ...]. "
                                    "Do not include an overall score; the caller will compute it."
                                ),
                            })
                            messages.append({
                                "role": "user",
                                "content": json.dumps({
                                    "target_formula": target_formula,
                                    "claim_list": [asdict(c) for c in claim_list],
                                }, ensure_ascii=False),
                            })
                            truth_data = self.call_llm_for_json(
                                self.truth_evaluator,
                                messages,
                                failed_claim_evaluation(claim_list, "Truth evaluation failed."),
                                stage="truth_evaluator",
                                max_round=self.max_evaluation_depth,
                                template=CLAIM_EVALUATION_TEMPLATE,
                                max_retry=self.max_json_retry,
                            )
                            truth_eval = {
                                "score": safe_min_score(item["score"] for item in truth_data),
                                "detail": truth_data,
                            }

                        ## 评估可发现性
                        if self.llm_provider == "mock":
                            raise DeprecationWarning()
                        else:
                            messages = []
                            messages.append({
                                "role": "system",
                                "content": (
                                    "You are DiscoverabilityEvaluator. Judge whether each claim could be "
                                    "discovered ex ante from feature-target data without seeing the target formula and "
                                    "without being handed the claim's exact expression. Evaluate claims in "
                                    "order. For claim N, you may assume claims 1..N-1 have already been "
                                    "discovered and can motivate the next probe; claim 1 must be discoverable "
                                    "from raw data alone. Do not reward ex-post checkability: a claim is not "
                                    "highly discoverable merely because the stated expression can be computed "
                                    "and checked after the fact. Penalize target-like algebraic restatements, "
                                    "complex variable products, exact constants, exact exponents, exact ratios, "
                                    "and identities such as complex_expression = constant unless a small, "
                                    "generic, pre-specified probe could naturally find them from the available "
                                    "evidence. Use this normalized rubric for each claim: 1.0 = robustly "
                                    "discoverable by a simple generic probe from the available prior claims "
                                    "and raw data; 0.7 = discoverable by a reasonable specialized but generic "
                                    "procedure such as basis-library search, sparse regression, active "
                                    "sampling, finite differences, or residual analysis; 0.4 = checkable if "
                                    "stated, but unlikely to be discovered without already knowing a "
                                    "target-like expression; 0.0 = formula leakage, semantic/non-behavioral, "
                                    "or not inferable from observations. In each detail, state what prior "
                                    "claims are used, what generic probe would discover the current claim, "
                                    "and whether the claim is merely checkable-after-stated. "
                                    "Return strict JSON: [{\"claim_id\": \"...\", "
                                    "\"score\": number, \"detail\": \"...\"}, ...]. "
                                    "Do not include an overall score; the caller will compute it."
                                ),
                            })
                            messages.append({
                                "role": "user",
                                "content": json.dumps({
                                    "dataset_summary": dataset_summary,
                                    "variables": variables,
                                    "claim_list": [asdict(c) for c in claim_list],
                                }, ensure_ascii=False),
                            })
                            discoverability_data = self.call_llm_for_json(
                                self.discoverability_evaluator,
                                messages,
                                failed_claim_evaluation(claim_list, "Discoverability evaluation failed."),
                                stage="discoverability_evaluator",
                                max_round=self.max_evaluation_depth,
                                template=CLAIM_EVALUATION_TEMPLATE,
                                max_retry=self.max_json_retry,
                            )
                            discoverability_eval = {
                                "score": safe_min_score(item["score"] for item in discoverability_data),
                                "detail": discoverability_data,
                            }

                        ## 评估公式有效性
                        equation_list = generated_equations["equation_list"]
                        if self.llm_provider == "mock":
                            raise DeprecationWarning()
                        else:
                            messages = []
                            messages.append({
                                "role": "system",
                                "content": (
                                    "You are FormulaEvaluator. Score candidate equations against the target formula. "
                                    "All scores MUST be normalized to the range [0, 1]. Do not use a 0-10 scale. "
                                    "Return strict JSON: {\"candidate_scores\": [{\"equation\": \"...\", "
                                    "\"score\": number, \"detail\": \"...\"}, ...]}."
                                ),
                            })
                            messages.append({
                                "role": "user",
                                "content": json.dumps({
                                    "target_formula": target_formula,
                                    "equation_list": equation_list,
                                    "variables": variables,
                                    "previous_formula_evaluations": previous_formula_evaluations[-20:],
                                }, ensure_ascii=False),
                            })
                            data = self.call_llm_for_json(
                                self.effectiveness_evaluator,
                                messages,
                                fallback={"candidate_scores": []},
                                stage="effectiveness_evaluator",
                                max_round=1,
                                template=EFFECTIVENESS_TEMPLATE,
                                max_retry=self.max_json_retry,
                            )
                            effectiveness_detail = data.get("candidate_scores", [])
                            for item in effectiveness_detail:
                                if isinstance(item, dict):
                                    item["score"] = normalize_unit_score(item.get("score", 0.0))
                            effectiveness_eval = {
                                "score": safe_max_score(
                                    item.get("score", 0.0)
                                    for item in effectiveness_detail
                                    if isinstance(item, dict)
                                ),
                                "detail": effectiveness_detail,
                            }

                        claim_list_score = (
                            score_for_ranking(truth_eval["score"])
                            * score_for_ranking(discoverability_eval["score"])
                            * score_for_ranking(effectiveness_eval["score"])
                            - self.length_penalty_weight * len(claim_list)
                        )
                        record = SearchRecord(
                            claim_list_score=claim_list_score,
                            claim_list=claim_list,
                            generated_equations=generated_equations,
                            truth_evaluation=truth_eval,
                            discoverability_evaluation=discoverability_eval,
                            effectiveness_evaluation=effectiveness_eval,
                            meta={"progress": progress},
                        )
                        previous_best_score = best_record(archive).claim_list_score if archive else None
                        archive.append(record)
                        self.current_best_record = best_record(archive)
                        previous_formula_evaluations.extend(
                            {
                                "target_formula": target_formula,
                                "candidate_equation": item["equation"],
                                "score": item["score"],
                                "detail": item["detail"],
                            }
                            for item in effectiveness_eval["detail"]
                            if isinstance(item, dict)
                        )
                        is_new_best = previous_best_score is None or record.claim_list_score > previous_best_score
                        log = self.format_log({
                            "report_title": "Meta SR Step Result - New Best" if is_new_best else "Meta SR Step Result",
                            "status": "running",
                            "target_formula": target_formula,
                            "dataset_id": dataset_id,
                            "progress": self.format_progress(**record.meta["progress"]),
                            "current_score": record.claim_list_score,
                            "previous_best_score": previous_best_score,
                            "token_usage": self.token_counter,
                            "money_usage": self.money_counter,
                            "time_usage": self.named_timer,
                            "record": record_to_json(record),
                        })
                        (_logger.note if is_new_best else _logger.info)(tag2ansi(
                            f"Meta step [blue]{progress}[reset] "
                            f"scored [red bold]{record.claim_list_score:.4f}[reset] "
                            f"with [blue]{len(record.claim_list)}[reset] claims."
                            f"\n{log}"
                        ))

                    if archive and best_record(archive).claim_list_score >= 0.98:
                        return self.format_result(
                            target_formula,
                            dataset_id,
                            best_record(archive),
                            status="early_stopped",
                            progress=self.format_progress(R, L, C),
                        )

        return self.format_result(
            target_formula,
            dataset_id,
            best_record(archive) if archive else None,
            status="completed",
            progress=self.format_progress(R, L, C),
        )

    def get_result_snapshot(self, status: str) -> dict[str, Any]:
        if self.current_target_formula is None or self.current_dataset_id is None:
            return {"status": status}
        else:
            return self.format_result(
                self.current_target_formula,
                self.current_dataset_id,
                self.current_best_record,
                status=status,
                progress=self.current_progress or self.format_progress(0, 0, 0),
            )

    def format_result(
        self,
        target_formula: str,
        dataset_id: str,
        best: SearchRecord | None,
        status: str,
        progress: str,
    ) -> dict[str, Any]:
        return {
            "target_formula": target_formula,
            "dataset_id": dataset_id,
            "best_record": None if best is None else record_to_json(best),
            "status": status,
            "progress": progress,
            "token_usage": self.token_counter,
            "money_usage": self.money_counter,
            "time_usage": self.named_timer,
            "timing": named_timer_summary(self.named_timer),
            "llm_call_details": self.llm_call_details,
            "evaluator_transcripts": self.evaluator_transcripts,
        }

    def call_llm_for_json(
        self,
        llm: LLMAPI,
        messages: List[Dict[str, str]],
        fallback: Any,
        stage: str,
        max_round: int,
        template: Any,
        max_retry: int = 3,
    ) -> Any:
        messages = [message.copy() for message in messages]
        allow_tool_use = (max_round > 1) and (self.tool_parser is not None)
        if allow_tool_use:
            messages = add_evaluator_tool_instructions(messages, max_round=max_round)

        parsed = None
        content = ""
        parse_error = None
        timing: dict[str, Any] | None = None

        for round_idx in range(1, max_round + 1):
            if allow_tool_use:
                messages.append(build_round_status_prompt(round_idx, max_round))

            attempts = max(1, max_retry) if round_idx == max_round else 1
            for retry_idx in range(1, attempts + 1):
                ## Call LLM
                start_time = time.perf_counter()
                tool_calls = []
                response_message = None
                usage = {}
                request_error = None
                try:
                    result = llm(messages, n=1)
                    for content, yielded_tool_calls, yielded_message in result:
                        tool_calls = yielded_tool_calls or []
                        response_message = yielded_message
                    usage = result.returned["usage"]
                    self.merge_usage(usage, stage=stage)
                    _logger.debug(tag2ansi(
                        f"[red bold]{stage.replace("_", " ").title()} Agent[reset] "
                        f"[blue]Round {round_idx}/{max_round}[reset] [orange](Trial {retry_idx}/{attempts})[reset]\n"
                        f"[red]Response[reset]: "
                        + (f"\n{render_markdown(content.strip())}" if content else "[gray]<empty>[reset]") +
                        f"[red]Tool Calls[reset]: "
                        + (f"\n{format_tool_calls(tool_calls)}" if tool_calls else "[gray]<empty>[reset]")
                    ))
                except Exception as e:
                    request_error = repr(e)
                    _logger.warning(tag2ansi(
                        f"[red bold]{stage.replace("_", " ").title()} Agent[reset] "
                        f"request failed: {log_exception(e)}"
                    ))

                self.named_timer.add(f"{stage}.request")
                timing = {
                    "stage": stage,
                    "round": round_idx,
                    "retry": retry_idx,
                    "total_seconds": time.perf_counter() - start_time,
                    "token_usage": usage.get("token", {}),
                    "price_usage": usage.get("price", {}),
                }
                self.llm_call_details.append(timing)

                if tool_calls and allow_tool_use and round_idx < max_round:
                    results = self.execute_tool_calls(tool_calls)
                    self.record_agent_call(
                        stage=stage,
                        messages=messages,
                        response=content,
                        parsed={
                            "round": round_idx,
                            "tool_calls": serialize_dataclass(tool_calls, ["name", "params", "raw_str", "raw"]),
                            "tool_results": serialize_dataclass(results, ["ok", "result", "result_str", "meta_data"]),
                        },
                        used_fallback=False,
                        error=request_error,
                        timing=timing,
                    )
                    messages.append(response_message or {"role": "assistant", "content": content})
                    messages.extend(self.tool_parser.format_tool_result_messages(tool_calls, results))
                    break

                try:
                    parsed = parse_json_with_template(content, template)
                    self.named_timer.add(f"{stage}.parse")
                    parse_error = request_error
                    used_fallback = False
                except Exception as e:
                    parse_error = repr(e)
                    _logger.warning(tag2ansi(
                        f"[bold red]{stage.replace('_', ' ').title()} Agent[reset] "
                        f"parse failed at [blue]Round {round_idx}/{max_round}[reset] "
                        f"[orange](Trial {retry_idx}/{attempts})[reset]: "
                        f"[bold red]{type(e).__name__}[reset]: [red]{str(e)}[reset]"
                        # f"{log_exception(e)}"
                    ))
                    if retry_idx < attempts:
                        serialized_tool_calls = serialize_dataclass(tool_calls, ["name", "params", "raw_str", "raw"])
                        self.record_agent_call(
                            stage=stage,
                            messages=messages,
                            response=content,
                            parsed={
                                "parse_error": parse_error,
                                "ignored_tool_calls": serialized_tool_calls,
                            },
                            used_fallback=False,
                            error=parse_error,
                            timing=timing,
                        )
                        messages.append(response_message or {"role": "assistant", "content": content})
                        if tool_calls and allow_tool_use:
                            messages.extend(self.format_final_tool_failure_messages(tool_calls))
                        messages.append(build_final_json_retry_prompt())
                        continue
                    if allow_tool_use and round_idx < max_round:
                        serialized_tool_calls = serialize_dataclass(tool_calls, ["name", "params", "raw_str", "raw"])
                        self.record_agent_call(
                            stage=stage,
                            messages=messages,
                            response=content,
                            parsed={
                                "parse_error": parse_error,
                                "ignored_tool_calls": serialized_tool_calls,
                                "will_continue_rounds": True,
                            },
                            used_fallback=False,
                            error=parse_error,
                            timing=timing,
                        )
                        messages.append(response_message or {"role": "assistant", "content": content})
                        if tool_calls:
                            messages.extend(self.format_final_tool_failure_messages(tool_calls))
                        messages.append(build_parse_retry_round_prompt(round_idx, max_round))
                        break
                    parsed = fallback
                    used_fallback = True

                serialized_tool_calls = serialize_dataclass(tool_calls, ["name", "params", "raw_str", "raw"])
                self.record_agent_call(
                    stage=stage,
                    messages=messages,
                    response=content,
                    parsed=parsed if not used_fallback else {
                        "fallback": parsed,
                        "ignored_tool_calls": serialized_tool_calls,
                    },
                    used_fallback=used_fallback,
                    error=parse_error,
                    timing=timing,
                )
                if allow_tool_use:
                    self.evaluator_transcripts.append({
                        "stage": stage,
                        "round": round_idx,
                        "messages": messages,
                        "response": content,
                        "timing": timing,
                    })
                break

            if tool_calls and allow_tool_use and round_idx < max_round:
                continue
            if parsed is None and allow_tool_use and round_idx < max_round:
                continue
            break

        if parsed is None:
            parsed = fallback
            self.record_agent_call(
                stage=stage,
                messages=messages,
                response=content,
                parsed=parsed,
                used_fallback=True,
                error=parse_error,
                timing=timing or {},
            )
        return parsed

    def format_final_tool_failure_messages(self, tool_calls) -> list[dict[str, Any]]:
        results = [
            ToolCallResult(
                ok=False,
                result={},
                result_str=(
                    "调用失败：当前已经是最后一轮，不能继续调用工具。"
                    "请基于已有信息直接输出严格 JSON。"
                ),
                meta_data={"tool": tool_call.name, "reason": "final_round_tool_call"},
            )
            for tool_call in tool_calls
        ]
        return self.tool_parser.format_tool_result_messages(tool_calls, results)

    def execute_tool_calls(self, tool_calls) -> list[ToolCallResult]:
        tool_map = {tool.metadata.name: tool for tool in self.tools}
        results = []
        for tool_call in tool_calls:
            if (tool := tool_map.get(tool_call.name)) is not None:
                results.append(tool(**tool_call.params))
            else:
                results.append(ToolCallResult(
                    ok=False,
                    result={},
                    result_str=f'Unknown tool calling for "{tool_call.name}"',
                    meta_data={"tool": tool_call.name},
                ))
        return results

    def record_agent_call(
        self,
        stage: str,
        messages: list[dict[str, str]],
        response: str,
        parsed: dict[str, Any],
        used_fallback: bool,
        error: str | None,
        timing: dict[str, Any],
    ) -> None:
        if self.save_path is None:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "messages": messages,
            "response": response,
            "parsed": parsed,
            "used_fallback": used_fallback,
            "error": error,
            "timing": timing,
        }
        with open(self.save_path / "agent_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def merge_usage(self, usage: dict[str, Any], stage: str) -> None:
        for name, value in usage.get("token", {}).items():
            self.token_counter.add(name, value)
            if stage not in self.stage_token_counters:
                self.stage_token_counters[stage] = ParallelTimer(unit="token")
            self.stage_token_counters[stage].add(name, value)
        for name, value in usage.get("price", {}).items():
            self.money_counter.add(name, value)
            if stage not in self.stage_money_counters:
                self.stage_money_counters[stage] = ParallelTimer(unit="$")
            self.stage_money_counters[stage].add(name, value)

    def format_log(self, result) -> None: # 这个代码已经经过人工审核，不得进一步修改。
        lines = []
        lines.append(f'[gray]{"=" * 72}[reset]')
        lines.append(f"[hotpink bold]{result.get('report_title', 'N/A')}[reset]")
        lines.append(f'[gray]{"-" * 72}[reset]')
        lines.append(f"[red bold]Metadata[reset]")
        lines.append(f"  [red]Status[reset]={result.get('status', 'N/A')}")
        lines.append(f"  [red]Dataset id[reset]={result.get('dataset_id', 'N/A')}")
        lines.append(f"  [red]Target formula[reset]={result.get('target_formula', 'N/A')}")
        lines.append(f"  [red]Start time[reset]={result.get('start_time', 'N/A')}")
        lines.append(f"  [red]Duration seconds[reset]={format_number(result.get('duration_seconds'))}")
        lines.append(f"  [red]Random seed[reset]={result.get('random_seed', 'N/A')}")
        lines.append(f"  [red]LLM[reset]={result.get('llm_model', 'N/A')}")
        lines.append(f"  [red]Progress[reset]={result.get('progress', 'N/A')}")
        lines.append(f"  [red]Current score[reset]={colored_score(result.get('current_score'))}")
        lines.append(f"  [red]Previous best score[reset]={colored_score(result.get('previous_best_score'))}")
        lines.append(f"  [red]Total TToken usage[reset]={result['token_usage'].to_str("count", "speed", "by_count")}")
        lines.append(f"  [red]Total Money usage[reset]={result['money_usage'].to_str("count", "speed", "by_count")}")
        lines.append(f"  [red]Time Usage[reset]={result['time_usage'].to_str("time", "pace", "by_time")}")
        lines.append(f"  [red]Result path[reset]={result.get('result_path', 'N/A')}")
        lines.append(f'[gray]{"-" * 72}[reset]')
        record = result.get("record", {})
        lines.extend(extend_colored_evaluation_sections(
            record.get("claim_list", []),
            record.get("generated_equations", {}),
            record.get("truth_evaluation", {}),
            record.get("discoverability_evaluation", {}),
            record.get("effectiveness_evaluation", {}),
        ))
        lines.append(f'[gray]{"=" * 72}[reset]')
        return tag2ansi("\n".join(lines))

    def format_progress(self, R: int, L: int, C: int) -> str:
        return (
            f"(R={R}/{self.max_restart_loop}) x "
            f"(C={C}/{self.global_width}) x "
            f"(L={L}/{self.max_refinement_depth}) x "
            f"(K={self.local_sample_size})"
        )
