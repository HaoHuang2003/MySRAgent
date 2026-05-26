# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Command-line entry point for the experimental MetaSRAgent."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "experimental"))

import re
import json
import shlex
import logging
import argparse
import numpy as np
import nd2py as nd
from datetime import datetime
from socket import gethostname
from meta_sr_agent import MetaSRAgent
from sr_agent.utils import (
    add_minus_flags,
    add_negation_flags,
    log_exception,
    seed_all,
    setup_logging,
    tag2ansi,
)

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the experimental MetaSRAgent on a synthetic problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=SCRIPT_NAME, help="Task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/experimental/{SCRIPT_NAME}", help="Root directory for logs.")
    parser.add_argument("-f", "--equation", default="y = sin(x1 - x2)", help="Target equation.")
    parser.add_argument("--problem_description", default=None, help="Problem description passed to the agent.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. -1 means current time.")
    parser.add_argument("--x_low", type=float, default=0.0, help="Lower bound for random features.")
    parser.add_argument("--x_high", type=float, default=1.0, help="Upper bound for random features.")
    parser.add_argument("--noise_std_ratio", type=float, default=0.0, help="Gaussian noise ratio added to y.")
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider name.")
    parser.add_argument("--llm_model", default="deepseek/deepseek-v4-pro", help="LLM model name.")
    parser.add_argument("-K", "--local_sample_size", type=int, default=2, help="Claim lists sampled per step.")
    parser.add_argument("-L", "--max_refinement_depth", type=int, default=2, help="Maximum refinement depth.")
    parser.add_argument("-C", "--global_width", type=int, default=1, help="Independent branches per restart.")
    parser.add_argument("-R", "--max_restart_loop", type=int, default=1, help="Best-record restart loops.")
    parser.add_argument("--restart_top_k", type=int, default=3, help="Top records reused by restarts.")
    parser.add_argument("--max_claim_list_length", type=int, default=6, help="Maximum claims per proposal.")
    parser.add_argument("--num_equations_per_claim_list", type=int, default=5, help="Candidate equations per proposal.")
    parser.add_argument("--max_evaluation_depth", type=int, default=3, help="Maximum tool-use rounds for Truth/Discoverability evaluators.")
    parser.add_argument("--max_json_retry", type=int, default=3, help="Retries on the final round when the LLM response cannot be parsed as the required JSON.")
    parser.add_argument("--save_path", default=None, help="Path to save logs and artifacts.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--debug", action="store_true", default=True, help="Raise caught exceptions.")
    parser = add_minus_flags(parser)
    parser = add_negation_flags(parser)
    return parser


def sanitize_filename(value: str) -> str:
    value = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub("_", value.strip())
    return (value or "unnamed")[:255]


def save_args(args: argparse.Namespace, args_path: Path) -> None:
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists():
            i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning("args.json already exists, backup to args.json.%s", i)
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


def make_dataset(args: argparse.Namespace):
    if "=" not in args.equation:
        if "target" in args.equation or "target" in args.equation:
            raise ValueError("Equation contains a target-like name but no '='. Use e.g. 'target = sin(x1)'.")
        args.equation = f"target = {args.equation}"

    target, formula_str = args.equation.split("=", 1)
    target = target.strip()
    formula_str = formula_str.strip()
    if nd is not None:
        formula = nd.parse(formula_str)
        features = sorted(
            {var.name for var in formula.iter_preorder() if isinstance(var, nd.Variable)}
        )
    else:
        formula = formula_str
        known_funcs = {"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "pi", "e"}
        names = set(re.findall(r"\b[A-Za-z_]\w*\b", formula_str))
        features = sorted(name for name in names if name not in known_funcs)
    if target not in features:
        pass
    elif "target" not in features:
        _logger.warning(
            f"Target name {target!r} also appears in the right-hand side; renaming the "
            f"dependent variable to 'target' while keeping feature variables unchanged.",
        )
        target = "target"
    else:
        raise ValueError(
            "The left-hand target name also appears as a feature, and 'target' "
            "is already used in the formula. Please choose a distinct target name."
        )

    rng = np.random.default_rng(args.seed)
    data = {name: rng.uniform(args.x_low, args.x_high, size=args.n_samples) for name in features}
    if nd is not None:
        data[target] = formula.eval(data)
    else:
        namespace = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e,
        }
        expression = formula_str.replace("^", "**")
        data[target] = eval(expression, {"__builtins__": {}}, namespace | data)
    if args.noise_std_ratio > 0:
        data[target] += rng.normal(
            0.0,
            args.noise_std_ratio * np.std(data[target]),
            size=data[target].shape,
        )
    return features, target, formula, data


def main(args: argparse.Namespace) -> dict:
    features, target, formula, data = make_dataset(args)
    X = {name: data[name] for name in features}
    y = {target: data[target]}
    target_formula = f"{target} = {formula}"
    problem_description = args.problem_description or (
        f"Find the relationship {target} = f({', '.join(features)}). "
        "The target was generated from a hidden symbolic formula."
    )

    _logger.note(
        f"Starting experiment {args.exp_name}\n"
        f"Equation: {target_formula}\n"
        f"Feature variables: {', '.join(features)}\n"
        f"Generated {args.n_samples} samples with seed {args.seed}\n"
    )

    agent = MetaSRAgent(
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        local_sample_size=args.local_sample_size,
        max_refinement_depth=args.max_refinement_depth,
        global_width=args.global_width,
        max_restart_loop=args.max_restart_loop,
        restart_top_k=args.restart_top_k,
        max_claim_list_length=args.max_claim_list_length,
        num_equations_per_claim_list=args.num_equations_per_claim_list,
        max_evaluation_depth=args.max_evaluation_depth,
        max_json_retry=args.max_json_retry,
        save_path=args.save_path,
    )

    result = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": None,
        "target_formula": target_formula,
        "noise_std_ratio": args.noise_std_ratio,
        "random_seed": args.seed,
        "status": "not_started",
        "llm_model": f"{args.llm_model} @ {args.llm_provider}",
    }

    try:
        result |= agent.fit(
            X=X,
            y=y,
            problem_description=problem_description,
            target_formula=target_formula,
            dataset_id=args.exp_name,
        )
    except KeyboardInterrupt:
        _logger.note("Experiment interrupted by user.")
        result |= agent.get_result_snapshot(status="interrupted")
    except Exception as exc:
        _logger.error(f"Experiment failed with an exception: {log_exception(exc)}")
        result |= agent.get_result_snapshot(status="failed")
        result["error"] = repr(exc)
        if args.debug:
            raise
    finally:
        start_time = datetime.strptime(result["start_time"], "%Y-%m-%d %H:%M:%S")
        result["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        result_path = Path(args.save_path) / "result.jsonl"
        result["token_usage"] = agent.token_counter
        result["money_usage"] = agent.money_counter
        result["time_usage"] = agent.named_timer
        result["current_score"] = (result.get("best_record") or {}).get("claim_list_score")
        result["record"] = result.pop("best_record")
        result["result_path"] = str(result_path)
        result["report_title"] = "Meta SR Result"
        _logger.note("\n" + agent.format_log(result))

        result_to_save = result | {
            "token_usage": agent.token_counter.named_count,
            "money_usage": agent.money_counter.named_count,
            "time_usage": agent.named_timer.to_str("time", "pace", "by_time"),
        }
        with open(result_path, "a", encoding="utf-8") as f:
            json.dump(result_to_save, f, ensure_ascii=False)
            f.write("\n")

    return result


if __name__ == "__main__":
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}")
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    if args.debug:
        args.verbose = True
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    seed_all(args.seed)

    save_path = Path(args.save_dir) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    args.command = " ".join(map(shlex.quote, [sys.executable, *sys.argv]))

    setup_logging(
        info_level="debug" if args.verbose else "info",
        exp_name=args.exp_name,
        save_path=save_path / "info.log",
        force=True,
    )

    if unknown:
        _logger.warning("Unknown args: %s", unknown)
    _logger.note(f"Args: {args}")
    save_args(args, save_path / "args.json")

    main(args)
    _logger.note(tag2ansi(f"Experiment completed. Re-run with [green bold]{args.command}[reset]"))
