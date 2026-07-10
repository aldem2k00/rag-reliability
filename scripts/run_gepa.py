#!/usr/bin/env python
"""Evolve the Method 3 judge instruction with GEPA (DSPy).

Optimization only; inference stays in scripts/run_m3.py
(--mode gepa --prompt-file <output of this script>).

Requires the "gepa" extra: uv pip install -e ".[gepa]"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from rag_reliability.dataset import load_jsonl
from rag_reliability.guard import assert_cloud_safe
from rag_reliability.methods.m3.gepa import (
    build_examples,
    build_program,
    extract_instruction,
    load_marker_gloss,
    make_metric,
    serialize_detailed,
    subsample_train,
)
from rag_reliability.methods.m3.prompts import SEED_INSTRUCTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, help="Train RagSample JSONL")
    parser.add_argument("--val-data", required=True, help="Validation RagSample JSONL")
    parser.add_argument("--variant", choices=["markers", "plain"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=None, help="Subsample train to N")
    parser.add_argument(
        "--train-marker-share",
        type=float,
        default=0.0,
        help="Raise the share of marker-carrying samples in the train subsample",
    )
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--reflection-model", default=None, help="Defaults to --model")
    parser.add_argument("--reflection-api-base", default=None, help="Defaults to --api-base")
    parser.add_argument("--reflection-max-tokens", type=int, default=8000)
    parser.add_argument("--markers-gloss", default="configs/markers.yaml")
    parser.add_argument("--max-context-chars", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N")
    parser.add_argument(
        "--output-prompt",
        default=None,
        help="Defaults to results/gepa/m3_optimized_prompt_{variant}_seed{seed}.txt",
    )
    parser.add_argument(
        "--profile",
        choices=["local", "cloud"],
        default="local",
        help="cloud enables the data-leak guard before any request",
    )
    parser.add_argument(
        "--allow-real-data",
        action="store_true",
        help="explicit data-owner opt-in to send real samples with --profile cloud",
    )
    return parser.parse_args()


def _git_hash() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:  # noqa: PLR0915
    args = parse_args()

    train_samples = load_jsonl(args.train_data)
    val_samples = load_jsonl(args.val_data)
    if args.limit is not None:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: args.limit]

    # Guard BEFORE the first LLM call: with the cloud profile only synthetic data.
    assert_cloud_safe(train_samples, args.profile, allow_real=args.allow_real_data)
    assert_cloud_safe(val_samples, args.profile, allow_real=args.allow_real_data)

    if args.train_size is not None:
        train_samples = subsample_train(
            train_samples, args.train_size, args.train_marker_share, args.seed
        )

    try:
        import dspy  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError('Install GEPA deps with: uv pip install -e ".[gepa]"') from exc

    api_key = os.environ.get(args.api_key_env, "")
    task_lm = dspy.LM(
        f"openai/{args.model}",
        api_base=args.api_base,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    reflection_lm = dspy.LM(
        f"openai/{args.reflection_model or args.model}",
        api_base=args.reflection_api_base or args.api_base,
        api_key=api_key,
        temperature=1.0,
        max_tokens=args.reflection_max_tokens,
    )
    dspy.configure(lm=task_lm)

    program = build_program()
    metric = make_metric(
        use_markers=args.variant == "markers",
        gloss=load_marker_gloss(args.markers_gloss),
    )
    trainset = build_examples(train_samples, args.max_context_chars)
    valset = build_examples(val_samples, args.max_context_chars)

    gepa = dspy.GEPA(
        metric=metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        track_stats=True,
        seed=args.seed,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    # Save the instruction (txt), the program (json) and the evolution stats.
    instruction = extract_instruction(optimized)
    out_prompt = Path(
        args.output_prompt
        or f"results/gepa/m3_optimized_prompt_{args.variant}_seed{args.seed}.txt"
    )
    out_prompt.parent.mkdir(parents=True, exist_ok=True)
    out_prompt.write_text(instruction, encoding="utf-8")
    optimized.save(str(out_prompt.with_suffix(".program.json")))

    stats = {
        "variant": args.variant,
        "seed": args.seed,
        "auto": args.auto,
        "train_size": len(trainset),
        "val_size": len(valset),
        "use_marker_feedback": args.variant == "markers",
        "task_model": args.model,
        "reflection_model": args.reflection_model or args.model,
        "task_lm_calls": len(getattr(task_lm, "history", []) or []),
        "reflection_lm_calls": len(getattr(reflection_lm, "history", []) or []),
        "profile": args.profile,
        "git_hash": _git_hash(),
        "seed_instruction": SEED_INSTRUCTION,
        "best_instruction": instruction,
        "detailed_results": serialize_detailed(getattr(optimized, "detailed_results", None)),
    }
    stats_path = out_prompt.parent / f"m3_gepa_stats_{args.variant}_seed{args.seed}.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"prompt: {out_prompt}\nstats: {stats_path}\n"
        f"calls: task={stats['task_lm_calls']}, reflection={stats['reflection_lm_calls']}"
    )


if __name__ == "__main__":
    main()
