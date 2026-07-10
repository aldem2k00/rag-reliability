#!/usr/bin/env python
"""Render a GEPA stats json (from scripts/run_gepa.py) into a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_reliability.methods.m3.gepa_report import render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["markers", "plain"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stats-dir", default="results/gepa")
    parser.add_argument(
        "--stats", default=None, help="Stats json path (otherwise built from --variant/--seed)"
    )
    parser.add_argument("--out", default=None, help="Markdown path (defaults next to stats)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stats:
        stats_path = Path(args.stats)
    elif args.variant is not None and args.seed is not None:
        stats_path = Path(args.stats_dir) / f"m3_gepa_stats_{args.variant}_seed{args.seed}.json"
    else:
        raise SystemExit("pass --stats OR both --variant and --seed")
    if not stats_path.exists():
        raise SystemExit(f"no stats file: {stats_path} (run scripts/run_gepa.py first)")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    variant, seed = stats.get("variant", "?"), stats.get("seed", "?")
    out_path = (
        Path(args.out)
        if args.out
        else stats_path.parent / f"m3_gepa_report_{variant}_seed{seed}.md"
    )
    out_path.write_text(render_report(stats), encoding="utf-8")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
