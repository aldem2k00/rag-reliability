#!/usr/bin/env python
"""Recompute every figure in the Methods 1-2 article from prediction artifacts.

Reads the two zero-shot prediction files plus the converted corpus and prints
each diagnostic under the LaTeX table label it feeds in
`reliability_prompted_lora_judges_article.tex`. Inference is never rerun: the
canonical split is a deterministic function of the corpus and the seed, so
every per-split number is recoverable from the full-corpus prediction files.

Example:
    python scripts/analyze_prompted_judges.py \
        --data data/organizers.jsonl \
        --direct results/organizers_qwen_direct_predictions.jsonl \
        --marker results/organizers_qwen_marker_predictions.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from rag_reliability.dataset import load_jsonl, split_samples
from rag_reliability.formatting import resolve_marker
from rag_reliability.prompts import build_direct_prompt, build_marker_prompt
from rag_reliability.schema import ALLOWED_MARKERS, Prediction, RagSample

AXES = ("faithfulness", "relevance", "reliable")
# SFT caps applied by scripts/prepare_lora_benchmark.py.
MAX_QUESTION_CHARS = 2000
MAX_CONTEXT_CHARS = 5000


# --------------------------------------------------------------------------
# arithmetic (pure, covered by tests/test_analyze_prompted_judges.py)
# --------------------------------------------------------------------------


def macro_f1_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    """Binary macro-F1 from a confusion matrix; matches sklearn zero_division=0."""
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return (f1_neg + f1_pos) / 2


def confusion(gold: list[int], pred: list[int]) -> tuple[int, int, int, int]:
    """(TN, FP, FN, TP) with 1 as the positive class."""
    tn = sum(1 for g, p in zip(gold, pred, strict=True) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred, strict=True) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == 1 and p == 0)
    tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == 1 and p == 1)
    return tn, fp, fn, tp


def per_class_f1(tn: int, fp: int, fn: int, tp: int) -> tuple[float, float]:
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1_neg, f1_pos


def youden_j(tn: int, fp: int, fn: int, tp: int) -> float:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return tpr + tnr - 1


def reconstruct_confusion(
    n_rows: int, n_gold_pos: int, n_pred_pos: int, target_macro: float
) -> tuple[int, int, int, int]:
    """Recover a confusion matrix from reported macro-F1 plus prediction counts.

    Used for the LoRA rows, whose per-row predictions were not persisted. With
    the predicted-positive and gold-positive counts fixed, macro-F1 is strictly
    monotone in TP, so the matrix is unique.
    """
    best: tuple[float, int, int, int, int] | None = None
    for tp in range(min(n_pred_pos, n_gold_pos) + 1):
        fp = n_pred_pos - tp
        fn = n_gold_pos - tp
        tn = n_rows - n_gold_pos - fp
        if tn < 0:
            continue
        macro = macro_f1_from_counts(tn, fp, fn, tp)
        if best is None or abs(macro - target_macro) < abs(best[0] - target_macro):
            best = (macro, tn, fp, fn, tp)
    if best is None:
        raise ValueError("no admissible confusion matrix for the given counts")
    return best[1], best[2], best[3], best[4]


def marker_macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    """Macro-F1 over an explicit label set, so the averaging set is auditable."""
    if not labels:
        return float("nan")
    total = 0.0
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p != label)
        total += 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return total / len(labels)


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path.resolve()}")
    by_id: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            prediction = Prediction.model_validate_json(line)
            by_id[prediction.id] = prediction
    return by_id


def axis_values(
    samples: list[RagSample], preds: dict[str, Prediction], axis: str
) -> tuple[list[int], list[int]]:
    gold = [getattr(s, axis) for s in samples]
    pred = [getattr(preds[s.id], f"{axis}_pred") for s in samples]
    return gold, pred


def raw_marker_provenance(zip_path: str | Path) -> dict[str, int]:
    """Marker counts straight from the organizer archive, before conversion."""
    csv.field_size_limit(10**8)
    with ZipFile(zip_path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        rows = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))

    annotated = multi = reliable = occurrences = 0
    for row in rows:
        text = (row.get("markers") or "").strip()
        try:
            markers = ast.literal_eval(text) if text else []
        except (SyntaxError, ValueError):
            markers = [text] if text else []
        if not isinstance(markers, list):
            markers = [markers]
        markers = [m for m in markers if isinstance(m, str) and m]
        if not markers:
            continue
        annotated += 1
        occurrences += len(markers)
        multi += len(markers) > 1
        faithful = str(row["binary_faithfulness"]).strip().lower() in ("true", "1")
        relevant = str(row["binary_relevancy"]).strip().lower() in ("true", "1")
        reliable += faithful and relevant
    return {
        "annotated_rows": annotated,
        "occurrences": occurrences,
        "multi_marker_rows": multi,
        "annotated_but_gold_reliable": reliable,
        "usable": annotated - reliable,
    }


# --------------------------------------------------------------------------
# report sections
# --------------------------------------------------------------------------


def section(title: str, table: str) -> None:
    print(f"\n{'=' * 78}\n{title}  [{table}]\n{'=' * 78}")


def report_splits(splits: dict[str, list[RagSample]]) -> None:
    section("Joint gold label distribution", "tab:joint-labels")
    print(f"{'Split':<12}{'Rows':>7}{'(0,0)':>8}{'(0,1)':>8}{'(1,0)':>8}{'(1,1)':>8}{'F!=R':>8}")
    for name, split in splits.items():
        cells = Counter((s.faithfulness, s.relevance) for s in split)
        diff = sum(1 for s in split if s.faithfulness != s.relevance)
        print(
            f"{name:<12}{len(split):>7}{cells[(0, 0)]:>8}{cells[(0, 1)]:>8}"
            f"{cells[(1, 0)]:>8}{cells[(1, 1)]:>8}{diff:>8}"
        )


def report_marker_provenance(splits: dict[str, list[RagSample]], zip_path: Path | None) -> None:
    section("Marker annotation provenance", "tab:marker-provenance")
    full = splits["Full"]
    usable = [s for s in full if s.marker not in (None, "none", "unknown")]
    unreliable = [s for s in full if s.reliable == 0]
    if zip_path is not None and zip_path.exists():
        for key, value in raw_marker_provenance(zip_path).items():
            print(f"  raw {key:<28} {value}")
    else:
        print(f"  raw archive not found at {zip_path}; skipping raw counts")
    print(f"  usable explicit-marker rows    {len(usable)}")
    print(f"    of corpus                    {len(usable) / len(full):.4f}")
    print(f"    of {len(unreliable)} unreliable rows        {len(usable) / len(unreliable):.4f}")
    for name, split in splits.items():
        explicit = [s for s in split if s.marker not in (None, "none", "unknown")]
        classes = {s.marker for s in explicit}
        print(f"  {name:<12} explicit rows {len(explicit):>4}  distinct classes {len(classes)}")
    taxonomy = {m for m in ALLOWED_MARKERS if m.startswith("reason_")}
    populated = {s.marker for s in usable}
    print(f"  taxonomy classes populated     {len(populated & taxonomy)} of {len(taxonomy)}")
    print(f"  never populated                {sorted(taxonomy - populated)}")


def report_confusions(splits: dict[str, list[RagSample]], preds: dict[str, dict]) -> None:
    for mode, table in (("direct", "tab:direct-confusion"), ("marker", "tab:marker-confusion")):
        section(f"Zero-shot {mode} judge confusion matrices", table)
        header = f"{'Split':<12}{'Axis':<14}{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}"
        print(header + f"{'F1_0':>9}{'F1_1':>9}{'macro':>9}")
        for name, split in splits.items():
            for axis in AXES:
                gold, pred = axis_values(split, preds[mode], axis)
                tn, fp, fn, tp = confusion(gold, pred)
                f1_neg, f1_pos = per_class_f1(tn, fp, fn, tp)
                print(
                    f"{name:<12}{axis:<14}{tn:>6}{fp:>6}{fn:>6}{tp:>6}"
                    f"{f1_neg:>9.4f}{f1_pos:>9.4f}{macro_f1_from_counts(tn, fp, fn, tp):>9.4f}"
                )
            invalid = sum(preds[mode][s.id].invalid_output for s in split)
            print(f"{'':<12}{'invalid':<14}{invalid:>6} / {len(split)}")


def report_operating_point(splits: dict[str, list[RagSample]], preds: dict[str, dict]) -> None:
    section("Reliability operating point", "tab:operating-point")
    print(
        f"{'Mode':<8}{'Split':<12}{'base':>8}{'PASS':>7}{'leak':>8}"
        f"{'FAIL':>7}{'prec':>8}{'J':>9}"
    )
    for mode in ("direct", "marker"):
        for name, split in splits.items():
            gold, pred = axis_values(split, preds[mode], "reliable")
            tn, fp, fn, tp = confusion(gold, pred)
            base = (tn + fp) / len(split)
            leak = fp / (tp + fp) if (tp + fp) else float("nan")
            fail_prec = tn / (tn + fn) if (tn + fn) else float("nan")
            print(
                f"{mode:<8}{name:<12}{base:>8.4f}{tp + fp:>7}{leak:>8.4f}"
                f"{tn + fn:>7}{fail_prec:>8.4f}{youden_j(tn, fp, fn, tp):>+9.4f}"
            )


def report_axis_coupling(splits: dict[str, list[RagSample]], preds: dict[str, dict]) -> None:
    section("Axis coupling and joint predictions", "tab:axis-coupling / tab:joint-predictions")
    for name, split in splits.items():
        gold_diff = sum(1 for s in split if s.faithfulness != s.relevance)
        print(f"{name:<12} gold F!=R {gold_diff:>5} / {len(split)} = {gold_diff / len(split):.4f}")
        for mode in ("direct", "marker"):
            diff = sum(
                1
                for s in split
                if preds[mode][s.id].faithfulness_pred != preds[mode][s.id].relevance_pred
            )
            cells = Counter(
                (preds[mode][s.id].faithfulness_pred, preds[mode][s.id].relevance_pred)
                for s in split
            )
            print(
                f"{'':<12} {mode:<7} pred F!=R {diff:>5} = {diff / len(split):.4f}  "
                f"joint {dict(sorted(cells.items()))}"
            )
    print("\nPredictions on the rare gold (F=1, R=0) cell:")
    rare = [s for s in splits["Full"] if s.faithfulness == 1 and s.relevance == 0]
    for mode in ("direct", "marker"):
        cells = Counter(
            (preds[mode][s.id].faithfulness_pred, preds[mode][s.id].relevance_pred) for s in rare
        )
        print(f"  {mode:<7} n={len(rare)}  {dict(sorted(cells.items()))}")


def report_prompt_sensitivity(splits: dict[str, list[RagSample]], preds: dict[str, dict]) -> None:
    section("Prompt-format sensitivity", "sec:appendix-agreement")
    for name, split in splits.items():
        a = [preds["direct"][s.id].reliable_pred for s in split]
        b = [preds["marker"][s.id].reliable_pred for s in split]
        agree = sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(split)
        print(f"  {name:<12} agreement {agree:.4f}  kappa {cohen_kappa_score(a, b):.4f}")


def _bootstrap_indices(split: list[RagSample], reps: int, seed: int) -> np.ndarray:
    """Stratified resampling that preserves the reliable/unreliable class counts."""
    rng = np.random.default_rng(seed)
    idx0 = np.array([i for i, s in enumerate(split) if s.reliable == 0])
    idx1 = np.array([i for i, s in enumerate(split) if s.reliable == 1])
    return np.stack(
        [
            np.concatenate(
                [rng.choice(idx0, idx0.size, replace=True), rng.choice(idx1, idx1.size, replace=True)]
            )
            for _ in range(reps)
        ]
    )


def _macro_f1_np(gold: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((gold == 1) & (pred == 1)))
    fp = int(np.sum((gold == 0) & (pred == 1)))
    fn = int(np.sum((gold == 1) & (pred == 0)))
    tn = int(np.sum((gold == 0) & (pred == 0)))
    return macro_f1_from_counts(tn, fp, fn, tp)


def report_bootstrap(test: list[RagSample], preds: dict[str, dict], reps: int, seed: int) -> None:
    section(
        f"Paired stratified bootstrap on test ({reps} reps, seed {seed})",
        "tab:bootstrap-point / tab:bootstrap-delta",
    )
    boot = _bootstrap_indices(test, reps, seed)
    series: dict[tuple[str, str], np.ndarray] = {}
    for mode in ("direct", "marker"):
        for axis in AXES:
            gold, pred = axis_values(test, preds[mode], axis)
            g, p = np.array(gold), np.array(pred)
            values = np.array([_macro_f1_np(g[b], p[b]) for b in boot])
            series[(mode, axis)] = values
            lo, hi = np.percentile(values, [2.5, 97.5])
            print(
                f"  {mode:<7}{axis:<14} point {_macro_f1_np(g, p):.4f}  CI [{lo:.4f}, {hi:.4f}]"
            )

    gold_rel = np.array([s.reliable for s in test])
    majority = np.ones_like(gold_rel)
    maj = np.array([_macro_f1_np(gold_rel[b], majority[b]) for b in boot])
    lo, hi = np.percentile(maj, [2.5, 97.5])
    print(f"  {'majority':<7}{'reliable':<14} point {_macro_f1_np(gold_rel, majority):.4f} "
          f" CI [{lo:.4f}, {hi:.4f}]")

    print("\n  paired differences on the reliability axis:")
    points = {
        "direct": _macro_f1_np(*(np.array(x) for x in axis_values(test, preds["direct"], "reliable"))),
        "marker": _macro_f1_np(*(np.array(x) for x in axis_values(test, preds["marker"], "reliable"))),
        "majority": _macro_f1_np(gold_rel, majority),
    }
    comparisons = (
        ("direct - majority", "direct", "majority", series[("direct", "reliable")], maj),
        ("marker - majority", "marker", "majority", series[("marker", "reliable")], maj),
        (
            "direct - marker",
            "direct",
            "marker",
            series[("direct", "reliable")],
            series[("marker", "reliable")],
        ),
    )
    for label, first, second, dist_a, dist_b in comparisons:
        delta = dist_a - dist_b
        lo, hi = np.percentile(delta, [2.5, 97.5])
        # The point difference is the estimand; the bootstrap mean is printed
        # beside it so the (small) resampling bias stays visible.
        print(
            f"    {label:<20} point {points[first] - points[second]:+.4f}  "
            f"boot mean {np.mean(delta):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
            f"P(>0) {np.mean(delta > 0):.4f}"
        )


def report_per_marker_recall(full: list[RagSample], preds: dict[str, dict]) -> None:
    section("Per-marker recall of the unreliable class (full corpus)", "tab:per-marker-recall")
    counts = Counter(s.marker for s in full if s.reliable == 0)
    print(f"{'Gold marker':<36}{'n':>5}{'direct':>9}{'rec.':>8}{'marker':>9}{'rec.':>8}")
    for marker, n in counts.most_common():
        rows = [s for s in full if s.reliable == 0 and s.marker == marker]
        caught = {
            mode: sum(1 for s in rows if preds[mode][s.id].reliable_pred == 0)
            for mode in ("direct", "marker")
        }
        print(
            f"{str(marker):<36}{n:>5}{caught['direct']:>9}{caught['direct'] / n:>8.3f}"
            f"{caught['marker']:>9}{caught['marker'] / n:>8.3f}"
        )


def report_marker_quality(splits: dict[str, list[RagSample]], preds: dict[str, dict]) -> None:
    section("Marker vocabulary and metric decomposition", "tab:marker-vocab / tab:marker-decomposition")
    allowed = set(ALLOWED_MARKERS)
    for name, split in splits.items():
        gold = [resolve_marker(s) for s in split]
        pred = [preds["marker"][s.id].marker_pred or "unknown" for s in split]
        oov = [p for p in pred if p not in allowed]
        union = sorted(set(gold) | set(pred))
        gold_only = sorted(set(gold))
        explicit = sorted(set(gold) - {"none", "unknown"})
        explicit_rows = [(g, p) for g, p in zip(gold, pred, strict=True) if g not in ("none", "unknown")]
        hits = sum(1 for g, p in explicit_rows if g == p)
        print(f"\n  {name} (n={len(split)}):")
        print(f"    out-of-vocabulary predictions {len(oov)} = {len(oov) / len(split):.4f}")
        print(f"    OOV strings {dict(Counter(oov).most_common())}")
        print(f"    macro-F1 over gold+pred union   {marker_macro_f1(gold, pred, union):.4f}")
        print(f"    macro-F1 over gold-present      {marker_macro_f1(gold, pred, gold_only):.4f}")
        print(f"    macro-F1 over explicit gold     {marker_macro_f1(gold, pred, explicit):.4f}")
        print(f"    exact match on explicit rows    {hits} / {len(explicit_rows)}")
        family = {"reason_hallucinated_fact", "reason_hallucination", "reason_hallucinated",
                  "hallucination"}
        hall = [(g, p) for g, p in explicit_rows if g == "reason_hallucinated_fact"]
        near = sum(1 for _, p in hall if p in family)
        print(f"    hallucinated_fact -> family     {near} / {len(hall)}")
        if name == "Full":
            print(f"    predicted marker distribution {dict(Counter(pred).most_common())}")


def report_truncation(splits: dict[str, list[RagSample]]) -> None:
    section("SFT truncation mismatch", "tab:truncation")
    print(
        f"{'Split':<12}{'dialog med':>12}{'ctx med':>10}{'ctx p95':>10}"
        f"{'rows trunc':>12}{'ctx lost':>10}"
    )
    for name, split in splits.items():
        q = np.array([len(s.question) for s in split])
        c = np.array([len(s.context) for s in split])
        truncated = np.mean((q > MAX_QUESTION_CHARS) | (c > MAX_CONTEXT_CHARS))
        lost = np.sum(np.clip(c - MAX_CONTEXT_CHARS, 0, None)) / c.sum()
        print(
            f"{name:<12}{np.median(q):>12.0f}{np.median(c):>10.0f}{np.percentile(c, 95):>10.0f}"
            f"{truncated:>12.4f}{lost:>10.4f}"
        )
    print()
    for name in ("Train", "Full"):
        q = np.array([len(s.question) for s in splits[name]])
        lost_q = np.sum(np.clip(q - MAX_QUESTION_CHARS, 0, None)) / q.sum()
        print(f"  {name:<6} rows over the dialogue cap {np.mean(q > MAX_QUESTION_CHARS):.4f}, "
              f"dialogue chars lost {lost_q:.4f}")
    train = splits["Train"]
    direct = np.array([len(build_direct_prompt(s)) for s in train])
    marker = np.array([len(build_marker_prompt(s)) for s in train])
    print(f"  train direct prompt chars: median {np.median(direct):.0f} max {direct.max()}")
    print(f"  train marker prompt chars: median {np.median(marker):.0f} max {marker.max()}")


def report_length_confound(full: list[RagSample], preds: dict[str, dict]) -> None:
    section("Length as a confound", "sec:results-bias prose")
    answer = np.array([len(s.answer) for s in full], dtype=float)
    context = np.array([len(s.context) for s in full], dtype=float)
    for mode in ("direct", "marker"):
        flagged = np.array([1 - preds[mode][s.id].reliable_pred for s in full])
        print(
            f"  {mode:<7} AUC(answer len -> flagged) {roc_auc_score(flagged, answer):.4f}  "
            f"AUC(context len -> flagged) {roc_auc_score(flagged, context):.4f}"
        )
    gold = np.array([1 - s.reliable for s in full])
    print(
        f"  {'gold':<7} AUC(answer len -> unreliable) {roc_auc_score(gold, answer):.4f}  "
        f"AUC(context len -> unreliable) {roc_auc_score(gold, context):.4f}"
    )


def report_lora_reconstruction(test: list[RagSample]) -> None:
    section("Derived LoRA confusion matrices", "tab:lora-reconstruction")
    n = len(test)
    n_pred_faithful = 133  # reported: (1,1) on 133 rows, (0,1) on 92
    reported = {"faithfulness": 0.4669, "reliable": 0.4636}
    recovered: dict[str, tuple[int, int, int, int]] = {}
    for axis, target in reported.items():
        n_gold_pos = sum(getattr(s, axis) for s in test)
        tn, fp, fn, tp = reconstruct_confusion(n, n_gold_pos, n_pred_faithful, target)
        recovered[axis] = (tn, fp, fn, tp)
        print(
            f"  {axis:<14} TN={tn:>3} FP={fp:>3} FN={fn:>3} TP={tp:>3}  "
            f"macro {macro_f1_from_counts(tn, fp, fn, tp):.4f} (reported {target:.4f})"
        )
    n_relevant = sum(s.relevance for s in test)
    tn, fp, fn, tp = 0, n - n_relevant, 0, n_relevant
    print(
        f"  {'relevance':<14} TN={tn:>3} FP={fp:>3} FN={fn:>3} TP={tp:>3}  "
        f"macro {macro_f1_from_counts(tn, fp, fn, tp):.4f} (exact: all-positive)"
    )

    rare = sum(1 for s in test if s.faithfulness == 1 and s.relevance == 0)
    delta = recovered["faithfulness"][3] - recovered["reliable"][3]
    status = "consistent" if delta == rare else "INCONSISTENT"
    print(
        f"\n  consistency check: TP_faith - TP_reliable = {delta}, "
        f"gold (F=1,R=0) rows in test = {rare} -> {status}"
    )

    print("\n  all-(1,1) predictor (unbalanced LoRA, marker LoRA, always-reliable floor):")
    for axis in AXES:
        n_gold_pos = sum(getattr(s, axis) for s in test)
        print(
            f"    {axis:<14} macro "
            f"{macro_f1_from_counts(0, n - n_gold_pos, 0, n_gold_pos):.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/organizers.jsonl", help="Converted RagSample jsonl")
    parser.add_argument(
        "--direct",
        default="results/organizers_qwen_direct_predictions.jsonl",
        help="Method 1 predictions over the full corpus",
    )
    parser.add_argument(
        "--marker",
        default="results/organizers_qwen_marker_predictions.jsonl",
        help="Method 2 predictions over the full corpus",
    )
    parser.add_argument(
        "--raw-archive",
        default="from_organizators/data/data.zip",
        help="Organizer archive, for pre-conversion marker counts",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split and bootstrap seed")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap replicates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.data)
    train, val, test = split_samples(samples, seed=args.seed)
    splits = {"Train": train, "Validation": val, "Test": test, "Full": samples}
    preds = {"direct": load_predictions(args.direct), "marker": load_predictions(args.marker)}

    for mode, by_id in preds.items():
        missing = [s.id for s in samples if s.id not in by_id]
        if missing:
            raise ValueError(f"{mode}: missing {len(missing)} predictions, e.g. {missing[:3]}")

    print(f"corpus {len(samples)}  train {len(train)}  val {len(val)}  test {len(test)}")
    report_splits(splits)
    report_marker_provenance(splits, Path(args.raw_archive))
    report_confusions(splits, preds)
    report_operating_point(splits, preds)
    report_axis_coupling(splits, preds)
    report_prompt_sensitivity(splits, preds)
    report_bootstrap(test, preds, args.bootstrap, args.seed)
    report_per_marker_recall(samples, preds)
    report_marker_quality(splits, preds)
    report_truncation(splits)
    report_length_confound(samples, preds)
    report_lora_reconstruction(test)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
