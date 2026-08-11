"""Tests for the Methods 1-2 diagnostics arithmetic."""

import importlib.util
from pathlib import Path

from sklearn.metrics import f1_score

_SPEC = importlib.util.spec_from_file_location(
    "analyze_prompted_judges",
    Path(__file__).parents[1] / "scripts" / "analyze_prompted_judges.py",
)
assert _SPEC is not None and _SPEC.loader is not None
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)


def test_macro_f1_from_counts_matches_sklearn() -> None:
    gold = [1] * 163 + [0] * 62
    pred = [1] * 144 + [0] * 19 + [1] * 53 + [0] * 9
    tn, fp, fn, tp = analyze.confusion(gold, pred)
    assert (tn, fp, fn, tp) == (9, 53, 19, 144)
    expected = f1_score(gold, pred, average="macro", zero_division=0)
    assert abs(analyze.macro_f1_from_counts(tn, fp, fn, tp) - expected) < 1e-12


def test_macro_f1_of_all_positive_predictor_is_the_reported_floor() -> None:
    # 162 of 225 canonical test rows are reliable.
    assert round(analyze.macro_f1_from_counts(0, 63, 0, 162), 4) == 0.4186
    # 1622 of 2245 full-corpus rows are reliable.
    assert round(analyze.macro_f1_from_counts(0, 623, 0, 1622), 4) == 0.4194


def test_reconstruct_confusion_recovers_the_balanced_lora_matrices() -> None:
    # 225 test rows, 163 gold faithful / 162 gold reliable, 133 predicted positive.
    faith = analyze.reconstruct_confusion(225, 163, 133, 0.4669)
    reliable = analyze.reconstruct_confusion(225, 162, 133, 0.4636)
    assert faith == (23, 39, 69, 94)
    assert reliable == (23, 40, 69, 93)
    # Predicted relevance is 1 everywhere, so the two true-positive counts differ
    # by exactly the number of gold (faithful, irrelevant) test rows, which is 1.
    assert faith[3] - reliable[3] == 1


def test_youden_j_is_zero_for_a_constant_predictor() -> None:
    assert analyze.youden_j(0, 63, 0, 162) == 0.0


def test_marker_macro_f1_drops_when_invented_labels_join_the_average() -> None:
    gold = ["none", "none", "reason_hallucinated_fact", "reason_off_topic_answer"]
    pred = ["none", "none", "reason_hallucination", "reason_off_topic_answer"]
    gold_only = sorted(set(gold))
    union = sorted(set(gold) | set(pred))
    over_gold = analyze.marker_macro_f1(gold, pred, gold_only)
    over_union = analyze.marker_macro_f1(gold, pred, union)
    assert over_union < over_gold
    # The invented string enters the averaging set as a guaranteed-zero class.
    assert len(union) == len(gold_only) + 1
