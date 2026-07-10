"""GEPA evolution report rendering from a stats dict."""

from rag_reliability.methods.m3.gepa_report import render_report

STATS = {
    "variant": "markers",
    "seed": 0,
    "auto": "light",
    "train_size": 100,
    "val_size": 50,
    "use_marker_feedback": True,
    "task_model": "judge-7b",
    "reflection_model": "reflect-72b",
    "task_lm_calls": 321,
    "reflection_lm_calls": 12,
    "profile": "local",
    "git_hash": "abc1234",
    "best_instruction": "Ты — строгий аудитор.",
    "detailed_results": {
        "val_aggregate_scores": [0.61, 0.68],
        "best_idx": 1,
        "candidates": [
            {"judge": "Первая инструкция."},
            {"judge": "Вторая инструкция." + "x" * 900},
        ],
    },
}


def test_report_contains_header_scores_and_final_instruction() -> None:
    report = render_report(STATS)
    assert "variant=markers, seed=0" in report
    assert "| 0 | 0.610 |" in report
    assert "| 1 | 0.680 | ✅ |" in report
    assert "Кандидат 1 (лучший)" in report
    assert "Первая инструкция." in report
    assert "символов пропущено" in report  # long candidate is excerpted
    assert "Ты — строгий аудитор." in report


def test_report_without_candidates_falls_back() -> None:
    stats = {**STATS, "detailed_results": {}}
    report = render_report(stats)
    assert "Полные тексты кандидатов недоступны" in report
    assert "Ты — строгий аудитор." in report
