"""Отчёт об эволюции GEPA: diff кандидатов вместо одинаковых префиксов."""

from __future__ import annotations

from rag_reliability.methods.m3.gepa_report import (
    candidate_text,
    instruction_diff,
    render_report,
)

_SEED = "\n".join(
    [
        "Ты — строгий аудитор фактической опоры ответа.",
        "Проверь, подтверждён ли ответ фрагментами.",
        "Поставь FAIL, если ответ добавляет факты, которых нет в [CTX].",
        "FAITHFULNESS: PASS или FAIL",
    ]
)

_EVOLVED = _SEED.replace(
    "Поставь FAIL, если ответ добавляет факты, которых нет в [CTX].",
    "Поставь FAIL, если ответ добавляет факты, искажает числа или путает продукты.",
)

STATS = {
    "variant": "markers",
    "seed": 0,
    "axis": "faithfulness",
    "auto": "medium",
    "train_size": 300,
    "pareto_size": 300,
    "repeat": 0,
    "fold": 0,
    "held_out_size": 297,
    "use_marker_feedback": True,
    "task_model": "judge-7b",
    "reflection_model": "reflect-72b",
    "task_lm_calls": 1987,
    "reflection_lm_calls": 42,
    "git_hash": "abc1234",
    "seed_instruction": _SEED,
    "best_instruction": _EVOLVED,
    "detailed_results": {
        "val_aggregate_scores": [0.610, 0.684],
        "best_idx": 1,
        "candidates": [{"judge": _SEED}, {"judge": _EVOLVED}],
    },
}


# --------------------------------------------------------------------------- #
# Извлечение текста кандидата
# --------------------------------------------------------------------------- #


def test_dspy_repr_is_not_mistaken_for_an_instruction() -> None:
    """Прошлый дефект: бралось первое строковое значение, а им был ``repr``
    объекта ``Predict(StringSignature(...))`` — одинаковый у всех кандидатов."""
    cand = {
        "predict": "Predict(StringSignature(query, context -> faithfulness))",
        "predict.instruction": "Ты — строгий аудитор.",
    }
    assert candidate_text(cand) == "Ты — строгий аудитор."


def test_instruction_is_recovered_from_a_repr_when_nothing_else_is_there() -> None:
    cand = {
        "predict": "Predict(StringSignature(q -> v, instructions='Ты — аудитор.\\nFAITHFULNESS:'))"
    }
    assert candidate_text(cand) == "Ты — аудитор.\nFAITHFULNESS:"


def test_candidate_text_accepts_a_bare_string_and_rejects_junk() -> None:
    assert candidate_text("Инструкция.") == "Инструкция."
    assert candidate_text({"a": "   "}) is None
    assert candidate_text(42) is None


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #


def test_diff_shows_the_changed_line() -> None:
    diff = instruction_diff(_SEED, _EVOLVED)
    assert "-Поставь FAIL, если ответ добавляет факты, которых нет в [CTX]." in diff
    assert "+Поставь FAIL, если ответ добавляет факты, искажает числа или путает продукты." in diff


def test_identical_texts_diff_to_nothing() -> None:
    assert instruction_diff(_SEED, _SEED) == ""


def test_a_change_in_the_middle_of_a_long_prompt_stays_visible() -> None:
    """Регресс на срез головы и хвоста по 400 символов: он прятал ровно середину,
    из-за чего все кандидаты выглядели одинаковыми."""
    filler = ["Строка наполнителя номер {}.".format(index) for index in range(60)]
    seed = "\n".join(filler)
    changed = list(filler)
    changed[30] = "Строка, которую переписала эволюция."
    candidate = "\n".join(changed)

    report = render_report(
        {
            **STATS,
            "seed_instruction": seed,
            "best_instruction": candidate,
            "detailed_results": {
                "val_aggregate_scores": [0.61, 0.68],
                "best_idx": 1,
                "candidates": [{"judge": seed}, {"judge": candidate}],
            },
        }
    )
    assert "+Строка, которую переписала эволюция." in report
    assert "-Строка наполнителя номер 30." in report


def test_long_diff_is_truncated_with_a_notice_not_silently() -> None:
    seed = "\n".join(f"строка {index}" for index in range(500))
    candidate = "\n".join(f"иная строка {index}" for index in range(500))
    diff = instruction_diff(seed, candidate, max_lines=20)
    assert len(diff.splitlines()) == 21
    assert "полный текст в stats.json" in diff


# --------------------------------------------------------------------------- #
# Отчёт целиком
# --------------------------------------------------------------------------- #


def test_report_header_scores_and_deltas() -> None:
    report = render_report(STATS)
    assert "variant=markers, seed=0" in report
    assert "ось: `faithfulness`" in report
    assert "D_pareto: 300" in report
    assert "held-out: 297 кейсов" in report
    assert "| 0 | 0.610 | +0.000 |  |" in report
    assert "| 1 | 0.684 | +0.074 | ✅ |" in report


def test_report_diffs_candidates_against_the_seed() -> None:
    report = render_report(STATS)
    assert "Кандидат 0" in report
    assert "_совпадает с сидом_" in report  # нулевой кандидат — это и есть сид
    assert "Кандидат 1 (лучший)" in report
    assert "+Поставь FAIL, если ответ добавляет факты, искажает числа" in report
    assert "```diff" in report


def test_report_surfaces_prompt_defects() -> None:
    report = render_report({**STATS, "prompt_defects": ["посторонние домены: хаббл"]})
    assert "Санитария промпта" in report
    assert "хаббл" in report


def test_report_without_candidates_falls_back_to_the_final_instruction() -> None:
    report = render_report({**STATS, "detailed_results": {}})
    assert "Полные тексты кандидатов недоступны" in report
    assert "Ты — строгий аудитор фактической опоры ответа." in report
