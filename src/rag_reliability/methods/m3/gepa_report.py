"""Отчёт об эволюции GEPA: что именно изменилось в инструкции.

Прошлая версия печатала первые и последние 400 символов каждого кандидата. Два
дефекта складывались в бесполезный отчёт: ``_candidate_text`` возвращала первое
строковое значение словаря кандидата, которым оказывался ``repr`` объекта
``Predict(StringSignature(...))``, а срез головы и хвоста прятал изменившуюся
середину. В результате у всех кандидатов первые 400 символов были идентичны, и
по отчёту нельзя было понять, менялось ли хоть что-то.

Здесь кандидат сравнивается с сидом построчным diff'ом: видно ровно то, что
эволюция переписала.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from typing import Any

#: Сколько строк diff'а показывать на кандидата. Полный текст инструкции лежит
#: рядом в stats.json; отчёт читают глазами.
MAX_DIFF_LINES = 120

#: Контекст unified diff: одна строка вокруг изменения.
DIFF_CONTEXT = 1

#: ``repr`` DSPy-объектов: это не текст инструкции, а служебная строка.
_DSPY_REPR = re.compile(r"^\s*(Predict|ChainOfThought|ProgramOfThought|StringSignature)\s*\(")

#: Инструкция внутри ``repr(StringSignature(...))`` — последняя надежда, если
#: словарь кандидата не содержит ничего, кроме repr'а.
_INSTRUCTIONS_IN_REPR = re.compile(r"instructions=(['\"])(?P<text>.*?)(?<!\\)\1", re.DOTALL)


def _unescape(text: str) -> str:
    return text.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')


def candidate_text(cand: Any) -> str | None:
    """Текст инструкции кандидата из статистики DSPy.

    Формат ``detailed_results.candidates`` менялся между версиями dspy, поэтому
    разбираются оба: словарь ``{predictor_name: instruction}`` и голая строка.
    ``repr`` DSPy-объекта инструкцией НЕ считается — прошлая версия принимала
    именно его, и все кандидаты выглядели одинаково.
    """
    values: Sequence[Any]
    if isinstance(cand, dict):
        values = list(cand.values())
    elif isinstance(cand, str):
        values = [cand]
    else:
        return None

    fallback: str | None = None
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        if _DSPY_REPR.match(value):
            match = _INSTRUCTIONS_IN_REPR.search(value)
            if match is not None and fallback is None:
                fallback = _unescape(match.group("text"))
            continue
        return value
    return fallback


def instruction_diff(seed: str, candidate: str, *, max_lines: int = MAX_DIFF_LINES) -> str:
    """Построчный unified diff «сид → кандидат».

    Пустая строка означает «текст совпал с сидом»: это тоже результат прогона, и
    его надо видеть, а не выводить из одинаковых префиксов.
    """
    lines = list(
        difflib.unified_diff(
            seed.splitlines(),
            candidate.splitlines(),
            fromfile="seed",
            tofile="candidate",
            lineterm="",
            n=DIFF_CONTEXT,
        )
    )
    if not lines:
        return ""
    if len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = lines[:max_lines] + [f"… ещё {hidden} строк(и) diff'а, полный текст в stats.json"]
    return "\n".join(lines)


def _seed_text(stats: dict, candidates: Sequence[Any]) -> str:
    """Сид эволюции: явное поле, иначе нулевой кандидат."""
    seed = stats.get("seed_instruction")
    if isinstance(seed, str) and seed.strip():
        return seed
    if candidates:
        return candidate_text(candidates[0]) or ""
    return ""


def _score_cell(scores: Sequence[Any], index: int) -> str:
    if index >= len(scores) or scores[index] is None:
        return "—"
    return f"{float(scores[index]):.3f}"


def _delta_cell(scores: Sequence[Any], index: int) -> str:
    if index >= len(scores) or not scores or scores[index] is None or scores[0] is None:
        return "—"
    return f"{float(scores[index]) - float(scores[0]):+.3f}"


def render_report(stats: dict) -> str:
    """Markdown-отчёт: параметры прогона, таблица кандидатов, diff'ы, финал."""
    variant, seed_value = stats.get("variant", "?"), stats.get("seed", "?")
    dr = stats.get("detailed_results") or {}
    scores = dr.get("val_aggregate_scores") or []
    candidates = dr.get("candidates") or []
    best_idx = dr.get("best_idx")
    seed_text = _seed_text(stats, candidates)

    lines = [
        f"# Эволюция GEPA-промпта — variant={variant}, seed={seed_value}",
        "",
        f"- ось: `{stats.get('axis')}`, метрика: `{stats.get('metric', 'balanced accuracy')}`",
        f"- auto: `{stats.get('auto')}`, train_size: {stats.get('train_size')}, "
        f"D_pareto: {stats.get('pareto_size')}",
        f"- фолд: repeat {stats.get('repeat')}, fold {stats.get('fold')} "
        f"(held-out: {stats.get('held_out_size')} кейсов, в оптимизацию не попадают)",
        f"- use_marker_feedback: {stats.get('use_marker_feedback')}",
        f"- модели: task `{stats.get('task_model')}`, reflection `{stats.get('reflection_model')}`",
        f"- LM-вызовы: task {stats.get('task_lm_calls')}, "
        f"reflection {stats.get('reflection_lm_calls')}",
        f"- git: `{stats.get('git_hash')}`",
        "",
    ]

    defects = stats.get("prompt_defects") or []
    if defects:
        lines += ["> **Санитария промпта:**", ""]
        lines += [f"> - {defect}" for defect in defects]
        lines.append("")

    lines += [
        "## Кандидаты",
        "",
        "| # | val-score | Δ к сиду | лучший |",
        "|---|---|---|---|",
    ]
    for index in range(max(len(scores), len(candidates))):
        best = "✅" if best_idx == index else ""
        lines.append(
            f"| {index} | {_score_cell(scores, index)} | {_delta_cell(scores, index)} | {best} |"
        )

    lines += ["", "## Что менялось в инструкции", ""]
    if candidates:
        lines += [
            "Построчный diff кандидата против сида. Пустой diff — эволюция текст не тронула.",
            "",
        ]
        for index, cand in enumerate(candidates):
            text = candidate_text(cand)
            lines.append(f"### Кандидат {index}" + (" (лучший)" if best_idx == index else ""))
            lines.append("")
            if text is None:
                lines += ["_текст кандидата недоступен в статистике_", ""]
                continue
            diff = instruction_diff(seed_text, text)
            if not diff:
                lines += ["_совпадает с сидом_", ""]
                continue
            lines += ["```diff", diff, "```", ""]
    else:
        lines += [
            "_Полные тексты кандидатов недоступны — только скоры выше и финальная "
            "инструкция ниже._",
            "",
        ]

    best_instruction = stats.get("best_instruction", "")
    lines += ["## Финальная инструкция против сида", ""]
    final_diff = instruction_diff(seed_text, best_instruction)
    if final_diff:
        lines += ["```diff", final_diff, "```", ""]
    else:
        lines += ["_совпадает с сидом_", ""]
    lines += ["## Финальная инструкция", "", "```", best_instruction, "```", ""]
    return "\n".join(lines)
