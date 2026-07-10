"""Render the GEPA evolution stats json into a markdown report (ported from m3-m6)."""

from __future__ import annotations

_EXCERPT = 400  # characters kept from the head and tail of each candidate


def _candidate_text(cand) -> str | None:
    """Candidate instruction text from the detailed stats (dspy format may vary)."""
    if isinstance(cand, dict):
        for v in cand.values():  # {predictor_name: instruction}
            if isinstance(v, str) and v.strip():
                return v
        return None
    return cand if isinstance(cand, str) else None


def _excerpt(text: str) -> str:
    if len(text) <= 2 * _EXCERPT:
        return text
    return (
        text[:_EXCERPT]
        + f"\n…[{len(text) - 2 * _EXCERPT} символов пропущено]…\n"
        + text[-_EXCERPT:]
    )


def render_report(stats: dict) -> str:
    """Markdown report of the prompt evolution: run params, candidate table,
    per-candidate instruction excerpts and the final instruction."""
    variant, seed = stats.get("variant", "?"), stats.get("seed", "?")
    dr = stats.get("detailed_results") or {}
    scores = dr.get("val_aggregate_scores") or []
    candidates = dr.get("candidates") or []
    best_idx = dr.get("best_idx")

    lines = [
        f"# Эволюция GEPA-промпта — variant={variant}, seed={seed}",
        "",
        f"- auto: `{stats.get('auto')}`, train_size: {stats.get('train_size')}, "
        f"val_size: {stats.get('val_size')}",
        f"- use_marker_feedback: {stats.get('use_marker_feedback')}",
        f"- модели: task `{stats.get('task_model')}`, reflection `{stats.get('reflection_model')}`",
        f"- LM-вызовы: task {stats.get('task_lm_calls')}, "
        f"reflection {stats.get('reflection_lm_calls')}",
        f"- git: `{stats.get('git_hash')}`, profile: `{stats.get('profile')}`",
        "",
        "## Кандидаты",
        "",
        "| # | val-score | лучший |",
        "|---|---|---|",
    ]
    for i in range(max(len(scores), len(candidates))):
        sc = f"{scores[i]:.3f}" if i < len(scores) and scores[i] is not None else "—"
        lines.append(f"| {i} | {sc} | {'✅' if best_idx == i else ''} |")

    lines += ["", "## Что менялось в инструкции", ""]
    if candidates:
        for i, cand in enumerate(candidates):
            text = _candidate_text(cand)
            lines.append(f"### Кандидат {i}" + (" (лучший)" if best_idx == i else ""))
            lines.append("")
            lines.append("```" if text else "_текст кандидата недоступен в статистике_")
            if text:
                lines += [_excerpt(text), "```"]
            lines.append("")
    else:
        lines += [
            "_Полные тексты кандидатов недоступны — только скоры выше "
            "и финальная инструкция ниже._",
            "",
        ]

    lines += ["## Финальная инструкция", "", "```", stats.get("best_instruction", ""), "```", ""]
    return "\n".join(lines)
