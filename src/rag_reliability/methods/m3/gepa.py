"""GEPA prompt evolution for the Method 3 judge (ported from m3-m6).

DSPy is used ONLY for optimization; inference stays in ``scripts/run_m3.py``
(``--mode gepa --prompt-file ...``). The markers/plain variants differ
EXCLUSIVELY in whether curator marker glosses are appended to the metric
feedback — their gap at equal budget and seed tests the marker-feedback
hypothesis (H5 in the original project docs).

Pure helpers (score/feedback, subsampling, gloss loading) work without dspy
and are unit-tested; dspy is imported lazily inside the functions that need it
(install with the ``gepa`` extra).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

from rag_reliability.schema import RagSample

# Gold marker values that mean "no curator marker on this sample".
_NO_MARKER = (None, "", "none", "unknown")


def has_marker(sample: RagSample) -> bool:
    return sample.marker not in _NO_MARKER


def verdict(value: int) -> str:
    return "PASS" if value == 1 else "FAIL"


def load_marker_gloss(path: str | Path) -> dict[str, str]:
    """Marker glosses from configs/markers.yaml (NOT hardcoded: on the real corpus
    the file is replaced by the curators' dictionary without code changes)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items()}


def score_and_feedback(
    gold_faith: str,
    gold_rel: str,
    pred_faith: str,
    pred_rel: str,
    *,
    marker: str | None = None,
    use_markers: bool = False,
    gloss: dict[str, str] | None = None,
) -> tuple[float, str]:
    """GEPA metric core: score = (ok_faith + ok_rel) / 2; on a mistake the feedback
    carries the gold labels, in the markers variant extended with the sample's
    curator-marker gloss. Feedback is in Russian — it is read by the reflection LM
    evolving a Russian judge prompt."""
    ok_f = pred_faith.strip().upper() == gold_faith
    ok_r = pred_rel.strip().upper() == gold_rel
    score = (int(ok_f) + int(ok_r)) / 2
    if ok_f and ok_r:
        return score, "Обе оценки верны."
    feedback = (
        f"Ошибка. Правильный ответ: FAITHFULNESS={gold_faith}, RELEVANCE={gold_rel}."
    )
    if use_markers and marker not in _NO_MARKER:
        gloss = gloss or {}
        line = f"- {marker}: {gloss[marker]}" if marker in gloss else f"- {marker}"
        feedback += "\nМаркер ошибки от кураторов:\n" + line
    return score, feedback


def subsample_train(
    samples: list[RagSample],
    train_size: int,
    marker_share: float = 0.0,
    seed: int = 0,
) -> list[RagSample]:
    """Deterministic train subsample. Markers are sparse on the real corpus, so
    their share is raised to ``marker_share`` — the same policy for the markers
    and plain variants keeps the H5 comparison fair."""
    if train_size >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    marked = [s for s in samples if has_marker(s)]
    if marker_share > 0 and marked:
        n_marked = min(len(marked), int(round(train_size * marker_share)))
        rest = [s for s in samples if not has_marker(s)]
        picked = rng.sample(marked, n_marked) + rng.sample(rest, train_size - n_marked)
        rng.shuffle(picked)
        return picked
    return rng.sample(samples, train_size)


def make_metric(use_markers: bool, gloss: dict[str, str]):
    """Wrap score_and_feedback into the dspy.GEPA metric signature."""
    import dspy  # noqa: PLC0415

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):  # noqa: ARG001
        score, feedback = score_and_feedback(
            str(getattr(gold, "faithfulness", "")),
            str(getattr(gold, "relevance", "")),
            str(getattr(pred, "faithfulness", "")),
            str(getattr(pred, "relevance", "")),
            marker=getattr(gold, "marker", None),
            use_markers=use_markers,
            gloss=gloss,
        )
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def build_program():
    """ChainOfThought judge; the instruction is the CURRENT SEED_INSTRUCTION
    (already contains the axis-independence rule)."""
    import dspy  # noqa: PLC0415
    from typing import Literal  # noqa: PLC0415

    from rag_reliability.methods.m3.prompts import SEED_INSTRUCTION  # noqa: PLC0415

    class Judge(dspy.Signature):
        """(the instruction is substituted via with_instructions below)"""

        query: str = dspy.InputField(desc="вопрос клиента (с историей диалога)")
        context: str = dspy.InputField(desc="фрагменты документации")
        answer: str = dspy.InputField(desc="ответ ассистента")
        faithfulness: Literal["PASS", "FAIL"] = dspy.OutputField()
        relevance: Literal["PASS", "FAIL"] = dspy.OutputField()

    return dspy.ChainOfThought(Judge.with_instructions(SEED_INSTRUCTION))


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n[контекст усечён]"
    return text


def build_examples(samples: list[RagSample], max_context_chars: int | None = None) -> list:
    """dspy.Example from RagSample: the same fields the judge sees at inference."""
    import dspy  # noqa: PLC0415

    return [
        dspy.Example(
            query=s.question,
            context=_truncate(s.context, max_context_chars),
            answer=s.answer,
            faithfulness=verdict(s.faithfulness),
            relevance=verdict(s.relevance),
            marker=s.marker,
        ).with_inputs("query", "context", "answer")
        for s in samples
    ]


def extract_instruction(program) -> str:
    """Instruction of the single predictor of the optimized program."""
    _, predictor = next(iter(program.named_predictors()))
    return predictor.signature.instructions


def serialize_detailed(dr: Any) -> dict:
    """track_stats -> json-compatible digest (candidates, val scores, counters)."""
    if dr is None:
        return {}
    try:
        return json.loads(json.dumps(dr.to_dict(), default=str, ensure_ascii=False))
    except Exception:
        return {
            "val_aggregate_scores": getattr(dr, "val_aggregate_scores", None),
            "best_idx": getattr(dr, "best_idx", None),
            "total_metric_calls": getattr(dr, "total_metric_calls", None),
            "candidates": [
                {k: str(v) for k, v in c.items()} if isinstance(c, dict) else str(c)
                for c in (getattr(dr, "candidates", None) or [])
            ],
        }
