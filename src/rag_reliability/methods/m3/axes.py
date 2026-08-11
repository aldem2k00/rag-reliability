"""Одноосевые промпты Метода 3: faithfulness и relevance в двух отдельных вызовах.

Зачем разводить оси. В объединённом промпте судья переносит аргумент одной оси в
вердикт другой — прямая улика в артефактах
(``predictions/pseudo_debug/m3/gepa_plain_s0/val.jsonl``, кейс ``pseudo_00059``:
«**FAITHFULNESS:** Ответ не добавляет новых фактов, но не полностью отвечает на
вопрос клиента»). Побочная выгода: оси relevance не нужны чанки, её проход
стоит ~600 prompt-токенов вместо ~3000.

Текст промптов живёт в ``configs/prompts/{faithfulness,relevance}.yaml``, а не в
коде: так GEPA (D2) может эволюционировать промпт, не трогая модуль, а версия
промпта попадает в ``run.yaml`` и в предсказания.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from rag_reliability.schema import RagSample

# Единственная точка связи с logprobs.py (владение A4). Публичный
# extract_verdict_probs требует ОБЕ оси в одном ответе и потому не годится для
# одноосевого вызова; здесь переиспользуются те же примитивы, чтобы одноосевой
# путь не разошёлся с двухосевым. Публичная axis-функция запрошена у A4 в PR.
from rag_reliability.methods.m3.logprobs import (
    _anchor_bounds,
    _pass_prob,
    _verdict_position_between,
)

AXIS_FAITHFULNESS = "faithfulness"
AXIS_RELEVANCE = "relevance"
AXES: tuple[str, str] = (AXIS_FAITHFULNESS, AXIS_RELEVANCE)

MODES: tuple[str, ...] = ("zero_shot", "few_shot", "gepa")

DEFAULT_PROMPTS_DIR = Path("configs/prompts")
DEFAULT_MARKERS_PATH = Path("configs/markers.yaml")

_ANCHORS = {AXIS_FAITHFULNESS: "FAITHFULNESS", AXIS_RELEVANCE: "RELEVANCE"}
_MARKER_CHECKLIST_PLACEHOLDER = "{marker_checklist}"
_NO_MARKER = "none"

# Вердикт ищем последним вхождением: в ANALYSIS модель может процитировать ось.
_MARKER_RE = re.compile(r"^[-*\s>]*\**\s*MARKER\s*:?\**\s*([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

_FEW_SHOT_VERDICT_KEYS = {AXIS_FAITHFULNESS: "faith", AXIS_RELEVANCE: "rel"}


def axis_anchor(axis: str) -> str:
    """Якорь вердикта в выводе модели; ``logprobs`` ищет позицию именно по нему."""
    if axis not in _ANCHORS:
        raise ValueError(f"Unknown axis {axis!r}, expected one of {AXES}")
    return _ANCHORS[axis]


@dataclass(frozen=True)
class AxisPromptSpec:
    """Загруженный YAML-промпт одной оси.

    ``system`` хранится уже с подставленным чеклистом маркеров: рендер зависит
    только от файлов конфигурации, поэтому его дешевле сделать один раз при
    загрузке, чем на каждом кейсе.
    """

    axis: str
    version: str
    system: str
    user_template: str
    needs_context: bool
    markers: tuple[str, ...]
    examples: tuple[dict[str, str], ...]


def _render(template: str, values: Mapping[str, str]) -> str:
    """Подстановка через replace, а не str.format.

    Ответы бота и чанки содержат фигурные скобки (JSON, шаблоны сообщений), и
    эволюционировавший GEPA-промпт тоже может их содержать — str.format на таком
    тексте падает с KeyError.
    """
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def load_markers(markers_path: str | Path | None = None) -> dict[str, str]:
    """Коды маркеров кураторов и их глоссы из ``configs/markers.yaml``."""
    path = Path(markers_path) if markers_path is not None else DEFAULT_MARKERS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Markers taxonomy not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Markers file {path} must be a non-empty mapping code -> gloss")
    return {str(code): str(gloss) for code, gloss in payload.items()}


def build_marker_checklist(codes: Sequence[str], glosses: Mapping[str, str]) -> str:
    """Чеклист criteria injection: коды маркеров как строки промпта.

    Судья обязан назвать код, поэтому ``marker_pred`` перестаёт быть пустым и
    появляется вход в per-marker recall.
    """
    missing = [code for code in codes if code not in glosses]
    if missing:
        raise ValueError(
            f"Marker codes are absent from the taxonomy: {missing[:5]} "
            f"(known codes: {sorted(glosses)[:5]}...)"
        )
    lines = [
        "Проверь по списку. Если сработал хотя бы один пункт — FAIL, и назови его код в MARKER.",
        *[f"  - {code}: {glosses[code]}" for code in codes],
        f"  - {_NO_MARKER}: ни один пункт не сработал.",
    ]
    return "\n".join(lines)


def _spec_from_payload(
    payload: Mapping[str, object],
    axis: str,
    path: Path,
    markers_path: str | Path | None,
) -> AxisPromptSpec:
    required = ("version", "axis", "system", "user_template", "needs_context", "markers")
    absent = [key for key in required if key not in payload]
    if absent:
        raise ValueError(f"Axis prompt {path} is missing required key(s): {absent}")
    if payload["axis"] != axis:
        raise ValueError(f"Axis prompt {path} declares axis={payload['axis']!r}, expected {axis!r}")
    examples = payload["examples"] if "examples" in payload else []
    if not isinstance(examples, list):
        raise ValueError(f"Axis prompt {path}: 'examples' must be a list, got {type(examples)}")
    return AxisPromptSpec(
        axis=axis,
        version=str(payload["version"]),
        system=str(payload["system"]).strip(),
        user_template=str(payload["user_template"]),
        needs_context=bool(payload["needs_context"]),
        markers=_marker_codes(payload["markers"], path, markers_path),
        examples=tuple(dict(example) for example in examples),
    )


def _marker_codes(
    declared: object, path: Path, markers_path: str | Path | None
) -> tuple[str, ...]:
    if declared == "all":
        return tuple(load_markers(markers_path))
    if isinstance(declared, list):
        return tuple(str(code) for code in declared)
    raise ValueError(
        f"Axis prompt {path}: 'markers' must be a list of codes or 'all', got {declared!r}"
    )


@lru_cache(maxsize=None)
def _load_axis_prompt_cached(
    axis: str, prompts_dir: str, markers_path: str | None
) -> AxisPromptSpec:
    path = Path(prompts_dir) / f"{axis}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Axis prompt not found: {path}. Expected configs/prompts/{axis}.yaml "
            "(see docs/specs/30_PHASE2_метод3.md §1.2)"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Axis prompt {path} must be a YAML mapping")
    spec = _spec_from_payload(payload, axis, path, markers_path)
    checklist = build_marker_checklist(spec.markers, load_markers(markers_path))
    if _MARKER_CHECKLIST_PLACEHOLDER not in spec.system:
        raise ValueError(
            f"Axis prompt {path} has no {_MARKER_CHECKLIST_PLACEHOLDER} placeholder: "
            "criteria injection would silently do nothing"
        )
    system = spec.system.replace(_MARKER_CHECKLIST_PLACEHOLDER, checklist)
    if axis_anchor(axis) not in system:
        raise ValueError(
            f"Axis prompt {path} never asks for the {axis_anchor(axis)}: line — "
            "the logprobs anchor would be missing from every answer"
        )
    return AxisPromptSpec(
        axis=spec.axis,
        version=spec.version,
        system=system,
        user_template=spec.user_template,
        needs_context=spec.needs_context,
        markers=spec.markers,
        examples=spec.examples,
    )


def load_axis_prompt(
    axis: str,
    *,
    prompts_dir: str | Path | None = None,
    markers_path: str | Path | None = None,
) -> AxisPromptSpec:
    """YAML-промпт оси с уже подставленным чеклистом маркеров."""
    if axis not in AXES:
        raise ValueError(f"Unknown axis {axis!r}, expected one of {AXES}")
    directory = str(prompts_dir) if prompts_dir is not None else str(DEFAULT_PROMPTS_DIR)
    markers = str(markers_path) if markers_path is not None else None
    return _load_axis_prompt_cached(axis, directory, markers)


def prompt_versions(
    *, prompts_dir: str | Path | None = None, markers_path: str | Path | None = None
) -> dict[str, str]:
    """Версии обоих промптов — для ``run.yaml`` и для строки предсказания."""
    return {
        axis: load_axis_prompt(axis, prompts_dir=prompts_dir, markers_path=markers_path).version
        for axis in AXES
    }


def _example_verdict(example: Mapping[str, str], axis: str) -> str:
    """PASS/FAIL примера для нужной оси.

    Поддержаны и одноосевой ключ ``verdict``, и формат configs/few_shot.yaml
    (``faith``/``rel``) — чтобы семь размеченных вручную примеров не переписывать.
    """
    if "verdict" in example:
        return str(example["verdict"]).strip().upper()
    key = _FEW_SHOT_VERDICT_KEYS[axis]
    if key not in example:
        raise ValueError(
            f"Few-shot example for axis {axis!r} has neither 'verdict' nor {key!r}: "
            f"keys={sorted(example)}"
        )
    return str(example[key]).strip().upper()


def _example_block(example: Mapping[str, str], index: int, spec: AxisPromptSpec) -> str:
    absent = [key for key in ("q", "a", "analysis") if key not in example]
    if absent:
        raise ValueError(f"Few-shot example {index} is missing key(s) {absent}")
    marker = str(example["marker"]) if "marker" in example else _NO_MARKER
    lines = [f"\n\nПример {index}.", f"[Q] {example['q']}"]
    if spec.needs_context:
        if "ctx" not in example:
            raise ValueError(f"Few-shot example {index} for axis {spec.axis} is missing 'ctx'")
        lines.append(f"[CTX] {example['ctx']}")
    lines += [
        f"[A] {example['a']}",
        f"ANALYSIS: {example['analysis']}",
        f"MARKER: {marker}",
        f"{axis_anchor(spec.axis)}: {_example_verdict(example, spec.axis)}",
    ]
    return "\n".join(lines)


def load_examples(examples: Sequence[Mapping[str, str]] | str | Path) -> list[dict[str, str]]:
    """Примеры few-shot: список как есть либо YAML-файл с ключом ``examples``."""
    if isinstance(examples, str | Path):
        path = Path(examples)
        if not path.exists():
            raise FileNotFoundError(f"Few-shot examples file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "examples" not in payload:
            raise ValueError(f"Few-shot file {path} must be a mapping with an 'examples' key")
        return [dict(example) for example in payload["examples"]]
    return [dict(example) for example in examples]


def build_axis_system(
    axis: str,
    *,
    mode: str = "zero_shot",
    examples: Sequence[Mapping[str, str]] | str | Path | None = None,
    spec: AxisPromptSpec | None = None,
    prompts_dir: str | Path | None = None,
    prompt_file: str | Path | None = None,
) -> str:
    """System-инструкция одной оси для одной из ступеней zero_shot/few_shot/gepa."""
    if mode not in MODES:
        raise ValueError(f"Unknown Method 3 mode: {mode!r}, expected one of {MODES}")
    if mode == "gepa":
        if prompt_file is None:
            raise ValueError(f"gepa mode requires prompt_file for axis {axis!r}")
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(
                f"GEPA prompt file not found for axis {axis!r}: {path}. "
                "Generate one with scripts/run_gepa.py or drop the --prompt-file flag."
            )
        evolved = path.read_text(encoding="utf-8").strip()
        if axis_anchor(axis) not in evolved:
            raise ValueError(
                f"GEPA prompt {path} never asks for the {axis_anchor(axis)}: line — "
                "verdict logprobs would have no anchor"
            )
        return evolved

    resolved = spec if spec is not None else load_axis_prompt(axis, prompts_dir=prompts_dir)
    if mode == "zero_shot":
        return resolved.system
    payload = load_examples(examples) if examples is not None else list(resolved.examples)
    if not payload:
        raise ValueError(
            f"few_shot mode for axis {axis!r} got no examples: pass --examples or fill "
            f"'examples' in the axis YAML"
        )
    blocks = [
        _example_block(example, index + 1, resolved) for index, example in enumerate(payload)
    ]
    return resolved.system + "".join(blocks)


def build_axis_prompt(
    sample: RagSample,
    axis: str,
    *,
    mode: str = "zero_shot",
    examples: Sequence[Mapping[str, str]] | str | Path | None = None,
    spec: AxisPromptSpec | None = None,
    prompts_dir: str | Path | None = None,
    prompt_file: str | Path | None = None,
    max_context_chars: int | None = None,
) -> tuple[str, str]:
    """(system, user) для одной оси. Оси не видят критериев друг друга.

    Для relevance ``[CTX]`` не подставляется вовсе: определение оси опирается
    только на диалог, вопрос и ответ.
    """
    resolved = spec if spec is not None else load_axis_prompt(axis, prompts_dir=prompts_dir)
    system = build_axis_system(
        axis,
        mode=mode,
        examples=examples,
        spec=resolved,
        prompts_dir=prompts_dir,
        prompt_file=prompt_file,
    )
    values = {"question": sample.question, "answer": sample.answer}
    if resolved.needs_context:
        context = sample.context
        if max_context_chars is not None and len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[контекст усечён]"
        values["context"] = context
    user = _render(resolved.user_template, values).strip()
    return system, user


def parse_axis_verdict(text: str, axis: str) -> int | None:
    """1/0 по последнему вхождению ``AXIS: PASS|FAIL``; None — вердикта нет.

    Последнее вхождение, а не первое: ANALYSIS может процитировать ось, и
    ровно так же выбирает якорь ``logprobs._anchor_bounds``.
    """
    pattern = re.compile(
        rf"[-*\s>]*\**\s*{axis_anchor(axis)}\s*:?\**\s*(PASS|FAIL)", re.IGNORECASE
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(1).upper() == "PASS")


def parse_marker(text: str) -> str | None:
    """Код маркера из строки ``MARKER:``; None — строки нет."""
    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def axis_pass_prob(tokens: Sequence[dict], axis: str) -> float | None:
    """P(PASS) по логпробам на позиции вердикта одной оси; None — якоря нет."""
    token_list = list(tokens)
    if not token_list:
        return None
    bounds = _anchor_bounds(token_list, axis_anchor(axis))
    if bounds is None:
        return None
    position = _verdict_position_between(token_list, bounds[1], len(token_list))
    if position is None:
        return None
    return _pass_prob(token_list[position])


def extract_axis_verdict(
    text: str,
    tokens: Sequence[dict],
    axis: str,
    *,
    finish_reason: str | None = None,
) -> tuple[float, dict]:
    """Цепочка logprobs -> regex (0.9/0.1) -> 0.5. Кейс не может быть потерян.

    Та же цепочка и те же значения, что в ``judge_client._judge_verdict``, но
    для одной оси: одноосевой ответ не проходит двухосевой парсер.
    """
    verdict = parse_axis_verdict(text, axis)
    meta: dict = {
        "axis": axis,
        "raw": text[-400:],
        "truncated": finish_reason == "length",
        "marker": parse_marker(text),
        # Дискретный вердикт сэмпла: из него считается доля голосов PASS.
        "verdict": verdict,
    }
    probability = axis_pass_prob(tokens, axis)
    if probability is not None:
        return probability, {"method": "logprobs", **meta}
    if verdict is not None:
        return (0.9 if verdict == 1 else 0.1), {"method": "regex", **meta}
    return 0.5, {"method": "default", **meta}
