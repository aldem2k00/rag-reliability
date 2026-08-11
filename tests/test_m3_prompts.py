"""YAML-промпты осей: содержание критериев, версии, покрытие таксономии.

Тесты сверяют промпт с определением организаторов (from_organizators/readme.md),
а не с тем, что получилось: критерии relevance перечислены поимённо, оговорка о
неполноте обязана присутствовать, faithfulness обязан упоминать все 13 кодов.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_reliability.methods.m3.axes import (
    AXES,
    AXIS_FAITHFULNESS,
    AXIS_RELEVANCE,
    build_axis_prompt,
    build_marker_checklist,
    load_axis_prompt,
    load_markers,
    prompt_versions,
)
from rag_reliability.schema import RagSample

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "configs" / "prompts"
MARKERS_PATH = REPO_ROOT / "configs" / "markers.yaml"

# Критерии FAIL из from_organizators/readme.md (Binary Relevancy, Score 0)
# плюс два, вытекающих из примеров 2-3 того же раздела.
RELEVANCE_FAIL_CRITERIA = (
    "не адресует вопрос клиента",
    "отвечает на другой, пусть и смежный, вопрос",
    "информацию, предназначенную оператору",
    "уточняющий вопрос вместо ответа",
    "перенаправляет клиента в статью",
    "покрывает не все части вопроса",
)


def _spec(axis: str):
    return load_axis_prompt(axis, prompts_dir=PROMPTS_DIR, markers_path=MARKERS_PATH)


def _sample() -> RagSample:
    return RagSample(
        id="s1",
        question="Клиент: Какие проценты по вкладу «Максимальный»?",
        context="[CHUNK 1] СЕКРЕТНЫЙ_ТЕКСТ_ЧАНКА: ставка 19.84% на 62 дня.",
        answer="На 62 дня — 19.84% с капитализацией.",
        faithfulness=1,
        relevance=1,
        marker="none",
    )


@pytest.mark.parametrize("axis", AXES)
def test_axis_prompt_loads_with_version_and_anchor(axis: str) -> None:
    spec = _spec(axis)

    assert spec.axis == axis
    assert spec.version.startswith(f"{axis}/")
    assert spec.system
    assert "{marker_checklist}" not in spec.system  # чеклист подставлен


def test_prompt_versions_covers_both_axes() -> None:
    versions = prompt_versions(prompts_dir=PROMPTS_DIR, markers_path=MARKERS_PATH)

    assert set(versions) == set(AXES)
    assert all(value for value in versions.values())


@pytest.mark.parametrize("criterion", RELEVANCE_FAIL_CRITERIA)
def test_relevance_prompt_states_every_organizer_fail_criterion(criterion: str) -> None:
    assert criterion in _spec(AXIS_RELEVANCE).system


def test_relevance_prompt_keeps_incompleteness_caveat() -> None:
    """Без этой оговорки судья штрафует неполноту дважды: разметка ставит 1
    неполному, но релевантному ответу (пример «На 62 дня — 19.84%»)."""
    system = _spec(AXIS_RELEVANCE).system

    assert "даже если он неполон" in system
    assert "ось faithfulness, не твоя" in system
    assert "историю диалога" in system
    assert "19.84%" in system  # пример организаторов приведён дословно


def test_relevance_prompt_does_not_leak_faithfulness_criteria() -> None:
    system = _spec(AXIS_RELEVANCE).system

    assert "[CTX]" not in system
    assert "{context}" not in _spec(AXIS_RELEVANCE).user_template


def test_faithfulness_prompt_covers_every_marker_code() -> None:
    system = _spec(AXIS_FAITHFULNESS).system
    codes = load_markers(MARKERS_PATH)

    assert len(codes) == 13
    missing = [code for code in codes if code not in system]
    assert missing == []


def test_faithfulness_prompt_adds_taxonomy_criteria_missing_from_the_joint_prompt() -> None:
    system = _spec(AXIS_FAITHFULNESS).system

    for criterion in (
        "нерелевантном фрагменте",
        "проверил или вычислил",
        "устаревшую информацию",
        "сгенерирован ИИ",
        "неверной ссылке",
    ):
        assert criterion in system


@pytest.mark.parametrize("axis", AXES)
def test_axis_prompt_requests_marker_and_verdict_format(axis: str) -> None:
    system = _spec(axis).system

    assert "ANALYSIS:" in system
    assert "MARKER:" in system
    assert f"{axis.upper()}: PASS или FAIL" in system
    # Якорь вердикта идёт последним: MARKER не должен сдвигать позицию PASS/FAIL.
    assert system.rindex("MARKER:") < system.rindex(f"{axis.upper()}:")


def test_relevance_user_prompt_carries_no_chunks() -> None:
    sample = _sample()

    _, user = build_axis_prompt(sample, AXIS_RELEVANCE, prompts_dir=PROMPTS_DIR)

    assert "СЕКРЕТНЫЙ_ТЕКСТ_ЧАНКА" not in user
    assert sample.question in user
    assert sample.answer in user


def test_faithfulness_user_prompt_carries_chunks() -> None:
    sample = _sample()

    _, user = build_axis_prompt(sample, AXIS_FAITHFULNESS, prompts_dir=PROMPTS_DIR)

    assert "СЕКРЕТНЫЙ_ТЕКСТ_ЧАНКА" in user
    assert "[CTX]" in user


def test_relevance_prompt_is_cheaper_than_faithfulness_prompt() -> None:
    """Смысл разделения осей — не только качество: relevance не платит за чанки."""
    sample = _sample()

    _, faith_user = build_axis_prompt(sample, AXIS_FAITHFULNESS, prompts_dir=PROMPTS_DIR)
    _, rel_user = build_axis_prompt(sample, AXIS_RELEVANCE, prompts_dir=PROMPTS_DIR)

    assert len(rel_user) < len(faith_user)


def test_prompt_version_reaches_predictions_and_run_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Версия промпта обязана дожить до артефактов: иначе не восстановить,
    какой именно текст дал числа."""
    import argparse  # noqa: PLC0415
    import importlib.util  # noqa: PLC0415
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("run_m3", REPO_ROOT / "scripts" / "run_m3.py")
    assert spec is not None and spec.loader is not None
    run_m3 = importlib.util.module_from_spec(spec)
    sys.modules["run_m3"] = run_m3
    spec.loader.exec_module(run_m3)

    output = tmp_path / "predictions.jsonl"
    run_meta = tmp_path / "run.yaml"
    args = argparse.Namespace(
        data=str(REPO_ROOT / "data" / "dummy.jsonl"),
        output=str(output),
        mode="zero_shot",
        examples=None,
        prompt_file=None,
        backend="dummy",
        dummy_strategy="always_reliable",
        model="dummy-model",
        api_base="http://localhost:8000/v1",
        api_key_env="OPENAI_API_KEY",
        cache_dir=None,
        run_meta=str(run_meta),
        max_tokens=200,
        max_context_chars=None,
        limit=2,
        concurrency=1,
        prompt_style="axes",
        prompts_dir=str(PROMPTS_DIR),
        prompt_file_faithfulness=None,
        prompt_file_relevance=None,
        sc_n=1,
        sc_temperature=0.0,
        ablation_n=None,
        ablation_temperature="0.7",
        ablation_out=None,
        ablation_replicates=200,
    )
    monkeypatch.setattr(run_m3, "parse_args", lambda: args)

    run_m3.main()

    expected = prompt_versions(prompts_dir=PROMPTS_DIR, markers_path=MARKERS_PATH)
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert json.loads(first["raw_output"])["prompt_versions"] == expected
    assert expected[AXIS_RELEVANCE] in json.loads(run_meta.read_text(encoding="utf-8"))["args"][
        "prompt_versions"
    ]


def test_marker_checklist_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="absent from the taxonomy"):
        build_marker_checklist(["reason_not_a_real_code"], load_markers(MARKERS_PATH))


def test_axis_prompt_without_marker_placeholder_is_rejected(tmp_path: Path) -> None:
    """Промпт без плейсхолдера молча терял бы criteria injection."""
    payload = {
        "version": "relevance/test",
        "axis": "relevance",
        "needs_context": False,
        "markers": ["reason_other"],
        "system": "RELEVANCE: PASS или FAIL",
        "user_template": "{question}\n{answer}",
        "examples": [],
    }
    (tmp_path / "relevance.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="marker_checklist"):
        load_axis_prompt(AXIS_RELEVANCE, prompts_dir=tmp_path, markers_path=MARKERS_PATH)


def test_axis_prompt_without_verdict_anchor_is_rejected(tmp_path: Path) -> None:
    """Промпт без строки вердикта оставил бы logprobs без якоря."""
    payload = {
        "version": "relevance/test",
        "axis": "relevance",
        "needs_context": False,
        "markers": ["reason_other"],
        "system": "Оцени ответ.\n{marker_checklist}",
        "user_template": "{question}\n{answer}",
        "examples": [],
    }
    (tmp_path / "relevance.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="RELEVANCE"):
        load_axis_prompt(AXIS_RELEVANCE, prompts_dir=tmp_path, markers_path=MARKERS_PATH)
