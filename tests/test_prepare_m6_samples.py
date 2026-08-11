"""Tests for Method 6 sample-cache preparation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "prepare_m6_samples",
    Path(__file__).parents[1] / "scripts" / "prepare_m6_samples.py",
)
assert _SPEC is not None
prepare_m6_samples = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["prepare_m6_samples"] = prepare_m6_samples
_SPEC.loader.exec_module(prepare_m6_samples)


def make_rag_sample(sample_id: str) -> RagSample:
    return RagSample(
        id=sample_id,
        question="q",
        context="c",
        answer="a",
        faithfulness=1,
        relevance=1,
    )


class FakeLLMClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, float | int]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        n: int,
        max_tokens: int,
        top_p: float,
        logprobs: bool = False,
    ) -> list[dict[str, object]]:
        self.calls.append({"n": n, "temperature": temperature, "top_p": top_p})
        return [{"text": f"sample-{i}", "tokens": []} for i in range(n)]


def test_build_bot_prompt_uses_question_context_answer_task() -> None:
    sample = RagSample(
        id="s1",
        question="Как подключить услугу?",
        context="Подключение доступно в личном кабинете.",
        answer="Откройте личный кабинет.",
        faithfulness=1,
        relevance=1,
    )

    prompt = prepare_m6_samples.build_bot_prompt(sample)

    assert "Фрагменты документации" in prompt
    assert "Как подключить услугу?" in prompt
    assert "Подключение доступно" in prompt


def test_write_sample_cache_and_need_samples(tmp_path: Path) -> None:
    cache_file = tmp_path / "s1.json"

    assert prepare_m6_samples.need_samples(cache_file, target=2) == (2, [])

    prepare_m6_samples.write_sample_cache(tmp_path, "s1", ["a"])

    assert prepare_m6_samples.need_samples(cache_file, target=2) == (1, ["a"])
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {"id": "s1", "samples": ["a"]}


def test_build_sample_cache_extends_existing_samples(tmp_path: Path) -> None:
    sample = RagSample(
        id="s1",
        question="q",
        context="c",
        answer="a",
        faithfulness=1,
        relevance=1,
    )
    prepare_m6_samples.write_sample_cache(tmp_path, "s1", ["old"])

    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return f"new-{len(calls)}"

    prepare_m6_samples.build_sample_cache(
        [sample],
        output_dir=tmp_path,
        generate_fn=generate,
        n_samples=3,
    )

    payload = json.loads((tmp_path / "s1.json").read_text(encoding="utf-8"))
    assert payload["samples"] == ["old", "new-1", "new-2"]


def test_openai_batch_fn_uses_n_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLMClient()
    monkeypatch.setattr(prepare_m6_samples, "LLMClient", lambda **kw: fake)
    batch_fn = prepare_m6_samples.make_openai_batch_fn(
        model="m",
        api_base="http://x/v1",
        api_key="k",
        max_tokens=64,
        temperature=0.8,
        top_p=0.95,
    )

    out = batch_fn("prompt", 3)

    assert out == ["sample-0", "sample-1", "sample-2"]
    assert fake.calls == [{"n": 3, "temperature": 0.8, "top_p": 0.95}]


def test_build_sample_cache_batch_fn_tops_up_only_missing(tmp_path: Path) -> None:
    sample = make_rag_sample("a")
    prepare_m6_samples.write_sample_cache(tmp_path, "a", ["old"])
    calls: list[int] = []

    def batch_fn(prompt: str, k: int) -> list[str]:
        calls.append(k)
        return [f"new-{i}" for i in range(k)]

    prepare_m6_samples.build_sample_cache(
        [sample], output_dir=tmp_path, generate_batch_fn=batch_fn, n_samples=3
    )

    assert calls == [2]
    _, existing = prepare_m6_samples.need_samples(tmp_path / "a.json", 3)
    assert existing == ["old", "new-0", "new-1"]


def test_build_sample_cache_requires_exactly_one_generator(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prepare_m6_samples.build_sample_cache([], output_dir=tmp_path, n_samples=1)
