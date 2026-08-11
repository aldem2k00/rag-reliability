# tests/test_registry.py
import re
from dataclasses import replace
from pathlib import Path

import pytest

from rag_reliability.dataset import load_jsonl
from rag_reliability.methods import registry
from rag_reliability.schema import Prediction


def _ctx(tmp_path: Path) -> registry.CommandContext:
    run_dir = tmp_path / "m"
    return registry.CommandContext(
        data=Path("data/dummy.jsonl"),
        run_dir=run_dir,
        predictions_path=run_dir / "predictions.jsonl",
        python="python",
    )


def test_registry_has_twenty_methods() -> None:
    assert len(registry.METHODS) == 20
    assert set(registry.all_method_names()) == set(registry.METHODS)


def test_resolve_all_returns_every_method() -> None:
    assert registry.resolve_names("all") == list(registry.all_method_names())


def test_resolve_unknown_raises_with_available_set() -> None:
    try:
        registry.resolve_names("dummy_direct,nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_dummy_marker_build_command_matches_legacy_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    argv = registry.get("dummy_marker").build_command(ctx)
    assert argv == [
        "python",
        "scripts/run_prompt_baseline.py",
        "--data",
        "data/dummy.jsonl",
        "--output",
        str(ctx.predictions_path),
        "--mode",
        "marker",
        "--backend",
        "dummy",
        "--dummy-strategy",
        "keyword",
    ]


def test_independent_build_command(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    argv = registry.get("independent").build_command(ctx)
    assert argv[0:2] == ["python", "scripts/run_independent.py"]
    assert "--output" in argv
    assert str(ctx.predictions_path) in argv


def test_every_spec_builds_a_nonempty_argv(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    for name in registry.all_method_names():
        argv = registry.get(name).build_command(ctx)
        assert argv and argv[0] == "python"


def test_m3_gepa_default_prompt_is_committed_path(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    argv = registry.get("m3_gepa").build_command(ctx)

    assert "configs/m3_gepa_prompt.txt" in argv
    assert "configs/m3_gepa_prompt.txt" in registry.get("m3_gepa").requires


def test_named_m3_mode_with_openai_judge_backend_gets_api_args(tmp_path: Path) -> None:
    ctx = replace(_ctx(tmp_path), m3_backend="openai_judge")

    argv = registry.get("m3_zero_shot").build_command(ctx)

    assert "--api-base" in argv
    assert "--cache-dir" in argv
    assert "--concurrency" in argv


def test_m6_build_command_uses_single_command_pipeline(tmp_path: Path) -> None:
    ctx = replace(
        _ctx(tmp_path),
        model="remote-model",
        m6_backend="openai",
        m6_samples_dir="cache/m6",
        m6_n_samples=7,
        m6_api_base="https://example.test/v1",
    )

    argv = registry.get("m6_selfcheck").build_command(ctx)

    assert argv[0:2] == ["python", "scripts/run_m6_pipeline.py"]
    assert argv[argv.index("--samples-dir") + 1] == "cache/m6"
    assert argv[argv.index("--backend") + 1] == "openai"
    assert argv[argv.index("--n-samples") + 1] == "7"
    assert argv[argv.index("--model") + 1] == "remote-model"
    assert argv[argv.index("--api-base") + 1] == "https://example.test/v1"


def test_demo_runner_keys_are_known(tmp_path: Path) -> None:
    allowed = {"dummy", "prompt", "lora", "lettucedetect", "encoder", "m3", "independent"}
    for spec in registry.METHODS.values():
        assert spec.demo_runner is None or spec.demo_runner in allowed


# --------------------------------------------------------------------------- #
# Контракт scores: инварианты, а не захардкоженные примеры.
# --------------------------------------------------------------------------- #


def test_every_real_method_declares_score_keys() -> None:
    """Буквальный контракт карточки B2: у каждого реального метода score_keys непусты.

    Единственное исключение — дамми: они существуют ради смоука пайплайна.
    Ослаблять инвариант до «только у corpus_wide» нельзя: тогда метод молча
    выпадает из протокола, а тест остаётся зелёным.
    """
    missing = [
        spec.name
        for spec in registry.METHODS.values()
        if not spec.score_keys and spec.name not in registry.DUMMY_METHODS
    ]
    assert not missing, f"methods without score_keys: {missing}"


def test_score_keys_use_registered_method_prefixes() -> None:
    for spec in registry.METHODS.values():
        for key in spec.score_keys:
            assert key.startswith(registry.SCORE_PREFIXES), f"{spec.name}: bad prefix in {key!r}"
            assert key.count(".") == 1, f"{spec.name}: {key!r} must be '<method>.<signal>'"


def test_default_score_expr_uses_only_declared_keys() -> None:
    """Выражение по умолчанию не может ссылаться на сигнал, которого метод не даёт."""
    identifier = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)+")
    for spec in registry.METHODS.values():
        if spec.default_score_expr is None:
            continue
        assert spec.score_keys, f"{spec.name} has default_score_expr but no score_keys"
        referenced = set(identifier.findall(spec.default_score_expr))
        unknown = referenced - set(spec.score_keys)
        assert not unknown, f"{spec.name}: default_score_expr references undeclared {unknown}"


def test_every_corpus_wide_method_can_actually_be_run() -> None:
    """Заявленный corpus_wide обязан иметь исполнителя: покейсовый скорер или свой скрипт.

    OOF-методы (surface, majority) видят фолд целиком и в модель score.py
    «один кейс -> Prediction» не укладываются — им положен corpus_runner.
    """
    for spec in registry.METHODS.values():
        runnable = spec.build_scorer is not None or spec.corpus_runner is not None
        assert runnable == spec.corpus_wide, (
            f"{spec.name}: corpus_wide={spec.corpus_wide}, "
            f"build_scorer={'set' if spec.build_scorer else 'None'}, "
            f"corpus_runner={spec.corpus_runner}"
        )


def test_oof_methods_are_not_offered_as_per_case_scorers(tmp_path: Path) -> None:
    """score.py должен отказывать внятно, а не считать OOF по одному кейсу."""
    for name in ("surface", "majority"):
        with pytest.raises(ValueError, match="run_surface_baseline"):
            registry.build_scorer(name, _ctx(tmp_path))


def test_build_scorer_refuses_parked_methods_with_wave_three_pointer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C2"):
        registry.build_scorer("encoder", _ctx(tmp_path))


def test_independent_is_the_only_method_allowed_to_binarize(tmp_path: Path) -> None:
    """Карточка разрешает бинарное решение ровно одному методу.

    Реестр не вправе завести себе дополнительные исключения: у каждого метода
    со скорером решение обнуляется, кроме independent.
    """
    ctx = replace(_ctx(tmp_path), m3_backend="dummy")
    samples = load_jsonl("data/dummy.jsonl")

    binarizing = []
    for spec in registry.METHODS.values():
        if spec.build_scorer is None:
            continue
        try:
            scorer = registry.build_scorer(spec.name, ctx)
            predictions = [scorer(sample) for sample in samples]
        except (ImportError, FileNotFoundError, KeyError, SystemExit):
            # Нужна MLX-модель, сеть или артефакт (mlx_backend уходит в sys.exit
            # при отсутствии mlx-lm); общий путь покрыт тестом ниже.
            continue
        if any(p.faithfulness_pred or p.relevance_pred for p in predictions):
            binarizing.append(spec.name)

    assert binarizing == ["independent"]


def test_text_judges_share_one_unbinarizing_score_path() -> None:
    """prompt/lora/m3 идут через verdict_scores + scores_only — общий код, общий инвариант.

    Это и есть гарантия для методов, чей скорер нельзя собрать без MLX.
    """
    verdict = Prediction(id="x", faithfulness_pred=1, relevance_pred=1)
    for prefix in ("m3", "prompt", "lora"):
        scored = registry.scores_only(verdict, registry.verdict_scores(verdict, prefix))
        assert scored.faithfulness_pred == 0
        assert scored.relevance_pred == 0
        assert set(scored.scores) == {f"{prefix}.p_faith", f"{prefix}.p_rel"}


def test_contract_version_tracks_the_declared_contract() -> None:
    spec = registry.get("independent")
    same = registry.contract_version(spec)

    assert registry.contract_version(spec) == same
    changed = replace(spec, score_keys=(*spec.score_keys, "ind.extra"))
    assert registry.contract_version(changed) != same


def test_list_methods_prints_the_new_contract_fields() -> None:
    """rag-judge остаётся окном в реестр: новые поля должны быть видны оператору."""
    from typer.testing import CliRunner

    from rag_reliability.cli import app

    result = CliRunner().invoke(app, ["list-methods"])

    assert result.exit_code == 0
    assert "corpus-wide" in result.output
    assert "split-only" in result.output
    assert "m3.p_faith" in result.output
    assert "ind.faith_score" in result.output


def test_m3_mode_and_backend_shared_by_command_and_scorer(tmp_path: Path) -> None:
    """Одна точка разбора имени: subprocess и score.py не должны разъехаться."""
    ctx = replace(_ctx(tmp_path), m3_backend="mlx")
    assert registry.m3_mode_and_backend("m3_few_shot", ctx) == ("few_shot", "mlx")
    assert registry.m3_mode_and_backend("m3_openai_judge", ctx) == ("zero_shot", "openai_judge")

    argv = registry.get("m3_few_shot").build_command(ctx)
    assert argv[argv.index("--mode") + 1] == "few_shot"
