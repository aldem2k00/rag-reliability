"""Чистая логика GEPA: feedback, отбор D_pareto без утечки, санитария. Без dspy."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_reliability.methods.m3.gepa import (
    DEFAULT_PARETO_SIZE,
    FOREIGN_DOMAIN_TERMS,
    LeakageError,
    answer_snippet,
    axis_gold,
    build_optimization_sets,
    closest_chunk,
    fold_partition,
    foreign_domain_hits,
    gepa_feedback,
    has_marker,
    load_marker_gloss,
    load_seed_instruction,
    load_stopwords,
    prompt_defects,
    split_context_chunks,
    stratified_subsample,
    verdict,
)
from rag_reliability.schema import RagSample

_REPO = Path(__file__).parents[1]
_GLOSS_PATH = _REPO / "configs" / "markers.yaml"
_PROMPTS_DIR = _REPO / "configs" / "prompts"

_CONTEXT = (
    "[CHUNK 1] Ставка по накопительному счёту составляет 4 процента годовых.\n"
    "[CHUNK 2] Кэшбэк по дебетовой карте начисляется рублями до 5000 в месяц.\n"
    "[CHUNK 3] Перевыпуск карты в отделении занимает пять рабочих дней."
)


def _sample(
    id: str = "s1",
    *,
    faithfulness: int = 1,
    relevance: int = 1,
    marker: str | None = None,
    answer: str = "ответ",
    context: str = "контекст",
) -> RagSample:
    return RagSample(
        id=id,
        question="Клиент: какой кэшбэк по карте?",
        context=context,
        answer=answer,
        faithfulness=faithfulness,
        relevance=relevance,
        marker=marker,
    )


def _corpus(n_positive: int, n_negative: int) -> list[RagSample]:
    positives = [_sample(id=f"pos{index:04d}") for index in range(n_positive)]
    negatives = [
        _sample(id=f"neg{index:04d}", faithfulness=0, relevance=0)
        for index in range(n_negative)
    ]
    return positives + negatives


def _folds(samples: list[RagSample], n_folds: int = 5, n_repeats: int = 2) -> dict:
    """Round-robin раскладка: для проверки изоляции важна форма, а не качество."""
    assignment = {
        sample.id: [(index + repeat) % n_folds for repeat in range(n_repeats)]
        for index, sample in enumerate(samples)
    }
    return {
        "schema_version": 1,
        "corpus": {"sha256": "x", "n": len(samples)},
        "config": {"n_folds": n_folds, "n_repeats": n_repeats},
        "assignment": assignment,
    }


# --------------------------------------------------------------------------- #
# Мелочи
# --------------------------------------------------------------------------- #


def test_verdict_mapping() -> None:
    assert verdict(1) == "PASS"
    assert verdict(0) == "FAIL"


def test_axis_gold_reads_the_axis_not_reliable() -> None:
    sample = _sample(faithfulness=1, relevance=0)
    assert axis_gold(sample, "faithfulness") == 1
    assert axis_gold(sample, "relevance") == 0
    with pytest.raises(ValueError, match="Unknown axis"):
        axis_gold(sample, "reliable")


def test_has_marker() -> None:
    assert has_marker(_sample(marker="reason_hallucinated_fact"))
    assert not has_marker(_sample(marker=None))
    assert not has_marker(_sample(marker="none"))
    assert not has_marker(_sample(marker="unknown"))


def test_gloss_loaded_from_yaml() -> None:
    """Реальная таксономия кураторов: 13 маркеров с русскими глоссами."""
    gloss = load_marker_gloss(_GLOSS_PATH)
    assert len(gloss) == 13
    assert "факт" in gloss["reason_hallucinated_fact"]


# --------------------------------------------------------------------------- #
# Feedback с диагностикой
# --------------------------------------------------------------------------- #


def test_correct_prediction_gets_short_positive_feedback() -> None:
    assert gepa_feedback(axis="faithfulness", gold=1, pred=1, answer="ответ") == "Верно."


def test_feedback_carries_marker_gloss_when_enabled() -> None:
    feedback = gepa_feedback(
        axis="faithfulness",
        gold=0,
        pred=1,
        answer="Кэшбэк составляет 10000 рублей в месяц.",
        context=_CONTEXT,
        marker="reason_hallucinated_fact",
        use_markers=True,
        gloss=load_marker_gloss(_GLOSS_PATH),
    )
    assert "Тип ошибки:" in feedback
    assert "галлюцинировала" in feedback


def test_feedback_without_markers_is_the_only_h5_difference() -> None:
    """plain-вариант: та же диагностика, но без строки глосса."""
    common = dict(
        axis="faithfulness",
        gold=0,
        pred=1,
        answer="Кэшбэк составляет 10000 рублей в месяц.",
        context=_CONTEXT,
        marker="reason_hallucinated_fact",
        gloss=load_marker_gloss(_GLOSS_PATH),
    )
    markers = gepa_feedback(**common, use_markers=True)
    plain = gepa_feedback(**common, use_markers=False)

    assert "Тип ошибки:" in markers
    assert "Тип ошибки:" not in plain
    gloss_line = "Тип ошибки: модель галлюцинировала неверный факт. "
    assert markers.replace(gloss_line, "") == plain


def test_feedback_survives_a_sample_without_a_marker() -> None:
    feedback = gepa_feedback(
        axis="faithfulness",
        gold=0,
        pred=1,
        answer="Кэшбэк 10000 рублей.",
        context=_CONTEXT,
        marker=None,
        use_markers=True,
        gloss=load_marker_gloss(_GLOSS_PATH),
    )
    assert "Тип ошибки:" not in feedback
    assert "FAITHFULNESS=FAIL" in feedback


def test_feedback_quotes_the_answer() -> None:
    """Прежний feedback сообщал только правильную метку: что, но не почему."""
    feedback = gepa_feedback(
        axis="faithfulness",
        gold=0,
        pred=1,
        answer="Кэшбэк по дебетовой карте начисляется милями до 20000 в месяц.",
        context=_CONTEXT,
    )
    assert "Фрагмент ответа: «Кэшбэк по дебетовой карте начисляется милями" in feedback


def test_feedback_points_at_the_closest_chunk() -> None:
    feedback = gepa_feedback(
        axis="faithfulness",
        gold=0,
        pred=1,
        answer="Кэшбэк по дебетовой карте начисляется милями до 20000 в месяц.",
        context=_CONTEXT,
    )
    assert "В чанках это не подтверждается" in feedback
    assert "чанк 2" in feedback


def test_feedback_reports_a_broken_output_format() -> None:
    """``pred=None`` — вердикта в выводе нет; рефлексия должна узнать именно это."""
    feedback = gepa_feedback(axis="relevance", gold=1, pred=None, answer="ответ")
    assert "Вердикта в выводе не нашлось" in feedback
    assert "RELEVANCE" in feedback


def test_relevance_feedback_quotes_the_client_turn_instead_of_chunks() -> None:
    feedback = gepa_feedback(
        axis="relevance",
        gold=0,
        pred=1,
        answer="Оформите вклад в отделении.",
        context="",
        question="Ассистент: здравствуйте\nКлиент: какой кэшбэк по карте?",
    )
    assert "Вопрос клиента: «какой кэшбэк по карте?»" in feedback
    assert "чанк" not in feedback


def test_answer_snippet_collapses_and_truncates() -> None:
    assert answer_snippet("а  б\n в") == "а б в"
    assert answer_snippet("я" * 300, max_chars=10) == "я" * 10 + "…"


def test_split_context_chunks_reads_both_markups() -> None:
    assert [number for number, _ in split_context_chunks(_CONTEXT)] == [1, 2, 3]
    assert split_context_chunks("[Чанк 7] текст")[0][0] == 7
    assert split_context_chunks("без разметки") == []


def test_closest_chunk_is_none_without_overlap() -> None:
    assert closest_chunk("совершенно посторонние слова здесь", _CONTEXT) is None
    assert closest_chunk("что угодно", "контекст без чанков") is None


# --------------------------------------------------------------------------- #
# D_pareto и изоляция фолда
# --------------------------------------------------------------------------- #


def test_subsample_is_deterministic_and_stratified() -> None:
    samples = _corpus(72, 28)

    first = stratified_subsample(samples, 50, seed=0)
    second = stratified_subsample(list(reversed(samples)), 50, seed=0)

    assert [s.id for s in first] == [s.id for s in second]  # порядок входа не важен
    assert len(first) == 50
    assert sum(s.reliable for s in first) == 36  # 72% сохранены


def test_subsample_returns_everything_when_size_covers_the_pool() -> None:
    samples = _corpus(4, 1)
    assert len(stratified_subsample(samples, 10, seed=0)) == 5


def test_subsample_keeps_the_rare_class_present() -> None:
    """Стратификация нужна ровно затем, чтобы редкий класс не исчез: без него
    balanced accuracy неопределена."""
    picked = stratified_subsample(_corpus(950, 50), 100, seed=3)
    assert 0 < sum(1 for s in picked if s.reliable == 0)


def test_subsample_size_is_exact_at_extreme_imbalance() -> None:
    for size in (7, 33, 100):
        assert len(stratified_subsample(_corpus(990, 10), size, seed=1)) == size


def test_fold_partition_splits_by_the_requested_repeat_and_fold() -> None:
    samples = _corpus(60, 40)
    folds = _folds(samples)

    train, held_out = fold_partition(samples, folds, repeat=1, fold=2)

    assert len(train) + len(held_out) == len(samples)
    assert {s.id for s in train} & {s.id for s in held_out} == set()
    assert all(folds["assignment"][s.id][1] == 2 for s in held_out)


def test_fold_partition_keeps_unassigned_cases_in_train() -> None:
    """Кейсы oversized-группы отсутствуют в assignment, в held-out не попадают
    никогда и потому безопасны для оптимизации."""
    samples = _corpus(60, 40)
    folds = _folds(samples)
    del folds["assignment"]["pos0000"]

    train, held_out = fold_partition(samples, folds, repeat=0, fold=0)

    assert "pos0000" in {s.id for s in train}
    assert "pos0000" not in {s.id for s in held_out}


def test_fold_partition_rejects_out_of_range_indices() -> None:
    samples = _corpus(10, 10)
    folds = _folds(samples)
    with pytest.raises(ValueError, match="repeat must be in"):
        fold_partition(samples, folds, repeat=9, fold=0)
    with pytest.raises(ValueError, match="fold must be in"):
        fold_partition(samples, folds, repeat=0, fold=9)


def test_no_held_out_case_reaches_train_or_pareto() -> None:
    """Главный тест карточки: оптимизация не видит held-out фолда."""
    samples = _corpus(720, 280)
    folds = _folds(samples)

    for repeat in (0, 1):
        for fold in range(5):
            sets = build_optimization_sets(
                samples,
                folds,
                axis="faithfulness",
                repeat=repeat,
                fold=fold,
                pareto_size=DEFAULT_PARETO_SIZE,
                train_size=300,
                seed=0,
            )
            held_out = {s.id for s in sets.held_out}
            assert {s.id for s in sets.train} & held_out == set()
            assert {s.id for s in sets.pareto} & held_out == set()
            assert {s.id for s in sets.train} & {s.id for s in sets.pareto} == set()
            assert len(sets.pareto) == DEFAULT_PARETO_SIZE


def test_pareto_is_stratified_by_the_axis_label_not_by_reliable() -> None:
    """Ось faithfulness может быть 1 при relevance 0; стратифицировать надо ось."""
    samples = [
        _sample(id=f"a{index:04d}", faithfulness=1, relevance=0) for index in range(700)
    ] + [_sample(id=f"b{index:04d}", faithfulness=0, relevance=1) for index in range(300)]
    folds = _folds(samples)

    sets = build_optimization_sets(
        samples, folds, axis="faithfulness", pareto_size=200, train_size=100, seed=0
    )
    positives = sum(1 for s in sets.pareto if s.faithfulness == 1)
    assert 130 <= positives <= 150  # ~70% как в источнике


def test_optimization_sets_are_deterministic_by_seed() -> None:
    samples = _corpus(720, 280)
    folds = _folds(samples)
    kwargs = dict(axis="faithfulness", repeat=0, fold=0, pareto_size=300, train_size=200)

    first = build_optimization_sets(samples, folds, seed=0, **kwargs)
    second = build_optimization_sets(samples, folds, seed=0, **kwargs)
    other = build_optimization_sets(samples, folds, seed=1, **kwargs)

    assert [s.id for s in first.pareto] == [s.id for s in second.pareto]
    assert [s.id for s in first.train] == [s.id for s in second.train]
    assert [s.id for s in first.pareto] != [s.id for s in other.pareto]


def test_leakage_check_fires_when_the_invariant_is_broken() -> None:
    from rag_reliability.methods.m3.gepa import OptimizationSets, check_no_leakage

    shared = _sample(id="shared")
    with pytest.raises(LeakageError, match="leaked into pareto"):
        check_no_leakage(
            OptimizationSets(train=[], pareto=[shared], held_out=[shared], repeat=0, fold=0)
        )
    with pytest.raises(LeakageError, match="both"):
        check_no_leakage(
            OptimizationSets(train=[shared], pareto=[shared], held_out=[], repeat=0, fold=0)
        )


def test_pareto_may_not_eat_the_whole_train_part() -> None:
    samples = _corpus(30, 20)
    folds = _folds(samples)
    with pytest.raises(ValueError, match="consumed the whole train part"):
        build_optimization_sets(
            samples, folds, axis="faithfulness", pareto_size=1000, train_size=10, seed=0
        )


# --------------------------------------------------------------------------- #
# Санитария промптов
# --------------------------------------------------------------------------- #


def test_foreign_domain_detector_fires_on_hubble() -> None:
    """Ровно тот дефект, что уже уехал в закоммиченные промпты."""
    text = "Пример: телескоп «Хаббл» был запущен в 1990 году. FAITHFULNESS: PASS"
    hits = foreign_domain_hits(text)
    assert "хаббл" in hits
    assert foreign_domain_hits("Ставка по вкладу 4 процента. FAITHFULNESS: PASS") == []


def test_foreign_domain_detector_is_case_and_inflection_tolerant() -> None:
    assert foreign_domain_hits("снимок Хаббла") == ["хаббл"]
    assert foreign_domain_hits("руины ПАРФЕНОНА") == ["парфенон"]


def test_foreign_domain_terms_are_not_substrings_of_banking_words() -> None:
    """Подстрочное сравнение ловит словоформы, но и ложные срабатывания: «афин»
    сработал бы на «парафине». Список обязан оставаться чистым на банковском
    тексте — иначе предупреждение перестанут читать."""
    banking = (
        "Ставка по накопительному счёту 4 процента годовых. Кэшбэк начисляется "
        "рублями. Перевыпуск карты в отделении, парафин и марсианские сроки "
        "здесь ни при чём: комиссия за перевод, эквайринг, овердрафт, ипотека, "
        "страхование, брокерский счёт, реквизиты, СБП, тариф «Всё сразу»."
    )
    assert foreign_domain_hits(banking) == []


def test_prompt_defects_flags_a_missing_verdict_anchor() -> None:
    defects = prompt_defects("Оцени ответ и напиши вывод.", "faithfulness")
    assert any("FAITHFULNESS" in defect for defect in defects)


def test_prompt_defects_flags_the_empty_chunk_demonstration() -> None:
    text = "Пример контекста: [Чанк 2] (...)\nFAITHFULNESS: PASS"
    assert any("пустого чанка" in defect for defect in prompt_defects(text, "faithfulness"))


def test_clean_axis_prompt_has_no_defects() -> None:
    """Сид эволюции — YAML-промпт оси, который читает инференс, и он чистый."""
    for axis in ("faithfulness", "relevance"):
        instruction = load_seed_instruction(axis, prompts_dir=_PROMPTS_DIR)
        assert prompt_defects(instruction, axis) == []


def test_stopwords_default_to_the_module_list_and_load_from_file(tmp_path: Path) -> None:
    assert load_stopwords(None) == FOREIGN_DOMAIN_TERMS

    path = tmp_path / "stopwords.yaml"
    path.write_text("terms:\n  - Бетховен\n", encoding="utf-8")
    assert load_stopwords(path) == ("бетховен",)

    path.write_text("- Бетховен\n", encoding="utf-8")
    assert load_stopwords(path) == ("бетховен",)

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_stopwords(path)
