"""Pure GEPA helpers: gloss loading, score/feedback and train subsampling (no dspy)."""

from pathlib import Path

from rag_reliability.methods.m3.gepa import (
    has_marker,
    load_marker_gloss,
    score_and_feedback,
    subsample_train,
    verdict,
)
from rag_reliability.schema import RagSample

_GLOSS_PATH = Path(__file__).parents[1] / "configs" / "markers.yaml"


def _sample(id: str = "s1", marker: str | None = None) -> RagSample:
    return RagSample(
        id=id,
        question="q",
        context="c",
        answer="a",
        faithfulness=1,
        relevance=0,
        marker=marker,
    )


def test_gloss_loaded_from_yaml() -> None:
    """The real curators' taxonomy: 13 reason_* markers with Russian glosses."""
    gloss = load_marker_gloss(_GLOSS_PATH)
    assert len(gloss) == 13
    assert "reason_hallucinated_fact" in gloss
    assert "факт" in gloss["reason_hallucinated_fact"]
    assert "reason_incomplete_answer" in gloss


def test_verdict_mapping() -> None:
    assert verdict(1) == "PASS"
    assert verdict(0) == "FAIL"


def test_score_halves() -> None:
    score, _ = score_and_feedback("PASS", "FAIL", "PASS", "FAIL")
    assert score == 1.0  # both axes correct
    score, _ = score_and_feedback("PASS", "FAIL", "FAIL", "FAIL")
    assert score == 0.5  # one axis
    score, _ = score_and_feedback("FAIL", "PASS", "PASS", "FAIL")
    assert score == 0.0  # none


def test_feedback_contains_marker_when_enabled() -> None:
    gloss = {"reason_hallucinated_fact": "модель галлюцинировала неверный факт"}
    _, feedback = score_and_feedback(
        "FAIL",
        "FAIL",
        "PASS",
        "FAIL",
        marker="reason_hallucinated_fact",
        use_markers=True,
        gloss=gloss,
    )
    assert "reason_hallucinated_fact" in feedback
    assert "галлюцинировала" in feedback


def test_feedback_no_marker_when_disabled() -> None:
    """Plain variant: gold labels only, no markers (the single H5 difference)."""
    _, feedback = score_and_feedback(
        "FAIL",
        "FAIL",
        "PASS",
        "FAIL",
        marker="reason_hallucinated_fact",
        use_markers=False,
        gloss={"reason_hallucinated_fact": "x"},
    )
    assert "reason_hallucinated_fact" not in feedback
    assert "FAITHFULNESS=FAIL" in feedback


def test_none_marker_never_appended() -> None:
    _, feedback = score_and_feedback(
        "FAIL", "FAIL", "PASS", "FAIL", marker="none", use_markers=True, gloss={}
    )
    assert "Маркер" not in feedback


def test_correct_prediction_positive_feedback() -> None:
    _, feedback = score_and_feedback("PASS", "FAIL", "PASS", "FAIL", use_markers=True)
    assert "верн" in feedback.lower()


def test_has_marker() -> None:
    assert has_marker(_sample(marker="reason_hallucinated_fact"))
    assert not has_marker(_sample(marker=None))
    assert not has_marker(_sample(marker="none"))
    assert not has_marker(_sample(marker="unknown"))


def test_subsample_is_deterministic_and_raises_marker_share() -> None:
    samples = [
        _sample(id=f"m{i}", marker="reason_hallucinated_fact") for i in range(10)
    ] + [_sample(id=f"p{i}") for i in range(90)]

    first = subsample_train(samples, train_size=20, marker_share=0.5, seed=0)
    second = subsample_train(samples, train_size=20, marker_share=0.5, seed=0)
    assert [s.id for s in first] == [s.id for s in second]  # deterministic
    assert len(first) == 20
    assert sum(has_marker(s) for s in first) == 10  # share raised to 50%


def test_subsample_noop_when_size_covers_all() -> None:
    samples = [_sample(id=f"p{i}") for i in range(5)]
    assert len(subsample_train(samples, train_size=10, seed=0)) == 5
