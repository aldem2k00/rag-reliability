"""Tests for robust output parsing."""

import pytest

from rag_reliability.methods.m3.parsing import parse_m3_prediction
from rag_reliability.parsing import extract_json_object, normalize_binary, parse_prediction


def test_valid_json() -> None:
    pred = parse_prediction('{"faithfulness": 1, "relevance": 0}', "s1")
    assert pred.faithfulness_pred == 1
    assert pred.relevance_pred == 0
    assert pred.invalid_output is False
    assert pred.marker_pred is None


def test_json_with_surrounding_text() -> None:
    raw = 'Sure! Here is my evaluation:\n{"faithfulness": 0, "relevance": 1}\nHope this helps.'
    pred = parse_prediction(raw, "s2")
    assert pred.faithfulness_pred == 0
    assert pred.relevance_pred == 1
    assert pred.invalid_output is False


def test_string_labels() -> None:
    pred = parse_prediction('{"faithfulness": "1", "relevance": "0"}', "s3")
    assert pred.faithfulness_pred == 1
    assert pred.relevance_pred == 0
    assert pred.invalid_output is False


def test_boolean_labels() -> None:
    pred = parse_prediction('{"faithfulness": true, "relevance": false}', "s4")
    assert pred.faithfulness_pred == 1
    assert pred.relevance_pred == 0
    assert pred.invalid_output is False


def test_broken_json_regex_fallback() -> None:
    raw = '{"faithfulness": 1, "relevance": 0'  # missing closing brace
    pred = parse_prediction(raw, "s5")
    assert pred.faithfulness_pred == 1
    assert pred.relevance_pred == 0
    assert pred.invalid_output is False


def test_garbage_conservative_fallback() -> None:
    pred = parse_prediction("I cannot evaluate this answer.", "s6")
    assert pred.invalid_output is True
    assert pred.faithfulness_pred == 0
    assert pred.relevance_pred == 0
    assert pred.raw_output == "I cannot evaluate this answer."


def test_marker_parsing() -> None:
    raw = '{"marker": "hallucination", "faithfulness": 0, "relevance": 1}'
    pred = parse_prediction(raw, "s7", expect_marker=True)
    assert pred.marker_pred == "hallucination"
    assert pred.faithfulness_pred == 0
    assert pred.relevance_pred == 1


def test_marker_ignored_when_not_expected() -> None:
    raw = '{"marker": "hallucination", "faithfulness": 0, "relevance": 1}'
    pred = parse_prediction(raw, "s8", expect_marker=False)
    assert pred.marker_pred is None


def test_marker_missing_is_none() -> None:
    pred = parse_prediction('{"faithfulness": 1, "relevance": 1}', "s9", expect_marker=True)
    assert pred.marker_pred is None
    assert pred.invalid_output is False


def test_marker_regex_fallback() -> None:
    raw = 'marker: "off_topic_answer", faithfulness: 1, relevance: 0 (broken json'
    pred = parse_prediction(raw, "s10", expect_marker=True)
    assert pred.faithfulness_pred == 1
    assert pred.relevance_pred == 0
    assert pred.marker_pred == "off_topic_answer"


def test_extract_json_object() -> None:
    assert extract_json_object('foo {"a": 1} bar') == '{"a": 1}'
    assert extract_json_object('{"a": {"b": 2}}') == '{"a": {"b": 2}}'
    assert extract_json_object("no json here") is None
    # First brace opens broken JSON; parser should recover the later valid object.
    assert extract_json_object('{broken then {"a": 1}') == '{"a": 1}'


def test_normalize_binary() -> None:
    assert normalize_binary(1) == 1
    assert normalize_binary(0) == 0
    assert normalize_binary("1") == 1
    assert normalize_binary("0") == 0
    assert normalize_binary(True) == 1
    assert normalize_binary(False) == 0
    assert normalize_binary("true") == 1
    assert normalize_binary("FALSE") == 0
    assert normalize_binary(2) is None
    assert normalize_binary("yes") is None
    assert normalize_binary(None) is None


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("- **FAITHFULNESS:** PASS\n- **RELEVANCE:** FAIL", (1, 0)),
        ("* **FAITHFULNESS:** FAIL\n* **RELEVANCE:** PASS", (0, 1)),
        ("**faithfulness** pass\n**relevance** fail", (1, 0)),
    ],
)
def test_m3_regex_matches_markdown_verdicts(
    raw_output: str,
    expected: tuple[int, int],
) -> None:
    prediction = parse_m3_prediction(raw_output, "sample")
    assert (prediction.faithfulness_pred, prediction.relevance_pred) == expected
    assert prediction.invalid_output is False
