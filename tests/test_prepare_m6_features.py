"""Tests for the Method 6 feature preparation script."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "prepare_m6_features",
    Path(__file__).parents[1] / "scripts" / "prepare_m6_features.py",
)
assert _SPEC is not None
prepare_m6_features = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["prepare_m6_features"] = prepare_m6_features
_SPEC.loader.exec_module(prepare_m6_features)


class StubNLI:
    def score(self, pairs):
        return [{"entail": 0.9, "contra": 0.1} for _ in pairs]


class StubEmbedder:
    def encode(self, texts, normalize_embeddings=True):  # noqa: ARG002
        return np.array([[1.0, 0.0], [1.0, 0.0]])


class RecordingNLI(StubNLI):
    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def score(self, pairs):
        self.call_sizes.append(len(pairs))
        return super().score(pairs)


def test_build_feature_rows_from_cache(tmp_path: Path) -> None:
    sample = RagSample(
        id="s1",
        question="q",
        context="c",
        answer="a.",
        faithfulness=1,
        relevance=1,
    )
    (tmp_path / "s1.json").write_text(
        json.dumps({"id": "s1", "samples": ["sample"]}, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = prepare_m6_features.build_feature_rows(
        [sample],
        samples_dir=tmp_path,
        nli=StubNLI(),
        embedder=StubEmbedder(),
        entail_threshold=0.5,
    )

    assert rows[0]["id"] == "s1"
    assert rows[0]["selfcheck_contra_mean"] == 0.1
    assert rows[0]["cos_q_a"] == 1.0


def test_use_n_samples_slices_cache(tmp_path: Path) -> None:
    sample = RagSample(
        id="s1",
        question="q",
        context="c",
        answer="a.",
        faithfulness=1,
        relevance=1,
    )
    (tmp_path / "s1.json").write_text(
        json.dumps({"id": "s1", "samples": [f"sample-{i}" for i in range(5)]}),
        encoding="utf-8",
    )
    nli = RecordingNLI()

    prepare_m6_features.build_feature_rows(
        [sample],
        samples_dir=tmp_path,
        nli=nli,
        embedder=StubEmbedder(),
        entail_threshold=0.5,
        use_n_samples=3,
    )

    assert nli.call_sizes == [3, 12]
