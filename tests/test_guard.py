"""Guard: the cloud profile admits only synthetic samples (ported from m3-m6)."""

import pytest

from rag_reliability.guard import (
    DataLeakError,
    assert_cloud_safe,
    assert_sample_cloud_safe,
    is_synthetic,
)
from rag_reliability.schema import RagSample


def _sample(id: str = "case_001", synthetic: bool = False) -> RagSample:
    return RagSample(
        id=id,
        question="q",
        context="c",
        answer="a",
        faithfulness=1,
        relevance=1,
        synthetic=synthetic,
    )


def test_is_synthetic_by_prefix() -> None:
    assert is_synthetic(_sample(id="pseudo_00001"))


def test_is_synthetic_by_flag() -> None:
    assert is_synthetic(_sample(id="whatever", synthetic=True))


def test_not_synthetic() -> None:
    assert not is_synthetic(_sample(id="case_00317"))


def test_local_profile_allows_anything() -> None:
    assert_cloud_safe([_sample()], profile="local")  # must not raise


def test_cloud_profile_rejects_real_data() -> None:
    with pytest.raises(DataLeakError):
        assert_cloud_safe([_sample(id="pseudo_1"), _sample(id="case_00317")], profile="cloud")


def test_cloud_profile_allows_synthetic() -> None:
    assert_cloud_safe(
        [_sample(id="pseudo_1"), _sample(id="x", synthetic=True)],
        profile="cloud",
    )


def test_allow_real_opt_in_passes_and_default_blocks() -> None:
    real = _sample(id="alfa_ab12cd34ef56")
    assert_sample_cloud_safe(real, "cloud", allow_real=True)  # must not raise
    assert_cloud_safe([real], "cloud", allow_real=True)  # must not raise
    with pytest.raises(DataLeakError):
        assert_sample_cloud_safe(real, "cloud")  # strict default


def test_real_sample_blocked_in_cloud() -> None:
    real = _sample(id="alfa_ab12cd34ef56")
    assert not is_synthetic(real)
    with pytest.raises(DataLeakError):
        assert_sample_cloud_safe(real, "cloud")
