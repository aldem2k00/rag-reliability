"""Data-leak guard ported from the m3-m6 branch.

Project rule: with the "cloud" profile only synthetic samples (id prefix
"pseudo_" or an explicit ``synthetic: true`` flag) may be sent to an external
API. The curators' corpus never leaves the machine. The single exception is an
explicit opt-in (``allow_real=True``) granted by the data owner for a specific
endpoint and corpus; the default always blocks and the opt-in is logged once.
"""

from __future__ import annotations

import logging

from rag_reliability.schema import RagSample

_log = logging.getLogger(__name__)
_warned_once = False


class DataLeakError(RuntimeError):
    """Attempt to send non-synthetic data to an external API."""


def is_synthetic(sample: RagSample) -> bool:
    return sample.id.startswith("pseudo_") or sample.synthetic


def _warn_opt_in() -> None:
    global _warned_once
    if not _warned_once:
        _log.warning(
            "allow_real=True: real samples are sent to an external API "
            "by explicit permission of the data owner"
        )
        _warned_once = True


def assert_sample_cloud_safe(sample: RagSample, profile: str, allow_real: bool = False) -> None:
    """Per-sample guard — called by the LLM client before every request."""
    if profile == "cloud" and not is_synthetic(sample):
        if allow_real:
            _warn_opt_in()
            return
        raise DataLeakError(
            f"cloud profile: sample {sample.id!r} is not synthetic (no pseudo_ prefix "
            f"and no synthetic flag) — request blocked"
        )


def assert_cloud_safe(samples: list[RagSample], profile: str, allow_real: bool = False) -> None:
    """Whole-dataset guard — called by scripts right after loading, before any request."""
    if profile != "cloud":
        return
    bad = [s.id for s in samples if not is_synthetic(s)]
    if bad:
        if allow_real:
            _warn_opt_in()
            return
        raise DataLeakError(
            f"cloud profile: {len(bad)} non-synthetic samples (first: {bad[:5]}) — stop"
        )
