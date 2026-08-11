"""Parse Method 3 PASS/FAIL verdicts into shared Prediction records."""

from __future__ import annotations

import re

from rag_reliability.schema import Prediction

_VERDICT_RE = re.compile(
    r"[-*\s]*\**\s*FAITHFULNESS\s*:?\**\s*(PASS|FAIL)"
    r".*?"
    r"[-*\s]*\**\s*RELEVANCE\s*:?\**\s*(PASS|FAIL)",
    re.IGNORECASE | re.DOTALL,
)


def parse_m3_prediction(raw_output: str, sample_id: str) -> Prediction:
    match = _VERDICT_RE.search(raw_output)
    if match is None:
        return Prediction(
            id=sample_id,
            faithfulness_pred=0,
            relevance_pred=0,
            raw_output=raw_output,
            invalid_output=True,
        )
    return Prediction(
        id=sample_id,
        faithfulness_pred=int(match.group(1).upper() == "PASS"),
        relevance_pred=int(match.group(2).upper() == "PASS"),
        raw_output=raw_output,
        invalid_output=False,
    )
