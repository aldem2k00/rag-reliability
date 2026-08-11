"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ALLOWED_MARKERS: tuple[str, ...] = (
    "none",
    "unknown",
    "hallucination",
    "off_topic_answer",
    "incomplete_answer",
    "context_mixing",
    "contradiction",
    "unsupported_claim",
    "reason_hallucinated_fact",
    "reason_off_topic_answer",
    "reason_irrelevant_chunk_used",
    "reason_chunk_fact_mixup",
    "reason_incomplete_answer",
    "reason_false_verification",
    "reason_outdated_fact",
    "reason_answer_for_operator",
    "reason_other",
    "reason_reveals_ai_identity",
    "reason_wrong_navigation",
    "reason_missed_complaint_handoff",
    "reason_missed_chunk_conditions",
)


class RagSample(BaseModel):
    """One labeled (question, context, answer) triple."""

    id: str
    question: str
    context: str
    answer: str
    faithfulness: int = Field(ge=0, le=1)
    relevance: int = Field(ge=0, le=1)
    marker: str | None = None

    @field_validator("marker")
    @classmethod
    def _marker_allowed(cls, value: str | None) -> str | None:
        # Gold labels only; predicted markers (Prediction.marker_pred) stay
        # free-form so bad model outputs surface in metrics, not crashes.
        if value is not None and value not in ALLOWED_MARKERS:
            raise ValueError(f"marker must be one of {ALLOWED_MARKERS}, got {value!r}")
        return value

    @property
    def reliable(self) -> int:
        return int(self.faithfulness == 1 and self.relevance == 1)


class Prediction(BaseModel):
    """Parsed model output for one sample."""

    id: str
    faithfulness_pred: int = Field(ge=0, le=1)
    relevance_pred: int = Field(ge=0, le=1)
    marker_pred: str | None = None
    raw_output: str | None = None
    invalid_output: bool = False
    # Judge-method probabilities (Method 3 logprobs path). Binary *_pred fields
    # stay the evaluation contract; probabilities are extra evidence and record
    # how they were obtained ("logprobs", "regex" or "default").
    faithfulness_prob: float | None = Field(default=None, ge=0, le=1)
    relevance_prob: float | None = Field(default=None, ge=0, le=1)
    prob_method: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("scores")
    @classmethod
    def _score_keys(cls, value: dict[str, float]) -> dict[str, float]:
        """Не допускает коллизий сигналов при сборке фич по префиксу метода."""
        bad = [
            key
            for key in value
            if "." not in key or key.startswith(".") or key.endswith(".")
        ]
        if bad:
            raise ValueError(
                f"score keys must be '<method>.<signal>', got {bad[:5]}"
            )
        return value

    @property
    def reliable_pred(self) -> int:
        return int(self.faithfulness_pred == 1 and self.relevance_pred == 1)


class MetricWithCI(BaseModel):
    """Метрика с обязательным интервалом для воспроизводимого отчёта."""

    value: float
    ci95: tuple[float, float]
    null_percentile: float | None = None
    above_noise: bool | None = None


class EvaluationReport(BaseModel):
    """Схема report.json: primary физически не существует без 95% ДИ."""

    schema_version: int = 1
    method: str
    variant: str
    protocol: dict
    primary: MetricWithCI
    axes: dict[str, MetricWithCI] = Field(default_factory=dict)
    operational: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)
    comparisons: list[dict] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Aggregate metrics over a prediction set.

    Marker fields are populated only when at least one prediction carries a
    marker (marker mode); in direct mode they stay None.
    """

    reliable_f1_macro: float
    faithfulness_f1_macro: float
    relevance_f1_macro: float
    invalid_output_rate: float
    total: int
    invalid_count: int
    marker_f1_macro: float | None = None
    marker_per_class_f1: dict[str, float] | None = None
    marker_confusion: dict[str, dict[str, int]] | None = None
