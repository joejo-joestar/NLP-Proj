from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HallucinationSpan:
    start: int
    end: int
    label: str = "hallucinated"


@dataclass(slots=True)
class RawSample:
    sample_id: str
    question: str
    retrieved_context: list[str]
    answer: str
    hallucination_spans: list[HallucinationSpan] = field(default_factory=list)
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenAlignment:
    """Answer-token label aligned to answer-relative character offsets."""

    token: str
    start: int
    end: int
    is_hallucinated: bool
    hallucination_label: str


@dataclass(slots=True)
class FormattedSample:
    """Standardized Person 1 example before model tokenization."""

    sample_id: str
    split: str
    question: str
    context: list[str]
    answer: str
    full_input_text: str
    prompt: str
    answer_tokens: list[str]
    token_alignment: list[TokenAlignment]
    source_id: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ModelOutput:
    """Reusable forward-pass artifact for downstream Track B metrics."""

    sample_id: str
    split: str
    question: str
    context: list[str]
    answer: str
    full_input_text: str
    input_ids: list[int]
    attention_mask: list[int]
    token_length: int
    answer_start_token_idx: int
    answer_end_token_idx: int
    hidden_states: list[list[list[float]]]
    logits: list[list[float]]
    token_outputs: list[str]
    token_alignment: list[TokenAlignment]
    prompt: str
    metadata: dict[str, Any]
