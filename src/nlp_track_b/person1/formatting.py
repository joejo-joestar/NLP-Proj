from __future__ import annotations

import re

from .schemas import FormattedSample, RawSample, TokenAlignment

_TOKEN_PATTERN = re.compile(r"\S+")


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in _TOKEN_PATTERN.finditer(text):
        spans.append((match.group(0), match.start(), match.end()))
    return spans


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n".join(f"[{idx + 1}] {doc}" for idx, doc in enumerate(contexts))
    return (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved Context:\n"
        f"{context_text}\n\n"
        "Answer:\n"
    )


def format_full_input(question: str, context: list[str], answer: str) -> tuple[str, str]:
    """Return the exact prompt prefix and full text fed to the model."""
    assert isinstance(question, str)
    assert isinstance(context, list)
    assert all(isinstance(doc, str) for doc in context)
    assert isinstance(answer, str)

    prompt = _build_prompt(question, context)
    full_input_text = prompt + answer
    assert full_input_text.startswith(prompt)
    assert full_input_text[len(prompt) :] == answer
    return prompt, full_input_text


def build_formatted_sample(sample: RawSample, split: str) -> FormattedSample:
    """Build the standardized Person 1 example schema for one split."""
    assert sample.sample_id
    assert split in {"train", "val", "test"}

    prompt, full_input_text = format_full_input(
        question=sample.question,
        context=sample.retrieved_context,
        answer=sample.answer,
    )

    answer_spans = _token_spans(sample.answer)
    token_alignment: list[TokenAlignment] = []
    for token, start, end in answer_spans:
        matched = [
            span
            for span in sample.hallucination_spans
            if start < span.end and end > span.start
        ]
        is_hallucinated = len(matched) > 0
        label = matched[0].label if matched else "none"
        token_alignment.append(
            TokenAlignment(
                token=token,
                start=start,
                end=end,
                is_hallucinated=is_hallucinated,
                hallucination_label=label,
            )
        )

    return FormattedSample(
        sample_id=sample.sample_id,
        split=split,
        question=sample.question,
        context=sample.retrieved_context,
        answer=sample.answer,
        full_input_text=full_input_text,
        prompt=prompt,
        answer_tokens=[x[0] for x in answer_spans],
        token_alignment=token_alignment,
        source_id=sample.source_id,
        metadata=sample.metadata,
    )
