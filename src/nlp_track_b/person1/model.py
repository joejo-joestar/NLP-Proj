from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from .config import ModelConfig
from .schemas import FormattedSample, ModelOutput, TokenAlignment


@dataclass(slots=True)
class TokenizedInput:
    """Full model input plus the answer-only token span."""

    input_ids: list[int]
    attention_mask: list[int]
    token_offsets: list[tuple[int, int]]
    answer_start_token_idx: int
    answer_end_token_idx: int
    truncated_left_tokens: int = 0


@dataclass(slots=True)
class ForwardRunner:
    cfg: ModelConfig
    tokenizer: Any = field(init=False, default=None)
    model: Any = field(init=False, default=None)
    torch: Any = field(init=False, default=None)
    device: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Load the HF backend once per runner instead of once per sample."""
        if self.cfg.provider != "hf":
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "provider='hf' requires 'transformers' and 'torch'. "
                "Install them or switch to provider='mock'."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
        self.device = _resolve_device(torch, self.cfg.device)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def run(self, sample: FormattedSample) -> ModelOutput:
        if self.cfg.provider == "mock":
            return _run_mock_forward(sample, self.cfg)
        if self.cfg.provider == "hf":
            assert self.tokenizer is not None
            assert self.model is not None
            assert self.torch is not None
            return _run_hf_forward(
                sample=sample,
                cfg=self.cfg,
                tokenizer=self.tokenizer,
                model=self.model,
                torch=self.torch,
            )
        raise ValueError(f"Unknown model provider: {self.cfg.provider}")


def _resolve_device(torch: Any, device: str) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device {device!r} but CUDA is not available."
        )
    return resolved


def _seed_for(sample: FormattedSample) -> int:
    digest = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_mock_forward(sample: FormattedSample, cfg: ModelConfig) -> ModelOutput:
    seed = _seed_for(sample)
    rng = random.Random(seed)
    token_count = max(len(sample.answer_tokens), 1)
    input_ids = list(range(token_count))
    attention_mask = [1] * token_count

    hidden_states = [
        [
            [round(rng.uniform(-1.0, 1.0), 6) for _ in range(cfg.hidden_size)]
            for _ in range(token_count)
        ]
        for _ in range(cfg.num_layers)
    ]

    logits = [
        [round(rng.uniform(-7.0, 7.0), 6) for _ in range(cfg.vocab_size)]
        for _ in range(token_count)
    ]
    token_outputs = sample.answer_tokens if sample.answer_tokens else [""]

    return ModelOutput(
        sample_id=sample.sample_id,
        split=sample.split,
        question=sample.question,
        context=sample.context,
        answer=sample.answer,
        full_input_text=sample.full_input_text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_length=token_count,
        answer_start_token_idx=0,
        answer_end_token_idx=token_count,
        hidden_states=hidden_states,
        logits=logits,
        token_outputs=token_outputs,
        token_alignment=sample.token_alignment,
        prompt=sample.prompt,
        metadata=sample.metadata,
    )


def _find_answer_token_span(
    token_offsets: list[tuple[int, int]],
    answer_char_start: int,
    answer_char_end: int,
) -> tuple[int, int]:
    """Return the exclusive answer token span inside the full token sequence."""
    assert answer_char_end > answer_char_start
    answer_indices = [
        idx
        for idx, (start, end) in enumerate(token_offsets)
        if end > start and end > answer_char_start and start < answer_char_end
    ]
    assert answer_indices, "Could not align the answer to tokenizer offsets."
    return answer_indices[0], answer_indices[-1] + 1


def _tokenize_full_input(
    sample: FormattedSample, cfg: ModelConfig, tokenizer: Any
) -> TokenizedInput:
    """Tokenize the exact full input and keep a valid answer-token range."""
    assert sample.full_input_text == sample.prompt + sample.answer
    assert sample.answer

    encoded = tokenizer(
        sample.full_input_text,
        return_attention_mask=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    token_offsets = [tuple(offset) for offset in encoded["offset_mapping"]]

    assert len(input_ids) == len(attention_mask) == len(token_offsets)
    answer_char_start = len(sample.prompt)
    answer_char_end = len(sample.full_input_text)
    raw_answer_start, raw_answer_end = _find_answer_token_span(
        token_offsets=token_offsets,
        answer_char_start=answer_char_start,
        answer_char_end=answer_char_end,
    )

    window_start = 0
    window_end = len(input_ids)
    if len(input_ids) > cfg.max_seq_len:
        answer_token_len = raw_answer_end - raw_answer_start
        if answer_token_len >= cfg.max_seq_len:
            window_start = raw_answer_start
            window_end = raw_answer_start + cfg.max_seq_len
        else:
            window_end = raw_answer_end
            window_start = max(0, window_end - cfg.max_seq_len)

    input_ids = input_ids[window_start:window_end]
    attention_mask = attention_mask[window_start:window_end]
    token_offsets = token_offsets[window_start:window_end]
    answer_start = max(0, raw_answer_start - window_start)
    answer_end = min(raw_answer_end, window_end) - window_start

    assert len(input_ids) <= cfg.max_seq_len
    assert len(input_ids) == len(attention_mask) == len(token_offsets)
    assert 0 <= answer_start < answer_end <= len(input_ids)

    return TokenizedInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_offsets=token_offsets,
        answer_start_token_idx=answer_start,
        answer_end_token_idx=answer_end,
        truncated_left_tokens=window_start,
    )


def _align_answer_tokens(
    sample: FormattedSample,
    tokenized: TokenizedInput,
    token_outputs: list[str],
) -> list[TokenAlignment]:
    """Align answer HF tokens to answer-relative character offsets."""
    answer_char_start = len(sample.prompt)
    answer_char_end = len(sample.full_input_text)
    aligned: list[TokenAlignment] = []

    for idx in range(tokenized.answer_start_token_idx, tokenized.answer_end_token_idx):
        start, end = tokenized.token_offsets[idx]
        rel_start = max(0, start - answer_char_start)
        rel_end = min(end, answer_char_end) - answer_char_start
        matched = [
            span
            for span in sample.token_alignment
            if rel_start < span.end and rel_end > span.start
        ]
        label = matched[0].hallucination_label if matched else "none"
        aligned.append(
            TokenAlignment(
                token=token_outputs[idx],
                start=rel_start,
                end=rel_end,
                is_hallucinated=bool(matched),
                hallucination_label=label,
            )
        )

    assert (
        len(aligned)
        == tokenized.answer_end_token_idx - tokenized.answer_start_token_idx
    )
    return aligned


def _run_hf_forward(
    sample: FormattedSample,
    cfg: ModelConfig,
    tokenizer: Any,
    model: Any,
    torch: Any,
) -> ModelOutput:
    tokenized = _tokenize_full_input(sample=sample, cfg=cfg, tokenizer=tokenizer)
    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": torch.tensor([tokenized.input_ids], device=device),
        "attention_mask": torch.tensor([tokenized.attention_mask], device=device),
    }

    with torch.no_grad():
        result = model(**model_inputs, output_hidden_states=True)

    logits_tensor = result.logits[0]
    hidden_tensor = result.hidden_states

    token_count = logits_tensor.shape[0]
    assert token_count == len(tokenized.input_ids)
    logits = logits_tensor.tolist()
    hidden_states = [layer[0].tolist() for layer in hidden_tensor]
    token_outputs = tokenizer.convert_ids_to_tokens(tokenized.input_ids)
    aligned = _align_answer_tokens(
        sample=sample,
        tokenized=tokenized,
        token_outputs=token_outputs,
    )

    return ModelOutput(
        sample_id=sample.sample_id,
        split=sample.split,
        question=sample.question,
        context=sample.context,
        answer=sample.answer,
        full_input_text=sample.full_input_text,
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        token_length=token_count,
        answer_start_token_idx=tokenized.answer_start_token_idx,
        answer_end_token_idx=tokenized.answer_end_token_idx,
        hidden_states=hidden_states,
        logits=logits,
        token_outputs=token_outputs,
        token_alignment=aligned,
        prompt=sample.prompt,
        metadata={
            **sample.metadata,
            "hf_model": cfg.model_name,
            "sequence_length": token_count,
            "truncated_left_tokens": tokenized.truncated_left_tokens,
        },
    )
