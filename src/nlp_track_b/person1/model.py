from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .schemas import FormattedSample, ModelOutput

_HF_CACHE: dict[str, tuple[Any, Any, Any, str]] = {}


def _resolve_torch_device(requested_device: str, torch: Any) -> str:
    requested = requested_device.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in torch. "
                "Install a CUDA-enabled PyTorch build or use --device cpu."
            )
        if requested.startswith("cuda:"):
            try:
                device_idx = int(requested.split(":", maxsplit=1)[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid CUDA device format: {requested_device}"
                ) from exc
            cuda_count = torch.cuda.device_count()
            if device_idx < 0 or device_idx >= cuda_count:
                raise ValueError(
                    f"CUDA device index {device_idx} out of range (0..{cuda_count - 1})"
                )
        return requested
    raise ValueError(
        f"Unsupported device '{requested_device}'. Use auto, cpu, cuda, or cuda:N."
    )


@dataclass(slots=True)
class ForwardRunner:
    cfg: ModelConfig

    def run(self, sample: FormattedSample) -> ModelOutput:
        return self.run_batch([sample])[0]

    def run_batch(self, samples: list[FormattedSample]) -> list[ModelOutput]:
        if not samples:
            return []
        if self.cfg.provider == "mock":
            return [_run_mock_forward(sample, self.cfg) for sample in samples]
        if self.cfg.provider == "hf":
            return _run_hf_forward_batch(samples, self.cfg)
        raise ValueError(f"Unknown model provider: {self.cfg.provider}")


def _seed_for(sample: FormattedSample) -> int:
    digest = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_mock_forward(sample: FormattedSample, cfg: ModelConfig) -> ModelOutput:
    seed = _seed_for(sample)
    rng = random.Random(seed)
    token_count = max(len(sample.answer_tokens), 1)
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
    token_outputs = [sample.answer_tokens[idx] for idx in range(token_count)]

    return ModelOutput(
        sample_id=sample.sample_id,
        split=sample.split,
        hidden_states=hidden_states,
        logits=logits,
        token_outputs=token_outputs,
        token_alignment=sample.token_alignment,
        prompt=sample.prompt,
        metadata=sample.metadata,
    )


def _load_hf_backend(cfg: ModelConfig) -> tuple[Any, Any, Any, str]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        resolved_device = _resolve_torch_device(cfg.device, torch)
        cache_key = f"{cfg.model_name}::{resolved_device}"

        if cache_key not in _HF_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            model.to(torch.device(resolved_device))
            if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
            model.eval()
            _HF_CACHE[cache_key] = (tokenizer, model, torch, resolved_device)

        tokenizer, model, torch, resolved_device = _HF_CACHE[cache_key]
    except ImportError as exc:
        raise RuntimeError(
            "provider='hf' requires 'transformers' and 'torch'. "
            "Install them or switch to provider='mock'."
        ) from exc

    return tokenizer, model, torch, resolved_device


def _run_hf_forward_batch(
    samples: list[FormattedSample], cfg: ModelConfig
) -> list[ModelOutput]:
    tokenizer, model, torch, resolved_device = _load_hf_backend(cfg)

    full_texts = [sample.prompt + " ".join(sample.answer_tokens) for sample in samples]

    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_seq_len,
        padding=True,
    )
    device_obj = torch.device(resolved_device)
    encoded = {key: value.to(device_obj) for key, value in encoded.items()}

    with torch.no_grad():
        result = model(**encoded, output_hidden_states=True)

    logits_tensor = result.logits
    hidden_tensor = result.hidden_states

    outputs: list[ModelOutput] = []
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        raise RuntimeError(
            "Tokenizer did not return an attention_mask for batched HF inference."
        )

    for row_idx, sample in enumerate(samples):
        token_count = int(attention_mask[row_idx].sum().item())
        row_logits = logits_tensor[row_idx, :token_count]
        row_hidden = [layer[row_idx, :token_count] for layer in hidden_tensor]
        row_token_ids = (
            encoded["input_ids"][row_idx, :token_count].detach().cpu().tolist()
        )
        token_outputs = tokenizer.convert_ids_to_tokens(row_token_ids)

        aligned = sample.token_alignment[:token_count]

        metadata = {
            **sample.metadata,
            "hf_model": cfg.model_name,
            "hf_device": resolved_device,
            "sequence_length": token_count,
        }

        if cfg.compact_output:
            k = max(1, min(cfg.logits_topk, row_logits.shape[-1]))
            topk_values, topk_indices = torch.topk(row_logits, k=k, dim=-1)
            last_n_layers = min(3, len(row_hidden))
            hidden_states_last_n = [
                row_hidden[-(i + 1)].detach().cpu().tolist()
                for i in range(last_n_layers)
            ]
            metadata.update(
                {
                    "compact_output": True,
                    "logits_topk_k": k,
                    "logits_topk_indices": topk_indices.detach().cpu().tolist(),
                    "logits_topk_values": topk_values.detach().cpu().tolist(),
                    "hidden_states_last_n_layers": hidden_states_last_n,
                    "num_hidden_layers_saved": last_n_layers,
                }
            )
            logits = []
            hidden_states = []
        else:
            logits = row_logits.detach().cpu().tolist()
            hidden_states = [layer.detach().cpu().tolist() for layer in row_hidden]

        outputs.append(
            ModelOutput(
                sample_id=sample.sample_id,
                split=sample.split,
                hidden_states=hidden_states,
                logits=logits,
                token_outputs=token_outputs,
                token_alignment=aligned,
                prompt=sample.prompt,
                metadata=metadata,
            )
        )

    return outputs
