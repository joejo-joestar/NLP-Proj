from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


LayerSpec = str | Sequence[int]


def normalize_hidden_states(hidden_states: Any) -> list[torch.Tensor]:
    """Return hidden-state layers shaped as (seq_len, hidden_size)."""
    if isinstance(hidden_states, torch.Tensor):
        if hidden_states.ndim == 4:
            layers = [hidden_states[idx] for idx in range(hidden_states.shape[0])]
        elif hidden_states.ndim == 3:
            layers = [hidden_states[idx] for idx in range(hidden_states.shape[0])]
        else:
            raise ValueError(
                f"Unsupported hidden_states tensor shape: {tuple(hidden_states.shape)}"
            )
    elif isinstance(hidden_states, (list, tuple)):
        layers = list(hidden_states)
    else:
        raise ValueError("hidden_states must be a list, tuple, or torch.Tensor.")

    normalized: list[torch.Tensor] = []
    for layer in layers:
        tensor = torch.as_tensor(layer, dtype=torch.float32)
        if tensor.ndim == 2:
            normalized.append(tensor)
        elif tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"Only batch size 1 is supported, got batch={tensor.shape[0]}."
                )
            normalized.append(tensor[0])
        else:
            raise ValueError(f"Unsupported layer shape: {tuple(tensor.shape)}")

    if not normalized:
        raise ValueError("hidden_states has no layers.")
    return normalized


def select_layer_indices(n_layers: int, layers: LayerSpec = "last4") -> list[int]:
    """Resolve a layer selector to concrete layer indices."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    if layers == "all":
        return list(range(n_layers))
    if layers == "last4":
        return list(range(max(0, n_layers - 4), n_layers))
    if isinstance(layers, str):
        raise ValueError(f"Unsupported layer selector: {layers}")

    resolved: list[int] = []
    for layer in layers:
        idx = int(layer)
        if idx < 0:
            idx += n_layers
        if not 0 <= idx < n_layers:
            raise ValueError(f"Layer index {layer} out of range for {n_layers} layers.")
        resolved.append(idx)
    if not resolved:
        raise ValueError("At least one layer must be selected.")
    return resolved


def slice_answer(
    hidden_state_layers: list[torch.Tensor],
    answer_start: int,
    answer_end: int,
    *,
    layers: LayerSpec = "last4",
) -> tuple[list[int], list[torch.Tensor]]:
    """Slice selected hidden-state layers to answer tokens only."""
    selected = select_layer_indices(len(hidden_state_layers), layers)
    seq_len = int(hidden_state_layers[0].shape[0])
    for layer in hidden_state_layers:
        if layer.ndim != 2:
            raise ValueError(
                f"Expected layer shape (seq, hidden), got {tuple(layer.shape)}."
            )
        if int(layer.shape[0]) != seq_len:
            raise ValueError(
                "All hidden-state layers must have the same sequence length."
            )
    if not 0 <= answer_start < answer_end <= seq_len:
        raise ValueError(
            "Invalid answer token range: "
            f"start={answer_start}, end={answer_end}, seq_len={seq_len}"
        )
    return selected, [
        hidden_state_layers[idx][answer_start:answer_end] for idx in selected
    ]


def answer_hidden_by_layer(
    hidden_states: Any,
    answer_start: int,
    answer_end: int,
    *,
    layers: LayerSpec = "last4",
) -> tuple[list[int], list[torch.Tensor]]:
    """Normalize hidden states and return selected answer-token layer tensors."""
    normalized = normalize_hidden_states(hidden_states)
    return slice_answer(normalized, answer_start, answer_end, layers=layers)


def _aggregate_per_layer(per_layer: torch.Tensor, aggregate: str) -> torch.Tensor:
    if aggregate == "mean":
        return per_layer.mean(dim=0)
    raise ValueError(f"Unsupported aggregate mode: {aggregate}")


def cosine_drift_one_layer(hidden: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Compute intra-answer cosine drift for one layer with drift[0] = 0."""
    if hidden.ndim != 2:
        raise ValueError(
            f"Expected hidden shape (A, hidden), got {tuple(hidden.shape)}."
        )
    token_count = int(hidden.shape[0])
    if token_count == 0:
        raise ValueError("Cosine drift requires at least one answer token.")
    drift = torch.zeros(token_count, dtype=hidden.dtype)
    if token_count == 1:
        return drift
    similarity = F.cosine_similarity(hidden[1:], hidden[:-1], dim=-1, eps=eps)
    drift[1:] = 1.0 - similarity
    return drift


def compute_cosine_drift(
    hidden_states: Any,
    answer_start: int,
    answer_end: int,
    *,
    layers: LayerSpec = "last4",
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Compute token-level cosine drift over answer tokens only."""
    layers_used, answer_layers = answer_hidden_by_layer(
        hidden_states,
        answer_start,
        answer_end,
        layers=layers,
    )
    per_layer = torch.stack([cosine_drift_one_layer(layer) for layer in answer_layers])
    return {
        "cosine_drift": _aggregate_per_layer(per_layer, aggregate),
        "cosine_drift_per_layer": per_layer,
        "layers_used": layers_used,
        "answer_token_count": int(per_layer.shape[1]),
    }


def _record_answer_layers(
    record: Mapping[str, Any], layers: LayerSpec
) -> tuple[list[int], list[torch.Tensor]]:
    return answer_hidden_by_layer(
        record["hidden_states"],
        int(record["answer_start_token_idx"]),
        int(record["answer_end_token_idx"]),
        layers=layers,
    )


def _covariance_stats(
    records: Iterable[Mapping[str, Any]], layers: LayerSpec
) -> dict[str, Any]:
    layers_used: list[int] | None = None
    sums: list[torch.Tensor] = []
    sum_outers: list[torch.Tensor] = []
    counts: list[int] = []

    for record in records:
        record_layers, answer_layers = _record_answer_layers(record, layers)
        if layers_used is None:
            layers_used = record_layers
            for hidden in answer_layers:
                hidden_dim = int(hidden.shape[1])
                sums.append(torch.zeros(hidden_dim))
                sum_outers.append(torch.zeros(hidden_dim, hidden_dim))
                counts.append(0)
        elif record_layers != layers_used:
            raise ValueError("Layer selection changed across records.")

        for idx, hidden in enumerate(answer_layers):
            sums[idx] += hidden.sum(dim=0)
            sum_outers[idx] += hidden.T @ hidden
            counts[idx] += int(hidden.shape[0])

    if layers_used is None:
        raise ValueError("No records supplied for fitting.")
    if any(count == 0 for count in counts):
        raise ValueError("Cannot fit stats without answer tokens.")
    return {
        "layers_used": layers_used,
        "sums": sums,
        "sum_outers": sum_outers,
        "counts": counts,
    }


def fit_mahalanobis_stats(
    records: Iterable[Mapping[str, Any]],
    *,
    layers: LayerSpec = "last4",
    regularization: float = 1e-4,
) -> dict[str, Any]:
    """Fit per-layer train-only Mahalanobis mean and inverse covariance."""
    cov_stats = _covariance_stats(records, layers)
    means: list[torch.Tensor] = []
    inv_covariances: list[torch.Tensor] = []

    for total, outer, count in zip(
        cov_stats["sums"], cov_stats["sum_outers"], cov_stats["counts"]
    ):
        mean = total / count
        centered_outer = outer - count * torch.outer(mean, mean)
        covariance = centered_outer / max(count - 1, 1)
        covariance = covariance + regularization * torch.eye(covariance.shape[0])
        means.append(mean)
        inv_covariances.append(torch.linalg.pinv(covariance))

    return {
        "type": "mahalanobis",
        "layers_used": cov_stats["layers_used"],
        "means": means,
        "inv_covariances": inv_covariances,
        "regularization": regularization,
    }


def compute_mahalanobis(
    hidden_states: Any,
    answer_start: int,
    answer_end: int,
    stats: Mapping[str, Any],
    *,
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Compute token-level Mahalanobis distance from train-fitted stats."""
    layers_used = [int(layer) for layer in stats["layers_used"]]
    _, answer_layers = answer_hidden_by_layer(
        hidden_states,
        answer_start,
        answer_end,
        layers=layers_used,
    )
    per_layer_scores: list[torch.Tensor] = []
    for hidden, mean, inv_cov in zip(
        answer_layers, stats["means"], stats["inv_covariances"]
    ):
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        inv_cov_t = torch.as_tensor(inv_cov, dtype=torch.float32)
        diff = hidden - mean_t
        squared = (diff @ inv_cov_t * diff).sum(dim=-1).clamp_min(0.0)
        per_layer_scores.append(torch.sqrt(squared))
    per_layer = torch.stack(per_layer_scores)
    return {
        "mahalanobis_distance": _aggregate_per_layer(per_layer, aggregate),
        "mahalanobis_per_layer": per_layer,
        "layers_used": layers_used,
        "answer_token_count": int(per_layer.shape[1]),
    }


def fit_pca_stats(
    records: Iterable[Mapping[str, Any]],
    *,
    layers: LayerSpec = "last4",
    n_components: int = 16,
) -> dict[str, Any]:
    """Fit per-layer PCA bases from train split answer tokens only."""
    if n_components < 0:
        raise ValueError("n_components must be non-negative.")
    cov_stats = _covariance_stats(records, layers)
    means: list[torch.Tensor] = []
    components: list[torch.Tensor] = []

    for total, outer, count in zip(
        cov_stats["sums"], cov_stats["sum_outers"], cov_stats["counts"]
    ):
        mean = total / count
        centered_outer = outer - count * torch.outer(mean, mean)
        covariance = centered_outer / max(count - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        max_rank = min(n_components, eigenvectors.shape[0], max(count - 1, 0))
        selected = eigenvectors[:, order[:max_rank]].T.contiguous()
        means.append(mean)
        components.append(selected)

    return {
        "type": "pca",
        "layers_used": cov_stats["layers_used"],
        "means": means,
        "components": components,
        "n_components": n_components,
    }


def compute_pca_deviation(
    hidden_states: Any,
    answer_start: int,
    answer_end: int,
    stats: Mapping[str, Any],
    *,
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Compute token-level PCA reconstruction error from train-fitted PCA stats."""
    layers_used = [int(layer) for layer in stats["layers_used"]]
    _, answer_layers = answer_hidden_by_layer(
        hidden_states,
        answer_start,
        answer_end,
        layers=layers_used,
    )
    per_layer_scores: list[torch.Tensor] = []
    for hidden, mean, components in zip(
        answer_layers, stats["means"], stats["components"]
    ):
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        components_t = torch.as_tensor(components, dtype=torch.float32)
        diff = hidden - mean_t
        if components_t.numel() == 0:
            reconstructed = mean_t.expand_as(hidden)
        else:
            weights = diff @ components_t.T
            reconstructed = weights @ components_t + mean_t
        per_layer_scores.append(torch.linalg.norm(hidden - reconstructed, dim=-1))
    per_layer = torch.stack(per_layer_scores)
    return {
        "pca_deviation": _aggregate_per_layer(per_layer, aggregate),
        "pca_deviation_per_layer": per_layer,
        "layers_used": layers_used,
        "answer_token_count": int(per_layer.shape[1]),
    }


def load_hf_model(model_name: str, *, device: str = "auto") -> Any:
    """Load a Hugging Face causal LM for logit-lens projection."""
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested CUDA device {device!r} but CUDA is not available."
            )
    model.to(target_device)
    model.eval()
    return model


def compute_logit_lens_divergence(
    hidden_states: Any,
    final_logits: Any,
    answer_start: int,
    answer_end: int,
    *,
    model: Any | None = None,
    model_name: str | None = None,
    layers: LayerSpec = "last4",
    aggregate: str = "mean",
) -> dict[str, Any]:
    """Compare layerwise logit-lens distributions to final logits using KL divergence."""
    if model is None:
        if model_name is None:
            raise ValueError(
                "compute_logit_lens_divergence requires model or model_name."
            )
        model = load_hf_model(model_name)
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise ValueError("Model does not expose output embeddings for logit lens.")

    layers_used, answer_layers = answer_hidden_by_layer(
        hidden_states,
        answer_start,
        answer_end,
        layers=layers,
    )
    final_logits_t = torch.as_tensor(final_logits, dtype=torch.float32)[
        answer_start:answer_end
    ]
    final_probs = F.softmax(final_logits_t, dim=-1)

    device = next(model.parameters()).device
    per_layer_scores: list[torch.Tensor] = []
    with torch.no_grad():
        for hidden in answer_layers:
            projected = output_head(hidden.to(device)).cpu()
            if projected.shape != final_logits_t.shape:
                raise ValueError(
                    "Logit-lens projection shape mismatch: "
                    f"projected={tuple(projected.shape)}, final={tuple(final_logits_t.shape)}"
                )
            layer_log_probs = F.log_softmax(projected, dim=-1)
            kl = F.kl_div(layer_log_probs, final_probs, reduction="none").sum(dim=-1)
            per_layer_scores.append(kl)

    per_layer = torch.stack(per_layer_scores)
    return {
        "logit_lens_divergence": _aggregate_per_layer(per_layer, aggregate),
        "logit_lens_divergence_per_layer": per_layer,
        "layers_used": layers_used,
        "answer_token_count": int(per_layer.shape[1]),
    }


def fit_normalizer_stats(
    metric_values: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    """Fit mean/std normalizers for composite scoring from train metric vectors."""
    normalizers: dict[str, dict[str, float]] = {}
    for name, values in metric_values.items():
        flat_values = [
            torch.as_tensor(value, dtype=torch.float32).reshape(-1) for value in values
        ]
        flat_values = [value for value in flat_values if value.numel() > 0]
        if not flat_values:
            raise ValueError(f"Cannot fit normalizer for empty metric: {name}")
        merged = torch.cat(flat_values)
        std = float(merged.std(unbiased=False))
        normalizers[name] = {
            "mean": float(merged.mean()),
            "std": std if std > 1e-8 else 1.0,
        }
    return normalizers


def compute_composite_score(
    metrics: Mapping[str, torch.Tensor],
    normalizer_stats: Mapping[str, Mapping[str, float]],
    *,
    metric_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Compute an equal-weight composite score from train-normalized metrics."""
    names = list(metric_names or normalizer_stats.keys())
    if not names:
        raise ValueError("At least one metric is required for composite scoring.")

    normalized: list[torch.Tensor] = []
    expected_shape: torch.Size | None = None
    for name in names:
        if name not in metrics:
            raise ValueError(f"Composite metric missing value: {name}")
        if name not in normalizer_stats:
            raise ValueError(f"Composite metric missing train normalizer stats: {name}")
        value = torch.as_tensor(metrics[name], dtype=torch.float32)
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(
                "All metric tensors must have the same shape for composite scoring."
            )
        stats = normalizer_stats[name]
        std = float(stats["std"])
        if std <= 0:
            raise ValueError(f"Normalizer std must be positive for metric: {name}")
        normalized.append((value - float(stats["mean"])) / std)
    return torch.stack(normalized).mean(dim=0)
