"""Core hallucination detection metrics"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_cosine_drift(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
    layers: str | list[int] = "last4",
) -> dict[str, Any]:
    """
    Compute cosine drift: similarity between context and answer representations.

    Args:
        hidden_states: (num_layers, seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
        layers: "last4", "all", or list of layer indices

    Returns:
        Dictionary with cosine_drift and per-layer values.
    """
    if isinstance(layers, str):
        if layers == "last4":
            layers_to_use = list(
                range(max(0, hidden_states.shape[0] - 4), hidden_states.shape[0])
            )
        elif layers == "all":
            layers_to_use = list(range(hidden_states.shape[0]))
        else:
            layers_to_use = []
    else:
        layers_to_use = layers

    if not layers_to_use:
        layers_to_use = list(range(hidden_states.shape[0]))

    # Edge case: if answer_start == 0, use first token as context proxy
    if answer_start == 0:
        context_reps = hidden_states[:, 0:1, :].mean(dim=1)
    else:
        context_reps = hidden_states[:, :answer_start, :].mean(dim=1)

    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(
        dim=1
    )  # (num_layers, hidden_dim)

    cosine_drifts = []
    for layer_idx in layers_to_use:
        ctx = F.normalize(context_reps[layer_idx], p=2, dim=0)
        ans = F.normalize(answer_reps[layer_idx], p=2, dim=0)
        cos_sim = torch.dot(ctx, ans).item()
        # Clamp to avoid NaN from numerical errors
        cos_sim = max(-1.0, min(1.0, cos_sim))
        cosine_drifts.append(1.0 - cos_sim)

    return {
        "cosine_drift": torch.tensor(
            sum(cosine_drifts) / len(cosine_drifts), dtype=torch.float32
        ),
        "cosine_drift_per_layer": torch.tensor(cosine_drifts, dtype=torch.float32),
        "layers_used": layers_to_use,
    }


def compute_mahalanobis(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
    stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute Mahalanobis distance using fitted statistics."""
    mean = stats.get("mean")
    inv_cov = stats.get("inv_cov")

    if mean is None or inv_cov is None:
        return {
            "mahalanobis_distance": torch.tensor(0.0),
            "mahalanobis_per_layer": torch.tensor([]),
        }

    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(
        dim=1
    )  # (num_layers, hidden_dim)

    diffs = answer_reps - mean
    distances = []
    for layer_idx in range(len(answer_reps)):
        diff = diffs[layer_idx]
        if inv_cov[layer_idx].numel() > 0:
            dist = torch.sqrt(diff @ inv_cov[layer_idx] @ diff.T)
            distances.append(dist.item())
        else:
            distances.append(0.0)

    return {
        "mahalanobis_distance": torch.tensor(
            sum(distances) / len(distances) if distances else 0.0, dtype=torch.float32
        ),
        "mahalanobis_per_layer": torch.tensor(distances, dtype=torch.float32),
    }


def compute_pca_deviation(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Compute PCA deviation using fitted PCA model."""
    pca_models = stats.get("pca_models", [])

    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(
        dim=1
    )  # (num_layers, hidden_dim)

    deviations = []
    for layer_idx, pca_model in enumerate(pca_models):
        if layer_idx < len(answer_reps):
            rep = (
                answer_reps[layer_idx].cpu().numpy()
                if hasattr(answer_reps[layer_idx], "cpu")
                else answer_reps[layer_idx]
            )
            projected = pca_model.transform([rep])[0]
            reconstructed = pca_model.inverse_transform([projected])[0]
            deviation = ((rep - reconstructed) ** 2).sum()
            deviations.append(deviation)

    return {
        "pca_deviation": torch.tensor(
            sum(deviations) / len(deviations) if deviations else 0.0,
            dtype=torch.float32,
        ),
        "pca_deviation_per_layer": torch.tensor(deviations, dtype=torch.float32),
    }


def compute_logit_lens_divergence(
    hidden_states: torch.Tensor,
    logits: torch.Tensor | tuple,
    answer_start: int,
    answer_end: int,
    model: Any,
    layers: str | list[int] = "last4",
) -> dict[str, Any]:
    """Compute logit lens divergence across layers."""
    # Placeholder for logit lens computation
    # This would typically project hidden states to logits and compare
    return {
        "logit_lens_divergence": torch.tensor(0.0, dtype=torch.float32),
        "logit_lens_divergence_per_layer": torch.tensor([], dtype=torch.float32),
    }


def compute_composite_score(
    metric_values: dict[str, torch.Tensor],
    normalizers: dict[str, tuple[float, float]],
) -> torch.Tensor:
    """Compute weighted composite score from normalized metrics."""
    scores = []
    for metric_name, (mean, std) in normalizers.items():
        if metric_name in metric_values:
            val = (
                metric_values[metric_name].item()
                if hasattr(metric_values[metric_name], "item")
                else metric_values[metric_name]
            )
            normalized = (val - mean) / (std + 1e-8)
            scores.append(normalized)

    return torch.tensor(
        sum(scores) / len(scores) if scores else 0.0, dtype=torch.float32
    )


def load_hf_model(model_name: str, device: str = "auto") -> Any:
    """Load a Hugging Face model for logit lens computation."""
    try:
        from transformers import AutoModelForCausalLM

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load HF model {model_name}: {e}")


def fit_mahalanobis_stats(
    records: list[dict] | Any,
    layers: str | list[int] = "last4",
    regularization: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Fit Mahalanobis statistics (mean, covariance) from training records.

    Args:
        records: Iterable of artifact dicts with 'hidden_states' key
        layers: "last4", "all", or list of layer indices
        regularization: Ridge regularization for covariance matrix

    Returns:
        Dict with 'mean' and 'inv_cov' tensors (per-layer).
    """
    if isinstance(records, list):
        records_list = records
    else:
        records_list = list(records)

    if not records_list:
        return {"mean": torch.tensor([]), "inv_cov": torch.tensor([])}

    # Collect all representations
    all_reps = []
    for record in records_list:
        hidden_states = record.get("hidden_states")
        if hidden_states is None or hidden_states.numel() == 0:
            continue
        # Average across sequence and token dimension: (num_layers, seq_len, hidden_dim) -> (num_layers, hidden_dim)
        rep = hidden_states.mean(dim=1)
        all_reps.append(rep)

    if not all_reps:
        return {"mean": torch.tensor([]), "inv_cov": torch.tensor([])}

    # Stack and compute per-layer statistics
    all_reps = torch.stack(all_reps, dim=1)  # (num_layers, num_samples, hidden_dim)
    mean = all_reps.mean(dim=1)  # (num_layers, hidden_dim)

    # Compute covariance and invert
    inv_covs = []
    for layer_idx in range(all_reps.shape[0]):
        reps = all_reps[layer_idx]  # (num_samples, hidden_dim)
        centered = reps - mean[layer_idx]
        cov = (centered.T @ centered) / max(1, len(reps) - 1)

        # Add regularization
        cov += regularization * torch.eye(
            cov.shape[0], device=cov.device, dtype=cov.dtype
        )

        try:
            inv_cov = torch.linalg.inv(cov)
            inv_covs.append(inv_cov)
        except:
            inv_covs.append(torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype))

    return {
        "mean": mean,
        "inv_cov": torch.stack(inv_covs, dim=0) if inv_covs else torch.tensor([]),
    }


def fit_pca_stats(
    records: list[dict] | Any,
    layers: str | list[int] = "last4",
    n_components: int = 16,
) -> dict[str, Any]:
    """Fit PCA models per-layer from training records.

    Args:
        records: Iterable of artifact dicts with 'hidden_states' key
        layers: "last4", "all", or list of layer indices
        n_components: Number of PCA components to keep

    Returns:
        Dict with 'pca_models' list (one fitted PCA per layer).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise RuntimeError("scikit-learn required for PCA fitting")

    if isinstance(records, list):
        records_list = records
    else:
        records_list = list(records)

    if not records_list:
        return {"pca_models": []}

    # Collect all representations
    all_reps = []
    for record in records_list:
        hidden_states = record.get("hidden_states")
        if hidden_states is None or hidden_states.numel() == 0:
            continue
        rep = hidden_states.mean(dim=1)
        all_reps.append(rep)

    if not all_reps:
        return {"pca_models": []}

    all_reps = torch.stack(all_reps, dim=1)  # (num_layers, num_samples, hidden_dim)

    # Fit PCA per layer
    pca_models = []
    for layer_idx in range(all_reps.shape[0]):
        reps = all_reps[layer_idx].cpu().numpy()  # (num_samples, hidden_dim)
        pca = PCA(n_components=min(n_components, reps.shape[1]))
        pca.fit(reps)
        pca_models.append(pca)

    return {"pca_models": pca_models}


def fit_normalizer_stats(
    metric_values: dict[str, list[torch.Tensor]],
) -> dict[str, tuple[float, float]]:
    """Fit mean and std dev for each metric.

    Args:
        metric_values: Dict mapping metric names to lists of tensor values

    Returns:
        Dict mapping metric names to (mean, std_dev) tuples.
    """
    normalizers = {}
    for metric_name, values in metric_values.items():
        if not values:
            normalizers[metric_name] = (0.0, 1.0)
            continue

        # Convert to tensor and compute stats
        tensor_values = torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]
        )
        mean_val = tensor_values.mean().item()
        std_val = tensor_values.std().item()

        # Avoid zero std
        if std_val == 0:
            std_val = 1.0

        normalizers[metric_name] = (mean_val, std_val)

    return normalizers
