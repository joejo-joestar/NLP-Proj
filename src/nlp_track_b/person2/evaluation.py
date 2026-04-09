from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from tqdm import tqdm

import torch


METRIC_KEYS = (
    "cosine_drift",
    "mahalanobis_distance",
    "pca_deviation",
    "logit_lens_divergence",
    "composite_score",
)

PER_LAYER_KEYS = {
    "cosine_drift": "cosine_drift_per_layer",
    "mahalanobis_distance": "mahalanobis_per_layer",
    "pca_deviation": "pca_deviation_per_layer",
    "logit_lens_divergence": "logit_lens_divergence_per_layer",
}


def _as_1d_float(values: Any) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1)


def _as_1d_int(values: Any) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.int64).reshape(-1)


def compute_binary_auroc(
    labels: Sequence[int] | torch.Tensor, scores: Any
) -> float | None:
    """Return AUROC for binary labels; None when only one class is present."""
    y_true = _as_1d_int(labels)
    y_score = _as_1d_float(scores)
    if y_true.numel() != y_score.numel():
        raise ValueError(
            "labels and scores must have the same length: "
            f"labels={y_true.numel()}, scores={y_score.numel()}"
        )

    positives = int((y_true == 1).sum().item())
    negatives = int((y_true == 0).sum().item())
    if positives == 0 or negatives == 0:
        return None

    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("AUROC evaluation requires scikit-learn.") from exc

    return float(roc_auc_score(y_true.numpy(), y_score.numpy()))


def extract_answer_labels(record: Mapping[str, Any]) -> torch.Tensor:
    """Extract binary hallucination labels for answer tokens from Person 1 artifacts."""
    answer_start = int(record["answer_start_token_idx"])
    answer_end = int(record["answer_end_token_idx"])
    answer_count = answer_end - answer_start
    token_alignment = list(record.get("token_alignment") or [])
    if not token_alignment:
        raise ValueError("Person 1 artifact is missing token_alignment labels.")

    if len(token_alignment) >= answer_end:
        answer_alignment = token_alignment[answer_start:answer_end]
    elif len(token_alignment) == answer_count:
        answer_alignment = token_alignment
    else:
        raise ValueError(
            "token_alignment length is incompatible with answer range: "
            f"alignment={len(token_alignment)}, answer_count={answer_count}, "
            f"answer_start={answer_start}, answer_end={answer_end}"
        )

    labels: list[int] = []
    for item in answer_alignment:
        if "is_hallucinated" in item:
            labels.append(1 if bool(item["is_hallucinated"]) else 0)
        else:
            label_name = str(item.get("hallucination_label", "")).strip().lower()
            labels.append(
                0 if label_name in {"", "faithful", "non_hallucinated"} else 1
            )
    return _as_1d_int(labels)


def aggregate_metric_aurocs(
    metric_records: Iterable[Mapping[str, Any]],
    labels_by_sample: Mapping[str, torch.Tensor],
    *,
    split: str | None = "test",
) -> dict[str, Any]:
    """Aggregate token-level metric AUROCs over all provided samples."""
    y_true_by_metric: dict[str, list[torch.Tensor]] = {key: [] for key in METRIC_KEYS}
    y_score_by_metric: dict[str, list[torch.Tensor]] = {key: [] for key in METRIC_KEYS}
    y_true_by_layer: dict[str, dict[int, list[torch.Tensor]]] = {
        key: {} for key in PER_LAYER_KEYS
    }
    y_score_by_layer: dict[str, dict[int, list[torch.Tensor]]] = {
        key: {} for key in PER_LAYER_KEYS
    }

    processed = 0
    skipped_split = 0
    skipped_missing_labels = 0
    length_adjustments = 0

    for record in tqdm(metric_records, desc="Aggregating AUROC inputs", unit="sample"):
        if split and record.get("split") != split:
            skipped_split += 1
            continue

        sample_id = str(record.get("sample_id") or record.get("id") or "")
        if not sample_id or sample_id not in labels_by_sample:
            skipped_missing_labels += 1
            continue

        labels = _as_1d_int(labels_by_sample[sample_id])
        processed += 1

        for metric_name in METRIC_KEYS:
            if metric_name not in record:
                continue
            scores = _as_1d_float(record[metric_name])
            if labels.numel() != scores.numel():
                limit = min(labels.numel(), scores.numel())
                labels_metric = labels[:limit]
                scores_metric = scores[:limit]
                length_adjustments += 1
            else:
                labels_metric = labels
                scores_metric = scores

            y_true_by_metric[metric_name].append(labels_metric)
            y_score_by_metric[metric_name].append(scores_metric)

        layers_used = [int(layer) for layer in record.get("layers_used", [])]
        for metric_name, per_layer_key in PER_LAYER_KEYS.items():
            if per_layer_key not in record:
                continue
            per_layer_values = torch.as_tensor(
                record[per_layer_key], dtype=torch.float32
            )
            if per_layer_values.ndim != 2:
                continue
            if len(layers_used) != int(per_layer_values.shape[0]):
                layers_used = list(range(int(per_layer_values.shape[0])))

            for idx, layer_id in enumerate(layers_used):
                scores_layer = _as_1d_float(per_layer_values[idx])
                if labels.numel() != scores_layer.numel():
                    limit = min(labels.numel(), scores_layer.numel())
                    labels_layer = labels[:limit]
                    scores_layer = scores_layer[:limit]
                    length_adjustments += 1
                else:
                    labels_layer = labels

                y_true_by_layer[metric_name].setdefault(layer_id, []).append(
                    labels_layer
                )
                y_score_by_layer[metric_name].setdefault(layer_id, []).append(
                    scores_layer
                )

    metrics_summary: dict[str, Any] = {}
    for metric_name in tqdm(
        METRIC_KEYS, desc="Computing AUROC metrics", unit="metric", leave=False
    ):
        all_labels = y_true_by_metric[metric_name]
        all_scores = y_score_by_metric[metric_name]
        if not all_labels:
            continue
        merged_labels = torch.cat(all_labels)
        merged_scores = torch.cat(all_scores)
        metrics_summary[metric_name] = {
            "auroc": compute_binary_auroc(merged_labels, merged_scores),
            "num_tokens": int(merged_labels.numel()),
            "num_positive": int((merged_labels == 1).sum().item()),
            "num_negative": int((merged_labels == 0).sum().item()),
        }

    per_layer_summary: dict[str, dict[str, Any]] = {}
    for metric_name, by_layer in tqdm(
        y_true_by_layer.items(),
        desc="Computing layer AUROC",
        unit="metric",
        leave=False,
    ):
        if not by_layer:
            continue
        per_layer_summary[metric_name] = {}
        for layer_id in sorted(by_layer):
            merged_labels = torch.cat(by_layer[layer_id])
            merged_scores = torch.cat(y_score_by_layer[metric_name][layer_id])
            per_layer_summary[metric_name][f"layer_{layer_id}"] = {
                "auroc": compute_binary_auroc(merged_labels, merged_scores),
                "num_tokens": int(merged_labels.numel()),
                "num_positive": int((merged_labels == 1).sum().item()),
                "num_negative": int((merged_labels == 0).sum().item()),
            }

    return {
        "split": split,
        "processed_samples": processed,
        "skipped_split": skipped_split,
        "skipped_missing_labels": skipped_missing_labels,
        "length_adjustments": length_adjustments,
        "metrics": metrics_summary,
        "per_layer": per_layer_summary,
    }
