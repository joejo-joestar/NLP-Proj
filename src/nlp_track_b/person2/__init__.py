"""Person 2 hidden-state metrics for Track B."""

from .artifacts import iter_artifact_paths, load_person1_artifact, save_metric_artifact
from .evaluation import (
    aggregate_metric_aurocs,
    compute_binary_auroc,
    extract_answer_labels,
)
from .metrics import (
    compute_composite_score,
    compute_cosine_drift,
    compute_logit_lens_divergence,
    compute_mahalanobis,
    compute_pca_deviation,
    fit_mahalanobis_stats,
    fit_normalizer_stats,
    fit_pca_stats,
)

__all__ = [
    "compute_composite_score",
    "compute_binary_auroc",
    "compute_cosine_drift",
    "compute_logit_lens_divergence",
    "compute_mahalanobis",
    "compute_pca_deviation",
    "extract_answer_labels",
    "aggregate_metric_aurocs",
    "fit_mahalanobis_stats",
    "fit_normalizer_stats",
    "fit_pca_stats",
    "iter_artifact_paths",
    "load_person1_artifact",
    "save_metric_artifact",
]
