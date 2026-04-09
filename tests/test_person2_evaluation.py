from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.evaluation import (  # noqa: E402
    aggregate_metric_aurocs,
    compute_binary_auroc,
    extract_answer_labels,
)


class Person2EvaluationTests(unittest.TestCase):
    def test_compute_binary_auroc_perfect_separation(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        value = compute_binary_auroc(labels, scores)
        self.assertIsNotNone(value)
        self.assertAlmostEqual(float(value), 1.0, places=6)

    def test_extract_answer_labels_uses_answer_slice(self) -> None:
        record = {
            "answer_start_token_idx": 1,
            "answer_end_token_idx": 4,
            "token_alignment": [
                {"is_hallucinated": False},
                {"is_hallucinated": True},
                {"is_hallucinated": False},
                {"is_hallucinated": True},
            ],
        }
        labels = extract_answer_labels(record)
        self.assertEqual(labels.tolist(), [1, 0, 1])

    def test_aggregate_metric_aurocs_includes_per_layer(self) -> None:
        metric_records = [
            {
                "sample_id": "s1",
                "split": "test",
                "layers_used": [0, 1],
                "cosine_drift": torch.tensor([0.1, 0.8]),
                "composite_score": torch.tensor([0.2, 0.9]),
                "cosine_drift_per_layer": torch.tensor(
                    [
                        [0.2, 0.7],
                        [0.1, 0.9],
                    ]
                ),
            },
            {
                "sample_id": "s2",
                "split": "test",
                "layers_used": [0, 1],
                "cosine_drift": torch.tensor([0.2, 0.7]),
                "composite_score": torch.tensor([0.3, 0.8]),
                "cosine_drift_per_layer": torch.tensor(
                    [
                        [0.3, 0.6],
                        [0.2, 0.8],
                    ]
                ),
            },
        ]
        labels_by_sample = {
            "s1": torch.tensor([0, 1]),
            "s2": torch.tensor([0, 1]),
        }

        summary = aggregate_metric_aurocs(
            metric_records, labels_by_sample, split="test"
        )

        self.assertEqual(summary["processed_samples"], 2)
        self.assertIn("cosine_drift", summary["metrics"])
        self.assertIn("composite_score", summary["metrics"])
        self.assertIn("cosine_drift", summary["per_layer"])
        self.assertIn("layer_0", summary["per_layer"]["cosine_drift"])


if __name__ == "__main__":
    unittest.main()
