from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.metrics import (
    compute_composite_score,
    compute_cosine_drift,
    compute_logit_lens_divergence,
    compute_mahalanobis,
    compute_pca_deviation,
    fit_mahalanobis_stats,
    fit_pca_stats,
    normalize_hidden_states,
    slice_answer,
)


class Person2MetricsTests(unittest.TestCase):
    def test_single_token_answer_cosine_drift_is_zero(self) -> None:
        hidden_states = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]

        result = compute_cosine_drift(
            hidden_states,
            answer_start=0,
            answer_end=1,
            layers="all",
        )

        self.assertEqual(tuple(result["cosine_drift"].shape), (1,))
        self.assertEqual(result["cosine_drift"].tolist(), [0.0])

    def test_multi_token_cosine_shapes(self) -> None:
        hidden_states = [
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [4.0, 2.0],
                ]
            ),
        ]

        result = compute_cosine_drift(
            hidden_states,
            answer_start=1,
            answer_end=4,
            layers="all",
        )

        self.assertEqual(tuple(result["cosine_drift"].shape), (3,))
        self.assertEqual(tuple(result["cosine_drift_per_layer"].shape), (2, 3))

    def test_out_of_bounds_answer_span_raises(self) -> None:
        layers = normalize_hidden_states([torch.zeros(3, 2)])

        with self.assertRaises(ValueError):
            slice_answer(layers, answer_start=2, answer_end=4, layers="all")

    def test_hidden_state_batch_dim(self) -> None:
        accepted = normalize_hidden_states([torch.zeros(1, 3, 2)])
        self.assertEqual(tuple(accepted[0].shape), (3, 2))

        with self.assertRaises(ValueError):
            normalize_hidden_states([torch.zeros(2, 3, 2)])

    def test_mahalanobis_stats_produce_finite_distances(self) -> None:
        records = [
            {
                "hidden_states": [torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])],
                "answer_start_token_idx": 0,
                "answer_end_token_idx": 3,
            },
            {
                "hidden_states": [torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 2.0]])],
                "answer_start_token_idx": 0,
                "answer_end_token_idx": 3,
            },
        ]
        stats = fit_mahalanobis_stats(records, layers="all", regularization=1e-3)

        result = compute_mahalanobis(
            records[0]["hidden_states"],
            answer_start=0,
            answer_end=3,
            stats=stats,
        )

        self.assertEqual(tuple(result["mahalanobis_distance"].shape), (3,))
        self.assertTrue(torch.isfinite(result["mahalanobis_distance"]).all())

    def test_pca_deviation_reconstructs_exact_line(self) -> None:
        record = {
            "hidden_states": [torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])],
            "answer_start_token_idx": 0,
            "answer_end_token_idx": 3,
        }
        stats = fit_pca_stats([record], layers="all", n_components=1)

        result = compute_pca_deviation(
            record["hidden_states"],
            answer_start=0,
            answer_end=3,
            stats=stats,
        )

        self.assertTrue(torch.allclose(result["pca_deviation"], torch.zeros(3), atol=1e-5))

    def test_logit_lens_divergence_matches_final_logits(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = torch.nn.Linear(2, 3, bias=False)
                with torch.no_grad():
                    self.head.weight.copy_(
                        torch.tensor(
                            [
                                [1.0, 0.0],
                                [0.0, 1.0],
                                [1.0, 1.0],
                            ]
                        )
                    )

            def get_output_embeddings(self) -> torch.nn.Linear:
                return self.head

        hidden_states = [torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])]
        model = TinyModel()
        final_logits = model.get_output_embeddings()(hidden_states[0])

        result = compute_logit_lens_divergence(
            hidden_states,
            final_logits,
            answer_start=0,
            answer_end=3,
            model=model,
            layers="all",
        )

        self.assertEqual(tuple(result["logit_lens_divergence"].shape), (3,))
        self.assertTrue(torch.allclose(result["logit_lens_divergence"], torch.zeros(3), atol=1e-6))

    def test_composite_requires_train_normalizer_stats(self) -> None:
        metrics = {"cosine_drift": torch.tensor([0.0, 1.0])}

        with self.assertRaises(ValueError):
            compute_composite_score(metrics, {})


if __name__ == "__main__":
    unittest.main()
