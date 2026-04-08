from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.artifacts import load_person1_artifact
from nlp_track_b.person2.metrics import (
    compute_cosine_drift,
    compute_logit_lens_divergence,
)


def _parse_layers(text: str) -> str | list[int]:
    if text in {"all", "last4"}:
        return text
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Person 2 token-level metrics on one sample")
    parser.add_argument("input", type=Path, help="Person 1 output artifact (.json or .pt)")
    parser.add_argument("--layers", default="last4", help="'last4', 'all', or comma-separated indices")
    parser.add_argument(
        "--include-logit-lens",
        action="store_true",
        help="Also compute logit lens divergence by loading an HF model",
    )
    parser.add_argument("--model-name", default=None, help="HF model name for logit lens")
    return parser.parse_args()


def main() -> None:
    """Print shape checks for one sample without reporting evaluation metrics."""
    args = parse_args()
    layers = _parse_layers(args.layers)
    record = load_person1_artifact(args.input, require_logits=args.include_logit_lens)
    sample_id = record.get("sample_id", record.get("id"))
    answer_start = int(record["answer_start_token_idx"])
    answer_end = int(record["answer_end_token_idx"])
    answer_count = answer_end - answer_start

    cosine = compute_cosine_drift(
        record["hidden_states"],
        answer_start,
        answer_end,
        layers=layers,
    )

    print(f"example_id={sample_id}")
    print(f"seq_len={len(record['input_ids'])}")
    print(f"answer_start={answer_start}, answer_end={answer_end}, A={answer_count}")
    print(f"hidden_states_layers={len(record['hidden_states'])}")
    print(f"layers_used={cosine['layers_used']}")
    print(f"cosine_drift={tuple(cosine['cosine_drift'].shape)}")
    print(f"cosine_drift_per_layer={tuple(cosine['cosine_drift_per_layer'].shape)}")
    print(f"first_5_cosine_drift={cosine['cosine_drift'][:5].tolist()}")

    if args.include_logit_lens:
        model_name = args.model_name or record.get("metadata", {}).get("hf_model")
        if not model_name:
            raise ValueError("--model-name is required when metadata.hf_model is missing.")
        logit_lens = compute_logit_lens_divergence(
            record["hidden_states"],
            record["logits"],
            answer_start,
            answer_end,
            model_name=model_name,
            layers=layers,
        )
        print(f"logit_lens_divergence={tuple(logit_lens['logit_lens_divergence'].shape)}")
        print(
            "logit_lens_divergence_per_layer="
            f"{tuple(logit_lens['logit_lens_divergence_per_layer'].shape)}"
        )
        print(f"first_5_logit_lens={logit_lens['logit_lens_divergence'][:5].tolist()}")


if __name__ == "__main__":
    main()
