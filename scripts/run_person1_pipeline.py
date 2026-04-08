from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

# Ensure src-layout imports work when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

config_mod = import_module("nlp_track_b.person1.config")
pipeline_mod = import_module("nlp_track_b.person1.pipeline")
ModelConfig = config_mod.ModelConfig
PipelineConfig = config_mod.PipelineConfig
SplitConfig = config_mod.SplitConfig
run_person1_pipeline = pipeline_mod.run_person1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Person 1 data + inference pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to raw JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for split manifests and model outputs",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "hf"],
        default="hf",
        help="Forward-pass backend. 'mock' is deterministic and dependency-free.",
    )
    parser.add_argument(
        "--model-name",
        default="distilgpt2",
        help="HF model name if provider='hf'",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        raw_dataset_path=args.dataset,
        output_dir=args.output_dir,
        split=SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ),
        model=ModelConfig(provider=args.provider, model_name=args.model_name),
    )
    summary = run_person1_pipeline(cfg)
    print(summary)


if __name__ == "__main__":
    main()
