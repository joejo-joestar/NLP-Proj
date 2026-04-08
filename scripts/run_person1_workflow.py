from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person1.config import ModelConfig, PipelineConfig, SplitConfig
from nlp_track_b.person1.conversion import convert_ragtruth_to_person1
from nlp_track_b.person1.pipeline import run_person1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAGTruth data and run the Person 1 pipeline end to end"
    )
    parser.add_argument(
        "--response-jsonl",
        type=Path,
        default=Path("dataset/response.jsonl"),
        help="Path to RAGTruth response.jsonl",
    )
    parser.add_argument(
        "--source-info-jsonl",
        type=Path,
        default=Path("dataset/source_info.jsonl"),
        help="Path to RAGTruth source_info.jsonl",
    )
    parser.add_argument(
        "--converted-jsonl",
        type=Path,
        default=None,
        help="Intermediate converted JSONL in the Person 1 schema",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/person1"),
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
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for smoke tests (0 = all rows)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted_jsonl = args.converted_jsonl or (args.output_dir / "converted_raw.jsonl")

    conversion_summary = convert_ragtruth_to_person1(
        response_jsonl=args.response_jsonl,
        source_info_jsonl=args.source_info_jsonl,
        output_jsonl=converted_jsonl,
        limit=args.limit,
    )

    cfg = PipelineConfig(
        raw_dataset_path=converted_jsonl,
        output_dir=args.output_dir,
        split=SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ),
        model=ModelConfig(provider=args.provider, model_name=args.model_name),
    )
    pipeline_summary = run_person1_pipeline(cfg)

    print(
        json.dumps(
            {
                "conversion": conversion_summary,
                "pipeline": pipeline_summary,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
