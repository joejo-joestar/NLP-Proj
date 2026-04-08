from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person1.config import ModelConfig
from nlp_track_b.person1.data import load_jsonl_dataset, normalize_samples
from nlp_track_b.person1.formatting import build_formatted_sample
from nlp_track_b.person1.model import ForwardRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one Person 1 HF input/output sample")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/person1/converted_raw.jsonl"),
        help="Converted Person 1 JSONL dataset",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Debug split label to attach to the sample",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "hf"],
        default="hf",
        help="Forward-pass backend",
    )
    parser.add_argument("--model-name", default="distilgpt2", help="HF model name")
    parser.add_argument("--max-context-docs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512)
    return parser.parse_args()


def _shape_2d(rows: list[list[float]]) -> tuple[int, int]:
    return (len(rows), len(rows[0]) if rows else 0)


def _shape_3d(rows: list[list[list[float]]]) -> tuple[int, int, int]:
    if not rows or not rows[0]:
        return (len(rows), 0, 0)
    return (len(rows), len(rows[0]), len(rows[0][0]))


def main() -> None:
    """Print one sample's formatted text, answer-token range, and tensor shapes."""
    args = parse_args()
    assert args.index >= 0
    assert args.max_context_docs >= 1
    assert args.max_seq_len >= 1

    raw_samples = load_jsonl_dataset(args.dataset)
    samples = normalize_samples(raw_samples, max_context_docs=args.max_context_docs)
    assert args.index < len(samples), f"Index {args.index} out of range for {len(samples)} samples"

    sample = build_formatted_sample(samples[args.index], split=args.split)
    runner = ForwardRunner(
        ModelConfig(
            provider=args.provider,
            model_name=args.model_name,
            max_context_docs=args.max_context_docs,
            max_seq_len=args.max_seq_len,
        )
    )
    output = runner.run(sample)

    print("id:", output.sample_id)
    print("split:", output.split)
    print("question:", output.question)
    print("context:")
    for idx, doc in enumerate(output.context, start=1):
        print(f"[{idx}] {doc}")
    print("answer:", output.answer)
    print("formatted input:")
    print(output.full_input_text)
    print("token count:", output.token_length)
    print(
        "answer token range:",
        output.answer_start_token_idx,
        output.answer_end_token_idx,
    )
    print("logits shape:", _shape_2d(output.logits))
    print("hidden states shape:", _shape_3d(output.hidden_states))


if __name__ == "__main__":
    main()
