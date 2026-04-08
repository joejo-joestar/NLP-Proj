from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from tqdm import tqdm

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.artifacts import iter_artifact_paths, load_person1_artifact
from nlp_track_b.person2.evaluation import (
    aggregate_metric_aurocs,
    extract_answer_labels,
)


def _iter_metric_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Metric path not found: {path}")
    files = [
        item
        for item in path.rglob("*")
        if item.suffix in {".json", ".pt", ".pth"} and ".person2_metrics" in item.name
    ]
    return sorted(files)


def _load_metric_artifact(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    elif path.suffix in {".pt", ".pth"}:
        record = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported metric artifact extension: {path.suffix}")
    if not isinstance(record, dict):
        raise ValueError(f"Metric artifact must contain a dictionary: {path}")
    return record


def _labels_index(person1_input: Path, *, split: str | None) -> dict[str, torch.Tensor]:
    index: dict[str, torch.Tensor] = {}
    paths = iter_artifact_paths(person1_input)
    iterator = tqdm(paths, desc="Indexing Person1 labels", unit="artifact")
    for path in iterator:
        record = load_person1_artifact(path, require_logits=False)
        if split and record.get("split") != split:
            continue
        sample_id = str(record.get("sample_id") or record.get("id") or "")
        if not sample_id:
            continue
        index[sample_id] = extract_answer_labels(record)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Person 2 token-level AUROC metrics"
    )
    parser.add_argument(
        "--metric-input",
        type=Path,
        default=Path("outputs/person2/metrics"),
        help="Person 2 metric artifact file or directory",
    )
    parser.add_argument(
        "--person1-input",
        type=Path,
        default=Path("outputs/person1/model_outputs"),
        help="Person 1 artifact file or directory for token labels",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to evaluate (train|val|test). Use empty string to include all splits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional metric artifact limit for smoke runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/person2/eval/auroc_summary.json"),
        help="Output JSON path for AUROC summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split or None

    metric_paths = _iter_metric_paths(args.metric_input)
    if args.limit:
        metric_paths = metric_paths[: args.limit]
    if not metric_paths:
        raise ValueError(
            f"No Person 2 metric artifacts found under: {args.metric_input}"
        )

    labels_by_sample = _labels_index(args.person1_input, split=split)
    if not labels_by_sample:
        raise ValueError("No Person 1 labels found for the selected split.")

    metric_records: list[dict[str, Any]] = []
    iterator = tqdm(metric_paths, desc="Loading metric artifacts", unit="artifact")
    for path in iterator:
        metric_records.append(_load_metric_artifact(path))

    summary = aggregate_metric_aurocs(metric_records, labels_by_sample, split=split)
    summary["metadata"] = {
        "metric_input": str(args.metric_input),
        "person1_input": str(args.person1_input),
        "metric_artifact_count": len(metric_paths),
        "label_sample_count": len(labels_by_sample),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    print(f"saved_auroc_summary={args.output}")
    print(f"processed_samples={summary['processed_samples']}")
    print(f"metrics={sorted(summary['metrics'])}")


if __name__ == "__main__":
    main()
