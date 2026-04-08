from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _sample_id(record: dict[str, Any]) -> str:
    sample_id = record.get("sample_id", record.get("id"))
    if not sample_id:
        raise ValueError("Person 1 artifact missing 'sample_id'/'id'.")
    return str(sample_id)


def _seq_len_from_hidden_states(hidden_states: Any) -> int:
    if not hidden_states:
        raise ValueError("Person 1 artifact has empty 'hidden_states'.")
    first_layer = torch.as_tensor(hidden_states[0])
    if first_layer.ndim == 2:
        return int(first_layer.shape[0])
    if first_layer.ndim == 3:
        if first_layer.shape[0] != 1:
            raise ValueError("Only batch size 1 artifacts are supported.")
        return int(first_layer.shape[1])
    raise ValueError(f"Unsupported hidden-state layer shape: {tuple(first_layer.shape)}")


def validate_person1_artifact(record: dict[str, Any], *, require_logits: bool = False) -> None:
    """Validate the minimum Person 1 fields needed for Person 2 metrics."""
    _sample_id(record)
    for key in ("hidden_states", "answer_start_token_idx", "answer_end_token_idx"):
        if key not in record:
            raise ValueError(f"Person 1 artifact missing required key: {key}")
    if require_logits and "logits" not in record:
        raise ValueError("Person 1 artifact missing required key: logits")

    seq_len = len(record.get("input_ids", [])) or _seq_len_from_hidden_states(record["hidden_states"])
    answer_start = int(record["answer_start_token_idx"])
    answer_end = int(record["answer_end_token_idx"])
    if not 0 <= answer_start < answer_end <= seq_len:
        raise ValueError(
            "Invalid answer token range: "
            f"start={answer_start}, end={answer_end}, seq_len={seq_len}"
        )

    if "attention_mask" in record and len(record["attention_mask"]) != seq_len:
        raise ValueError("attention_mask length does not match sequence length.")
    if require_logits and len(record["logits"]) != seq_len:
        raise ValueError("logits length does not match sequence length.")


def load_person1_artifact(path: Path, *, require_logits: bool = False) -> dict[str, Any]:
    """Load a Person 1 output artifact from JSON or compact torch .pt format."""
    path = Path(path)
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    elif path.suffix in {".pt", ".pth"}:
        record = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported artifact extension: {path.suffix}")

    if not isinstance(record, dict):
        raise ValueError(f"Artifact must contain a dictionary: {path}")
    validate_person1_artifact(record, require_logits=require_logits)
    return record


def iter_artifact_paths(path: Path) -> list[Path]:
    """Return sorted Person 1 artifact paths from a file or directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact path not found: {path}")
    files = [
        p
        for p in path.rglob("*")
        if p.suffix in {".json", ".pt", ".pth"}
        and ".person2_metrics" not in p.name
        and "person2_stats" not in p.name
    ]
    return sorted(files)


def save_metric_artifact(path: Path, artifact: dict[str, Any]) -> Path:
    """Save a small Person 2 metric artifact as JSON or torch .pt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        serializable = {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in artifact.items()
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=True, indent=2)
    elif path.suffix in {".pt", ".pth"}:
        torch.save(artifact, path)
    else:
        raise ValueError(f"Unsupported metric artifact extension: {path.suffix}")
    return path
