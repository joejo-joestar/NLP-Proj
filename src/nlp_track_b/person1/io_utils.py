from __future__ import annotations

import json
from pathlib import Path

from .schemas import ModelOutput


def save_model_output(base_dir: Path, output: ModelOutput) -> Path:
    target_dir = base_dir / "model_outputs" / output.split
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{output.sample_id}.json"

    payload = {
        "sample_id": output.sample_id,
        "split": output.split,
        "prompt": output.prompt,
        "token_outputs": output.token_outputs,
        "token_alignment": [
            {
                "token": x.token,
                "start": x.start,
                "end": x.end,
                "is_hallucinated": x.is_hallucinated,
                "hallucination_label": x.hallucination_label,
            }
            for x in output.token_alignment
        ],
        "hidden_states": output.hidden_states,
        "logits": output.logits,
        "metadata": output.metadata,
    }

    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)

    return out_file


def save_run_summary(base_dir: Path, summary: dict[str, int]) -> Path:
    out = base_dir / "run_summary.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    return out
