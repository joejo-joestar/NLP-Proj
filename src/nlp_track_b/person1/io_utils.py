from __future__ import annotations

import json
from pathlib import Path

from .schemas import ModelOutput


def save_model_output(base_dir: Path, output: ModelOutput) -> Path:
    """Save one reusable Person 1 forward-pass artifact."""
    assert output.token_length == len(output.input_ids) == len(output.attention_mask)
    assert 0 <= output.answer_start_token_idx < output.answer_end_token_idx <= output.token_length
    assert len(output.logits) == output.token_length
    assert output.hidden_states

    target_dir = base_dir / "model_outputs" / output.split
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{output.sample_id}.json"

    payload = {
        "id": output.sample_id,
        "sample_id": output.sample_id,
        "split": output.split,
        "question": output.question,
        "context": output.context,
        "answer": output.answer,
        "full_input_text": output.full_input_text,
        "prompt": output.prompt,
        "input_ids": output.input_ids,
        "attention_mask": output.attention_mask,
        "token_length": output.token_length,
        "answer_start_token_idx": output.answer_start_token_idx,
        "answer_end_token_idx": output.answer_end_token_idx,
        "answer_token_range": {
            "start": output.answer_start_token_idx,
            "end": output.answer_end_token_idx,
        },
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
