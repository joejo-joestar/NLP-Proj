from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm


def load_source_info(path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Indexing source_info", unit="row"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            index[source_id] = {
                "task_type": str(row.get("task_type", "")),
                "source": str(row.get("source", "")),
                "source_info": str(row.get("source_info", "")),
                "prompt": str(row.get("prompt", "")),
            }
    return index


def map_label_type(label_type: str) -> str:
    text = label_type.lower()
    if "conflict" in text:
        return "contradictory"
    if "baseless" in text or "introduction" in text:
        return "unsupported"
    if "fabric" in text:
        return "fabricated"
    return "hallucinated"


def choose_question(prompt: str, task_type: str) -> str:
    cleaned = prompt.strip()
    if cleaned:
        return cleaned
    if task_type:
        return f"Task: {task_type}"
    return ""


def convert_ragtruth_to_person1(
    response_jsonl: Path,
    source_info_jsonl: Path,
    output_jsonl: Path,
    limit: int = 0,
) -> dict[str, str | int]:
    if not response_jsonl.exists():
        raise FileNotFoundError(f"Missing file: {response_jsonl}")
    if not source_info_jsonl.exists():
        raise FileNotFoundError(f"Missing file: {source_info_jsonl}")

    source_map = load_source_info(source_info_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_source = 0
    with (
        response_jsonl.open("r", encoding="utf-8") as src,
        output_jsonl.open("w", encoding="utf-8") as out,
    ):
        for line in tqdm(src, desc="Converting RAGTruth", unit="row"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            source_id = str(row.get("source_id", "")).strip()
            source_info = source_map.get(source_id, {})
            if not source_info:
                missing_source += 1

            spans = []
            for label in row.get("labels", []):
                spans.append(
                    {
                        "start": int(label.get("start", 0)),
                        "end": int(label.get("end", 0)),
                        "label": map_label_type(str(label.get("label_type", ""))),
                    }
                )

            prompt = str(source_info.get("prompt", "")).strip()
            task_type = str(source_info.get("task_type", "")).strip()
            question = choose_question(prompt=prompt, task_type=task_type)
            context_text = str(source_info.get("source_info", "")).strip()

            out_row = {
                "sample_id": f"ragtruth_{row.get('id')}",
                "source_id": source_id or f"missing_source_{row.get('id')}",
                "question": question,
                "retrieved_context": [context_text] if context_text else [],
                "answer": str(row.get("response", "")),
                "hallucination_spans": spans,
                "metadata": {
                    "model": row.get("model"),
                    "temperature": row.get("temperature"),
                    "split_original": row.get("split"),
                    "quality": row.get("quality"),
                    "task_type": source_info.get("task_type"),
                    "source": source_info.get("source"),
                },
            }

            out.write(json.dumps(out_row, ensure_ascii=True) + "\n")
            written += 1

            if limit and written >= limit:
                break

    return {
        "output": str(output_jsonl),
        "rows_written": written,
        "rows_missing_source_info": missing_source,
    }
