from __future__ import annotations

from .config import PipelineConfig
from .data import load_jsonl_dataset, normalize_samples, save_split_manifests, split_samples
from .formatting import build_formatted_sample
from .io_utils import save_model_output, save_run_summary
from .model import ForwardRunner


def run_person1_pipeline(cfg: PipelineConfig) -> dict[str, int]:
    cfg.validate()

    raw = load_jsonl_dataset(cfg.raw_dataset_path)
    cleaned = normalize_samples(raw, max_context_docs=cfg.model.max_context_docs)
    split_map = split_samples(cleaned, cfg.split)
    save_split_manifests(split_map, cfg.output_dir)

    runner = ForwardRunner(cfg.model)
    summary: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for split_name, samples in split_map.items():
        for sample in samples:
            formatted = build_formatted_sample(sample, split=split_name)
            output = runner.run(formatted)
            save_model_output(cfg.output_dir, output)
            summary[split_name] += 1

    summary["total"] = summary["train"] + summary["val"] + summary["test"]
    save_run_summary(cfg.output_dir, summary)
    return summary
