from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PipelineConfig
from .data import (
    load_jsonl_dataset,
    normalize_samples,
    save_split_manifests,
    split_samples,
)
from .formatting import build_formatted_sample
from .io_utils import save_model_output, save_run_summary
from .model import ForwardRunner
from tqdm import tqdm


def run_person1_pipeline(cfg: PipelineConfig) -> dict[str, int]:
    cfg.validate()

    raw = load_jsonl_dataset(cfg.raw_dataset_path)
    cleaned = normalize_samples(raw, max_context_docs=cfg.model.max_context_docs)
    if cfg.limit_samples:
        cleaned = cleaned[: cfg.limit_samples]
    split_map = split_samples(cleaned, cfg.split)
    save_split_manifests(split_map, cfg.output_dir)

    del raw, cleaned
    gc.collect()

    runner = ForwardRunner(cfg.model)
    summary: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_samples = sum(len(samples) for samples in split_map.values())
    batch_size = cfg.model.batch_size if cfg.model.provider == "hf" else 1
    max_pending_writes = 16

    overall_bar = tqdm(
        total=total_samples,
        desc="Person1 total",
        unit="sample",
    )

    pending_writes = set()

    try:
        with ThreadPoolExecutor(max_workers=1) as writer_pool:
            for split_name, samples in split_map.items():
                split_iter = tqdm(
                    range(0, len(samples), batch_size),
                    desc=f"Person1 {split_name}",
                    unit="batch",
                    leave=False,
                )
                for start_idx in split_iter:
                    batch_samples = samples[start_idx : start_idx + batch_size]
                    formatted_batch = [
                        build_formatted_sample(sample, split=split_name)
                        for sample in batch_samples
                    ]
                    outputs = runner.run_batch(formatted_batch)
                    for output in outputs:
                        pending_writes.add(
                            writer_pool.submit(
                                save_model_output, cfg.output_dir, output
                            )
                        )
                        summary[split_name] += 1
                        overall_bar.update(1)

                        if len(pending_writes) >= max_pending_writes:
                            done_future = next(as_completed(pending_writes))
                            done_future.result()
                            pending_writes.discard(done_future)
                            gc.collect()
                            _clear_cuda_cache_if_available(cfg)

                    del batch_samples, formatted_batch, outputs
                    gc.collect()
                    _clear_cuda_cache_if_available(cfg)

                del samples
                gc.collect()

            for future in as_completed(pending_writes):
                future.result()
    finally:
        overall_bar.close()

    pending_writes.clear()
    del split_map
    gc.collect()
    _clear_cuda_cache_if_available(cfg)

    summary["total"] = summary["train"] + summary["val"] + summary["test"]
    save_run_summary(cfg.output_dir, summary)
    return summary


def _clear_cuda_cache_if_available(cfg: PipelineConfig) -> None:
    if cfg.model.provider != "hf":
        return
    if not cfg.model.device.startswith("cuda") and cfg.model.device != "auto":
        return

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
