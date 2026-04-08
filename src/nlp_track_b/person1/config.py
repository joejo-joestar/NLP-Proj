from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass(slots=True)
class ModelConfig:
    provider: str = "hf"
    model_name: str = "distilgpt2"
    max_context_docs: int = 3
    max_seq_len: int = 512
    hidden_size: int = 32
    num_layers: int = 4
    vocab_size: int = 128


@dataclass(slots=True)
class PipelineConfig:
    raw_dataset_path: Path
    output_dir: Path
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        self.split.validate()
        if not self.raw_dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.raw_dataset_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
