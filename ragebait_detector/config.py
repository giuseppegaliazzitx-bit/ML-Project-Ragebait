from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathConfig:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    unlabeled_dir: str = "data/unlabeled"
    labeled_dir: str = "data/labeled"
    output_dir: str = "outputs"
    unified_posts_path: str = "data/interim/unified_posts.csv"
    source_manifest_path: str = "data/interim/source_manifest.json"
    annotation_template_path: str = "data/interim/annotation_template.csv"
    labeled_posts_path: str = "data/interim/labeled_posts.csv"
    processed_dataset_path: str = "data/processed/processed_posts.csv"
    unlabeled_posts_path: str = "data/unlabeled/unified_unlabeled_posts.csv"
    vllm_labels_path: str = "data/labeled/vllm_ragebait_labels.csv"


@dataclass
class DataConfig:
    min_posts: int = 20000
    text_column: str = "text"
    label_column: str = "label"
    language_column: str = "language"
    supported_languages: list[str] = field(default_factory=lambda: ["en"])
    drop_non_english: bool = True
    min_text_length: int = 3
    augment_minority_class: bool = True
    augmentation_copies: int = 1
    max_baseline_features: int = 20000
    test_size: float = 0.10
    validation_size: float = 0.10


@dataclass
class BaselineConfig:
    enabled: bool = True
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)


@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    tokenizer_name: str | None = None
    hidden_dim: int = 256
    dropout: float = 0.2
    max_length: int = 256
    stride: int = 96


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    patience: int = 2
    min_delta: float = 0.001
    num_workers: int = 0
    decision_threshold: float = 0.50
    seed: int = 42
    device: str = "auto"


@dataclass
class VLLMConfig:
    model: str = "Qwen/Qwen2.5-3B-Instruct-AWQ"
    quantization: str = "awq"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 1024
    temperature: float = 0.0
    limit: int | None = 50000
    random_seed: int = 42
    enable_random: bool = True
    balance_by_source: bool = True


@dataclass
class Settings:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)


def _deep_update_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    for key, value in overrides.items():
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _deep_update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_settings(config_path: str | Path | None = None) -> Settings:
    settings = Settings()
    if config_path is None:
        return settings
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    return _deep_update_dataclass(settings, overrides)


def as_dict(settings: Settings) -> dict[str, Any]:
    return asdict(settings)
