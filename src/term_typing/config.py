from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class DatasetConfig:
    train_path: str
    test_path: str
    term_field: str = "term"
    context_field: str = "context"
    label_field: str = "label"


@dataclass
class PreprocessConfig:
    input_mode: str = "marked_context"  # context_only | context_plus_term | marked_context
    max_length: int = 128
    lowercase_term_match: bool = True
    add_special_tokens_markers: bool = True
    marker_left: str = "[TGT]"
    marker_right: str = "[/TGT]"


@dataclass
class ModelConfig:
    name: str
    num_labels: int = 4


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # None/[] => auto


@dataclass
class TrainConfig:
    output_dir: str
    run_name: str = "run"
    seed: int = 42

    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    fp16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200

    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True


@dataclass
class DataYAML:
    dataset: DatasetConfig
    preprocess: PreprocessConfig


@dataclass
class TrainYAML:
    model: ModelConfig
    lora: LoRAConfig
    train: TrainConfig


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data_config(path: str) -> DataYAML:
    cfg = _read_yaml(path)
    ds = DatasetConfig(**cfg["dataset"])
    pp = PreprocessConfig(**cfg.get("preprocess", {}))
    return DataYAML(dataset=ds, preprocess=pp)


def load_train_config(path: str) -> TrainYAML:
    cfg = _read_yaml(path)
    model = ModelConfig(**cfg["model"])
    lora_dict = cfg.get("lora", {})
    lora = LoRAConfig(**lora_dict)
    train = TrainConfig(**cfg["train"])
    # normalize target_modules
    if lora.target_modules is None:
        lora.target_modules = []
    return TrainYAML(model=model, lora=lora, train=train)
