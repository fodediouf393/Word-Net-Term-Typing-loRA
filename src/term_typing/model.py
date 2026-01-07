from __future__ import annotations

from typing import List, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType


def infer_lora_target_modules(model_name: str) -> List[str]:
    """Essaie de choisir des modules LoRA compatibles selon la famille."""
    name = model_name.lower()

    # BERT/RoBERTa: attention.self.query / attention.self.value
    if "bert" in name or "roberta" in name:
        return ["query", "value"]

    # DeBERTa-v2/v3: attention.query_proj / attention.value_proj
    if "deberta" in name:
        return ["query_proj", "value_proj"]

    # DistilBERT: attention.q_lin / attention.v_lin
    if "distilbert" in name:
        return ["q_lin", "v_lin"]

    # fallback: essayer query/value
    return ["query", "value"]


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_model(
    model_name: str,
    num_labels: int,
    lora_enabled: bool,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if not lora_enabled:
        return model

    target_modules = lora_target_modules or []
    if len(target_modules) == 0:
        target_modules = infer_lora_target_modules(model_name)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    # Affiche nb paramètres entraînables
    model.print_trainable_parameters()
    return model
