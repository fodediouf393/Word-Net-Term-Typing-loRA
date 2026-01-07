from __future__ import annotations

import argparse
from pathlib import Path
import inspect
import time

import torch
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from term_typing.config import load_data_config, load_train_config
from term_typing.data import load_datasets
from term_typing.features import FeatureBuilder
from term_typing.metrics import hf_compute_metrics
from term_typing.model import build_model, build_tokenizer
from term_typing.utils import ensure_dir, save_json, set_seed


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    return ds.map(tok, batched=True)


def dataset_has_labels(ds: Dataset) -> bool:
    if "label" not in ds.column_names:
        return False
    if "label_str" in ds.column_names:
        return any(x is not None for x in ds["label_str"])
    return any((isinstance(x, int) and x >= 0) for x in ds["label"])


def _get_eval_strategy_key() -> str:
    params = inspect.signature(TrainingArguments.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


def count_params(model) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Train config YAML (baseline.yaml ou lora.yaml)")
    parser.add_argument("--data", required=True, help="Data config YAML (data.yaml)")
    parser.add_argument("--resume_from_checkpoint", default=None)

    # Overrides pour sweeps
    parser.add_argument("--model_name", default=None, help="Override model name (e.g., bert-base-uncased)")
    parser.add_argument("--output_dir", default=None, help="Override output dir (e.g., outputs/runs/bert/baseline)")
    parser.add_argument("--run_name", default=None, help="Override run name for logging (TensorBoard)")
    parser.add_argument("--lora_enabled", default=None, choices=["true", "false"], help="Override LoRA enable/disable")

    args = parser.parse_args()

    data_cfg = load_data_config(args.data)
    train_cfg = load_train_config(args.config)

    # Apply overrides
    if args.model_name:
        train_cfg.model.name = args.model_name
    if args.output_dir:
        train_cfg.train.output_dir = args.output_dir
    if args.run_name:
        train_cfg.train.run_name = args.run_name
    if args.lora_enabled is not None:
        train_cfg.lora.enabled = (args.lora_enabled.lower() == "true")

    # Output structure
    output_dir = Path(train_cfg.train.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    ensure_dir(checkpoints_dir)
    ensure_dir(metrics_dir)
    ensure_dir(logs_dir)

    set_seed(train_cfg.train.seed)

    feature_builder = FeatureBuilder(
        input_mode=data_cfg.preprocess.input_mode,
        lowercase_term_match=data_cfg.preprocess.lowercase_term_match,
        add_special_tokens_markers=data_cfg.preprocess.add_special_tokens_markers,
        marker_left=data_cfg.preprocess.marker_left,
        marker_right=data_cfg.preprocess.marker_right,
    )

    train_ds, test_ds = load_datasets(
        train_path=data_cfg.dataset.train_path,
        test_path=data_cfg.dataset.test_path,
        term_field=data_cfg.dataset.term_field,
        context_field=data_cfg.dataset.context_field,
        label_field=data_cfg.dataset.label_field,
        feature_builder=feature_builder,
    )

    # test n'a pas de labels => on désactive eval pendant training (Option 1)
    do_eval = dataset_has_labels(test_ds)

    tokenizer = build_tokenizer(train_cfg.model.name)
    train_tok = tokenize_dataset(train_ds, tokenizer, data_cfg.preprocess.max_length)
    test_tok = tokenize_dataset(test_ds, tokenizer, data_cfg.preprocess.max_length)

    model = build_model(
        model_name=train_cfg.model.name,
        num_labels=train_cfg.model.num_labels,
        lora_enabled=train_cfg.lora.enabled,
        lora_r=train_cfg.lora.r,
        lora_alpha=train_cfg.lora.alpha,
        lora_dropout=train_cfg.lora.dropout,
        lora_target_modules=train_cfg.lora.target_modules,
    )

    trainable, total, pct = count_params(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_key = _get_eval_strategy_key()

    ta_kwargs = dict(
        output_dir=str(checkpoints_dir),
        run_name=train_cfg.train.run_name,

        num_train_epochs=train_cfg.train.epochs,
        per_device_train_batch_size=train_cfg.train.batch_size,
        per_device_eval_batch_size=train_cfg.train.eval_batch_size,

        learning_rate=train_cfg.train.learning_rate,
        weight_decay=train_cfg.train.weight_decay,
        warmup_ratio=train_cfg.train.warmup_ratio,

        fp16=train_cfg.train.fp16,
        gradient_accumulation_steps=train_cfg.train.gradient_accumulation_steps,
        max_grad_norm=train_cfg.train.max_grad_norm,

        logging_steps=train_cfg.train.logging_steps,

        save_strategy="steps",
        save_steps=train_cfg.train.save_steps,
        save_total_limit=3,

        report_to=["tensorboard"],
        logging_dir=str(logs_dir),
    )

    if do_eval:
        ta_kwargs[eval_key] = "steps"
        ta_kwargs["eval_steps"] = train_cfg.train.eval_steps
        ta_kwargs["load_best_model_at_end"] = train_cfg.train.load_best_model_at_end
        ta_kwargs["metric_for_best_model"] = train_cfg.train.metric_for_best_model
        ta_kwargs["greater_is_better"] = train_cfg.train.greater_is_better
    else:
        ta_kwargs[eval_key] = "no"
        ta_kwargs["load_best_model_at_end"] = False

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_compute_metrics if do_eval else None,
    )

    # VRAM peak tracking (si GPU)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    t1 = time.time()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        max_vram_mb = 0.0

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # No eval here (Option 1)
    if do_eval:
        eval_metrics = trainer.evaluate(eval_dataset=test_tok)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    else:
        eval_metrics = {"note": "No labels in test set; skipped evaluation."}
        save_json(metrics_dir / "eval_skipped.json", eval_metrics)

    # Save best-like checkpoint (last state) as "best" for predict
    best_dir = output_dir / "checkpoints" / "best"
    ensure_dir(best_dir)
    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    run_summary = {
        "run_name": train_cfg.train.run_name,
        "model_name": train_cfg.model.name,
        "mode": "lora" if train_cfg.lora.enabled else "baseline",
        "epochs": train_cfg.train.epochs,
        "batch_size": train_cfg.train.batch_size,
        "max_length": data_cfg.preprocess.max_length,
        "fp16": bool(train_cfg.train.fp16),
        "train_runtime_sec": float(t1 - t0),
        "train_metrics": train_result.metrics,
        "max_vram_mb": float(max_vram_mb),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": float(pct),
        "checkpoint_best": str(best_dir),
    }
    save_json(metrics_dir / "run_summary.json", run_summary)
    save_json(metrics_dir / "final_metrics.json", {"train": train_result.metrics, "eval": eval_metrics})

    print(f"\n Done. Checkpoint saved to: {best_dir}")
    print("TensorBoard:")
    print(f"  tensorboard --logdir {output_dir.parent} --port 6006")


if __name__ == "__main__":
    main()
