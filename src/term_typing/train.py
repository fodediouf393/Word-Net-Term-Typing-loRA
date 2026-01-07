from __future__ import annotations

import argparse
from pathlib import Path

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
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    # retire les colonnes non nécessaires au modèle après tokenization
    tokenized = ds.map(tok, batched=True)
    # Trainer utilise 'labels' ou 'label'. Ici 'label' ok.
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Train config YAML (baseline.yaml ou lora.yaml)")
    parser.add_argument("--data", required=True, help="Data config YAML (data.yaml)")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    data_cfg = load_data_config(args.data)
    train_cfg = load_train_config(args.config)

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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
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
        evaluation_strategy="steps",
        eval_steps=train_cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=train_cfg.train.save_steps,

        load_best_model_at_end=train_cfg.train.load_best_model_at_end,
        metric_for_best_model=train_cfg.train.metric_for_best_model,
        greater_is_better=train_cfg.train.greater_is_better,

        report_to=["tensorboard"],
        logging_dir=str(logs_dir),

        # Limiter le nombre de checkpoints (évite d’encombrer)
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_compute_metrics,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Eval final
    eval_metrics = trainer.evaluate(eval_dataset=test_tok)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Sauvegarde "best" lisible (model + tokenizer)
    best_dir = output_dir / "checkpoints" / "best"
    ensure_dir(best_dir)

    trainer.model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Sauvegarder aussi les configs utilisées pour reproductibilité
    save_json(metrics_dir / "final_metrics.json", {"train": train_result.metrics, "eval": eval_metrics})

    print(f"\n Done. Best checkpoint saved to: {best_dir}")
    print("TensorBoard:")
    print(f"  tensorboard --logdir {output_dir.parent} --port 6006")


if __name__ == "__main__":
    main()
