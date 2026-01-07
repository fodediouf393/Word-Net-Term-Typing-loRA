from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

from peft import PeftModel

from term_typing.config import load_data_config
from term_typing.data import load_datasets
from term_typing.features import FeatureBuilder
from term_typing.metrics import compute_metrics_from_logits
from term_typing.utils import ensure_dir, is_peft_checkpoint, save_json


def load_model_and_tokenizer(ckpt_dir: str):
    ckpt_dir = str(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)

    # Si c'est un adapter PEFT sauvegardé, il faut reconstruire base+adapter
    if is_peft_checkpoint(ckpt_dir):
        # adapter saved in ckpt_dir; base model name is stored in adapter config
        base = AutoModelForSequenceClassification.from_pretrained(
            PeftModel.from_pretrained(
                AutoModelForSequenceClassification.from_pretrained(ckpt_dir),
                ckpt_dir
            )
        )
       

    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Chemin checkpoint (ex: outputs/runs/lora/checkpoints/best)")
    parser.add_argument("--data", required=True, help="Data config YAML (data.yaml)")
    args = parser.parse_args()

    data_cfg = load_data_config(args.data)

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

    model, tokenizer = load_model_and_tokenizer(args.ckpt)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=data_cfg.preprocess.max_length)

    test_tok = test_ds.map(tok, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = Path(args.ckpt).resolve().parents[2]  # .../outputs/runs/<run>
    metrics_dir = out_dir / "metrics"
    ensure_dir(metrics_dir)

    targs = TrainingArguments(
        output_dir=str(out_dir / "tmp_eval"),
        per_device_eval_batch_size=64,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    preds = trainer.predict(test_tok)
    logits = preds.predictions
    labels = preds.label_ids

    res = compute_metrics_from_logits(np.asarray(logits), np.asarray(labels))

    save_json(metrics_dir / "eval_only.json", {
        "accuracy": res.accuracy,
        "macro_f1": res.macro_f1,
        "confusion_matrix": res.confusion.tolist(),
    })

    print(" Eval done:")
    print(f"  accuracy:  {res.accuracy:.4f}")
    print(f"  macro_f1:  {res.macro_f1:.4f}")
    print(f"  confusion: saved to {metrics_dir / 'eval_only.json'}")


if __name__ == "__main__":
    main()
