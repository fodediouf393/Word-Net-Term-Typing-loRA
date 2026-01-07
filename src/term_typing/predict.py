from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from term_typing.config import load_data_config
from term_typing.data import load_table
from term_typing.features import FeatureBuilder
from term_typing.constants import ID2LABEL, LABEL2ID


def is_peft_adapter_dir(ckpt_dir: Path) -> bool:
    # fichiers typiques PEFT
    return (ckpt_dir / "adapter_config.json").exists() or (ckpt_dir / "adapter_model.safetensors").exists() or (
        ckpt_dir / "adapter_model.bin"
    ).exists()


def load_model_for_inference(ckpt_dir: Path):
    """
    - Baseline: charge directement le modèle depuis ckpt_dir
    - LoRA/PEFT: charge base model avec num_labels=4 puis applique l'adapter
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_peft_adapter_dir(ckpt_dir):
        # Charge via PEFT
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(str(ckpt_dir))
        base_name = peft_cfg.base_model_name_or_path

        # IMPORTANT: forcer num_labels=4 + mapping
        cfg = AutoConfig.from_pretrained(
            base_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        base = AutoModelForSequenceClassification.from_pretrained(base_name, config=cfg)
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
    else:
        # Baseline (ou modèle merged)
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir))

    model.to(device)
    model.eval()
    return model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint dir (outputs/.../checkpoints/best)")
    parser.add_argument("--input", required=True, help="Test file path (json/jsonl/csv)")
    parser.add_argument("--output", required=True, help="Output predictions jsonl")
    parser.add_argument("--data", required=True, help="Data config YAML")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    data_cfg = load_data_config(args.data)

    ckpt = Path(args.ckpt)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tokenizer : si ckpt contient tokenizer, on le prend ; sinon base tokenizer auto.
    # (AutoTokenizer saura se débrouiller avec un dossier PEFT la plupart du temps,
    #  sinon il utilisera le base model indiqué dans l'adapter.)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    except Exception:
        # fallback: base tokenizer via peft config
        from peft import PeftConfig
        base_name = PeftConfig.from_pretrained(str(ckpt)).base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)

    model, device = load_model_for_inference(ckpt)

    feature_builder = FeatureBuilder(
        input_mode=data_cfg.preprocess.input_mode,
        lowercase_term_match=data_cfg.preprocess.lowercase_term_match,
        add_special_tokens_markers=data_cfg.preprocess.add_special_tokens_markers,
        marker_left=data_cfg.preprocess.marker_left,
        marker_right=data_cfg.preprocess.marker_right,
    )

    df = load_table(args.input)

    term_field = data_cfg.dataset.term_field
    context_field = data_cfg.dataset.context_field

    if term_field not in df.columns:
        raise KeyError(f"Missing term_field '{term_field}' in test columns: {list(df.columns)}")
    if context_field not in df.columns:
        df[context_field] = ""

    texts = []
    ids = df["ID"].tolist() if "ID" in df.columns else [None] * len(df)

    for _, row in df.iterrows():
        term = str(row[term_field]) if row[term_field] is not None else ""
        context = str(row[context_field]) if row[context_field] is not None else ""
        texts.append(feature_builder.build_text(term=term, context=context))

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=data_cfg.preprocess.max_length,
        padding=False,
        return_tensors=None,
    )

    # Build dataloader
    features = []
    for i in range(len(texts)):
        item = {k: enc[k][i] for k in enc.keys()}
        item["idx"] = i
        features.append(item)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dl = DataLoader(features, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    preds = [None] * len(texts)
    probs = [None] * len(texts)

    with torch.no_grad():
        for batch in dl:
            idx = batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.detach().cpu()
            p = torch.softmax(logits, dim=-1).numpy()
            y = logits.argmax(dim=-1).numpy()

            for j, ii in enumerate(idx.tolist()):
                preds[ii] = ID2LABEL[int(y[j])]
                probs[ii] = [float(x) for x in p[j].tolist()]

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(len(texts)):
            rec = {"ID": ids[i], "pred_label": preds[i], "probs": probs[i]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f" Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()
