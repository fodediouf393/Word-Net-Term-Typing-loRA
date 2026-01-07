from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from term_typing.config import load_data_config
from term_typing.constants import ID2LABEL
from term_typing.features import FeatureBuilder
from term_typing.utils import ensure_dir, write_jsonl


def load_input_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.lower().endswith(".json"):
        return pd.read_json(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


@torch.inference_mode()
def predict_rows(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    model.eval()
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Chemin checkpoint (best)")
    parser.add_argument("--input", required=True, help="Fichier JSONL/CSV d'entrées")
    parser.add_argument("--output", required=True, help="Fichier JSONL de sortie")
    parser.add_argument("--data", required=True, help="Data config YAML (pour input_mode)")
    args = parser.parse_args()

    data_cfg = load_data_config(args.data)

    df = load_input_table(args.input)

    term_field = data_cfg.dataset.term_field
    context_field = data_cfg.dataset.context_field

    if term_field not in df.columns:
        raise KeyError(f"Input is missing '{term_field}' column")
    if context_field not in df.columns:
        df[context_field] = ""

    feature_builder = FeatureBuilder(
        input_mode=data_cfg.preprocess.input_mode,
        lowercase_term_match=data_cfg.preprocess.lowercase_term_match,
        add_special_tokens_markers=data_cfg.preprocess.add_special_tokens_markers,
        marker_left=data_cfg.preprocess.marker_left,
        marker_right=data_cfg.preprocess.marker_right,
    )

    texts = [
        feature_builder.build_text(str(row[term_field]), str(row[context_field]))
        for _, row in df.iterrows()
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    probs = predict_rows(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=64,
        max_length=data_cfg.preprocess.max_length,
    )

    pred_ids = probs.argmax(axis=-1)
    pred_labels = [ID2LABEL[int(i)] for i in pred_ids]
    pred_confs = probs.max(axis=-1)

    rows_out: List[Dict] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        rows_out.append({
            "term": row[term_field],
            "context": row[context_field],
            "pred_label": pred_labels[idx],
            "pred_conf": float(pred_confs[idx]),
            "probs": {ID2LABEL[i]: float(probs[idx, i]) for i in range(probs.shape[1])},
        })

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    write_jsonl(out_path, rows_out)

    print(f" Predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
