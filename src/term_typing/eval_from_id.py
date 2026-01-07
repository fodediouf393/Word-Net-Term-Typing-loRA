from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


POS_TO_LABEL = {
    # nouns
    "NN": "noun", "NNS": "noun", "NNP": "noun", "NNPS": "noun",
    # verbs
    "VB": "verb", "VBD": "verb", "VBG": "verb", "VBN": "verb", "VBP": "verb", "VBZ": "verb",
    # adjectives
    "JJ": "adjective", "JJR": "adjective", "JJS": "adjective",
    # adverbs
    "RB": "adverb", "RBR": "adverb", "RBS": "adverb",
}

LABELS = ["noun", "verb", "adjective", "adverb"]


def gold_from_id(_id: str) -> str:
    # Exemple: "__telephone_VB_1"
    m = re.search(r"_([A-Z]{2,4})_\d+$", _id)
    if not m:
        raise ValueError(f"Cannot parse POS tag from ID: {_id}")
    pos = m.group(1)
    if pos not in POS_TO_LABEL:
        raise ValueError(f"Unknown POS tag '{pos}' in ID: {_id}")
    return POS_TO_LABEL[pos]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="Path to test.json (list of dicts with ID)")
    p.add_argument("--pred", required=True, help="Path to predictions jsonl (ID, pred_label)")
    p.add_argument("--run_name", required=True, help="Run name used for filenames")
    p.add_argument("--out_metrics_dir", default="outputs/metrics")
    p.add_argument("--out_images_dir", default="outputs/report_images")
    args = p.parse_args()

    metrics_dir = Path(args.out_metrics_dir)
    images_dir = Path(args.out_images_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    test = json.load(open(args.test, "r", encoding="utf-8"))
    gold_by_id = {x["ID"]: gold_from_id(x["ID"]) for x in test}

    pred_by_id = {}
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred_by_id[r["ID"]] = r["pred_label"]

    # align
    ids = [x["ID"] for x in test if x["ID"] in pred_by_id]
    y_true = [gold_by_id[i] for i in ids]
    y_pred = [pred_by_id[i] for i in ids]

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    out_json = {
        "run_name": args.run_name,
        "n": len(ids),
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    out_path = metrics_dir / f"{args.run_name}_test_metrics.json"
    out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # plot confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Pred")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion Matrix — {args.run_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig_path = images_dir / f"confusion_{args.run_name}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f" metrics: {out_path}")
    print(f" confusion: {fig_path}")


if __name__ == "__main__":
    main()
