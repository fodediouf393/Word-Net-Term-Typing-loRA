import os
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Chemins projet
ROOT = Path(__file__).resolve().parent

# Données WordNet
DATA_DIR = ROOT / "downloads" / "llms4ol_2024" / "TaskA-Term Typing" / "SubTask A.1(FS) - WordNet"
TRAIN_FILE = DATA_DIR / "A.1(FS)_WordNet_Train.json"
TEST_FILE = DATA_DIR / "A.1(FS)_WordNet_Test.json"
GT_FILE = DATA_DIR / "A.1(FS)_WordNet_Test_GT.json"

# Sorties communes
OUTPUT_DIR = ROOT / "output_llms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres
PROVIDER = "sml_local"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_FEWSHOT = 6

ALLOWED_LABELS = ["noun", "verb", "adjective", "adverb"]
ALLOWED = set(ALLOWED_LABELS)

random.seed(42)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def norm_type(t):
    if isinstance(t, list) and t:
        t = t[0]
    return str(t).strip().lower()

def build_fewshot_examples(train_data, n):
    n = min(n, len(train_data))
    samples = random.sample(train_data, n)
    block = ""
    for s in samples:
        block += (
            f"\nTerm: {s['term']}\n"
            f"Sentence: {s.get('sentence', '')}\n"
            f"Type: {norm_type(s['type'])}\n"
        )
    return block.strip()

def build_prompt(term, sentence, fewshot_block):
    return f"""
You are a linguistics expert.

Your task is to determine the part of speech of a lexical term.
Choose ONLY one of the following labels:
- noun
- verb
- adjective
- adverb

Here are some examples:
{fewshot_block}

Now classify the following term:
Term: {term}
Sentence: {sentence}

Answer with a single word from: noun, verb, adjective, adverb.
""".strip()

def postprocess(pred):
    pred = str(pred).strip().lower()
    pred = pred.replace(".", "").replace(",", "").replace(":", "")
    pred = pred.split()[0] if pred else ""
    return pred if pred in ALLOWED else "unknown"

def call_llm_local(generator, prompt):
    out = generator(
        prompt,
        max_new_tokens=8,
        do_sample=False,
        return_full_text=False
    )
    if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)

def save_confusion_matrix_png(cm, labels, path: Path, title: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    train_data = load_json(TRAIN_FILE)
    test_data = load_json(TEST_FILE)
    gt_data = load_json(GT_FILE)

    fewshot_block = build_fewshot_examples(train_data, N_FEWSHOT)

    safe_model = MODEL.replace("/", "_").replace(":", "_")
    pred_path = OUTPUT_DIR / f"A.1(FS)_WordNet_Predictions_{PROVIDER}_{safe_model}.json"
    metrics_path = OUTPUT_DIR / f"A.1(FS)_WordNet_Metrics_{PROVIDER}_{safe_model}.json"
    cm_png_path = OUTPUT_DIR / f"A.1(FS)_WordNet_ConfMat_{PROVIDER}_{safe_model}.png"

    generator = pipeline("text-generation", model=MODEL, device_map="auto")

    predictions = []
    for item in tqdm(test_data):
        prompt = build_prompt(item["term"], item.get("sentence", ""), fewshot_block)
        raw = call_llm_local(generator, prompt)
        pred = postprocess(raw)
        predictions.append({"ID": item["ID"], "predicted_type": pred})

    save_json(predictions, pred_path)

    gt_map = {x["ID"]: norm_type(x["type"]) for x in gt_data}
    pred_map = {x["ID"]: x["predicted_type"] for x in predictions}

    y_true = [gt_map[k] for k in gt_map.keys()]
    y_pred = [pred_map.get(k, "unknown") for k in gt_map.keys()]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true, y_pred, labels=ALLOWED_LABELS, digits=4, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    save_confusion_matrix_png(cm, ALLOWED_LABELS, cm_png_path, f"{PROVIDER} | {MODEL}")

    metrics = {
        "provider": PROVIDER,
        "model": MODEL,
        "n_fewshot": N_FEWSHOT,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": ALLOWED_LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "outputs": {
            "predictions_json": str(pred_path),
            "metrics_json": str(metrics_path),
            "confusion_matrix_png": str(cm_png_path),
        },
    }
    save_json(metrics, metrics_path)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {cm_png_path}")

if __name__ == "__main__":
    main()
