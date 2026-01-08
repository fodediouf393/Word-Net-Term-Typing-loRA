import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "downloads" / "llms4ol_2024" / "TaskA-Term Typing" / "SubTask A.1(FS) - WordNet"
TRAIN_FILE = DATA_DIR / "A.1(FS)_WordNet_Train.json"
TEST_FILE = DATA_DIR / "A.1(FS)_WordNet_Test.json"
GT_FILE = DATA_DIR / "A.1(FS)_WordNet_Test_GT.json"

OUTPUT_DIR = ROOT / "output_llms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROVIDER = "sml_local"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

N_FEWSHOT = 6

ALLOWED_LABELS = ["noun", "verb", "adjective", "adverb"]
ALLOWED = set(ALLOWED_LABELS)

random.seed(42)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_save_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

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
    return f"""Classify the POS of the term in context.
Return exactly one label from: noun, verb, adjective, adverb.

Examples:
{fewshot_block}

Term: {term}
Sentence: {sentence}
Label:""".strip()

def postprocess(pred):
    s = str(pred).strip().lower()
    s = s.replace(".", " ").replace(",", " ").replace(":", " ").replace(";", " ").replace("\n", " ")
    s = " ".join(s.split())

    mapping = {
        "nn": "noun", "nns": "noun", "nnp": "noun", "nnps": "noun",
        "vb": "verb", "vbd": "verb", "vbg": "verb", "vbn": "verb", "vbp": "verb", "vbz": "verb",
        "jj": "adjective", "jjr": "adjective", "jjs": "adjective", "adj": "adjective",
        "rb": "adverb", "rbr": "adverb", "rbs": "adverb", "adv": "adverb",
    }

    first = s.split()[0] if s else ""
    if first in ALLOWED:
        return first
    if first in mapping:
        return mapping[first]

    for lab in ["noun", "verb", "adjective", "adverb"]:
        if lab in s:
            return lab

    for k, v in mapping.items():
        if f" {k} " in f" {s} ":
            return v

    return "unknown"

def call_llm_local(generator, prompt):
    out = generator(prompt, max_new_tokens=4, do_sample=False, return_full_text=False)
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
    safe_model = MODEL.replace("/", "_").replace(":", "_")
    OUTPUT_PRED_FILE = OUTPUT_DIR / f"A.1(FS)_WordNet_Predictions_{PROVIDER}_{safe_model}.json"
    OUTPUT_METRICS_FILE = OUTPUT_DIR / f"A.1(FS)_WordNet_Metrics_{PROVIDER}_{safe_model}.json"
    OUTPUT_CM_PNG = OUTPUT_DIR / f"A.1(FS)_WordNet_ConfMat_{PROVIDER}_{safe_model}.png"

    if OUTPUT_PRED_FILE.exists():
        OUTPUT_PRED_FILE.unlink()
    if OUTPUT_METRICS_FILE.exists():
        OUTPUT_METRICS_FILE.unlink()
    if OUTPUT_CM_PNG.exists():
        OUTPUT_CM_PNG.unlink()

    train_data = load_json(TRAIN_FILE)
    test_data = load_json(TEST_FILE)
    gt_data = load_json(GT_FILE)

    fewshot_block = build_fewshot_examples(train_data, N_FEWSHOT)

    generator = pipeline("text-generation", model=MODEL, device_map="auto")

    predictions = []

    for item in tqdm(test_data[:5]):
        prompt = build_prompt(item["term"], item.get("sentence", ""), fewshot_block)
        raw = call_llm_local(generator, prompt)
        print("RAW:", repr(raw[:200]))
        print("POST:", postprocess(raw))
        print("---")

    for item in tqdm(test_data):
        prompt = build_prompt(item["term"], item.get("sentence", ""), fewshot_block)
        raw = call_llm_local(generator, prompt)
        pred = postprocess(raw)
        predictions.append({"ID": item["ID"], "predicted_type": pred})

    atomic_save_json(predictions, OUTPUT_PRED_FILE)

    gt_map = {x["ID"]: norm_type(x["type"]) for x in gt_data}
    pred_map = {x["ID"]: x["predicted_type"] for x in predictions}

    y_true = [gt_map[k] for k in gt_map.keys()]
    y_pred = [pred_map.get(k, "unknown") for k in gt_map.keys()]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true, y_pred, labels=ALLOWED_LABELS, digits=4, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    save_confusion_matrix_png(cm, ALLOWED_LABELS, OUTPUT_CM_PNG, f"{PROVIDER} | {MODEL}")

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
            "predictions_json": str(OUTPUT_PRED_FILE),
            "metrics_json": str(OUTPUT_METRICS_FILE),
            "confmat_png": str(OUTPUT_CM_PNG),
        },
    }
    atomic_save_json(metrics, OUTPUT_METRICS_FILE)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Saved: {OUTPUT_PRED_FILE}")
    print(f"Saved: {OUTPUT_METRICS_FILE}")
    print(f"Saved: {OUTPUT_CM_PNG}")

if __name__ == "__main__":
    main()
