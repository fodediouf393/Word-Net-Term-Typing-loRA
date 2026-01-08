import os
import json
import time
import random
from pathlib import Path

from tqdm import tqdm
from mistralai import Mistral

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "downloads" / "llms4ol_2024" / "TaskA-Term Typing" / "SubTask A.1(FS) - WordNet"
TRAIN_FILE = DATA_DIR / "A.1(FS)_WordNet_Train.json"
TEST_FILE = DATA_DIR / "A.1(FS)_WordNet_Test.json"
GT_FILE = DATA_DIR / "A.1(FS)_WordNet_Test_GT.json"

OUTPUT_DIR = ROOT / "output_llms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PRED_FILE = OUTPUT_DIR / "A.1(FS)_WordNet_Predictions_Mistral2025.json"
OUTPUT_METRICS_FILE = OUTPUT_DIR / "A.1(FS)_WordNet_Metrics_Mistral2025.json"

MODEL_MISTRAL = "mistral-small-latest"
N_FEWSHOT = 6

SLEEP_TIME = 1.2

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

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        )
    return str(content)

def postprocess(pred):
    pred = str(pred).strip().lower()
    pred = pred.replace(".", "").replace(",", "").replace(":", "")
    pred = pred.split()[0] if pred else ""
    return pred if pred in ALLOWED else "unknown"

def call_llm(client, prompt, max_retries=8, base_sleep=1.0):
    for attempt in range(max_retries):
        try:
            res = client.chat.complete(
                model=MODEL_MISTRAL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "text"},
            )
            return extract_text(res.choices[0].message.content)

        except Exception as e:
            msg = str(e)
            is_429 = ("Status 429" in msg) or ("rate_limited" in msg) or ('"code":"1300"' in msg)
            if not is_429:
                raise

            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Rate limit (429). Retry {attempt+1}/{max_retries} in {sleep_s:.2f}s")
            time.sleep(sleep_s)

    raise RuntimeError("Rate limit persisted after max retries.")

def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY non définie")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file introuvable: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file introuvable: {TEST_FILE}")
    if not GT_FILE.exists():
        raise FileNotFoundError(f"GT file introuvable: {GT_FILE}")

    train_data = load_json(TRAIN_FILE)
    test_data = load_json(TEST_FILE)
    gt_data = load_json(GT_FILE)

    fewshot_block = build_fewshot_examples(train_data, N_FEWSHOT)

    predictions = []

    with Mistral(api_key=api_key) as client:
        for item in tqdm(test_data):
            prompt = build_prompt(item["term"], item.get("sentence", ""), fewshot_block)
            pred = postprocess(call_llm(client, prompt))

            predictions.append({
                "ID": item["ID"],
                "predicted_type": pred,
            })

            time.sleep(SLEEP_TIME)

    save_json(predictions, OUTPUT_PRED_FILE)

    gt_map = {x["ID"]: norm_type(x["type"]) for x in gt_data}
    pred_map = {x["ID"]: x["predicted_type"] for x in predictions}

    y_true = [gt_map[k] for k in gt_map.keys()]
    y_pred = [pred_map.get(k, "unknown") for k in gt_map.keys()]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=ALLOWED_LABELS,
        digits=4,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS).tolist()

    metrics = {
        "model": MODEL_MISTRAL,
        "n_fewshot": N_FEWSHOT,
        "sleep_time": SLEEP_TIME,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": ALLOWED_LABELS,
        "confusion_matrix": cm,
        "classification_report": report,
        "paths": {
            "train": str(TRAIN_FILE),
            "test": str(TEST_FILE),
            "gt": str(GT_FILE),
            "predictions_out": str(OUTPUT_PRED_FILE),
            "metrics_out": str(OUTPUT_METRICS_FILE),
        }
    }

    save_json(metrics, OUTPUT_METRICS_FILE)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels:", ALLOWED_LABELS)
    print(confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS))
    print(f"Predictions saved to: {OUTPUT_PRED_FILE}")
    print(f"Metrics saved to: {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    main()
