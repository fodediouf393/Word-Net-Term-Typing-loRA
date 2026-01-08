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
CHECKPOINT_EVERY = 1

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

def call_llm(client, prompt, max_retries=10, base_sleep=1.0):
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
            msg = str(e).lower()
            retriable = (
                ("status 429" in msg) or ("rate_limited" in msg) or ('"code":"1300"' in msg) or
                ("status 503" in msg) or ("service unavailable" in msg) or ("overflow" in msg) or
                ("status 502" in msg) or ("bad gateway" in msg) or
                ("status 504" in msg) or ("gateway timeout" in msg) or
                ("disconnect" in msg) or ("reset" in msg) or ("connection" in msg)
            )
            if not retriable:
                raise

            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Transient API error. Retry {attempt+1}/{max_retries} in {sleep_s:.2f}s")
            time.sleep(sleep_s)

    raise RuntimeError("Transient API errors persisted after max retries.")

def load_existing_predictions(path: Path):
    if not path.exists():
        return {}
    try:
        preds = load_json(path)
        return {p["ID"]: p["predicted_type"] for p in preds if "ID" in p and "predicted_type" in p}
    except Exception:
        return {}

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

    print("OUTPUT_DIR =", OUTPUT_DIR)
    print("OUTPUT_PRED_FILE =", OUTPUT_PRED_FILE)
    print("OUTPUT_METRICS_FILE =", OUTPUT_METRICS_FILE)

    train_data = load_json(TRAIN_FILE)
    test_data = load_json(TEST_FILE)
    gt_data = load_json(GT_FILE)

    fewshot_block = build_fewshot_examples(train_data, N_FEWSHOT)

    pred_map = load_existing_predictions(OUTPUT_PRED_FILE)
    if pred_map:
        print(f"Resume enabled: {len(pred_map)} predictions already saved.")

    with Mistral(api_key=api_key) as client:
        for item in tqdm(test_data):
            item_id = item["ID"]
            if item_id in pred_map:
                continue

            prompt = build_prompt(item["term"], item.get("sentence", ""), fewshot_block)
            pred = postprocess(call_llm(client, prompt))
            pred_map[item_id] = pred

            out_list = [{"ID": k, "predicted_type": v} for k, v in pred_map.items()]
            atomic_save_json(out_list, OUTPUT_PRED_FILE)

            time.sleep(SLEEP_TIME)

    out_list = [{"ID": k, "predicted_type": v} for k, v in pred_map.items()]
    atomic_save_json(out_list, OUTPUT_PRED_FILE)

    gt_map = {x["ID"]: norm_type(x["type"]) for x in gt_data}
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
        "checkpoint_every": CHECKPOINT_EVERY,
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

    atomic_save_json(metrics, OUTPUT_METRICS_FILE)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("Labels:", ALLOWED_LABELS)
    print(confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS))
    print(f"Predictions saved to: {OUTPUT_PRED_FILE}")
    print(f"Metrics saved to: {OUTPUT_METRICS_FILE}")

if __name__ == "__main__":
    main()
