import os
import json
import time
import random
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from openai import OpenAI

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "downloads" / "llms4ol_2024" / "TaskA-Term Typing" / "SubTask A.1(FS) - WordNet"
TRAIN_FILE = DATA_DIR / "A.1(FS)_WordNet_Train.json"
TEST_FILE = DATA_DIR / "A.1(FS)_WordNet_Test.json"
GT_FILE = DATA_DIR / "A.1(FS)_WordNet_Test_GT.json"

OUTPUT_DIR = ROOT / "output_llms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROVIDER = "llama_groq"
MODEL = "llama-3.1-8b-instant"

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

def postprocess(pred):
    pred = str(pred).strip().lower()
    pred = pred.replace(".", "").replace(",", "").replace(":", "")
    pred = pred.split()[0] if pred else ""
    return pred if pred in ALLOWED else "unknown"

def backoff_sleep(base_sleep, attempt):
    return base_sleep * (2 ** attempt) + random.uniform(0, 0.5)

def call_llm_openai_compat(client, model, prompt, max_retries=10, base_sleep=1.0):
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            retriable = (
                ("429" in msg) or ("rate limit" in msg) or ("rate_limited" in msg) or
                ("503" in msg) or ("service unavailable" in msg) or ("overflow" in msg) or
                ("502" in msg) or ("bad gateway" in msg) or
                ("504" in msg) or ("gateway timeout" in msg) or
                ("timeout" in msg) or ("disconnect" in msg) or ("reset" in msg) or ("connection" in msg)
            )
            if not retriable:
                raise
            time.sleep(backoff_sleep(base_sleep, attempt))
    raise RuntimeError("Transient API errors persisted after max retries.")

def load_existing_predictions(path: Path):
    if not path.exists():
        return {}
    try:
        preds = load_json(path)
        return {p["ID"]: p["predicted_type"] for p in preds if "ID" in p and "predicted_type" in p}
    except Exception:
        return {}

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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY non définie")

    safe_model = MODEL.replace("/", "_").replace(":", "_")
    OUTPUT_PRED_FILE = OUTPUT_DIR / f"A.1(FS)_WordNet_Predictions_{PROVIDER}_{safe_model}.json"
    OUTPUT_METRICS_FILE = OUTPUT_DIR / f"A.1(FS)_WordNet_Metrics_{PROVIDER}_{safe_model}.json"
    OUTPUT_CM_PNG = OUTPUT_DIR / f"A.1(FS)_WordNet_ConfMat_{PROVIDER}_{safe_model}.png"

    train_data = load_json(TRAIN_FILE)
    test_data = load_json(TEST_FILE)
    gt_data = load_json(GT_FILE)

    fewshot_block = build_fewshot_examples(train_data, N_FEWSHOT)

    pred_map = load_existing_predictions(OUTPUT_PRED_FILE)
    if pred_map:
        print(f"Resume enabled: {len(pred_map)} predictions already saved.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )

    for item in tqdm(test_data):
        item_id = item["ID"]
        if item_id in pred_map:
            continue

        prompt = build_prompt(item["term"], item.get("sentence", ""), fewshot_block)
        raw = call_llm_openai_compat(client, MODEL, prompt)
        pred_map[item_id] = postprocess(raw)

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
        y_true, y_pred, labels=ALLOWED_LABELS, digits=4, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
    save_confusion_matrix_png(cm, ALLOWED_LABELS, OUTPUT_CM_PNG, f"{PROVIDER} | {MODEL}")

    metrics = {
        "provider": PROVIDER,
        "model": MODEL,
        "n_fewshot": N_FEWSHOT,
        "sleep_time": SLEEP_TIME,
        "checkpoint_every": CHECKPOINT_EVERY,
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
