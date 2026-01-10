import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS = ["noun", "verb", "adjective", "adverb"]

# COMMENTAIRE: renseigne ici le nombre de paramètres de Mistral si tu veux l'afficher (ex: "7B", "8B", "22B")
MISTRAL_PARAMS = "24B"

# COMMENTAIRE: Qwen2.5-0.5B est connu
QWEN_PARAMS = "0.5B"

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def norm_type(t):
    if isinstance(t, list) and t:
        t = t[0]
    return str(t).strip().lower()

def compute_metrics_and_cm(gt_map, pred_map):
    y_true = []
    y_pred = []
    for k in gt_map.keys():
        y_true.append(gt_map[k])
        y_pred.append(pred_map.get(k, "unknown"))
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return acc, macro, cm

def save_confusion(cm, out_png: Path, out_svg: Path, title: str):
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title, fontsize=11)
    plt.colorbar()
    plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close()

def save_bar_acc_and_f1(mistral_acc, mistral_f1, qwen_acc, qwen_f1, out_png: Path, out_svg: Path):
    metrics = ["Accuracy", "Macro-F1"]
    x = [0, 1]
    width = 0.35

    mistral_vals = [mistral_acc, mistral_f1]
    qwen_vals = [qwen_acc, qwen_f1]

    plt.figure(figsize=(7.2, 4.0))

    bars_m = plt.bar([i - width/2 for i in x], mistral_vals, width=width, color="green", label="Mistral")
    bars_q = plt.bar([i + width/2 for i in x], qwen_vals, width=width, color="red", label="Qwen2.5-0.5B")

    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Mistral vs Qwen (Few-shot) — Accuracy & Macro-F1")
    plt.legend()

    for b in list(bars_m) + list(bars_q):
        v = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close()

def save_llm_table_like_paper(df: pd.DataFrame, out_png: Path, out_svg: Path):
    col_labels = ["Model", "Params", "Accuracy", "Macro-F1"]

    df2 = df.copy()
    df2["Accuracy"] = df2["Accuracy"].map(lambda x: f"{float(x):.4f}")
    df2["Macro-F1"] = df2["Macro-F1"].map(lambda x: f"{float(x):.4f}")

    best_acc = df["Accuracy"].astype(float).max()
    best_f1 = df["Macro-F1"].astype(float).max()

    cell_text = df2[["Model", "Params", "Accuracy", "Macro-F1"]].values.tolist()

    fig, ax = plt.subplots(figsize=(11.0, 2.2))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_text_props(weight="bold", fontsize=14)
        else:
            cell.set_text_props(fontsize=13)

    for i in range(len(df)):
        if float(df.iloc[i]["Accuracy"]) == best_acc:
            tbl[(i + 1, 2)].set_text_props(weight="bold", fontsize=13)
        if float(df.iloc[i]["Macro-F1"]) == best_f1:
            tbl[(i + 1, 3)].set_text_props(weight="bold", fontsize=13)

    tbl.scale(1.15, 1.75)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

def main():
    root = Path(__file__).resolve().parent
    output_llms = root / "output_llms"

    out_dir = output_llms / "report_images_llms"
    out_dir.mkdir(parents=True, exist_ok=True)

    mistral_pred = output_llms / "A.1(FS)_WordNet_Predictions_Mistral2025.json"
    qwen_pred = output_llms / "A.1(FS)_WordNet_Predictions_sml_local_Qwen_Qwen2.5-0.5B-Instruct.json"

    if not mistral_pred.exists():
        raise FileNotFoundError(f"Introuvable: {mistral_pred}")
    if not qwen_pred.exists():
        raise FileNotFoundError(f"Introuvable: {qwen_pred}")

    gt_file = root / "downloads" / "llms4ol_2024" / "TaskA-Term Typing" / "SubTask A.1(FS) - WordNet" / "A.1(FS)_WordNet_Test_GT.json"
    if not gt_file.exists():
        raise FileNotFoundError(f"GT introuvable: {gt_file}")

    gt_data = load_json(gt_file)
    gt_map = {x["ID"]: norm_type(x["type"]) for x in gt_data}

    def load_pred_map(pred_path: Path):
        preds = load_json(pred_path)
        return {x["ID"]: str(x["predicted_type"]).strip().lower() for x in preds}

    mistral_map = load_pred_map(mistral_pred)
    qwen_map = load_pred_map(qwen_pred)

    mistral_acc, mistral_f1, mistral_cm = compute_metrics_and_cm(gt_map, mistral_map)
    qwen_acc, qwen_f1, qwen_cm = compute_metrics_and_cm(gt_map, qwen_map)

    save_bar_acc_and_f1(
        mistral_acc, mistral_f1,
        qwen_acc, qwen_f1,
        out_png=out_dir / "bar_mistral_vs_qwen_acc_and_macro_f1.png",
        out_svg=out_dir / "bar_mistral_vs_qwen_acc_and_macro_f1.svg",
    )

    save_confusion(
        mistral_cm,
        out_png=out_dir / "confusion_mistral.png",
        out_svg=out_dir / "confusion_mistral.svg",
        title="Mistral (few-shot)"
    )

    save_confusion(
        qwen_cm,
        out_png=out_dir / "confusion_qwen.png",
        out_svg=out_dir / "confusion_qwen.svg",
        title="Qwen2.5-0.5B (few-shot)"
    )

    llm_df = pd.DataFrame([
        {"Model": "Mistral", "Params": MISTRAL_PARAMS, "Accuracy": mistral_acc, "Macro-F1": mistral_f1},
        {"Model": "Qwen2.5-0.5B", "Params": QWEN_PARAMS, "Accuracy": qwen_acc, "Macro-F1": qwen_f1},
    ])

    save_llm_table_like_paper(
        llm_df,
        out_png=out_dir / "llm_results_table_like.png",
        out_svg=out_dir / "llm_results_table_like.svg",
    )

    summary = {
        "mistral": {"pred_file": mistral_pred.name, "params": MISTRAL_PARAMS, "accuracy": mistral_acc, "macro_f1": mistral_f1},
        "qwen": {"pred_file": qwen_pred.name, "params": QWEN_PARAMS, "accuracy": qwen_acc, "macro_f1": qwen_f1},
        "outputs_dir": str(out_dir),
    }
    with open(out_dir / "summary_mistral_qwen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved assets in: {out_dir}")
    print(f"Mistral: params={MISTRAL_PARAMS} acc={mistral_acc:.4f} macro_f1={mistral_f1:.4f}")
    print(f"Qwen:    params={QWEN_PARAMS} acc={qwen_acc:.4f} macro_f1={qwen_f1:.4f}")

if __name__ == "__main__":
    main()
