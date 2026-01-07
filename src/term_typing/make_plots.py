from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
BACKBONE_COLORS = {
    "bert-base-uncased": "tab:blue",
    "roberta-base": "tab:green",
    "distilbert-base-uncased": "tab:red",
}


def find_event_files(run_dir: Path) -> list[Path]:
    return list(run_dir.rglob("events.out.tfevents.*"))


def read_scalar(run_dir: Path, tag_candidates: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    ev_files = find_event_files(run_dir)
    if not ev_files:
        return None

    # pick latest file
    ev_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    ea = EventAccumulator(str(ev_files[0]))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    tag = None
    for cand in tag_candidates:
        if cand in tags:
            tag = cand
            break
    if tag is None:
        # try fuzzy
        for t in tags:
            for cand in tag_candidates:
                if cand in t:
                    tag = t
                    break
            if tag:
                break
    if tag is None:
        return None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, vals


def load_run_summary(run_dir: Path) -> dict | None:
    p = run_dir / "metrics" / "run_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_test_metrics(metrics_dir: Path, run_name: str) -> dict | None:
    p = metrics_dir / f"{run_name}_test_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plot_overlay(lines: list[tuple[str, np.ndarray, np.ndarray]], title: str, out_path: Path, ylabel: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for name, x, y in lines:
        color = BACKBONE_COLORS.get(name, None)  # fallback: cycle par défaut si inconnu
        if color is None:
            ax.plot(x, y, label=name)
        else:
            ax.plot(x, y, label=name, color=color)

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="outputs/runs")
    p.add_argument("--metrics_dir", default="outputs/metrics")
    p.add_argument("--images_dir", default="outputs/report_images")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    metrics_dir = Path(args.metrics_dir)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Discover runs: outputs/runs/{backbone}/{mode}/{run_name}
    run_dirs = []
    for run_dir in runs_root.rglob("metrics/run_summary.json"):
        run_dirs.append(run_dir.parent.parent)  # .../<run_name>

    # Group by mode
    by_mode = {"baseline": [], "lora": []}
    summaries = []
    for rd in run_dirs:
        s = load_run_summary(rd)
        if not s:
            continue
        summaries.append((rd, s))
        by_mode[s["mode"]].append((rd, s))

    # 1) Train loss overlays (3 models) separately for baseline vs lora
    for mode in ["baseline", "lora"]:
        lines = []
        for rd, s in sorted(by_mode[mode], key=lambda x: x[1]["model_name"]):
            scalar = read_scalar(rd, ["train/loss", "loss"])
            if scalar is None:
                continue
            steps, vals = scalar
            lines.append((s["model_name"], steps, vals))

        if lines:
            plot_overlay(
                lines,
                title=f"Train loss — {mode}",
                out_path=images_dir / f"train_loss_{mode}_overlay.png",
                ylabel="loss",
            )

    # 2) Learning rate overlays
    for mode in ["baseline", "lora"]:
        lines = []
        for rd, s in sorted(by_mode[mode], key=lambda x: x[1]["model_name"]):
            scalar = read_scalar(rd, ["train/learning_rate", "learning_rate"])
            if scalar is None:
                continue
            steps, vals = scalar
            lines.append((s["model_name"], steps, vals))

        if lines:
            plot_overlay(
                lines,
                title=f"Learning rate — {mode}",
                out_path=images_dir / f"lr_{mode}_overlay.png",
                ylabel="learning_rate",
            )

    # 3) Perf bar chart: macro-F1 test baseline vs lora per backbone
    # build table: backbone short from run_name prefix (bert/roberta/distilbert)
    rows = []
    for rd, s in summaries:
        run_name = s["run_name"]
        test_m = load_test_metrics(metrics_dir, run_name)
        if not test_m:
            continue
        # infer backbone short
        backbone = run_name.split("-")[0]
        rows.append({
            "backbone": backbone,
            "mode": s["mode"],
            "macro_f1": float(test_m["macro_f1"]),
            "accuracy": float(test_m["accuracy"]),
            "train_time_sec": float(s["train_runtime_sec"]),
            "max_vram_mb": float(s["max_vram_mb"]),
            "trainable_pct": float(s["trainable_pct"]),
        })

    if rows:
        backbones = sorted(set(r["backbone"] for r in rows))
        x = np.arange(len(backbones))
        width = 0.35

        def get_val(bb, mode, key):
            for r in rows:
                if r["backbone"] == bb and r["mode"] == mode:
                    return r[key]
            return np.nan

        macro_baseline = [get_val(bb, "baseline", "macro_f1") for bb in backbones]
        macro_lora = [get_val(bb, "lora", "macro_f1") for bb in backbones]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(x - width/2, macro_baseline, width, label="baseline", color="tab:blue")
        ax.bar(x + width/2, macro_lora, width, label="lora", color="tab:orange")

        ax.set_xticks(x)
        ax.set_xticklabels(backbones)
        ax.set_ylabel("macro-F1 (test)")
        ax.set_title("Macro-F1 — baseline vs LoRA (per backbone)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(images_dir / "perf_macro_f1_by_backbone.png", dpi=200)
        plt.close(fig)

        # 4) Costs table (csv + image)
        # Create a stable ordering
        table_rows = []
        for bb in backbones:
            for mode in ["baseline", "lora"]:
                table_rows.append([
                    bb, mode,
                    get_val(bb, mode, "train_time_sec"),
                    get_val(bb, mode, "max_vram_mb"),
                    get_val(bb, mode, "trainable_pct"),
                    get_val(bb, mode, "accuracy"),
                    get_val(bb, mode, "macro_f1"),
                ])

        csv_path = images_dir / "costs_table.csv"
        header = "backbone,mode,train_time_sec,max_vram_mb,trainable_pct,accuracy,macro_f1\n"
        lines = [header] + [
            f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n"
            for r in table_rows
        ]
        csv_path.write_text("".join(lines), encoding="utf-8")

        # render table image
        fig = plt.figure(figsize=(10, 2 + 0.35*len(table_rows)))
        ax = fig.add_subplot(111)
        ax.axis("off")

        col_labels = ["backbone", "mode", "train_time(s)", "max_vram(MB)", "trainable(%)", "acc", "macro-F1"]
        cell_text = []
        for r in table_rows:
            cell_text.append([
                r[0], r[1],
                f"{r[2]:.1f}",
                f"{r[3]:.0f}",
                f"{r[4]:.3f}",
                f"{r[5]:.3f}",
                f"{r[6]:.3f}",
            ])

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        fig.tight_layout()
        fig.savefig(images_dir / "costs_table.png", dpi=200)
        plt.close(fig)

    print(f" Images saved to: {images_dir}")


if __name__ == "__main__":
    main()
