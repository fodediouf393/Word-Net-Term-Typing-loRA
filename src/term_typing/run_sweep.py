from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime


MODELS = [
    ("bert", "bert-base-uncased"),
    ("roberta", "roberta-base"),
    ("distilbert", "distilbert-base-uncased"),
]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".", help="Repo root")
    p.add_argument("--data_yaml", required=True)
    p.add_argument("--baseline_yaml", required=True)
    p.add_argument("--lora_yaml", required=True)
    p.add_argument("--test_path", required=True, help="Path to test.json (for eval_from_id)")
    p.add_argument("--epochs_note", default="e5", help="Just for naming runs")
    args = p.parse_args()

    repo = Path(args.repo).resolve()
    out_root = repo / "outputs" / "runs"
    out_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ensure python module path is correct
    env = dict(**{**__import__("os").environ})
    env["PYTHONPATH"] = str(repo / "src")

    for short, model_name in MODELS:
        for mode in ["baseline", "lora"]:
            cfg = args.baseline_yaml if mode == "baseline" else args.lora_yaml

            run_name = f"{short}-{mode}-{args.epochs_note}-{timestamp}"
            output_dir = out_root / short / mode / run_name

            # Train
            run([
                "python", "-m", "term_typing.train",
                "--config", str(Path(cfg).resolve()),
                "--data", str(Path(args.data_yaml).resolve()),
                "--model_name", model_name,
                "--output_dir", str(output_dir),
                "--run_name", run_name,
                "--lora_enabled", "true" if mode == "lora" else "false",
            ])

            # Predict on test
            pred_path = repo / "outputs" / "predictions" / f"{run_name}.jsonl"
            pred_path.parent.mkdir(parents=True, exist_ok=True)

            run([
                "python", "-m", "term_typing.predict",
                "--ckpt", str(output_dir / "checkpoints" / "best"),
                "--input", str(Path(args.test_path).resolve()),
                "--output", str(pred_path),
                "--data", str(Path(args.data_yaml).resolve()),
            ])

            # Eval from ID
            run([
                "python", "scripts/eval_from_id.py",
                "--test", str(Path(args.test_path).resolve()),
                "--pred", str(pred_path),
                "--run_name", run_name,
                "--out_metrics_dir", str(repo / "outputs" / "metrics"),
                "--out_images_dir", str(repo / "outputs" / "report_images"),
            ])

    print("\n Sweep complete.")
    print(f"Runs in: {out_root}")
    print(f"Images in: {repo / 'outputs' / 'report_images'}")
    print(f"Metrics in: {repo / 'outputs' / 'metrics'}")


if __name__ == "__main__":
    main()
