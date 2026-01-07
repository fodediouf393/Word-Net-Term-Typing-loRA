from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import os


MODELS = [
    ("bert", "bert-base-uncased"),
    ("roberta", "roberta-base"),
    ("distilbert", "distilbert-base-uncased"),
]


def run(cmd: list[str], env: dict) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd, env=env)


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

    # Always ensure module import works
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")

    data_yaml = str(Path(args.data_yaml).resolve())
    baseline_yaml = str(Path(args.baseline_yaml).resolve())
    lora_yaml = str(Path(args.lora_yaml).resolve())
    test_path = str(Path(args.test_path).resolve())

    for short, model_name in MODELS:
        for mode in ["baseline", "lora"]:
            cfg = baseline_yaml if mode == "baseline" else lora_yaml

            run_name = f"{short}-{mode}-{args.epochs_note}-{timestamp}"
            output_dir = out_root / short / mode / run_name

            # 1) Train
            run(
                [
                    "python", "-m", "term_typing.train",
                    "--config", cfg,
                    "--data", data_yaml,
                    "--model_name", model_name,
                    "--output_dir", str(output_dir),
                    "--run_name", run_name,
                    "--lora_enabled", "true" if mode == "lora" else "false",
                ],
                env=env,
            )

            # 2) Predict on test
            pred_path = repo / "outputs" / "predictions" / f"{run_name}.jsonl"
            pred_path.parent.mkdir(parents=True, exist_ok=True)

            run(
                [
                    "python", "-m", "term_typing.predict",
                    "--ckpt", str(output_dir / "checkpoints" / "best"),
                    "--input", test_path,
                    "--output", str(pred_path),
                    "--data", data_yaml,
                ],
                env=env,
            )

            # 3) Eval from ID (gold parsed from ID)
            run(
                [
                    "python", "-m", "term_typing.eval_from_id",
                    "--test", test_path,
                    "--pred", str(pred_path),
                    "--run_name", run_name,
                    "--out_metrics_dir", str(repo / "outputs" / "metrics"),
                    "--out_images_dir", str(repo / "outputs" / "report_images"),
                ],
                env=env,
            )

    print("\n Sweep complete.")
    print(f"Runs:   {out_root}")
    print(f"Images: {repo / 'outputs' / 'report_images'}")
    print(f"Metrics:{repo / 'outputs' / 'metrics'}")
    print(f"Preds:  {repo / 'outputs' / 'predictions'}")


if __name__ == "__main__":
    main()
