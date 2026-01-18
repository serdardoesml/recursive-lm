"""
Minimal CORE eval runner for RecursiveGPT.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from recursive_lm.common import get_base_dir
from recursive_lm.model import ModelConfig, RecursiveGPT
from recursive_lm.tokenizer import RustBPETokenizer
from evals.core_eval import evaluate_task


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _load_random_baselines(path: Path):
    if not path.exists():
        return {}
    baselines = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Eval Task")
            value = row.get("Random baseline")
            if not name or value is None or value == "":
                continue
            try:
                baselines[name] = float(value)
            except ValueError:
                continue
    return baselines


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CORE evals for a RecursiveGPT checkpoint.")
    parser.add_argument("--model", required=True, help="Model filename under models/")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer filename under tokenizers/")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda, cuda:0, cpu).")
    args = parser.parse_args()

    base_dir = get_base_dir()
    evals_dir = base_dir / "evals" / "data"
    if not evals_dir.exists() or not any(evals_dir.iterdir()):
        print("Error: evals/data is missing or empty.", file=sys.stderr)
        return 1

    core_dir = evals_dir / "core"
    config_path = core_dir / "core.yaml"
    data_base = core_dir / "eval_data"
    meta_path = core_dir / "eval_meta_data.csv"
    if not config_path.exists() or not data_base.exists():
        print("Error: evals/data/core is missing core.yaml or eval_data/", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config.get("icl_tasks", [])
    if not tasks:
        print("Error: core.yaml has no tasks.", file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = base_dir / "models" / args.model
    tokenizer_path = base_dir / "tokenizers" / args.tokenizer
    if not model_path.exists():
        print(f"Error: model not found at {model_path}", file=sys.stderr)
        return 1
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found at {tokenizer_path}", file=sys.stderr)
        return 1

    tokenizer = RustBPETokenizer.load_from_dir(str(tokenizer_path))
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "config" not in checkpoint:
        print("Error: checkpoint missing config/state_dict.", file=sys.stderr)
        return 1
    config = ModelConfig(**checkpoint["config"])
    model = RecursiveGPT(config)
    model.load_state_dict(checkpoint["state_dict"])
    if "cuda" in str(device):
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    model.eval()

    random_baselines = _load_random_baselines(meta_path)

    results = {}
    centered_results = {}

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in str(device) else torch.no_grad()
    with autocast_ctx:
        for task in tasks:
            label = task["label"]
            task_meta = {
                "task_type": task["icl_task_type"],
                "dataset_uri": task["dataset_uri"],
                "num_fewshot": task["num_fewshot"][0] if isinstance(task["num_fewshot"], list) else task["num_fewshot"],
                "continuation_delimiter": task.get("continuation_delimiter", " "),
            }
            data_path = data_base / task_meta["dataset_uri"]
            if not data_path.exists():
                print(f"Warning: missing dataset {data_path}, skipping.", file=sys.stderr)
                continue

            start = time.time()
            data = _load_jsonl(data_path)
            accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
            elapsed = time.time() - start
            results[label] = accuracy

            if label in random_baselines:
                baseline = random_baselines[label]
                centered = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
                centered_results[label] = centered
                print(f"{label}: acc {accuracy:.4f} | centered {centered:.4f} | {elapsed:.2f}s")
            else:
                print(f"{label}: acc {accuracy:.4f} | {elapsed:.2f}s")

    mean_acc = sum(results.values()) / len(results) if results else 0.0
    core_metric = sum(centered_results.values()) / len(centered_results) if centered_results else None

    log_dir = base_dir / "evals" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"core_eval_{Path(args.model).stem}_{ts}.json"
    payload = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "device": device,
        "results": results,
        "centered_results": centered_results,
        "mean_accuracy": mean_acc,
        "core_metric": core_metric,
        "timestamp": ts,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote results to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
