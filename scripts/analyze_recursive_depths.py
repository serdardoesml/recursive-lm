"""Analyze how recursive-depth predictions evolve over shuffled batches."""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from recursive_lm.common import get_base_dir
from recursive_lm.dataloader import batch_iterator
from recursive_lm.model import ModelConfig, RecursiveGPT
from recursive_lm.tokenizer import RustBPETokenizer


@dataclass(frozen=True)
class AnalysisJob:
    model: str
    dataset: str
    tokenizer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run shuffled-batch diagnostics for a RecursiveGPT checkpoint and "
            "read out logits after every recursive depth."
        )
    )
    parser.add_argument(
        "--job",
        nargs=3,
        action="append",
        metavar=("MODEL", "DATASET", "TOKENIZER"),
        help=(
            "Analyze one checkpoint. MODEL is under models/, DATASET is under "
            "data/tokenized/, TOKENIZER is under tokenizers/. Can be repeated."
        ),
    )
    parser.add_argument("--model", help="Single-model shorthand: model filename under models/.")
    parser.add_argument("--dataset", help="Single-model shorthand: parquet filename under data/tokenized/.")
    parser.add_argument("--tokenizer", help="Single-model shorthand: tokenizer filename under tokenizers/.")
    parser.add_argument("--batch-tokens", type=int, default=8192, help="Packed tokens per batch.")
    parser.add_argument("--batches", type=int, default=8, help="Number of shuffled batches to analyze per model.")
    parser.add_argument("--sequence-len", type=int, default=256, help="Maximum segment length used by the dataloader.")
    parser.add_argument("--seed", type=int, default=12345, help="Dataset shuffle seed.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, cpu. Defaults to cuda if available.")
    parser.add_argument("--token-col", default="tokens", help="Parquet token column.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k accuracy to report.")
    parser.add_argument("--max-examples", type=int, default=8, help="Interesting last-token trajectories to include.")
    parser.add_argument("--context-tokens", type=int, default=32, help="Context tokens shown for examples.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to analysis/recursive_depths/<timestamp>.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    return parser.parse_args()


def parse_jobs(args: argparse.Namespace) -> list[AnalysisJob]:
    jobs = [AnalysisJob(*job) for job in args.job or []]
    single_values = [args.model, args.dataset, args.tokenizer]
    if any(v is not None for v in single_values):
        if not all(v is not None for v in single_values):
            raise SystemExit("--model, --dataset, and --tokenizer must be provided together.")
        jobs.append(AnalysisJob(args.model, args.dataset, args.tokenizer))
    if not jobs:
        raise SystemExit("Provide at least one --job MODEL DATASET TOKENIZER.")
    return jobs


def add_mean(stat: dict[str, float], name: str, value: float | None, weight: int | float) -> None:
    if value is None or not math.isfinite(float(value)):
        return
    stat[f"{name}_sum"] = stat.get(f"{name}_sum", 0.0) + float(value) * float(weight)
    stat[f"{name}_weight"] = stat.get(f"{name}_weight", 0.0) + float(weight)


def mean(stat: dict[str, float], name: str) -> float | None:
    weight = stat.get(f"{name}_weight", 0.0)
    if weight <= 0:
        return None
    return stat.get(f"{name}_sum", 0.0) / weight


def safe_ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def safe_exp(x: float | None) -> float | None:
    if x is None:
        return None
    if x > 80:
        return float("inf")
    return math.exp(x)


def rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a_f = a.detach().float()
    b_f = b.detach().float()
    denom = torch.linalg.norm(a_f) * torch.linalg.norm(b_f)
    if float(denom.item()) <= 1e-12:
        return None
    return float(torch.sum(a_f * b_f).item() / denom.item())


def readout_logits(model: RecursiveGPT, hidden: torch.Tensor) -> torch.Tensor:
    x = model.norm_out(hidden)
    if getattr(model, "use_factorized", False):
        x = model.h_to_e(x)
    if model.config.tie_embed:
        return F.linear(x, model.embedding.weight)
    return model.lm_head(x)


def load_model(base_dir: Path, model_name: str, device: str, sequence_len: int) -> RecursiveGPT:
    model_path = base_dir / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"Checkpoint {model_path} is missing config/state_dict.")

    config = ModelConfig(**checkpoint["config"])
    config.rope_cache_len = max(int(config.rope_cache_len), int(sequence_len))
    model = RecursiveGPT(config)
    model.load_state_dict(checkpoint["state_dict"])
    if "cuda" in str(device):
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    model.eval()
    return model


def make_depth_stats(depth_count: int) -> list[dict[str, float]]:
    rows = []
    for _ in range(depth_count):
        rows.append(
            {
                "tokens": 0.0,
                "last_tokens": 0.0,
                "loss_sum": 0.0,
                "last_loss_sum": 0.0,
                "top1_correct": 0.0,
                "topk_correct": 0.0,
                "last_top1_correct": 0.0,
                "last_topk_correct": 0.0,
                "margin_sum": 0.0,
                "last_margin_sum": 0.0,
                "changed_from_prev": 0.0,
                "last_changed_from_prev": 0.0,
                "final_agree": 0.0,
                "last_final_agree": 0.0,
                "stable_from_depth": 0.0,
                "last_stable_from_depth": 0.0,
            }
        )
    return rows


def add_logit_metrics(
    stat: dict[str, float],
    logits: torch.Tensor,
    targets: torch.Tensor,
    last_idx: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    n_tokens = int(targets.numel())
    k = max(1, min(int(top_k), int(logits.shape[-1])))

    loss = F.cross_entropy(logits, targets, reduction="mean")
    top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
    top1 = top_idx[:, 0]
    top1_correct = top1.eq(targets)
    topk_correct = top_idx.eq(targets[:, None]).any(dim=-1)

    stat["tokens"] += n_tokens
    stat["loss_sum"] += float(loss.detach().float().item()) * n_tokens
    stat["top1_correct"] += float(top1_correct.sum().item())
    stat["topk_correct"] += float(topk_correct.sum().item())
    if k >= 2:
        stat["margin_sum"] += float((top_vals[:, 0] - top_vals[:, 1]).detach().float().sum().item())

    if int(last_idx.numel()) > 0:
        last_targets = targets.index_select(0, last_idx)
        last_logits = logits.index_select(0, last_idx)
        last_loss = F.cross_entropy(last_logits, last_targets, reduction="mean")
        last_top_idx = top_idx.index_select(0, last_idx)
        last_top_vals = top_vals.index_select(0, last_idx)
        last_top1 = last_top_idx[:, 0]
        last_top1_correct = last_top1.eq(last_targets)
        last_topk_correct = last_top_idx.eq(last_targets[:, None]).any(dim=-1)

        n_last = int(last_idx.numel())
        stat["last_tokens"] += n_last
        stat["last_loss_sum"] += float(last_loss.detach().float().item()) * n_last
        stat["last_top1_correct"] += float(last_top1_correct.sum().item())
        stat["last_topk_correct"] += float(last_topk_correct.sum().item())
        if k >= 2:
            stat["last_margin_sum"] += float((last_top_vals[:, 0] - last_top_vals[:, 1]).detach().float().sum().item())

    return top1.detach()


def safe_decode(tokenizer: RustBPETokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except Exception:
        return "<decode failed>"


def md_escape(value: Any, max_len: int | None = None) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("|", "\\|")
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def token_piece(tokenizer: RustBPETokenizer, token_id: int) -> str:
    decoded = safe_decode(tokenizer, [int(token_id)])
    return md_escape(repr(decoded), max_len=28)


def compress_trajectory(tokenizer: RustBPETokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    pieces = []
    start = 0
    current = ids[0]
    for idx, token_id in enumerate(ids[1:], start=1):
        if token_id == current:
            continue
        label = f"{start}" if start == idx - 1 else f"{start}-{idx - 1}"
        pieces.append(f"{label}: {token_piece(tokenizer, current)}")
        start = idx
        current = token_id
    label = f"{start}" if start == len(ids) - 1 else f"{start}-{len(ids) - 1}"
    pieces.append(f"{label}: {token_piece(tokenizer, current)}")
    return " -> ".join(pieces)


def collect_examples(
    examples: list[dict[str, Any]],
    tokenizer: RustBPETokenizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cu_seqlens: torch.Tensor,
    top1_stack: torch.Tensor,
    flip_counts: torch.Tensor,
    stable_depth: torch.Tensor,
    last_idx: torch.Tensor,
    max_examples: int,
    context_tokens: int,
) -> None:
    if max_examples <= 0 or int(last_idx.numel()) == 0:
        return

    candidate_scores = flip_counts.index_select(0, last_idx).float() * 1000.0
    candidate_scores = candidate_scores + stable_depth.index_select(0, last_idx).float()
    k = min(max_examples, int(candidate_scores.numel()))
    _, local_positions = torch.topk(candidate_scores, k=k)

    input_cpu = input_ids.detach().cpu()
    targets_cpu = targets.detach().cpu()
    cu_cpu = cu_seqlens.detach().cpu()
    top1_cpu = top1_stack.detach().cpu()
    flip_cpu = flip_counts.detach().cpu()
    stable_cpu = stable_depth.detach().cpu()

    for local_position in local_positions.detach().cpu().tolist():
        seq_idx = int(local_position)
        pos = int(cu_cpu[seq_idx + 1].item() - 1)
        seq_start = int(cu_cpu[seq_idx].item())
        context_start = max(seq_start, pos - context_tokens + 1)
        context_ids = input_cpu[context_start : pos + 1].tolist()
        trajectory = [int(v) for v in top1_cpu[:, pos].tolist()]
        target_id = int(targets_cpu[pos].item())
        flips = int(flip_cpu[pos].item())
        settle = int(stable_cpu[pos].item())
        examples.append(
            {
                "score": flips * 1000 + settle,
                "flips": flips,
                "settle_depth": settle,
                "context": safe_decode(tokenizer, [int(x) for x in context_ids]),
                "target_id": target_id,
                "target": safe_decode(tokenizer, [target_id]),
                "final_prediction_id": trajectory[-1],
                "final_prediction": safe_decode(tokenizer, [trajectory[-1]]),
                "correct_depths": [idx for idx, token_id in enumerate(trajectory) if token_id == target_id],
                "trajectory_ids": trajectory,
                "trajectory": compress_trajectory(tokenizer, trajectory),
            }
        )


def analyze_batch(
    model: RecursiveGPT,
    tokenizer: RustBPETokenizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    depth_stats: list[dict[str, float]],
    residual_stats: list[dict[str, float]],
    settle_hist: torch.Tensor,
    last_settle_hist: torch.Tensor,
    flip_hist: torch.Tensor,
    last_flip_hist: torch.Tensor,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    input_ids, targets, cu_seqlens, position_ids = batch
    rec_depth = int(model.config.rec_depth)
    depth_count = rec_depth + 1
    last_idx = (cu_seqlens[1:].long() - 1).to(targets.device)

    top1_by_depth: list[torch.Tensor] = []
    prev_delta: torch.Tensor | None = None

    x = model.embedding(input_ids)
    if getattr(model, "use_factorized", False):
        x = model.e_to_h(x)
    if model.config.std_depth != 0:
        x = model.std_blocks(x, cu_seqlens, model.config.rope_cache_len, position_ids)

    logits = readout_logits(model, x)
    top1_by_depth.append(add_logit_metrics(depth_stats[0], logits, targets, last_idx, args.top_k))
    del logits

    if rec_depth > 0:
        rec_blocks = model.rec_blocks
        block = rec_blocks.recursive_block

        for depth_idx in range(rec_depth):
            x_before = x
            if rec_blocks.moe:
                attn_out = block.attn(
                    rec_blocks.attn_norms[depth_idx](x),
                    cu_seqlens,
                    model.config.rope_cache_len,
                    position_ids,
                    rec_blocks.qk_norms[depth_idx],
                )
                x_after_attn = x + attn_out
                mlp_out = block.moe(rec_blocks.mlp_norms[depth_idx](x_after_attn), rec_blocks.routers[depth_idx])
            else:
                attn_out = block.attn(
                    rec_blocks.attn_norms[depth_idx](x),
                    cu_seqlens,
                    model.config.rope_cache_len,
                    position_ids,
                    rec_blocks.qk_norms[depth_idx],
                )
                x_after_attn = x + attn_out
                mlp_out = block.mlp(rec_blocks.mlp_norms[depth_idx](x_after_attn))
            x = x_after_attn + mlp_out
            delta = x - x_before

            residual_stat = residual_stats[depth_idx]
            n_tokens = int(targets.numel())
            x_before_rms = rms(x_before)
            x_after_rms = rms(x)
            delta_rms = rms(delta)
            attn_out_rms = rms(attn_out)
            mlp_out_rms = rms(mlp_out)
            add_mean(residual_stat, "x_before_rms", x_before_rms, n_tokens)
            add_mean(residual_stat, "x_after_rms", x_after_rms, n_tokens)
            add_mean(residual_stat, "delta_rms", delta_rms, n_tokens)
            add_mean(residual_stat, "delta_over_x", delta_rms / (x_before_rms + 1e-12), n_tokens)
            add_mean(residual_stat, "attn_out_rms", attn_out_rms, n_tokens)
            add_mean(residual_stat, "mlp_out_rms", mlp_out_rms, n_tokens)
            add_mean(residual_stat, "attn_over_delta", attn_out_rms / (delta_rms + 1e-12), n_tokens)
            add_mean(residual_stat, "mlp_over_delta", mlp_out_rms / (delta_rms + 1e-12), n_tokens)
            add_mean(residual_stat, "cos_x_delta", cosine(x_before, delta), n_tokens)
            add_mean(residual_stat, "cos_attn_mlp", cosine(attn_out, mlp_out), n_tokens)
            if prev_delta is not None:
                add_mean(residual_stat, "cos_delta_prev", cosine(prev_delta, delta), n_tokens)
            prev_delta = delta.detach()

            logits = readout_logits(model, x)
            top1_by_depth.append(add_logit_metrics(depth_stats[depth_idx + 1], logits, targets, last_idx, args.top_k))
            del logits, attn_out, mlp_out, delta, x_before

    top1_stack = torch.stack(top1_by_depth, dim=0)
    if depth_count > 1:
        changed = top1_stack[1:] != top1_stack[:-1]
        flip_counts = changed.sum(dim=0)
        for depth_idx in range(1, depth_count):
            depth_stats[depth_idx]["changed_from_prev"] += float(changed[depth_idx - 1].sum().item())
            if int(last_idx.numel()) > 0:
                depth_stats[depth_idx]["last_changed_from_prev"] += float(
                    changed[depth_idx - 1].index_select(0, last_idx).sum().item()
                )
    else:
        flip_counts = torch.zeros_like(top1_stack[0], dtype=torch.long)

    final_top1 = top1_stack[-1]
    eq_final = top1_stack.eq(final_top1.unsqueeze(0))
    stable_suffix = torch.cumprod(eq_final.flip(0).int(), dim=0).flip(0).bool()
    stable_depth = stable_suffix.int().argmax(dim=0)

    for depth_idx in range(depth_count):
        depth_stats[depth_idx]["final_agree"] += float(eq_final[depth_idx].sum().item())
        depth_stats[depth_idx]["stable_from_depth"] += float(stable_suffix[depth_idx].sum().item())
        if int(last_idx.numel()) > 0:
            depth_stats[depth_idx]["last_final_agree"] += float(eq_final[depth_idx].index_select(0, last_idx).sum().item())
            depth_stats[depth_idx]["last_stable_from_depth"] += float(
                stable_suffix[depth_idx].index_select(0, last_idx).sum().item()
            )

    settle_hist += torch.bincount(stable_depth.detach().cpu(), minlength=depth_count)
    flip_hist += torch.bincount(flip_counts.detach().cpu(), minlength=depth_count)
    if int(last_idx.numel()) > 0:
        last_settle_hist += torch.bincount(stable_depth.index_select(0, last_idx).detach().cpu(), minlength=depth_count)
        last_flip_hist += torch.bincount(flip_counts.index_select(0, last_idx).detach().cpu(), minlength=depth_count)

    collect_examples(
        examples,
        tokenizer,
        input_ids,
        targets,
        cu_seqlens,
        top1_stack,
        flip_counts,
        stable_depth,
        last_idx,
        args.max_examples,
        args.context_tokens,
    )

    changed_once = float((flip_counts > 0).sum().item())
    last_changed_once = 0.0
    if int(last_idx.numel()) > 0:
        last_changed_once = float((flip_counts.index_select(0, last_idx) > 0).sum().item())
    return {
        "tokens": float(targets.numel()),
        "last_tokens": float(last_idx.numel()),
        "flip_sum": float(flip_counts.sum().item()),
        "last_flip_sum": float(flip_counts.index_select(0, last_idx).sum().item()) if int(last_idx.numel()) > 0 else 0.0,
        "changed_once": changed_once,
        "last_changed_once": last_changed_once,
        "settle_depth_sum": float(stable_depth.sum().item()),
        "last_settle_depth_sum": float(stable_depth.index_select(0, last_idx).sum().item())
        if int(last_idx.numel()) > 0
        else 0.0,
    }


def build_depth_rows(depth_stats: list[dict[str, float]], top_k: int) -> list[dict[str, Any]]:
    rows = []
    for depth, stat in enumerate(depth_stats):
        tokens = stat["tokens"]
        last_tokens = stat["last_tokens"]
        loss = safe_ratio(stat["loss_sum"], tokens)
        last_loss = safe_ratio(stat["last_loss_sum"], last_tokens)
        rows.append(
            {
                "depth": depth,
                "tokens": int(tokens),
                "loss": loss,
                "ppl": safe_exp(loss),
                "top1_acc": safe_ratio(stat["top1_correct"], tokens),
                f"top{top_k}_acc": safe_ratio(stat["topk_correct"], tokens),
                "margin": safe_ratio(stat["margin_sum"], tokens),
                "change_from_prev": None if depth == 0 else safe_ratio(stat["changed_from_prev"], tokens),
                "final_agreement": safe_ratio(stat["final_agree"], tokens),
                "stable_from_depth": safe_ratio(stat["stable_from_depth"], tokens),
                "last_loss": last_loss,
                "last_ppl": safe_exp(last_loss),
                "last_top1_acc": safe_ratio(stat["last_top1_correct"], last_tokens),
                f"last_top{top_k}_acc": safe_ratio(stat["last_topk_correct"], last_tokens),
                "last_margin": safe_ratio(stat["last_margin_sum"], last_tokens),
                "last_change_from_prev": None if depth == 0 else safe_ratio(stat["last_changed_from_prev"], last_tokens),
                "last_final_agreement": safe_ratio(stat["last_final_agree"], last_tokens),
                "last_stable_from_depth": safe_ratio(stat["last_stable_from_depth"], last_tokens),
            }
        )
    return rows


def build_residual_rows(residual_stats: list[dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    metric_names = [
        "x_before_rms",
        "x_after_rms",
        "delta_rms",
        "delta_over_x",
        "attn_out_rms",
        "mlp_out_rms",
        "attn_over_delta",
        "mlp_over_delta",
        "cos_x_delta",
        "cos_attn_mlp",
        "cos_delta_prev",
    ]
    for idx, stat in enumerate(residual_stats, start=1):
        row = {"depth": idx}
        for metric in metric_names:
            row[metric] = mean(stat, metric)
        rows.append(row)
    return rows


def percentile_from_hist(hist: torch.Tensor, percentile: float) -> int | None:
    total = int(hist.sum().item())
    if total <= 0:
        return None
    target = math.ceil(total * percentile)
    running = 0
    for idx, count in enumerate(hist.tolist()):
        running += int(count)
        if running >= target:
            return idx
    return int(hist.numel() - 1)


def first_depth_at_or_above(rows: list[dict[str, Any]], key: str, threshold: float) -> int | None:
    for row in rows:
        value = row.get(key)
        if value is not None and value >= threshold:
            return int(row["depth"])
    return None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    out_dir: Path,
    depth_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    settle_hist: torch.Tensor,
    last_settle_hist: torch.Tensor,
    top_k: int,
    no_plots: bool,
) -> tuple[list[Path], str | None]:
    if no_plots:
        return [], "disabled by --no-plots"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [], f"matplotlib unavailable: {exc}"

    plot_paths: list[Path] = []
    depths = [int(row["depth"]) for row in depth_rows]

    def values(key: str) -> list[float]:
        return [float(row[key]) if row.get(key) is not None else float("nan") for row in depth_rows]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(depths, values("loss"), label="loss", color="#305cde")
    ax1.set_xlabel("recursive readout depth")
    ax1.set_ylabel("cross entropy")
    ax2 = ax1.twinx()
    ax2.plot(depths, values("top1_acc"), label="top-1 acc", color="#168a4a")
    ax2.plot(depths, values(f"top{top_k}_acc"), label=f"top-{top_k} acc", color="#7a4cc2")
    ax2.set_ylabel("accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("Prediction quality by recursive depth")
    fig.tight_layout()
    path = out_dir / "prediction_quality.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(depths, values("change_from_prev"), label="changed from previous")
    ax.plot(depths, values("final_agreement"), label="matches final prediction")
    ax.plot(depths, values("stable_from_depth"), label="stable from this depth")
    ax.set_xlabel("recursive readout depth")
    ax.set_ylabel("fraction of tokens")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.set_title("Prediction changes and stabilization")
    fig.tight_layout()
    path = out_dir / "prediction_dynamics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    if residual_rows:
        residual_depths = [int(row["depth"]) for row in residual_rows]

        def residual_values(key: str) -> list[float]:
            return [float(row[key]) if row.get(key) is not None else float("nan") for row in residual_rows]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(residual_depths, residual_values("x_before_rms"), label="x before RMS")
        ax.plot(residual_depths, residual_values("delta_rms"), label="residual update RMS")
        ax.plot(residual_depths, residual_values("attn_out_rms"), label="attention out RMS")
        ax.plot(residual_depths, residual_values("mlp_out_rms"), label="MLP out RMS")
        ax.set_xlabel("recursive step")
        ax.set_ylabel("RMS")
        ax.legend(loc="best")
        ax.set_title("Residual stream and module output norms")
        fig.tight_layout()
        path = out_dir / "residual_norms.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(residual_depths, residual_values("delta_over_x"), label="update / x")
        ax.plot(residual_depths, residual_values("cos_x_delta"), label="cos(x, update)")
        ax.plot(residual_depths, residual_values("cos_delta_prev"), label="cos(update, previous)")
        ax.set_xlabel("recursive step")
        ax.legend(loc="best")
        ax.set_title("Residual update scale and direction")
        fig.tight_layout()
        path = out_dir / "residual_direction.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    hist_depths = list(range(int(settle_hist.numel())))
    total = float(settle_hist.sum().item())
    last_total = float(last_settle_hist.sum().item())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if total > 0:
        ax.bar([d - 0.2 for d in hist_depths], [float(v) / total for v in settle_hist.tolist()], width=0.4, label="all tokens")
    if last_total > 0:
        ax.bar(
            [d + 0.2 for d in hist_depths],
            [float(v) / last_total for v in last_settle_hist.tolist()],
            width=0.4,
            label="last tokens",
        )
    ax.set_xlabel("first depth whose prediction stays final")
    ax.set_ylabel("fraction")
    ax.legend(loc="best")
    ax.set_title("Prediction settle depth")
    fig.tight_layout()
    path = out_dir / "settle_depth_histogram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    return plot_paths, None


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(fmt(value)) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    job: AnalysisJob,
    summary: dict[str, Any],
    depth_rows: list[dict[str, Any]],
    residual_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    plot_paths: list[Path],
    plot_note: str | None,
    top_k: int,
) -> Path:
    report_path = out_dir / "report.md"
    final = depth_rows[-1]
    first = depth_rows[0]

    lines = [
        f"# Recursive Depth Analysis: `{job.model}`",
        "",
        f"- Dataset: `{job.dataset}`",
        f"- Tokenizer: `{job.tokenizer}`",
        f"- Batches: `{summary['batches']}` shuffled batches, `{summary['batch_tokens']}` tokens each",
        f"- Total analyzed tokens: `{summary['tokens']}`; last-token positions: `{summary['last_tokens']}`",
        f"- Recursive depth: `{summary['rec_depth']}`; standard depth: `{summary['std_depth']}`",
        f"- Seed: `{summary['seed']}`",
        "",
        "## Executive Summary",
        "",
        markdown_table(
            ["metric", "value"],
            [
                ["depth-0 loss", first["loss"]],
                ["final loss", final["loss"]],
                ["loss improvement", summary["loss_improvement"]],
                ["final perplexity", final["ppl"]],
                ["final top-1 acc", final["top1_acc"]],
                [f"final top-{top_k} acc", final[f"top{top_k}_acc"]],
                ["tokens changing at least once", summary["changed_once_frac"]],
                ["mean prediction flips / token", summary["mean_flips"]],
                ["mean settle depth", summary["mean_settle_depth"]],
                ["median settle depth", summary["median_settle_depth"]],
                ["90% stable by depth", summary["stable_90_depth"]],
                ["99% stable by depth", summary["stable_99_depth"]],
                ["last-token changed at least once", summary["last_changed_once_frac"]],
                ["last-token mean settle depth", summary["last_mean_settle_depth"]],
            ],
        ),
        "",
    ]

    if plot_paths:
        lines.extend(["## Plots", ""])
        for path in plot_paths:
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")
    elif plot_note:
        lines.extend(["## Plots", "", f"No plots written: {plot_note}.", ""])

    lines.extend(
        [
            "## Per-Depth Prediction Metrics",
            "",
            markdown_table(
                [
                    "depth",
                    "loss",
                    "ppl",
                    "top1",
                    f"top{top_k}",
                    "change",
                    "final agree",
                    "stable",
                    "last top1",
                    "last change",
                    "last stable",
                ],
                [
                    [
                        row["depth"],
                        row["loss"],
                        row["ppl"],
                        row["top1_acc"],
                        row[f"top{top_k}_acc"],
                        row["change_from_prev"],
                        row["final_agreement"],
                        row["stable_from_depth"],
                        row["last_top1_acc"],
                        row["last_change_from_prev"],
                        row["last_stable_from_depth"],
                    ]
                    for row in depth_rows
                ],
            ),
            "",
        ]
    )

    if residual_rows:
        lines.extend(
            [
                "## Residual Metrics",
                "",
                markdown_table(
                    [
                        "step",
                        "x rms",
                        "delta rms",
                        "delta/x",
                        "attn rms",
                        "mlp rms",
                        "cos x,delta",
                        "cos delta,prev",
                    ],
                    [
                        [
                            row["depth"],
                            row["x_before_rms"],
                            row["delta_rms"],
                            row["delta_over_x"],
                            row["attn_out_rms"],
                            row["mlp_out_rms"],
                            row["cos_x_delta"],
                            row["cos_delta_prev"],
                        ]
                        for row in residual_rows
                    ],
                ),
                "",
            ]
        )

    if examples:
        lines.extend(["## Interesting Last-Token Trajectories", ""])
        lines.append(
            markdown_table(
                ["flips", "settle", "target", "final", "correct depths", "trajectory", "context"],
                [
                    [
                        ex["flips"],
                        ex["settle_depth"],
                        repr(ex["target"]),
                        repr(ex["final_prediction"]),
                        ",".join(str(v) for v in ex["correct_depths"]) or "-",
                        ex["trajectory"],
                        repr(ex["context"]),
                    ]
                    for ex in examples
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `metrics.json`: full machine-readable payload",
            "- `depth_metrics.csv`: per-depth prediction metrics",
            "- `residual_metrics.csv`: per-recursive-step residual diagnostics",
            "- `examples.json`: selected last-token prediction trajectories",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def analyze_job(
    base_dir: Path,
    root_out_dir: Path,
    job: AnalysisJob,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    model_stem = Path(job.model).stem
    out_dir = root_out_dir / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = base_dir / "tokenizers" / job.tokenizer
    dataset_path = base_dir / "data" / "tokenized" / job.dataset
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer = RustBPETokenizer.load_from_dir(str(tokenizer_path))
    model = load_model(base_dir, job.model, device, args.sequence_len)

    rec_depth = int(model.config.rec_depth)
    depth_count = rec_depth + 1
    depth_stats = make_depth_stats(depth_count)
    residual_stats = [{} for _ in range(rec_depth)]
    settle_hist = torch.zeros(depth_count, dtype=torch.long)
    last_settle_hist = torch.zeros(depth_count, dtype=torch.long)
    flip_hist = torch.zeros(depth_count, dtype=torch.long)
    last_flip_hist = torch.zeros(depth_count, dtype=torch.long)
    examples: list[dict[str, Any]] = []
    aggregate = {
        "tokens": 0.0,
        "last_tokens": 0.0,
        "flip_sum": 0.0,
        "last_flip_sum": 0.0,
        "changed_once": 0.0,
        "last_changed_once": 0.0,
        "settle_depth_sum": 0.0,
        "last_settle_depth_sum": 0.0,
    }

    use_autocast = "cuda" in str(device)
    start = time.time()
    batches_seen = 0
    with torch.no_grad():
        for batch in batch_iterator(
            str(dataset_path),
            tokens_per_batch=args.batch_tokens,
            max_sl=args.sequence_len,
            token_col=args.token_col,
            drop_last=True,
            device=device,
            fix_length=True,
            seed=args.seed,
        ):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
            with autocast_ctx:
                batch_metrics = analyze_batch(
                    model,
                    tokenizer,
                    batch,
                    depth_stats,
                    residual_stats,
                    settle_hist,
                    last_settle_hist,
                    flip_hist,
                    last_flip_hist,
                    examples,
                    args,
                )
            for key, value in batch_metrics.items():
                aggregate[key] += value
            batches_seen += 1
            print(
                f"[analysis] {job.model}: batch {batches_seen}/{args.batches} "
                f"({int(aggregate['tokens'])} tokens)",
                flush=True,
            )
            if batches_seen >= args.batches:
                break

    if batches_seen == 0:
        raise RuntimeError(f"No batches yielded for {dataset_path}")

    elapsed = time.time() - start
    examples = sorted(examples, key=lambda ex: ex["score"], reverse=True)[: args.max_examples]
    depth_rows = build_depth_rows(depth_stats, args.top_k)
    residual_rows = build_residual_rows(residual_stats)

    final = depth_rows[-1]
    first = depth_rows[0]
    summary = {
        "model": job.model,
        "dataset": job.dataset,
        "tokenizer": job.tokenizer,
        "device": device,
        "seed": args.seed,
        "batch_tokens": args.batch_tokens,
        "batches": batches_seen,
        "sequence_len": args.sequence_len,
        "tokens": int(aggregate["tokens"]),
        "last_tokens": int(aggregate["last_tokens"]),
        "rec_depth": rec_depth,
        "std_depth": int(model.config.std_depth),
        "n_hidden": int(model.config.n_hidden),
        "n_wembed": int(model.config.n_wembed),
        "n_head": int(model.config.n_head),
        "top_k": int(args.top_k),
        "elapsed_s": elapsed,
        "loss_improvement": None
        if first["loss"] is None or final["loss"] is None
        else float(first["loss"] - final["loss"]),
        "top1_gain": None
        if first["top1_acc"] is None or final["top1_acc"] is None
        else float(final["top1_acc"] - first["top1_acc"]),
        "changed_once_frac": safe_ratio(aggregate["changed_once"], aggregate["tokens"]),
        "last_changed_once_frac": safe_ratio(aggregate["last_changed_once"], aggregate["last_tokens"]),
        "mean_flips": safe_ratio(aggregate["flip_sum"], aggregate["tokens"]),
        "last_mean_flips": safe_ratio(aggregate["last_flip_sum"], aggregate["last_tokens"]),
        "mean_settle_depth": safe_ratio(aggregate["settle_depth_sum"], aggregate["tokens"]),
        "last_mean_settle_depth": safe_ratio(aggregate["last_settle_depth_sum"], aggregate["last_tokens"]),
        "median_settle_depth": percentile_from_hist(settle_hist, 0.5),
        "p90_settle_depth": percentile_from_hist(settle_hist, 0.9),
        "last_median_settle_depth": percentile_from_hist(last_settle_hist, 0.5),
        "last_p90_settle_depth": percentile_from_hist(last_settle_hist, 0.9),
        "stable_90_depth": first_depth_at_or_above(depth_rows, "stable_from_depth", 0.90),
        "stable_99_depth": first_depth_at_or_above(depth_rows, "stable_from_depth", 0.99),
        "last_stable_90_depth": first_depth_at_or_above(depth_rows, "last_stable_from_depth", 0.90),
        "last_stable_99_depth": first_depth_at_or_above(depth_rows, "last_stable_from_depth", 0.99),
    }

    plot_paths, plot_note = make_plots(
        out_dir,
        depth_rows,
        residual_rows,
        settle_hist,
        last_settle_hist,
        args.top_k,
        args.no_plots,
    )

    write_csv(out_dir / "depth_metrics.csv", depth_rows)
    write_csv(out_dir / "residual_metrics.csv", residual_rows)
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    payload = {
        "summary": summary,
        "depth_metrics": depth_rows,
        "residual_metrics": residual_rows,
        "settle_hist": settle_hist.tolist(),
        "last_settle_hist": last_settle_hist.tolist(),
        "flip_hist": flip_hist.tolist(),
        "last_flip_hist": last_flip_hist.tolist(),
        "examples": examples,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = write_report(
        out_dir,
        job,
        summary,
        depth_rows,
        residual_rows,
        examples,
        plot_paths,
        plot_note,
        args.top_k,
    )
    print(f"[analysis] wrote {report_path}", flush=True)
    return {
        "summary": summary,
        "report_path": str(report_path),
        "out_dir": str(out_dir),
    }


def write_index(root_out_dir: Path, results: list[dict[str, Any]]) -> Path:
    index_path = root_out_dir / "index.md"
    rows = []
    for result in results:
        summary = result["summary"]
        report_rel = Path(result["report_path"]).relative_to(root_out_dir)
        rows.append(
            [
                f"[`{summary['model']}`]({report_rel.as_posix()})",
                summary["dataset"],
                summary["tokens"],
                summary["rec_depth"],
                summary.get("loss_improvement"),
                summary.get("changed_once_frac"),
                summary.get("mean_flips"),
                summary.get("mean_settle_depth"),
                summary.get("stable_90_depth"),
            ]
        )

    lines = [
        "# Recursive Depth Analysis Index",
        "",
        markdown_table(
            [
                "model",
                "dataset",
                "tokens",
                "rec depth",
                "loss improvement",
                "changed once",
                "mean flips",
                "mean settle depth",
                "90% stable by",
            ],
            rows,
        ),
        "",
    ]
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def main() -> int:
    args = parse_args()
    jobs = parse_jobs(args)
    base_dir = get_base_dir()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.batch_tokens < args.sequence_len:
        raise SystemExit("--batch-tokens must be >= --sequence-len.")

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        root_out_dir = base_dir / "analysis" / "recursive_depths" / ts
    else:
        root_out_dir = Path(args.output_dir)
        if not root_out_dir.is_absolute():
            root_out_dir = base_dir / root_out_dir
    root_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] device={device} output={root_out_dir}", flush=True)
    results = []
    for job in jobs:
        results.append(analyze_job(base_dir, root_out_dir, job, args, device))
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    index_path = write_index(root_out_dir, results)
    print(f"[analysis] wrote {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
