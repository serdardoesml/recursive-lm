"""
Quick vibecoded tool to test the base model and do some analysis. Can only run on gpu.
Analysis mode requires uv group 'analysis'.
"""

import argparse
import os

import torch
import torch.nn.functional as F

import json
import time
from dataclasses import dataclass
from typing import Any, Optional


from recursive_lm.common import get_base_dir
from recursive_lm.model import ModelConfig, RecursiveGPT
from recursive_lm.tokenizer import RustBPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Prompt a trained RecursiveGPT model")
    parser.add_argument("--model", type=str, required=True, help="Model filename under models/")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer filename under tokenizers/")
    parser.add_argument("--gen_tok_count", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run a single-step (one next-token) analysis and save metrics under analysis/",
    )
    parser.add_argument(
        "--analysis_out",
        type=str,
        default=None,
        help="Optional output stem (no extension) under analysis/; defaults to auto timestamped name",
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", args.model)
    tokenizer_path = os.path.join(base_dir, "tokenizers", args.tokenizer)

    tokenizer = RustBPETokenizer.load_from_dir(tokenizer_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint is missing config/state_dict; re-save with the updated trainer.")
    state = checkpoint["state_dict"]
    config = ModelConfig(**checkpoint["config"])
    model = RecursiveGPT(config)
    device = "cuda" # Cuda is required due to varlen-attn
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    base_analysis_dir = os.path.join(base_dir, "analysis")
    os.makedirs(base_analysis_dir, exist_ok=True)

    @dataclass
    class StepMetrics:
        step: int
        x_in_rms: float
        x_out_rms: float
        delta_rms: float
        delta_over_in: float
        # Last-token directional metrics
        cos_x_delta_last: Optional[float] = None
        cos_delta_prev_last: Optional[float] = None
        # Raw module output RMS (pre-RMSNorm in your block)
        attn_out_rms: Optional[float] = None
        mlp_out_rms: Optional[float] = None
        # Logit dynamics for the last token (using model readout on x_out)
        top1_id: Optional[int] = None
        top1_logit: Optional[float] = None
        logit_l2_from_prev: Optional[float] = None

    class ForwardAnalyzer:
        """Collect per-recursion-step metrics via forward hooks.

        Notes:
        - `model.recursive_block` is called once per recursion step, so its hook gives us x^(r) -> x^(r+1).
        - We also hook attn/mlp to attribute update magnitude (roughly) to each submodule.
        """

        def __init__(self, model: RecursiveGPT):
            self.model = model
            self.handles = []
            self.reset()

        def reset(self):
            self._step_idx = 0
            self._pending_attn: Optional[float] = None
            self._pending_mlp: Optional[float] = None
            self._prev_delta_last: Optional[torch.Tensor] = None
            self._prev_logits_last: Optional[torch.Tensor] = None
            self.steps: list[StepMetrics] = []
        @staticmethod
        def _rmsnorm_vec(v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
            # v is 1D [d]
            denom = (v.float().pow(2).mean() + eps).sqrt()
            return v.float() / denom

        @staticmethod
        def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
            a = a.float().flatten()
            b = b.float().flatten()
            na = torch.linalg.norm(a)
            nb = torch.linalg.norm(b)
            if float(na.item()) < eps or float(nb.item()) < eps:
                return float("nan")
            return float((a @ b).item() / (na.item() * nb.item() + eps))

        def _readout_last_logits(self, x_last: torch.Tensor) -> torch.Tensor:
            # x_last: [d]
            # Mirror model readout: h -> e -> vocab
            x_n = self.model.norm_out(x_last)
            if getattr(self.model, "use_factorized", False):
                x_n = self.model.h_to_e(x_n)
            if getattr(self.model.config, "tie_embed", False):
                W = self.model.embedding.weight
            else:
                W = self.model.lm_head.weight
            # Return logits [V] in float32
            return F.linear(x_n, W).float()

        def close(self):
            for h in self.handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self.handles = []

        @staticmethod
        def _rms(t: torch.Tensor) -> float:
            # Root-mean-square over all elements; robust across shapes.
            return float(t.detach().float().pow(2).mean().sqrt().item())

        def _attn_hook(self, _mod, _inp, out):
            try:
                self._pending_attn = self._rms(out)
            except Exception:
                self._pending_attn = None

        def _mlp_hook(self, _mod, _inp, out):
            try:
                self._pending_mlp = self._rms(out)
            except Exception:
                self._pending_mlp = None

        def _block_hook(self, _mod, inp, out):
            # Block forward signature is (x, cu_seqlens, position_ids, max_seqlen)
            x_in = inp[0]
            x_out = out

            x_in_rms = self._rms(x_in)
            x_out_rms = self._rms(x_out)

            delta = (x_out - x_in)
            delta_rms = self._rms(delta)
            delta_over_in = float(delta_rms / (x_in_rms + 1e-12))

            # Directional metrics on the last token only (keeps memory/compute small)
            x_in_last = x_in[-1]
            delta_last = delta[-1]
            cos_x_delta_last = self._cos(x_in_last, delta_last)

            cos_delta_prev_last = None
            if self._prev_delta_last is not None:
                cos_delta_prev_last = self._cos(self._prev_delta_last, delta_last)

            # Logit dynamics for last token: top-1 and drift from previous step
            logits_last = self._readout_last_logits(x_out[-1])
            top1_id = int(torch.argmax(logits_last).item())
            top1_logit = float(logits_last[top1_id].item())

            logit_l2_from_prev = None
            if self._prev_logits_last is not None:
                # RMS L2 drift of logits (normalized by sqrt(V) for scale stability)
                diff = (logits_last - self._prev_logits_last)
                logit_l2_from_prev = float(torch.linalg.norm(diff).item() / (diff.numel() ** 0.5 + 1e-12))

            self.steps.append(
                StepMetrics(
                    step=int(self._step_idx),
                    x_in_rms=x_in_rms,
                    x_out_rms=x_out_rms,
                    delta_rms=delta_rms,
                    delta_over_in=delta_over_in,
                    cos_x_delta_last=float(cos_x_delta_last),
                    cos_delta_prev_last=(float(cos_delta_prev_last) if cos_delta_prev_last is not None else None),
                    attn_out_rms=self._pending_attn,
                    mlp_out_rms=self._pending_mlp,
                    top1_id=top1_id,
                    top1_logit=top1_logit,
                    logit_l2_from_prev=logit_l2_from_prev,
                )
            )

            self._pending_attn = None
            self._pending_mlp = None
            self._prev_delta_last = delta_last.detach()
            self._prev_logits_last = logits_last.detach()
            self._step_idx += 1

        def enable(self):
            # Ensure hooks are only registered once
            if self.handles:
                return
            self.handles.append(self.model.recursive_block.attn.register_forward_hook(self._attn_hook))
            self.handles.append(self.model.recursive_block.mlp.register_forward_hook(self._mlp_hook))
            self.handles.append(self.model.recursive_block.register_forward_hook(self._block_hook))

    analyzer = ForwardAnalyzer(model) if args.analysis else None
    if analyzer is not None:
        analyzer.enable()

    def generate(prompt):
        tokens = tokenizer.encode(prompt)

        # In analysis mode we run exactly one next-token prediction, capture per-step metrics,
        # save them to analysis/, print a short summary, and return.
        if args.analysis and analyzer is not None:
            import matplotlib.pyplot as plt # type: ignore

            analyzer.reset()

            ctx = tokens[-config.rope_cache_len:]
            input_ids = torch.tensor(ctx, dtype=torch.long, device=device)
            cu_seqlens = torch.tensor([0, input_ids.numel()], dtype=torch.int32, device=device)
            position_ids = torch.arange(input_ids.numel(), dtype=torch.long, device=device)

            with torch.no_grad():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(input_ids, cu_seqlens, position_ids)
                else:
                    logits = model(input_ids, cu_seqlens, position_ids)

            next_id = int(torch.argmax(logits[-1]).item())
            tokens_plus = tokens + [next_id]

            # Build payload
            ts = time.strftime("%Y%m%d-%H%M%S")
            model_stem = os.path.splitext(os.path.basename(args.model))[0]
            out_stem = args.analysis_out or f"prompt_analysis_{model_stem}_{ts}"
            out_json_path = os.path.join(base_analysis_dir, f"{out_stem}.json")

            next_piece = None
            try:
                next_piece = tokenizer.decode([next_id])
            except Exception:
                next_piece = None

            payload: dict[str, Any] = {
                "model": args.model,
                "tokenizer": args.tokenizer,
                "rec_depth": int(config.rec_depth),
                "n_head": int(config.n_head),
                "n_hidden": int(config.n_hidden),
                "sequence_len": int(config.rope_cache_len),
                "prompt": prompt,
                "prompt_token_count": int(len(tokens)),
                "ctx_token_count": int(len(ctx)),
                "next_id": int(next_id),
                "next_piece": next_piece,
                "steps": [s.__dict__ for s in analyzer.steps],
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            # --- Plot metrics (save only; no textual report) ---
            steps = [s.step for s in analyzer.steps]
            x_in = [s.x_in_rms for s in analyzer.steps]
            x_out = [s.x_out_rms for s in analyzer.steps]
            delta = [s.delta_rms for s in analyzer.steps]
            delta_ratio = [s.delta_over_in for s in analyzer.steps]
            attn_rms = [s.attn_out_rms if s.attn_out_rms is not None else float("nan") for s in analyzer.steps]
            mlp_rms = [s.mlp_out_rms if s.mlp_out_rms is not None else float("nan") for s in analyzer.steps]
            cos_xd = [s.cos_x_delta_last if s.cos_x_delta_last is not None else float("nan") for s in analyzer.steps]
            cos_dd = [s.cos_delta_prev_last if s.cos_delta_prev_last is not None else float("nan") for s in analyzer.steps]
            top1_id = [s.top1_id if s.top1_id is not None else -1 for s in analyzer.steps]
            top1_logit = [s.top1_logit if s.top1_logit is not None else float("nan") for s in analyzer.steps]
            logit_drift = [s.logit_l2_from_prev if s.logit_l2_from_prev is not None else float("nan") for s in analyzer.steps]

            # Figure 1: norms and relative update
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(steps, x_in, label="x_in_rms")
            ax1.plot(steps, x_out, label="x_out_rms")
            ax1.plot(steps, delta, label="delta_rms")
            ax1.set_title("Recursive block trajectory norms")
            ax1.set_xlabel("recursion step")
            ax1.set_ylabel("RMS")
            ax1.legend()
            fig1.tight_layout()
            p1 = os.path.join(base_analysis_dir, f"{out_stem}_norms.png")
            fig1.savefig(p1, dpi=150)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(steps, delta_ratio, label="delta/x_in")
            ax2.set_title("Relative update size (delta/x_in)")
            ax2.set_xlabel("recursion step")
            ax2.set_ylabel("ratio")
            ax2.legend()
            fig2.tight_layout()
            p2 = os.path.join(base_analysis_dir, f"{out_stem}_delta_ratio.png")
            fig2.savefig(p2, dpi=150)
            plt.close(fig2)

            # Figure 2: module raw RMS (pre-RMSNorm outputs)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(steps, attn_rms, label="attn_out_rms (raw)")
            ax3.plot(steps, mlp_rms, label="mlp_out_rms (raw)")
            ax3.set_title("Module output RMS (raw; before RMSNorm)")
            ax3.set_xlabel("recursion step")
            ax3.set_ylabel("RMS")
            ax3.legend()
            fig3.tight_layout()
            p3 = os.path.join(base_analysis_dir, f"{out_stem}_module_rms.png")
            fig3.savefig(p3, dpi=150)
            plt.close(fig3)

            # Figure 3: directional metrics on last token
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(steps, cos_xd, label="cos(x, delta) last")
            ax4.plot(steps, cos_dd, label="cos(delta, prev_delta) last")
            ax4.set_title("Directional metrics (last token)")
            ax4.set_xlabel("recursion step")
            ax4.set_ylabel("cosine")
            ax4.legend()
            fig4.tight_layout()
            p4 = os.path.join(base_analysis_dir, f"{out_stem}_cosines.png")
            fig4.savefig(p4, dpi=150)
            plt.close(fig4)

            # Figure 4: logit dynamics for last token
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            ax5.plot(steps, logit_drift, label="logit drift (RMS L2) from prev")
            ax5.set_title("Last-token logit drift across recursion")
            ax5.set_xlabel("recursion step")
            ax5.set_ylabel("RMS L2 drift")
            ax5.legend()
            fig5.tight_layout()
            p5 = os.path.join(base_analysis_dir, f"{out_stem}_logit_drift.png")
            fig5.savefig(p5, dpi=150)
            plt.close(fig5)

            fig6, ax6 = plt.subplots(figsize=(10, 4))
            ax6.plot(steps, top1_logit, label="top-1 logit")
            ax6.set_title("Last-token top-1 logit across recursion")
            ax6.set_xlabel("recursion step")
            ax6.set_ylabel("logit")
            ax6.legend()
            fig6.tight_layout()
            p6 = os.path.join(base_analysis_dir, f"{out_stem}_top1_logit.png")
            fig6.savefig(p6, dpi=150)
            plt.close(fig6)

            fig7, ax7 = plt.subplots(figsize=(10, 4))
            ax7.plot(steps, top1_id, label="top-1 token id")
            ax7.set_title("Last-token argmax id across recursion")
            ax7.set_xlabel("recursion step")
            ax7.set_ylabel("token id")
            ax7.legend()
            fig7.tight_layout()
            p7 = os.path.join(base_analysis_dir, f"{out_stem}_top1_id.png")
            fig7.savefig(p7, dpi=150)
            plt.close(fig7)

            # Save a minimal manifest so you can find outputs quickly
            manifest = {
                "json": out_json_path,
                "plots": [p1, p2, p3, p4, p5, p6, p7],
                "out_stem": out_stem,
            }
            manifest_path = os.path.join(base_analysis_dir, f"{out_stem}_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            return

        # Normal generation mode (unchanged behavior)
        with torch.no_grad():
            for _ in range(args.gen_tok_count):
                ctx = tokens[-config.rope_cache_len:]
                input_ids = torch.tensor(ctx, dtype=torch.long, device=device)
                cu_seqlens = torch.tensor([0, input_ids.numel()], dtype=torch.int32, device=device)
                position_ids = torch.arange(input_ids.numel(), dtype=torch.long, device=device)
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(input_ids, cu_seqlens, position_ids)
                else:
                    logits = model(input_ids, cu_seqlens, position_ids)
                next_id = int(torch.argmax(logits[-1]).item())
                tokens.append(next_id)

        print(tokenizer.decode(tokens))


    if args.prompt is not None:
        generate(args.prompt)
        return

    while True:
        try:
            prompt = input("Prompt: ")
        except (EOFError, KeyboardInterrupt):
            break
        generate(prompt)


if __name__ == "__main__":
    if not torch.cuda.is_available(): # Cuda is required due to varlen-attn
        raise Exception("Cuda not available!")
    main()
