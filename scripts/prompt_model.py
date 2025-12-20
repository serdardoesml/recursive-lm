"""Quick vibecoded tool to test the base model. Can only run on gpu."""

import argparse
import os

import torch

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
    parser.add_argument("--rec_depth", type=int, default=ModelConfig.rec_depth)
    parser.add_argument("--n_head", type=int, default=ModelConfig.n_head)
    parser.add_argument("--sequence_len", type=int, default=ModelConfig.sequence_len)
    args = parser.parse_args()

    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", args.model)
    tokenizer_path = os.path.join(base_dir, "tokenizers", args.tokenizer)

    tokenizer = RustBPETokenizer.load_from_dir(tokenizer_path)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    tie_embed = "lm_head.weight" not in state
    embed_w = state["embedding.weight"]
    vocab_size, n_embd = embed_w.shape
    mlp_mul = state["recursive_block.mlp.c_fc.weight"].shape[0] // n_embd
    config = ModelConfig(
        sequence_len=args.sequence_len,
        vocab_size=vocab_size,
        n_head=args.n_head,
        n_embd=n_embd,
        mlp_mul=mlp_mul,
        rec_depth=args.rec_depth,
        tie_embed=tie_embed,
    )
    model = RecursiveGPT(config)
    device = "cuda" # Cuda is required due to flash-attn
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
        attn_out_rms: Optional[float] = None
        mlp_out_rms: Optional[float] = None

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
            self.steps: list[StepMetrics] = []

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
            delta_rms = self._rms(x_out - x_in)
            delta_over_in = float(delta_rms / (x_in_rms + 1e-12))

            self.steps.append(
                StepMetrics(
                    step=int(self._step_idx),
                    x_in_rms=x_in_rms,
                    x_out_rms=x_out_rms,
                    delta_rms=delta_rms,
                    delta_over_in=delta_over_in,
                    attn_out_rms=self._pending_attn,
                    mlp_out_rms=self._pending_mlp,
                )
            )

            self._pending_attn = None
            self._pending_mlp = None
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
            analyzer.reset()

            ctx = tokens[-config.sequence_len:]
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
            out_txt_path = os.path.join(base_analysis_dir, f"{out_stem}.txt")

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
                "n_embd": int(config.n_embd),
                "sequence_len": int(config.sequence_len),
                "prompt": prompt,
                "prompt_token_count": int(len(tokens)),
                "ctx_token_count": int(len(ctx)),
                "next_id": int(next_id),
                "next_piece": next_piece,
                "steps": [s.__dict__ for s in analyzer.steps],
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            # Human-readable summary
            lines = []
            lines.append(f"model: {args.model}")
            lines.append(f"tokenizer: {args.tokenizer}")
            lines.append(f"prompt_token_count: {len(tokens)} (ctx {len(ctx)})")
            lines.append(f"next_id: {next_id} next_piece: {repr(next_piece)}")
            lines.append("")
            lines.append("Per-step metrics (RMS norms; delta is x_out - x_in):")
            lines.append("step\tx_in_rms\tx_out_rms\tdelta_rms\tdelta/x_in\tattn_out_rms\tmlp_out_rms")
            for s in analyzer.steps:
                lines.append(
                    f"{s.step}\t{s.x_in_rms:.6f}\t{s.x_out_rms:.6f}\t{s.delta_rms:.6f}\t{s.delta_over_in:.6f}\t"
                    f"{(s.attn_out_rms if s.attn_out_rms is not None else float('nan')):.6f}\t"
                    f"{(s.mlp_out_rms if s.mlp_out_rms is not None else float('nan')):.6f}"
                )

            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print("\n".join(lines[:8]))
            print(f"Saved analysis JSON to {out_json_path}")
            print(f"Saved analysis TXT  to {out_txt_path}")
            print("\nCompletion:")
            print(tokenizer.decode(tokens_plus))
            return

        # Normal generation mode (unchanged behavior)
        with torch.no_grad():
            for _ in range(args.gen_tok_count):
                ctx = tokens[-config.sequence_len:]
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
    if not torch.cuda.is_available(): # Cuda is required due to flash-attn
        raise Exception("Cuda not available!")
    main()
