"""Spectral / norm analysis utilities for a (tied) transformer block.

This script is intentionally simple: it loads a saved state_dict and computes
- singular value spectra (SVD) for key weight matrices
- optional per-head spectra for Wq/Wk/Wv (requires --n-heads)
- optional eigenvalue magnitude spectrum for square matrices

Outputs PNG plots under analysis/.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np

from recursive_lm.common import get_base_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Plot spectra of transformer block weights")
    ap.add_argument("--model", type=str, required=True, help="Model filename under models/")
    ap.add_argument("--out", type=str, default="spectral_analysis", help="Output stem under analysis/ (no extension)")
    ap.add_argument("--logy", action="store_true", help="Use log-scale for singular value plots")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window (useful on clusters)")
    ap.add_argument("--n-heads", type=int, default=None, help="If set, compute per-head spectra for Wq/Wk/Wv")
    ap.add_argument("--eig", action="store_true", help="Also compute eigenvalue magnitude spectra for square matrices")
    return ap.parse_args()


def load_state(model_name: str) -> dict:
    model_path = Path(model_name)
    if not model_path.is_file():
        model_path = get_base_dir() / "models" / model_name
    return torch.load(model_path, map_location="cpu", weights_only=True)


def _split_qkv(Wqkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a [3d, d] (out,in) projection into (Wq, Wk, Wv), each [d, d]."""
    if Wqkv.ndim != 2 or Wqkv.shape[0] % 3 != 0:
        raise ValueError(f"Expected Wqkv shape [3d, d], got {tuple(Wqkv.shape)}")
    d3, din = Wqkv.shape
    d = d3 // 3
    Wq = Wqkv[:d]
    Wk = Wqkv[d : 2 * d]
    Wv = Wqkv[2 * d :]
    if Wq.shape != (d, din) or Wk.shape != (d, din) or Wv.shape != (d, din):
        raise ValueError("Unexpected split shapes for Wq/Wk/Wv")
    return Wq, Wk, Wv


def _per_head_svd(W: torch.Tensor, n_heads: int) -> np.ndarray:
    """Return an array of shape [n_heads, head_dim] with singular values per head.

    Assumes W is [d, din] with d = n_heads * head_dim, and heads are contiguous in output dim.
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {W.ndim}D")
    d, din = W.shape
    if d % n_heads != 0:
        raise ValueError(f"d={d} not divisible by n_heads={n_heads}")
    head_dim = d // n_heads
    W_h = W.reshape(n_heads, head_dim, din)
    out = []
    for h in range(n_heads):
        s = torch.linalg.svdvals(W_h[h].float()).cpu().numpy()
        out.append(s)
    # pad to same length (svdvals length is min(head_dim, din))
    maxlen = max(len(x) for x in out) if out else 0
    out_pad = np.zeros((n_heads, maxlen), dtype=np.float64)
    for i, s in enumerate(out):
        out_pad[i, : len(s)] = s
    return out_pad


def _eig_magnitudes(W: torch.Tensor) -> np.ndarray:
    """Eigenvalue magnitudes for a square matrix W."""
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got {tuple(W.shape)}")
    eig = torch.linalg.eigvals(W.float()).cpu().numpy()
    mag = np.abs(eig.astype(np.complex128))
    return np.sort(mag)[::-1]


def main():
    args = parse_args()
    state = load_state(args.model)

    weight_keys = {
        "attn.Wqkv": "recursive_block.attn.Wqkv.weight",
        "attn.Wo": "recursive_block.attn.Wo.weight",
        "mlp.c_fc": "recursive_block.mlp.c_fc.weight",
        "mlp.c_proj": "recursive_block.mlp.c_proj.weight",
    }
    missing = [key for key in weight_keys.values() if key not in state]
    if missing:
        raise KeyError(f"Missing weights: {', '.join(missing)}")

    base_dir = get_base_dir()
    out_dir = base_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    out_stem = args.out

    spectra: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    eig_spectra: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for name, key in weight_keys.items():
            W = state[key].float()
            shapes[name] = tuple(W.shape)
            s = torch.linalg.svdvals(W).cpu().numpy()
            spectra[name] = s
            if args.eig and W.ndim == 2 and W.shape[0] == W.shape[1]:
                try:
                    eig_spectra[name] = _eig_magnitudes(W)
                except Exception as e:
                    print(f"[warn] eig failed for {name}: {e}")

    # --- Plot 1: singular values ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    for name in weight_keys:
        s = np.sort(np.asarray(spectra[name]))[::-1]
        ax1.plot(s, label=f"{name} {shapes[name]}")
    ax1.set_title("Transformer block weight singular values")
    ax1.set_xlabel("Singular value index")
    ax1.set_ylabel("Singular value")
    if args.logy:
        ax1.set_yscale("log")
    ax1.legend()
    fig1.tight_layout()
    out_path1 = out_dir / f"{out_stem}_svd.png"
    fig1.savefig(out_path1, dpi=150)
    print(f"Saved plot to {out_path1}")

    # --- Plot 2: cumulative energy (s^2) ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for name in weight_keys:
        s = np.sort(np.asarray(spectra[name]))[::-1]
        e = s * s
        ce = np.cumsum(e) / (np.sum(e) + 1e-12)
        ax2.plot(ce, label=f"{name} {shapes[name]}")
    ax2.set_title("Cumulative spectral energy (s^2)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Energy fraction in top-k")
    ax2.set_ylim(0.0, 1.02)
    ax2.legend()
    fig2.tight_layout()
    out_path2 = out_dir / f"{out_stem}_energy.png"
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved plot to {out_path2}")

    # --- Optional: per-head spectra for Q/K/V ---
    if args.n_heads is not None:
        Wqkv = state[weight_keys["attn.Wqkv"]].float()
        Wq, Wk, Wv = _split_qkv(Wqkv)
        per_head = {
            "Wq": _per_head_svd(Wq, args.n_heads),
            "Wk": _per_head_svd(Wk, args.n_heads),
            "Wv": _per_head_svd(Wv, args.n_heads),
        }
        # summarize: top singular per head
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        heads = np.arange(args.n_heads)
        ax3.plot(heads, per_head["Wq"][:, 0], label="Q: top σ")
        ax3.plot(heads, per_head["Wk"][:, 0], label="K: top σ")
        ax3.plot(heads, per_head["Wv"][:, 0], label="V: top σ")
        ax3.set_title("Per-head spectral norm (top singular value) for Wq/Wk/Wv")
        ax3.set_xlabel("Head index")
        ax3.set_ylabel("Top singular value")
        ax3.legend()
        fig3.tight_layout()
        out_path3 = out_dir / f"{out_stem}_per_head_qkv.png"
        fig3.savefig(out_path3, dpi=150)
        print(f"Saved plot to {out_path3}")

    # --- Optional: eigenvalue magnitude spectra plot ---
    if args.eig and eig_spectra:
        fig4, ax4 = plt.subplots(figsize=(9, 5))
        for name, mag in eig_spectra.items():
            ax4.plot(mag, label=f"{name} eig|·| {shapes[name]}")
        ax4.set_title("Eigenvalue magnitude spectra (square matrices)")
        ax4.set_xlabel("Index")
        ax4.set_ylabel("|eigenvalue|")
        ax4.legend()
        fig4.tight_layout()
        out_path4 = out_dir / f"{out_stem}_eig.png"
        fig4.savefig(out_path4, dpi=150)
        print(f"Saved plot to {out_path4}")

    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
