"""
Minimal, token-only inference module for RecursiveGPT.
Caller owns tokenizer/model lifecycle; this module is stateless.
Currently only does batched inference, and is not really optimized.

TODO: Add kv-cached inference for efficient chatting.
TODO: Explore more efficient ways to load model during inference.
TODO: Support alternative attention backends for inference.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

def pack_varlen(token_seqs: Sequence[Sequence[int]] | Sequence[int], device=None):
    """
    Pack token sequences into varlen-attn format.

    Returns:
      flat_input   [total_tokens] (long)
      cu_seqlens   [B+1]          (int32)
      position_ids[total_tokens] (long, resets each sequence)
      lengths      list[int]
      offsets      list[int]  (start index of each sequence in flat_input)
    """
    if not token_seqs:
        flat_input = torch.empty((0,), dtype=torch.long)
        cu_seqlens = torch.zeros((1,), dtype=torch.int32)
        position_ids = torch.empty((0,), dtype=torch.long)
        return flat_input, cu_seqlens, position_ids, [], []

    if isinstance(token_seqs[0], int):
        token_seqs = [token_seqs]

    lengths: List[int] = [len(seq) for seq in token_seqs]
    offsets: List[int] = []
    cu: List[int] = [0]
    flat: List[int] = []
    pos: List[int] = []
    running = 0
    for seq in token_seqs:
        offsets.append(running)
        L = len(seq)
        flat.extend(seq)
        pos.extend(range(L))
        running += L
        cu.append(running)

    flat_input = torch.tensor(flat, dtype=torch.long)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    position_ids = torch.tensor(pos, dtype=torch.long)

    if device is not None:
        flat_input = flat_input.to(device)
        cu_seqlens = cu_seqlens.to(device)
        position_ids = position_ids.to(device)

    return flat_input, cu_seqlens, position_ids, lengths, offsets


@torch.no_grad()
def forward_logits(model, flat_input: torch.Tensor, cu_seqlens: torch.Tensor, position_ids: torch.Tensor):
    """Forward a packed batch and return flat logits [total_tokens, vocab_size]."""
    if flat_input.numel() == 0:
        vocab = int(getattr(model.config, "vocab_size", 0))
        return flat_input.new_empty((0, vocab))
    return model(flat_input, cu_seqlens, position_ids)
