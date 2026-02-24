"""
Data loader that streams tokenized data from parquet files and packs into (micro)batches to be used in training.

The goal is to fill each microbatch with a fixed number of tokens across any number of segments,
with each segment having a maximum length to ensure a minimum diversity within each batch, and to prevent attention memory exploding.

Sequences can be split further to fill up the microbatch with an exact number of tokens in order to retain fixed shapes.
"""

import os
import random

import pyarrow.parquet as pq
import torch

from .common import get_base_dir


def parquet_doc_segments(parquet_path, token_col="tokens", T=512, seed=None):
    """
    Yields token segments (as plain python lists), each chunk is from ONE document only.
    Chunk length is in [2, T+1].
    Row groups are read in random order to keep streaming simple while avoiding
    deterministic ordering across files. 
    This shuffling is necessary since the way we sample climbmix leads to our raw data
    being clustered. Choosing a low enough row group size is also important to ensure
    a diversity of clusters in each batch.
    """
    pf = pq.ParquetFile(parquet_path)

    if pf.num_row_groups == 0:
        return

    max_chunk_len = T + 1
    
    rng = random.Random(seed)
    row_group_perm = rng.sample(range(pf.num_row_groups), pf.num_row_groups)

    for rg_idx in row_group_perm:
        rb = pf.read_row_group(rg_idx, columns=[token_col])
        col = rb.column(0)

        for row in col:
            toks = row.as_py()
            if not toks or len(toks) < 2: # Skip empty rows or rows with less than 2 tokens. This should never happen.
                continue

            for start in range(0, len(toks) - 1, max_chunk_len):
                chunk = toks[start : start + max_chunk_len]
                if len(chunk) >= 2:
                    yield chunk


def count_dataset_tokens(parquet_path, token_col="tokens"):
    """Count total training tokens (next-token targets) in a parquet dataset."""
    pf = pq.ParquetFile(parquet_path)
    total_tokens = 0
    for rg_idx in range(pf.num_row_groups):
        rb = pf.read_row_group(rg_idx, columns=[token_col])
        col = rb.column(0)
        for row in col:
            toks = row.as_py()
            if toks and len(toks) >= 2:
                total_tokens += len(toks) - 1
    return total_tokens


def pack_batch(segments, device):
    """Pack Python-list segments into a varlen-attn batch.

    segments: list[list[int]], each segment length >= 2 and <= T

    Returns:
      input_ids    [total_tokens]  (long)
      targets      [total_tokens]  (long)
      cu_seqlens   [B+1]           (int32)
      position_ids [total_tokens]  (long, resets each segment)

    """

    input_ids: list[int] = []
    targets: list[int] = []
    position_ids: list[int] = []
    cu: list[int] = [0]

    offset = 0

    for s in segments:
        L = len(s) - 1  # after shift

        # x = s[:-1], y = s[1:]
        input_ids.extend(s[:-1])
        targets.extend(s[1:])
        position_ids.extend(range(L))

        offset += L
        cu.append(offset)

    # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
    # I have absolutely no idea if doing it this way is any faster than creating the tensor on GPU directly
    # It probably does not matter much, and i spent way too much time on it, so i am leaving it as it is
    use_cuda_optimizations = device == "cuda"
    input_ids_t = torch.tensor(input_ids, dtype=torch.long, pin_memory=use_cuda_optimizations, device="cpu")
    targets_t = torch.tensor(targets, dtype=torch.long, pin_memory=use_cuda_optimizations, device="cpu")
    cu_seqlens_t = torch.tensor(cu, dtype=torch.int32, pin_memory=use_cuda_optimizations, device="cpu")
    position_ids_t = torch.tensor(position_ids, dtype=torch.long, pin_memory=use_cuda_optimizations, device="cpu")

    if device is not None:
        input_ids_t = input_ids_t.to(device, non_blocking=use_cuda_optimizations)
        targets_t = targets_t.to(device, non_blocking=use_cuda_optimizations)
        cu_seqlens_t = cu_seqlens_t.to(device, non_blocking=use_cuda_optimizations)
        position_ids_t = position_ids_t.to(device, non_blocking=use_cuda_optimizations)

    return input_ids_t, targets_t, cu_seqlens_t, position_ids_t


def batch_iterator(
    parquet_path,
    *,
    tokens_per_batch: int,
    max_sl: int = 512,
    token_col: str = "tokens",
    drop_last: bool = True,
    device="cuda",
    fix_length = True, # Default True: no observed accuracy downside, and fixed-length batches allow us to compute exact step counts.
    seed=None,
):
    """Yield packed (micro)batches with token budget `tokens_per_batch`.

    We stream doc-segments (each <= T+1 tokens raw, so <= T after shift) and pack
    them until the sum of training positions (len(chunk)-1) reaches
    tokens_per_batch.

    If `fix_length` is True, chunks may be split to exactly fit the token budget.
    If `fix_length` is False, chunks are never split and batch lengths may vary.

    Yields the output of pack_batch(buf).
    """

    assert max_sl <= tokens_per_batch

    buf: list[list[int]] = []
    tok = 0  # sum of (len(chunk)-1) in buf

    for chunk in parquet_doc_segments(parquet_path, token_col=token_col, T=max_sl, seed=seed):
        seglen = len(chunk) - 1
        if seglen <= 0:
            continue

        # If adding this would exceed budget:
        # - fix_length=True: split the current chunk to exactly fill.
        # - fix_length=False: flush current batch and keep chunk intact.
        if buf and tok + seglen > tokens_per_batch:
            remaining = tokens_per_batch - tok
            if fix_length and remaining > 0:
                # Split within the same chunk; overlap by one token to preserve last target.
                head = chunk[: remaining + 1]
                tail = chunk[remaining:]
                buf.append(head)
                tok += remaining
                yield pack_batch(buf, device=device)
                buf.clear()
                tok = 0
                chunk = tail
                seglen = len(chunk) - 1
            else:
                yield pack_batch(buf, device=device)
                buf.clear()
                tok = 0

        buf.append(chunk)
        tok += seglen

    if buf and not drop_last:
        yield pack_batch(buf, device=device)


# DEBUG
if __name__ == "__main__":
    parquet_file = os.path.join(get_base_dir(), "data", "tokenized", "climbmix100Mwords.parquet")
    batch_count = 5
    i = 0
    for input_ids, targets, cu_seqlens, position_ids in batch_iterator(
        parquet_file,
        tokens_per_batch=8192,
        max_sl=256,
        token_col="tokens",
        drop_last=True
    ):
        i += 1
        print(input_ids.shape, targets.shape, cu_seqlens.shape, position_ids.shape)
        if i > batch_count:
            break
