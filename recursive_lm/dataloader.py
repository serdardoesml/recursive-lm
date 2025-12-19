"""
Data loader that streams tokenized data from parquet files and packs into (micro)batches to be used in training.

The goal is to fill each microbatch with a fixed number of tokens across any number of segments,
with each segment having a maximum length to ensure a minimum diversity within each batch, and to prevent attention memory exploding.

With a max segment length of 512 and a target token per batch of 65536 and grad_acc of 8 (Meaning our batch is split into 8 microbatches)
Each microbatch has a minimum of 7681/8192 tokens in the worst case. Meaning we get a minimum fill-rate of 93% for this configuration.
This could be improved if we split up a segment further to fill the microbatch,
but this is probably unnecessary and could trade performance for speed slightly.

TODO: Add batch utilization rate metric tracking.
"""

import os
import random

import pyarrow.parquet as pq
import torch

from common import get_base_dir


def parquet_doc_segments(parquet_path, token_col="tokens", T=512):
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
    row_group_perm = random.sample(range(pf.num_row_groups), pf.num_row_groups)

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


def pack_batch(segments, device):
    """Pack Python-list segments into a FlashAttention-varlen batch.

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
    device="cpu",
):
    """Yield packed (micro)batches with fixed token budget.

    We stream doc-segments (each <= T+1 tokens raw, so <= T after shift) and pack
    them until the sum of training positions (len(chunk)-1) reaches
    tokens_per_batch.

    Yields the output of pack_batch(buf).
    """

    assert max_sl <= tokens_per_batch

    buf: list[list[int]] = []
    tok = 0  # sum of (len(chunk)-1) in buf

    for chunk in parquet_doc_segments(parquet_path, token_col=token_col, T=max_sl):
        seglen = len(chunk) - 1
        if seglen <= 0:
            continue

        # If adding this would exceed budget, flush current batch first.
        if buf and tok + seglen > tokens_per_batch:
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
