"""Data loader that streams tokenized data from parquet files into memory to be used in training."""

import os
import random

import pyarrow.parquet as pq
import torch

from common import get_base_dir


def parquet_doc_chunks(parquet_path, token_col="tokens", T=512):
    """
    Yields 1D LongTensor chunks, each chunk is from ONE document only.
    Chunk length is in [2, T+1].
    Row groups are read in random order to keep streaming simple while avoiding
    deterministic ordering across files.
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
            if not toks or len(toks) < 2:
                continue

            for start in range(0, len(toks) - 1, max_chunk_len):
                chunk = toks[start : start + max_chunk_len]
                if len(chunk) >= 2:
                    yield torch.tensor(chunk, dtype=torch.long)


def pack_batch(chunks):
    """
    chunks: list of 1D LongTensor, each length >= 2
    Returns:
      input_ids  [total_tokens]
      targets    [total_tokens]
      cu_seqlens [B+1] int32
      max_seqlen int
      position_ids [total_tokens] int32 (resets each seq)
    """
    xs = [c[:-1] for c in chunks]
    ys = [c[1:]  for c in chunks]
    seqlens = torch.tensor([x.numel() for x in xs], dtype=torch.int32)
    B = seqlens.numel()
    max_seqlen = int(seqlens.max().item())

    cu_seqlens = torch.empty((B + 1,), dtype=torch.int32)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    input_ids = torch.cat(xs, dim=0)
    targets   = torch.cat(ys, dim=0)

    position_ids = torch.cat(
        [torch.arange(int(l), dtype=torch.int32) for l in seqlens.tolist()],
        dim=0
    )

    return input_ids, targets, cu_seqlens, max_seqlen, position_ids


def batch_iterator(parquet_path, batch_size_seqs=32, T=512, token_col="tokens", drop_last=True):
    """
    Pure-python iterator that yields packed varlen batches.

    batch_size_seqs: number of sequences (doc-chunks) per batch.
    """
    buf = []
    for chunk in parquet_doc_chunks(parquet_path, token_col=token_col, T=T):
        buf.append(chunk)
        if len(buf) == batch_size_seqs:
            yield pack_batch(buf)
            buf.clear()

    if buf and not drop_last:
        yield pack_batch(buf)


# ---- Example usage ----
if __name__ == "__main__":
    parquet_file = os.path.join(get_base_dir(), "data", "tokenized", "climbmix100Mwords.parquet")

    for input_ids, targets, cu_seqlens, max_seqlen, position_ids in batch_iterator(parquet_file, batch_size_seqs=32, T=512, token_col="tokens", drop_last=True):
        print(input_ids.shape, targets.shape, cu_seqlens.shape, max_seqlen, position_ids.shape)
        break
