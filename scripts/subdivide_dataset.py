import argparse
import math
import os
import random
from bisect import bisect_left

import pyarrow as pa
import pyarrow.parquet as pq

from recursive_lm.common import get_base_dir


def resolve_dataset_path(dataset: str) -> str:
    # Allow either an explicit path or a filename under data/tokenized.
    if os.path.isabs(dataset) or os.path.exists(dataset):
        return dataset
    return os.path.join(get_base_dir(), "data", "tokenized", dataset)


def prefix_table(table: pa.Table, lengths: list[int], tokens_to_keep: int) -> pa.Table:
    # Keep exactly the requested token-prefix, truncating only the final row if needed.
    cumulative = []
    running = 0
    for length in lengths:
        running += length
        cumulative.append(running)

    # Find the first row whose inclusion would meet or exceed the token budget.
    stop_idx = bisect_left(cumulative, tokens_to_keep)
    if stop_idx >= len(lengths):
        return table

    prefix = table.slice(0, stop_idx + 1)
    if cumulative[stop_idx] == tokens_to_keep:
        return prefix

    remaining = tokens_to_keep - (cumulative[stop_idx - 1] if stop_idx > 0 else 0)
    length_idx = prefix.schema.get_field_index("length")
    token_idx = prefix.schema.get_field_index("tokens")

    prefix_lengths = prefix.column(length_idx).to_pylist()
    prefix_tokens = prefix.column(token_idx).to_pylist()
    prefix_lengths[-1] = remaining
    prefix_tokens[-1] = prefix_tokens[-1][:remaining]

    prefix = prefix.set_column(
        length_idx,
        prefix.schema.field(length_idx),
        pa.array(prefix_lengths, type=prefix.schema.field(length_idx).type),
    )
    prefix = prefix.set_column(
        token_idx,
        prefix.schema.field(token_idx),
        pa.array(prefix_tokens, type=prefix.schema.field(token_idx).type),
    )
    return prefix


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Tokenized parquet file or filename under data/tokenized")
parser.add_argument("--seed", type=int, default=0, help="Seed used to shuffle document order before subdivision")
args = parser.parse_args()

input_path = resolve_dataset_path(args.dataset)
table = pq.read_table(input_path)
if table.num_rows == 0:
    raise ValueError(f"No rows found in {input_path}")

# Shuffle document order once up front so each output is a random prefix of rows.
rng = random.Random(args.seed)
permutation = list(range(table.num_rows))
rng.shuffle(permutation)
shuffled_table = table.take(pa.array(permutation, type=pa.int64()))

length_idx = shuffled_table.schema.get_field_index("length")
if length_idx == -1:
    raise ValueError(f"Dataset {input_path} is missing required 'length' column")

lengths = shuffled_table.column(length_idx).to_pylist()
total_tokens = sum(lengths)

if total_tokens == 0:
    raise ValueError(f"No tokens found in {input_path}")

fractions = [("10pct", 0.10), ("25pct", 0.25), ("50pct", 0.50)]
targets = [(name, min(total_tokens, math.ceil(total_tokens * fraction))) for name, fraction in fractions]

stem, _ = os.path.splitext(input_path)
for name, target in targets:
    # Reuse the same shuffled order for all subsets so 10% is contained in 25%, etc.
    subset = prefix_table(shuffled_table, lengths, target)
    output_path = f"{stem}_{name}.parquet"
    pq.write_table(subset, output_path)
    print(f"{output_path}: {target} tokens")
