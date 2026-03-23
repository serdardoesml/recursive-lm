import argparse
import math
import os

import pyarrow as pa
import pyarrow.parquet as pq

from recursive_lm.common import get_base_dir


def resolve_dataset_path(dataset: str) -> str:
    # Allow either an explicit path or a filename under data/tokenized.
    if os.path.isabs(dataset) or os.path.exists(dataset):
        return dataset
    return os.path.join(get_base_dir(), "data", "tokenized", dataset)


def prefix_batch(batch, tokens_to_keep: int):
    # Keep exactly the requested token-prefix from this batch, truncating only the last row if needed.
    length_idx = batch.schema.get_field_index("length")
    token_idx = batch.schema.get_field_index("tokens")
    lengths = batch.column(length_idx).to_pylist()

    kept_tokens = []
    kept_lengths = []
    kept_total = 0

    for row_idx, length in enumerate(lengths):
        if kept_total + length <= tokens_to_keep:
            kept_tokens.append(batch.column(token_idx)[row_idx].as_py())
            kept_lengths.append(length)
            kept_total += length
            continue

        remaining = tokens_to_keep - kept_total
        if remaining > 0:
            kept_tokens.append(batch.column(token_idx)[row_idx].as_py()[:remaining])
            kept_lengths.append(remaining)
            kept_total += remaining
        break

    return pa.record_batch(
        [pa.array(kept_tokens), pa.array(kept_lengths)],
        names=["tokens", "length"],
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Tokenized parquet file or filename under data/tokenized")
args = parser.parse_args()

input_path = resolve_dataset_path(args.dataset)
pf = pq.ParquetFile(input_path)

total_tokens = 0
# Count total tokens first so the subset cutoffs are absolute.
for batch in pf.iter_batches(columns=["length"]):
    total_tokens += sum(batch.column(0).to_pylist())

if total_tokens == 0:
    raise ValueError(f"No tokens found in {input_path}")

fractions = [("10pct", 0.10), ("25pct", 0.25), ("50pct", 0.50)]
targets = [(name, min(total_tokens, math.ceil(total_tokens * fraction))) for name, fraction in fractions]

stem, _ = os.path.splitext(input_path)
writers = {
    name: pq.ParquetWriter(f"{stem}_{name}.parquet", pf.schema_arrow)
    for name, _ in targets
}
written_tokens = {name: 0 for name, _ in targets}
finished = set()
seen_tokens = 0

try:
    for batch in pf.iter_batches():
        lengths = batch.column(batch.schema.get_field_index("length")).to_pylist()
        batch_tokens = sum(lengths)

        for name, target in targets:
            if name in finished:
                continue

            if seen_tokens + batch_tokens <= target:
                writers[name].write_batch(batch)
                written_tokens[name] += batch_tokens
                continue

            # This batch crosses the cutoff, so write only the needed prefix.
            partial = prefix_batch(batch, target - seen_tokens)
            writers[name].write_batch(partial)
            written_tokens[name] = target
            finished.add(name)

        seen_tokens += batch_tokens
        if len(finished) == len(targets):
            break
finally:
    for writer in writers.values():
        writer.close()

for name, _ in targets:
    print(f"{stem}_{name}.parquet: {written_tokens[name]} tokens")
