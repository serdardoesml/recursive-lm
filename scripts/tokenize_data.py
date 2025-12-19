"""Script to tokenize the data with a given tokenizer and store it as a parquet table."""
import argparse
import os, time
import pyarrow as pa
import pyarrow.parquet as pq
from recursive_lm.common import get_base_dir
from recursive_lm.tokenizer import RustBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset filename (Has to be located under data directory)')
parser.add_argument('--tokenizer', type=str, help='Tokenizer filename (Has to be located under tokenizers directory)')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to tokenize (default: 10B)')

args = parser.parse_args()

base_dir = get_base_dir()

tok = RustBPETokenizer.load_from_dir(os.path.join(base_dir, "tokenizers", args.tokenizer))

# Iterator that yields an entire plain text file (Limited by max chars). Assumes utf-8 encoding.
def raw_text_iterator():
    with open(os.path.join(base_dir, "data", args.dataset), "r", encoding="utf-8") as f:
        total = 0
        buf = ""
        for line in f:
            if total >= args.max_chars:
                break
            if total + len(line) > args.max_chars:
                line = line[: args.max_chars - total]
            total += len(line)

            # treat a newline + BOS marker as a document boundary
            if line.startswith("<|bos|> "):
                if buf:
                    yield buf
                buf = line[len("<|bos|> ") :]
                continue

            buf += line
        if buf:
            yield buf

t0 = time.time()
docs = []
lengths = []
out_path = os.path.join(base_dir, "data", "tokenized", os.path.splitext(args.dataset)[0] + ".parquet")
writer = None
total_docs = 0
for doc in raw_text_iterator():
    tokens = tok.encode(doc)
    docs.append(tokens)
    lengths.append(len(tokens))
    total_docs += 1

    if len(docs) >= 10_000:  # write in chunks to avoid holding everything in memory
        table = pa.Table.from_arrays([pa.array(docs), pa.array(lengths)], names=["tokens", "length"])
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table, row_group_size=4)
        docs.clear()
        lengths.clear()

if docs:
    table = pa.Table.from_arrays([pa.array(docs), pa.array(lengths)], names=["tokens", "length"])
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table, row_group_size=4)

if writer is not None:
    writer.close()
t1 = time.time()
total_time = t1 - t0
print(f"Wrote {total_docs} documents to {out_path} in {total_time:.2f}s")