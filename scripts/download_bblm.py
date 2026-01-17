"""
Download BabyLM data and write it as plaintext with BOS markers between docs.
"""

import argparse
import os
from datasets import load_dataset
from recursive_lm.common import get_base_dir


def iter_splits(dataset):
    if hasattr(dataset, "items"):
        for split_name, split_ds in dataset.items():
            yield split_name, split_ds
    else:
        yield "train", dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--10M",
        "-10M",
        dest="ten_m",
        action="store_true",
        help="Download the 10M dataset instead of 100M.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.ten_m:
        dataset_name = "ltg/babylm-2024-baby-cosmo-fine-10m"
        out_name = "bblm10M.txt"
    else:
        dataset_name = "ltg/babylm-2024-baby-cosmo-fine-100m"
        out_name = "bblm100M.txt"

    ds = load_dataset(dataset_name)
    base_dir = get_base_dir()
    out_path = os.path.join(base_dir, "data", out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total_docs = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for split_name, split_ds in iter_splits(ds):
            print(f"Writing split: {split_name}")
            for row in split_ds:
                if "text" not in row:
                    raise KeyError("Expected a 'text' field in dataset rows.")

                text = row["text"]
                if not text:
                    continue

                fout.write(text.rstrip("\n"))
                fout.write("\n")
                fout.write("<|bos|> ")
                total_docs += 1

    print(f"DONE. Wrote {total_docs:,} docs to {out_path}")


if __name__ == "__main__":
    main()
