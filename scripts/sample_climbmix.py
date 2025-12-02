"""
This script samples gvlassis/ClimbMix to download a tiny, detokenized subset of Nvidia's ClimbMix dataset in the same cluster ratios.
"""

import argparse
import os
import re
from datasets import load_dataset
from recursive_lm.common import get_base_dir

# Cluster sizes (row counts) from HuggingFace viewer for gvlassis/ClimbMix.
# These provide accurate enough cluster proportions.
CLUSTER_ROWS = {
    1: 4785103,
    2: 6684586,
    3: 8003099,
    4: 21348980,
    5: 10450928,
    6: 98368523,
    7: 92561323,
    8: 6455507,
    9: 4493536,
    10: 40603579,
    11: 8630635,
    12: 142111098,
    13: 5004064,
    14: 1530996,
    15: 1296383,
    16: 40297278,
    17: 38854459,
    18: 12586375,
    19: 6437288,
    20: 2811316,
}

PUNCT_DASH_CHARS = "!\"#$%&'()*+,\\-./:;<=>?@[\\]^_`{|}~—–"
LONG_PUNCT_RE = re.compile(rf"[{re.escape(PUNCT_DASH_CHARS)}]{{6,}}")


def clamp_punctuation_runs(text: str) -> str:
    return LONG_PUNCT_RE.sub(lambda m: m.group(0)[:3], text)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-words", type=int, default=100_000_000,
                    help="Desired output size in words (default: 100000000).")
    ap.add_argument("--buffer-size", type=int, default=100_000,
                    help="Shuffle buffer size for streaming dataset.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for per-cluster shuffle.")
    return ap.parse_args()


def main():
    args = parse_args()

    target_words_total = int(args.target_words)

    total_rows = sum(CLUSTER_ROWS.values())
    cluster_frac = {cid: CLUSTER_ROWS[cid] / total_rows for cid in CLUSTER_ROWS}
    cluster_target_words = {
        cid: target_words_total * cluster_frac[cid] for cid in CLUSTER_ROWS
    }

    print("Target total words:", target_words_total)
    print("\nCluster mixture (approx):")
    for cid in sorted(cluster_frac):
        print(f" cluster {cid:2d}: {cluster_frac[cid]*100:6.2f}%")

    written_total_words = 0
    written_per_cluster = {cid: 0 for cid in CLUSTER_ROWS}

    base_dir = get_base_dir()
    filename = f"climbmix{target_words_total}words.txt"

    with open(os.path.join(base_dir, "data", filename), "w", encoding="utf-8") as fout:
        for cid in sorted(CLUSTER_ROWS):
            if written_total_words >= target_words_total:
                break

            print(f"\nSampling cluster_id={cid} ...")

            target_c = cluster_target_words[cid]

            ds = load_dataset(
                "gvlassis/ClimbMix",
                f"cluster_id={cid}",
                split="train",
                streaming=True
            ).shuffle(seed=args.seed + cid, buffer_size=args.buffer_size)

            for row in ds:
                if written_total_words >= target_words_total:
                    break
                if written_per_cluster[cid] >= target_c:
                    break

                text = clamp_punctuation_runs(row["text"])

                word_count = len(text.split())
                if word_count == 0:
                    continue

                # Ensure docs are separated by at least one newline.
                # Preserve text content otherwise.
                if not text.endswith("\n"):
                    text += "\n"
                fout.write(text)

                written_total_words += word_count
                written_per_cluster[cid] += word_count

            print(f"  Wrote {written_per_cluster[cid]:,} words "
                  f"(target ~{int(target_c):,} words)")

    print("\nDONE.")
    print(f"Total written: {written_total_words:,} words")
    print("Per-cluster words:")
    for cid in sorted(written_per_cluster):
        print(f" cluster {cid:2d}: {written_per_cluster[cid]:,} words")

if __name__ == "__main__":
    main()
