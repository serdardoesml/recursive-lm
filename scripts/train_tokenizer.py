""" Script to train a tokenizer for a given dataset using the nanochat RustBPETokenizer. """
from recursive_lm.tokenizer import RustBPETokenizer
from recursive_lm.common import get_base_dir
import argparse
import time
import os

parser = argparse.ArgumentParser(description='Train a BPE tokenizer from a dataset')
parser.add_argument('--dataset', type=str, help='Dataset filename (Has to be located under data directory)')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
args = parser.parse_args()

base_dir = get_base_dir()

# Iterator that yields an entire plain text file (Limited by max chars). Assumes utf-8 encoding.
def raw_text_iterator():
    with open(os.path.join(base_dir, "data", args.dataset), "r", encoding="utf-8") as f:
        yield f.read(args.max_chars)

text_iter = raw_text_iterator()

# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# Save the tokenizer to disk
tokenizer_dir = os.path.join(base_dir, "tokenizers", os.path.splitext(args.dataset)[0] + "_tokenizer.pkl")
tokenizer.save(tokenizer_dir)

# Quick sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

"""
# Debug
token_strings = [tokenizer.decode([token_id]) for token_id in range(tokenizer.get_vocab_size())]
print(token_strings)
"""
