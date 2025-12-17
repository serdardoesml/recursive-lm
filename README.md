# Random Notes

We shuffle by rowgroups in the dataloader, for climbmix we use a row group size of 32 so with a seq length of around 1 mean doc length and batch size of 256, we sample from 8 random spots per batch.

We use FlashAttention to have variable length fast attention to prevent cross-doc attention.
FlashAttention expects fp16/bf16 on CUDA for the fast path.

bb24.train has 7,195,269 words

# Commands

Build rustbpe with:
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

Download 100M word plain text sample of climbix with:
python -m scripts.sample_climbmix

Train tokenizer with:
python -m scripts.train_tokenizer --dataset climbmix100Mwords.txt

Tokenize data with:
python -m scripts.tokenize_data --dataset climbmix100Mwords.txt --tokenizer climbmix100Mwords_tokenizer.pkl