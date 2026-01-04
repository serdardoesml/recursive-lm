# Random Notes

We shuffle by rowgroups in the dataloader, for climbmix we use a row group size of 4 so with a batch size of 8192 we sample from around 4 documents.

We use FlashAttention to have variable length fast attention to prevent cross-doc attention.
FlashAttention expects fp16/bf16 on CUDA for the fast path.

bb24.train has 7,195,269 words

flash_attn_varlen_qkvpacked_func signature
(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=False)

# Commands

Build rustbpe with:
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

Download 100M word plain text sample of climbix with:
python -m scripts.sample_climbmix

Train tokenizer with:
python -m scripts.train_tokenizer --dataset climbmix100Mwords.txt

Tokenize data with:
python -m scripts.tokenize_data --dataset climbmix100Mwords.txt --tokenizer climbmix100Mwords_tokenizer.pkl

Train model with (Requires uv group 'gpu'):
python -m scripts.train_model --dataset climbmix100Mwords.parquet

Convert to hf with (Requires uv group 'hf'):
python -m scripts.convert_hf model_name.pth --tokenizer tokenizer_name.pkl
