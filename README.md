General code structure inspired heavily from Andrej Karpathy's nanochat and nanogpt repositories.

Switched from FA2 to torch varlen-attn (same backend) to simplify dependencies, however this means torch 2.10 is required.

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
