bb24.train has 7,195,269 words

Build rustbpe with:
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

Download 100M word plain text sample of climbix with:
python -m scripts.sample_climbmix

Train tokenizer with:
python -m scripts.train_tokenizer --dataset climbmix100Mwords.txt

Tokenize data with:
python -m scripts.tokenize_data --dataset climbmix100Mwords.txt --tokenizer climbmix100Mwords_tokenizer.pkl