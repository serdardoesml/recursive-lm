"""Convert a RecursiveGPT checkpoint into a Hugging Face folder."""

import argparse

from recursive_lm.common import get_base_dir
from recursive_lm.huggingface.hf_tokenizer import RecursiveLMTokenizer
from recursive_lm.huggingface.hf_wrapper import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert a checkpoint to Hugging Face format")
    parser.add_argument("model", type=str, help="Model filename under models/")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer filename under tokenizers/")
    args = parser.parse_args()

    base_dir = get_base_dir()
    model_path = base_dir / "models" / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_root = base_dir / "models" / "huggingface"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / model_path.stem
    convert_checkpoint(str(model_path), str(out_dir))

    tok_path = base_dir / "tokenizers" / args.tokenizer
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")
    tokenizer = RecursiveLMTokenizer(tokenizer_file=str(tok_path))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved Hugging Face model to {out_dir}")


if __name__ == "__main__":
    main()
