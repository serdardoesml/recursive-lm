"""Convert a RecursiveGPT checkpoint into a Hugging Face folder."""

import argparse
import shutil
from pathlib import Path

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
    (out_dir / "processor_config.json").unlink(missing_ok=True)
    (out_dir / "hf_processor.py").unlink(missing_ok=True)

    tok_path = base_dir / "tokenizers" / args.tokenizer
    if not tok_path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")
    tokenizer = RecursiveLMTokenizer(tokenizer_file=str(tok_path))
    tokenizer.save_pretrained(str(out_dir))

    generation_config_path = out_dir / "generation_config.json"
    generation_config = {
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with generation_config_path.open("w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)

    hf_dir = Path(__file__).resolve().parents[1] / "recursive_lm" / "huggingface"
    shutil.copyfile(hf_dir / "hf_wrapper.py", out_dir / "hf_wrapper.py")
    shutil.copyfile(hf_dir / "hf_tokenizer.py", out_dir / "hf_tokenizer.py")
    print(f"Saved Hugging Face model to {out_dir}")


if __name__ == "__main__":
    main()
