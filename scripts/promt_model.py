"""Quick vibecoded tool to test the base model. Can only run on gpu."""

import argparse
import os

import torch

from recursive_lm.common import get_base_dir
from recursive_lm.model import ModelConfig, RecursiveGPT
from recursive_lm.tokenizer import RustBPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Prompt a trained RecursiveGPT model")
    parser.add_argument("--model", type=str, required=True, help="Model filename under models/")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer filename under tokenizers/")
    parser.add_argument("--gen_tok_count", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    args = parser.parse_args()

    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", args.model)
    tokenizer_path = os.path.join(base_dir, "tokenizers", args.tokenizer)

    tokenizer = RustBPETokenizer.load_from_dir(tokenizer_path)
    config = ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = RecursiveGPT(config)
    state = torch.load(model_path, map_location="cpu")
    device = "cuda" # Cuda is required due to flash-attn
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    def generate(prompt):
        tokens = tokenizer.encode(prompt)

        with torch.no_grad():
            for _ in range(args.gen_tok_count):
                ctx = tokens[-config.sequence_len:]
                input_ids = torch.tensor(ctx, dtype=torch.long, device=device)
                cu_seqlens = torch.tensor([0, input_ids.numel()], dtype=torch.int32, device=device)
                position_ids = torch.arange(input_ids.numel(), dtype=torch.long, device=device)
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(input_ids, cu_seqlens, position_ids)
                else:
                    logits = model(input_ids, cu_seqlens, position_ids)
                next_id = int(torch.argmax(logits[-1]).item())
                tokens.append(next_id)

        print(tokenizer.decode(tokens))

    if args.prompt is not None:
        generate(args.prompt)
        return

    while True:
        try:
            prompt = input("Prompt: ")
        except (EOFError, KeyboardInterrupt):
            break
        generate(prompt)


if __name__ == "__main__":
    if not torch.cuda.is_available(): # Cuda is required due to flash-attn
        raise Exception("Cuda not available!")
    main()
