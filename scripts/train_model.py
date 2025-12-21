import argparse
import os

from recursive_lm.common import get_base_dir
from recursive_lm.model import ModelConfig
from recursive_lm.training import TrainingConfig, train

parser = argparse.ArgumentParser(description="Train RecursiveGPT")
parser.add_argument("--dataset", type=str, required=True, help="Tokenized parquet filename under data/tokenized")

parser.add_argument("--lr_embed", type=float, default=TrainingConfig.lr_embed)
parser.add_argument("--lr_block", type=float, default=TrainingConfig.lr_block)
parser.add_argument("--microbatch_tok", type=int, default=TrainingConfig.microbatch_tok)
parser.add_argument("--grad_acc", type=int, default=TrainingConfig.grad_acc)
parser.add_argument("--max_tok_count", type=int, default=TrainingConfig.max_tok_count)
parser.add_argument("--epoch", type=int, default=TrainingConfig.epoch)
parser.add_argument("--warmup_steps", type=int, default=TrainingConfig.warmup_steps)
parser.add_argument("--wandb", type=str, choices=["true", "false"], default="false")
parser.add_argument("--wandb_project", type=str, default=TrainingConfig.wandb_project)
parser.add_argument("--wandb_run_name", type=str, default="")

parser.add_argument("--sequence_len", type=int, default=ModelConfig.sequence_len)
parser.add_argument("--vocab_size", type=int, default=ModelConfig.vocab_size)
parser.add_argument("--n_head", type=int, default=ModelConfig.n_head)
parser.add_argument("--n_embd", type=int, default=ModelConfig.n_embd)
parser.add_argument("--mlp_mul", type=int, default=ModelConfig.mlp_mul)
parser.add_argument("--rec_depth", type=int, default=ModelConfig.rec_depth)
parser.add_argument("--tie_embed", type=str, choices=["true", "false"], default="true")
parser.add_argument("--standard_gpt", type=str, choices=["true", "false"], default="false") # For experiments
parser.add_argument("--save", type=str, choices=["true", "false"], default="true")

args = parser.parse_args()

model_config = ModelConfig(
    sequence_len=args.sequence_len,
    vocab_size=args.vocab_size,
    n_head=args.n_head,
    n_embd=args.n_embd,
    mlp_mul=args.mlp_mul,
    rec_depth=args.rec_depth,
    tie_embed=args.tie_embed == "true",
    standard_gpt=args.standard_gpt == "true",
)

train_config = TrainingConfig(
    model_config=model_config,
    lr_embed=args.lr_embed,
    lr_block=args.lr_block,
    microbatch_tok=args.microbatch_tok,
    grad_acc=args.grad_acc,
    max_tok_count=args.max_tok_count,
    epoch=args.epoch,
    warmup_steps=args.warmup_steps,
    use_wandb=args.wandb == "true",
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name or None,
)

parquet_path = os.path.join(get_base_dir(), "data", "tokenized", args.dataset)
train(train_config, parquet_path, device="cuda", save=args.save == "true")
