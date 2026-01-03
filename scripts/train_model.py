import argparse
import os

import torch
import torch.distributed as dist

from recursive_lm.common import get_base_dir
from recursive_lm.model import ModelConfig
from recursive_lm.training import TrainingConfig, train

parser = argparse.ArgumentParser(description="Train RecursiveGPT")
parser.add_argument("--dataset", type=str, required=True, help="Tokenized parquet filename under data/tokenized")

parser.add_argument("--lr_embed", type=float, default=TrainingConfig.lr_embed)
parser.add_argument("--lr_block", type=float, default=TrainingConfig.lr_block)
parser.add_argument("--min_lr_embed", type=float, default=TrainingConfig.min_lr_embed)
parser.add_argument("--min_lr_block", type=float, default=TrainingConfig.min_lr_block)
parser.add_argument("--wd_adam", type=float, default=TrainingConfig.wd_adam)
parser.add_argument("--wd_muon", type=float, default=TrainingConfig.wd_muon)
parser.add_argument("--microbatch_tok", type=int, default=TrainingConfig.microbatch_tok)
parser.add_argument("--grad_acc", type=int, default=TrainingConfig.grad_acc)
parser.add_argument("--max_tok_count", type=int, default=TrainingConfig.max_tok_count)
parser.add_argument("--epoch", type=int, default=TrainingConfig.epoch)
parser.add_argument("--warmup_steps", type=int, default=TrainingConfig.warmup_steps)
parser.add_argument("--wandb", type=str, choices=["true", "false"], default="false")
parser.add_argument("--wandb_project", type=str, default=TrainingConfig.wandb_project)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--grad_clip", type=str, choices=["true", "false"], default="true")
parser.add_argument("--max_grad_norm", type=float, default=TrainingConfig.max_grad_norm)

parser.add_argument("--sequence_len", type=int, default=ModelConfig.sequence_len)
parser.add_argument("--vocab_size", type=int, default=ModelConfig.vocab_size)
parser.add_argument("--n_head", type=int, default=ModelConfig.n_head)
parser.add_argument("--n_hidden", type=int, default=ModelConfig.n_hidden)
parser.add_argument("--n_wembed", type=int, default=ModelConfig.n_wembed)
parser.add_argument("--mlp_mul", type=int, default=ModelConfig.mlp_mul)
parser.add_argument("--rec_depth", type=int, default=ModelConfig.rec_depth)
parser.add_argument("--tie_embed", type=str, choices=["true", "false"], default="true")
parser.add_argument("--standard_gpt", type=str, choices=["true", "false"], default="false") # For experiments
parser.add_argument("--grad_checkpointing", type=str, choices=["true", "false"], default="false")
parser.add_argument("--save", type=str, choices=["true", "false"], default="true")

args = parser.parse_args()

model_config = ModelConfig(
    sequence_len=args.sequence_len,
    vocab_size=args.vocab_size,
    n_head=args.n_head,
    n_hidden=args.n_hidden,
    n_wembed=args.n_wembed,
    mlp_mul=args.mlp_mul,
    rec_depth=args.rec_depth,
    tie_embed=args.tie_embed == "true",
    standard_gpt=args.standard_gpt == "true",
)

train_config = TrainingConfig(
    model_config=model_config,
    lr_embed=args.lr_embed,
    lr_block=args.lr_block,
    min_lr_embed=args.min_lr_embed,
    min_lr_block=args.min_lr_block,
    wd_adam=args.wd_adam,
    wd_muon=args.wd_muon,
    microbatch_tok=args.microbatch_tok,
    grad_acc=args.grad_acc,
    max_tok_count=args.max_tok_count,
    epoch=args.epoch,
    warmup_steps=args.warmup_steps,
    use_wandb=args.wandb == "true",
    wandb_project=args.wandb_project,
    run_name=args.run_name or None,
    grad_clip=args.grad_clip == "true",
    max_grad_norm=args.max_grad_norm,
    grad_checkpointing=args.grad_checkpointing == "true",
)

parquet_path = os.path.join(get_base_dir(), "data", "tokenized", args.dataset)
train(train_config, parquet_path, device="cuda", save=args.save == "true")
