import torch
import torch.nn.functional as F
from .optimizer import SingleDeviceNorMuonWithAuxAdam
from .model import RecursiveGPT, ModelConfig
from .dataloader import batch_iterator
from .common import get_base_dir

from dataclasses import dataclass, asdict
from datetime import datetime
import os
import time

@dataclass
class TrainingConfig:
    model_config: ModelConfig
    lr_embed: float = 0.007
    lr_block: float = 0.02 # Muon
    wd_adam: float = 0.005
    wd_muon: float = 0.1

    # MASSIVE reduction in memory use as memory usage essentially reduces to single depth.
    # However, adds some compute overhead (roughly 30% at depth 48) that cannot be easily recovered by reducing grad_acc
    # Essentially makes training compute bound, useful for super high depths or large MLP multipliers on small GPUs.
    grad_checkpointing: bool = False 

    # Default target batch size: 65536 tok
    microbatch_tok: int = 32768
    grad_acc: int = 2
    sequence_len: int = 256 # Only for training, does not change model itself.

    # TODO: Add feature so max_tok_count is optional and by default determined from full dataset size.

    # 130.3M default max tok count, 4000 microbatches, 2000 updates per epoch
    max_tok_count: int = 130300589 # Update when changing dataset or tokenizer
    epoch: int = 10 # 10 epochs by default
    warmup_steps: int = 50
    min_lr_embed: float = 0.0
    min_lr_block: float = 0.0
    use_wandb: bool = False
    wandb_project: str = "recursive-lm"
    run_name: str = "rec-d24-small"
    grad_clip: bool = True
    max_grad_norm: float = 2.0

def train(train_config: TrainingConfig, parquet_path, device, save=False):
    model = RecursiveGPT(
        train_config.model_config,
        grad_checkpointing=train_config.grad_checkpointing,
    ).to(device)

    # Set up parameter groups
    embed_params = list(model.embedding.parameters())
    embed_params += list(model.e_to_h.parameters())
    embed_params += list(model.h_to_e.parameters())
    embed_params += list(model.norm_out.parameters())
    if hasattr(model, "lm_head"):
        embed_params += list(model.lm_head.parameters())
    if hasattr(model, "rec_layer_embedding"):
        embed_params += list(model.rec_layer_embedding.parameters())
    if train_config.model_config.standard_gpt:
        for block in model.blocks:
            embed_params += list(block.norm_attn.parameters())
            embed_params += list(block.norm_mlp.parameters())
            embed_params += list(block.attn.norm_qk.parameters())
        block_params = []
        for block in model.blocks:
            block_params += list(block.attn.Wqkv.parameters())
            block_params += list(block.attn.Wo.parameters())
            block_params += list(block.mlp.parameters())
    else:
        embed_params += list(model.recursive_block.norm_attn.parameters())
        embed_params += list(model.recursive_block.norm_mlp.parameters())
        embed_params += list(model.recursive_block.attn.norm_qk.parameters())
        block_params = list(model.recursive_block.attn.Wqkv.parameters())
        block_params += list(model.recursive_block.attn.Wo.parameters())
        block_params += list(model.recursive_block.mlp.parameters())

    opt = SingleDeviceNorMuonWithAuxAdam(
        [
            {"params": embed_params, "lr": train_config.lr_embed, "use_muon": False, "weight_decay": train_config.wd_adam},
            {"params": block_params, "lr": train_config.lr_block, "use_muon": True, "weight_decay": train_config.wd_muon},
        ]
    ) # NorMuon optimizer with CWD (References in optimizer.py)

    total_steps = int(
        (train_config.max_tok_count * train_config.epoch)
        / (train_config.microbatch_tok * train_config.grad_acc)
    )
    step = 0
    micro_step = 0
    accum_loss = 0.0
    opt.zero_grad(set_to_none=True)
    start_time = time.time()
    last_step_time = start_time
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=total_steps,
        min_lrs=[train_config.min_lr_embed, train_config.min_lr_block],
    )
    wandb_run = None
    if train_config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=train_config.wandb_project,
            name=train_config.run_name,
        )
        wandb.define_metric("tokens_processed")
        wandb.define_metric("*", step_metric="tokens_processed")

    print(
        "Training summary | "
        f"epochs {train_config.epoch} | "
        f"total_steps {total_steps} | "
        f"grad_checkpointing {train_config.grad_checkpointing} | "
        f"lr_embed {train_config.lr_embed:.6g} | "
        f"lr_block {train_config.lr_block:.6g} | "
        f"wd_adam {train_config.wd_adam:.6g} | "
        f"wd_muon {train_config.wd_muon:.6g} | "
        f"distinct params {model.total_param_size:,} | "
        f"unrolled params {model.total_unrolled_param_size:,}"
    )

    for epoch_idx in range(train_config.epoch):
        for input_ids, targets, cu_seqlens, position_ids in batch_iterator(
            parquet_path,
            tokens_per_batch=train_config.microbatch_tok,
            max_sl=train_config.sequence_len,
            device=device
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, cu_seqlens, position_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss_float = float(loss.detach())
            accum_loss += loss_float
            (loss / train_config.grad_acc).backward()
            micro_step += 1

            if micro_step % train_config.grad_acc == 0:
                if train_config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                opt.step()
                scheduler.step()
                step += 1
                avg_loss = accum_loss / train_config.grad_acc
                now = time.time()
                step_time = now - last_step_time
                avg_step_time = (now - start_time) / step
                remaining = avg_step_time * (total_steps - step)
                lr_embed, lr_block = scheduler.get_last_lr()
                tok_per_s = (train_config.microbatch_tok * train_config.grad_acc) / step_time
                tokens_processed = step * train_config.microbatch_tok * train_config.grad_acc
                print(
                    f"Epoch {epoch_idx + 1}/{train_config.epoch} "
                    f"Step {step}/{total_steps} training loss: {avg_loss:.4f} "
                    f"lr_embed {lr_embed:.6g} lr_block {lr_block:.6g} "
                    f"step_time {step_time:.2f}s tok/s {tok_per_s:.0f} "
                    f"eta {remaining:.0f}s "
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch_idx + 1,
                            "step": step,
                            "total_steps": total_steps,
                            "tokens_processed": tokens_processed,
                            "loss": avg_loss,
                            "lr_embed": lr_embed,
                            "lr_block": lr_block,
                            "step_time_s": step_time,
                            "tok_per_s": tok_per_s,
                        },
                        step=tokens_processed,
                    )
                last_step_time = now
                if step >= total_steps:
                    break
                opt.zero_grad(set_to_none=True)
                accum_loss = 0.0
        if step >= total_steps:
            break

    if save:
        save_model(model, train_config.run_name)
    if wandb_run is not None:
        wandb_run.finish()

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lrs: list[float],
    last_epoch: int = -1,
):
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    min_factors = []
    for lr, min_lr in zip(base_lrs, min_lrs, strict=True):
        if lr <= 0:
            min_factor = 1.0
        else:
            min_factor = min_lr / lr
            if not 0 <= min_factor <= 1:
                raise ValueError(f"min_lr must be in [0, base_lr], got min_lr={min_lr} base_lr={lr}")
        min_factors.append(min_factor)
    warmup_steps = max(0, num_warmup_steps)
    total_steps = max(1, num_training_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    warmup_start = 1e-8

    def build_lambda(min_factor: float):
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return warmup_start + (1.0 - warmup_start) * (step / warmup_steps)
            decay_step = step - warmup_steps
            if decay_step >= decay_steps:
                return min_factor
            decay_progress = decay_step / decay_steps
            return (1.0 - decay_progress) * (1.0 - min_factor) + min_factor
        return lr_lambda

    lr_lambdas = [build_lambda(min_factor) for min_factor in min_factors]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambdas,
        last_epoch=last_epoch,
    )

def save_model(model, run_name: str | None):
    if run_name:
        filename = f"{run_name}.pth"
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"model_{timestamp}.pth"
    path = os.path.join(get_base_dir(), "models", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
        },
        path,
    )
