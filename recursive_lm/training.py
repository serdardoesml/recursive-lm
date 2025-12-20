import torch
import torch.nn.functional as F
from .model import RecursiveGPT, ModelConfig
from .dataloader import batch_iterator
from .common import get_base_dir

from dataclasses import dataclass
from datetime import datetime
import os
import time

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    if num_warmup_steps <= 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=max(1, num_training_steps),
            last_epoch=last_epoch,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=num_warmup_steps,
        last_epoch=last_epoch,
    )
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=max(1, num_training_steps - num_warmup_steps),
        last_epoch=last_epoch,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[num_warmup_steps],
        last_epoch=last_epoch,
    )

@dataclass
class TrainingConfig:
    model_config: ModelConfig
    lr: float = 7e-3

    # Default target batch size: 65536 tok
    microbatch_tok: int = 32768
    grad_acc: int = 2

    max_tok_count: int = 130300589 # 130.3M default max tok count, 16000 microbatches, 2000 updates per epoch
    epoch: int = 10 # 10 epochs by default
    warmup_steps: int = 0
    use_wandb: bool = False
    wandb_project: str = "recursive-lm"
    wandb_run_name: str | None = None

def train(train_config: TrainingConfig, parquet_path, device, save=False):
    model = RecursiveGPT(train_config.model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_config.lr) # TODO: Try muon optimizer

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
    )
    wandb_run = None
    if train_config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
        )

    print(
        "Training summary | "
        f"epochs {train_config.epoch} | "
        f"total_steps {total_steps} | "
        f"lr {train_config.lr:.6g} | "
        f"distinct params {train_config.model_config.total_param_size:,} | "
        f"unrolled params {train_config.model_config.total_unrolled_param_size:,}"
    )

    for epoch_idx in range(train_config.epoch):
        for input_ids, targets, cu_seqlens, position_ids in batch_iterator(
            parquet_path,
            tokens_per_batch=train_config.microbatch_tok,
            max_sl=train_config.model_config.sequence_len,
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
                opt.step()
                scheduler.step()
                step += 1
                avg_loss = accum_loss / train_config.grad_acc
                now = time.time()
                step_time = now - last_step_time
                avg_step_time = (now - start_time) / step
                remaining = avg_step_time * (total_steps - step)
                lr = scheduler.get_last_lr()[0]
                tok_per_s = train_config.microbatch_tok / step_time
                print(
                    f"Epoch {epoch_idx + 1}/{train_config.epoch} "
                    f"Step {step}/{total_steps} training loss: {avg_loss:.4f} "
                    f"lr {lr:.6g} step_time {step_time:.2f}s tok/s {tok_per_s} "
                    f"eta {remaining:.0f}s "
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch_idx + 1,
                            "step": step,
                            "total_steps": total_steps,
                            "loss": avg_loss,
                            "lr": lr,
                            "step_time_s": step_time,
                            "tok_per_s": tok_per_s,
                        },
                        step=step,
                    )
                last_step_time = now
                if step >= total_steps:
                    break
                opt.zero_grad(set_to_none=True)
                accum_loss = 0.0
        if step >= total_steps:
            break

    if save:
        save_model(model)
    if wandb_run is not None:
        wandb_run.finish()

def save_model(model):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    path = os.path.join(get_base_dir(), "models", f"model_{timestamp}.pth")
    torch.save(model.state_dict(), path)
