"""
Simple training loop, most of the code is boilerplate metrics and initializing stuff.

Note: At some point i tried prefetching the batch iterator, does not improve total step time.
"""

import torch
import torch.nn.functional as F
from .optimizer import SingleDeviceNorMuonWithAuxAdam
from .model import RecursiveGPT, ModelConfig, MoE
from .dataloader import batch_iterator
from .common import get_base_dir

from dataclasses import dataclass, asdict
from datetime import datetime
import math
import os
import time

@dataclass
class TrainingConfig:
    model_config: ModelConfig
    lr_embed: float = 0.007
    lr_block: float = 0.02 # Muon requires higher learning rate.
    wd_adam: float = 0.005
    wd_muon: float = 0.1
    lb_coef: float = 0.004

    # Default target batch size: 65536 tok
    microbatch_tok: int = 32768
    grad_acc: int = 2
    sequence_len: int = 256 # Only for training, does not change model itself.

    # TODO: Add feature so max_tok_count is optional and by default determined from full dataset size.

    # 130.3M default max tok count, 4000 microbatches, 2000 updates per epoch
    max_tok_count: int = 130300589 # Update when changing dataset or tokenizer
    epoch: int = 10 # 10 epochs by default
    warmup_steps: int = 50
    cooldown_steps: int = 400
    min_lr_embed: float = 0.0
    min_lr_block: float = 0.0
    use_wandb: bool = False
    wandb_project: str = "recursive-lm"
    run_name: str = "rec-d24-small"
    grad_clip: bool = True
    max_grad_norm: float = 2.0

    # Massive speedup with torch 2.9.1 (33% reduction in step time), however with torch 2.4 it was constantly triggering recompilation.
    # Another weird thing is without compilation, 2.4 was faster than 2.9.1 at least on H100s, no idea why.
    # Options: "false", "true", "max-autotune" 
    torch_compile: str = "true" 

    # Limits step count to 10 and disables saving.
    profile: bool = False

def train(train_config: TrainingConfig, parquet_path, device, save=False):
    if train_config.profile:
        save = False
    train_config.model_config.rope_cache_len = train_config.sequence_len # Avoids inefficiency as this gets passed into varlen_attn as max_seqlen, probably better way to handle this though
    model = RecursiveGPT(
        train_config.model_config,
    ).to(device) # Init model and move to device

    moe_modules = [m for m in model.modules() if isinstance(m, MoE)] # Keep track for balance stats

    if train_config.torch_compile != "false":
        compile_kwargs = {} # After fixing token shape, setting dynamic to either true or false makes performance worse, as it's mostly static except for cu_seqlens
        if train_config.torch_compile == "max-autotune":
            compile_kwargs["mode"] = "max-autotune-no-cudagraphs"
        model = torch.compile(model, **compile_kwargs)

    # Set up param groups.
    # We split params so only block params use Muon,
    # and everything else (embeddings and norms) uses AdamW.
    adam_params, muon_params = model.get_param_groups()

    opt = SingleDeviceNorMuonWithAuxAdam(
        [
            {"params": adam_params, "lr": train_config.lr_embed, "use_muon": False, "weight_decay": train_config.wd_adam},
            {"params": muon_params, "lr": train_config.lr_block, "use_muon": True, "weight_decay": train_config.wd_muon},
        ]
    ) # NorMuon optimizer with CWD (References in optimizer.py)

    # Warn if any trainable params are missing from optimizer groups.
    # I wasted a lot of time forgetting to add params to the lr groups, hopefully this prevents that lol.
    param_names = {id(p): name for name, p in model.named_parameters()}
    opt_param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    missing = [param_names[id(p)] for p in model.parameters() if id(p) not in opt_param_ids]
    if missing:
        print(f"Warning: {len(missing)} params not in optimizer: {', '.join(missing)}")

    total_steps = int(
        (train_config.max_tok_count * train_config.epoch)
        / (train_config.microbatch_tok * train_config.grad_acc)
    )
    if train_config.profile:
        total_steps = min(total_steps, 10)

    # Initializing stuff
    step = 0
    micro_step = 0
    accum_loss = 0.0
    accum_lb_loss = 0.0
    opt.zero_grad(set_to_none=True)
    start_time = time.time()
    last_step_time = start_time
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=train_config.warmup_steps,
        num_cooldown_steps=train_config.cooldown_steps,
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

    profiler = None
    profiler_started = False
    if train_config.profile:
        print("Warning: Profiling mode enabled!")
        import torch.profiler as profiler_mod
        activities = [profiler_mod.ProfilerActivity.CPU]
        if str(device).startswith("cuda") and torch.cuda.is_available():
            activities.append(profiler_mod.ProfilerActivity.CUDA)
        profiler = profiler_mod.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        )

    print(
        "Training summary | "
        f"epochs {train_config.epoch} | "
        f"total_steps {total_steps} | "
        f"torch_compile {train_config.torch_compile} | "
        f"lr_embed {train_config.lr_embed:.6g} | "
        f"lr_block {train_config.lr_block:.6g} | "
        f"wd_adam {train_config.wd_adam:.6g} | "
        f"wd_muon {train_config.wd_muon:.6g} | "
        f"distinct params {model.total_param_size:,}"
    )

    try:
        for epoch_idx in range(train_config.epoch):
            # Batch iterator re-initializes at each epoch with a different random shuffle
            for input_ids, targets, cu_seqlens, position_ids in batch_iterator(
                parquet_path,
                tokens_per_batch=train_config.microbatch_tok,
                max_sl=train_config.sequence_len,
                device=device
            ):
                # Cast to bf16 for fast training with A100 and H100s .
                # Varlen-attn doesn't support anything else, so no need to change this really. 
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, cu_seqlens, position_ids)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                    # SimBal load balancing loss (https://arxiv.org/pdf/2506.14038v2)
                    # Modified to normalize router weights with l2 norm before calculating
                    lb_loss = logits.new_tensor(0.0)
                    for m in moe_modules:
                        W = m.router.weight.float()  # [E, D]
                        W = F.normalize(W, p=2, dim=1)
                        G = W @ W.t()  # [E, E]
                        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                        lb_loss = lb_loss + torch.norm(G - I, p=1)
                    total_loss = loss + train_config.lb_coef * lb_loss

                # Accumulate gradients
                loss_float = float(loss.detach())
                lb_loss_float = float(lb_loss.detach())
                accum_loss += loss_float
                accum_lb_loss += lb_loss_float
                (total_loss / train_config.grad_acc).backward()
                micro_step += 1

                if micro_step % train_config.grad_acc == 0:
                    # Optimizer Step
                    if train_config.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                    opt.step()
                    scheduler.step()
                    step += 1

                    # Metrics and logging
                    if step == 1:
                        now = time.time()
                        first_step_time = now - start_time
                        if train_config.torch_compile != "false":
                            print(f"Compile time: {first_step_time:.2f}s")
                        else:
                            print(f"First step time: {first_step_time:.2f}s")
                        start_time = now
                        last_step_time = now
                        if profiler is not None and not profiler_started:
                            profiler.__enter__()
                            profiler_started = True
                    else:
                        now = time.time()
                        last_step_time = report_step(now, epoch_idx, step, total_steps, accum_loss, accum_lb_loss, train_config, scheduler, last_step_time, start_time, wandb_run)

                        if profiler_started:
                            profiler.step()

                    if step >= total_steps:
                        break
                    opt.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    accum_lb_loss = 0.0
            if step >= total_steps:
                break
    finally:
        if profiler_started:
            profiler.__exit__(None, None, None)
            sort_key = "cuda_time_total" if str(device).startswith("cuda") and torch.cuda.is_available() else "self_cpu_time_total"
            print(
                profiler.key_averages().table(
                    sort_by=sort_key,
                    row_limit=30,
                )
            )

    if save:
        save_model(model, train_config.run_name) 
    if wandb_run is not None:
        wandb_run.finish()

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_cooldown_steps: int,
    num_training_steps: int,
    min_lrs: list[float],
    last_epoch: int = -1,
):
    # Linear warmup, constant phase, then linear cooldown to min_lrs.
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
    cooldown_steps = max(0, num_cooldown_steps)
    total_steps = max(1, num_training_steps)
    constant_steps = max(0, total_steps - warmup_steps - cooldown_steps)
    warmup_start = 1e-8

    # LambdaLR lets us apply separate schedules for embed and block LRs.
    def build_lambda(min_factor: float):
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return warmup_start + (1.0 - warmup_start) * (step / warmup_steps)
            if step < warmup_steps + constant_steps:
                return 1.0
            decay_step = step - warmup_steps - constant_steps
            if cooldown_steps <= 0:
                return min_factor
            if decay_step >= cooldown_steps:
                return min_factor
            decay_progress = decay_step / cooldown_steps
            return (1.0 - decay_progress) * (1.0 - min_factor) + min_factor
        return lr_lambda

    lr_lambdas = [build_lambda(min_factor) for min_factor in min_factors]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambdas,
        last_epoch=last_epoch,
    )

def save_model(model, run_name: str | None):
    # Saves weights and model config directly. 
    # Can be converted to a hf model later with a wrapper using convert_hf.py (Dirty implementation for now)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model # For compile
    if run_name:
        filename = f"{run_name}.pth"
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"model_{timestamp}.pth"
    path = os.path.join(get_base_dir(), "models", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": raw_model.state_dict(),
            "config": asdict(raw_model.config),
        },
        path,
    )

# Print metrics and optionally report to wandb
def report_step(
    now: float,
    epoch_idx: int,
    step: int,
    total_steps: int,
    accum_loss: float,
    accum_lb_loss: float,
    train_config: TrainingConfig,
    scheduler,
    last_step_time: float,
    start_time: float,
    wandb_run,
) -> float:
    avg_loss = accum_loss / train_config.grad_acc
    avg_lb_loss = accum_lb_loss / train_config.grad_acc
    step_time = now - last_step_time
    avg_step_time = (now - start_time) / (step - 1)
    remaining = avg_step_time * (total_steps - step)
    lr_embed, lr_block = scheduler.get_last_lr()
    tok_per_s = (train_config.microbatch_tok * train_config.grad_acc) / step_time
    tokens_processed = step * train_config.microbatch_tok * train_config.grad_acc
    print(
        f"Epoch {epoch_idx + 1}/{train_config.epoch} "
        f"Step {step}/{total_steps} training loss: {avg_loss:.4f} "
        f"lb_loss: {avg_lb_loss:.4f} "
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
                "lb_loss": avg_lb_loss,
                "lr_embed": lr_embed,
                "lr_block": lr_block,
                "step_time_s": step_time,
                "tok_per_s": tok_per_s,
            },
            step=tokens_processed, # By default, x axis is token count rather than step count on wandb graphs
        )
    return now
