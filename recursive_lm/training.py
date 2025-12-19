import torch
import torch.nn.functional as F
from model import RecursiveGPT, ModelConfig
from dataloader import batch_iterator
from common import get_base_dir

from dataclasses import dataclass
from datetime import datetime
import os
import time

@dataclass
class TrainingConfig:
    model_config: ModelConfig
    lr: float = 7e-3

    # Default target batch size: 65536 tok
    microbatch_tok: int = 32768
    grad_acc: int = 2

    max_tok_count: int = 13107200 # 13.1M default max tok count, 1600 microbatches, 200 updates per epoch
    epoch: int = 10 # 10 epochs by default

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
            step += 1
            avg_loss = accum_loss / train_config.grad_acc
            now = time.time()
            step_time = now - last_step_time
            avg_step_time = (now - start_time) / step
            remaining = avg_step_time * (total_steps - step)
            print(
                f"Step {step}/{total_steps} training loss: {avg_loss:.4f} "
                f"step_time {step_time:.2f}s eta {remaining:.0f}s"
            )
            last_step_time = now
            if step >= total_steps:
                break
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0

    if save:
        save_model(model)

def save_model(model):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    path = os.path.join(get_base_dir(), "models", f"model_{timestamp}.pth")
    torch.save(model.state_dict(), path)
