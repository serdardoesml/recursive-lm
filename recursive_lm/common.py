from pathlib import Path

import os

def get_base_dir() -> Path:
    # Return the project root directory (parent of the recursive_lm package).
    return Path(__file__).resolve().parent.parent

def print_banner() -> None:
    banner_path = get_base_dir() / "banner.txt"
    print(banner_path.read_text(encoding="utf-8").rstrip(), flush=True)

def compute_init(device_type: str = "cuda"):
    # Read torchrun env vars; defaults keep normal single-process training unchanged.
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world_size > 1

    if ddp:
        # One process owns one local GPU. NCCL handles gradient all-reduce for DDP.
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = f"cuda:{local_rank}"
    else:
        device = device_type

    return ddp, rank, local_rank, world_size, device

def compute_cleanup() -> None:
    # No-op for normal single-process training, cleanup for DDP.
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def print0(*args, rank: int = 0, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs)
