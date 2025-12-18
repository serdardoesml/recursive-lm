import torch
import torch.nn.functional as F
from model import RecursiveGPT, ModelConfig
from dataloader import batch_iterator

def train(model_config: ModelConfig, tokens_per_batch, lr, parquet_path, device):
    model = RecursiveGPT(model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for input_ids, targets, cu_seqlens, max_seqlen, position_ids in batch_iterator(
        parquet_path, 
        tokens_per_batch=tokens_per_batch,
        max_sl=model_config.sequence_len,
        device=device
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, cu_seqlens, max_seqlen, position_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

