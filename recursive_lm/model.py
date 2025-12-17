"""Definition of the recursive model. A lot of building blocks copied from nanochat."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_qkvpacked_func # type: ignore

from dataclasses import dataclass

@dataclass
class ModelConfig:
    sequence_len: int = 256
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 16 # number of attention heads
    n_embd: int = 1024

class CausalVarlenSelfAttention(nn.Module):
    def __init(self, config: ModelConfig):
        # One fused projection for QKV
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.Wo   = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x, cu_seqlens, max_seqlen):
        """
        x: [total_tokens, n_embd] (flattened packed tokens)
        cu_seqlens: [n_seqs+1] int32
        max_seqlen: int
        returns: [total_tokens, d_model]
        """
        # Project to QKV and reshape to [total, 3, n_heads, head_dim]
        qkv = self.Wqkv(x)
        qkv = qkv.view(-1, 3, self.n_heads, self.head_dim).contiguous()


        out = flash_attn_varlen_qkvpacked_func(
            qkv=qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0, # 0 dropout, not something i expect to use, can be added as hyperparam later
            causal=True,
        )  # out: [total_tokens, n_heads, head_dim] 

        out = out.reshape(-1, self.d_model)
        return self.Wo(out)


class RecursiveGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Assert config is correct
        assert self.n_embd % self.n_head == 0

