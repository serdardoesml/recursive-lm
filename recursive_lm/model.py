"""Definition of the recursive model."""

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
    mlp_mul: int = 8
    rec_depth: int = 4

    @property
    def n_headdim(self) -> int:
        return self.n_embd // self.n_head

def norm(x):
    # x: [..., n_embd], purely functional rmsnorm with no learnable params
    # TODO: Try Derf (https://arxiv.org/pdf/2512.10938)
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    """
    x:   [N, H, D]   (e.g., q or k from packed qkv tensor)
    cos: [N, 1, D//2]
    sin: [N, 1, D//2]

    Returns: [N, H, D]
    """
    assert x.ndim == 3
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:] # split last dim into halves
    y1 = x1 * cos + x2 * sin # rotate pairs
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], dim=-1)
    return out.to(dtype=x.dtype)

class CausalVarlenSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        # One fused projection for QKV
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.Wo   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_headdim
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
    
    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        """
        x: [total_tokens, n_embd] (flattened packed tokens)
        cu_seqlens: [n_seqs+1] int32
        max_seqlen: int
        position_ids:[N]

        Returns: [total_tokens, n_embd]
        """
        # Project to QKV and reshape to [total, 3, n_heads, head_dim]
        qkv = self.Wqkv(x)
        qkv = qkv.view(-1, 3, self.n_head, self.head_dim).contiguous()

        # pick cos/sin for each token position, then broadcast over heads
        cos = self.cos_cache[position_ids].unsqueeze(1)  # [N, 1, D//2]
        sin = self.sin_cache[position_ids].unsqueeze(1)  # [N, 1, D//2]

        # qkv[:, 0] is Q: [N, H, D], qkv[:, 1] is K: [N, H, D]
        # We also apply QK norm
        qkv[:, 0] = norm(apply_rotary_emb(qkv[:, 0], cos, sin))
        qkv[:, 1] = norm(apply_rotary_emb(qkv[:, 1], cos, sin))

        out = flash_attn_varlen_qkvpacked_func(
            qkv=qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0, # 0 dropout, not something i expect to use, can be added as hyperparam later
            causal=True,
        )  # out: [total_tokens, n_heads, head_dim] 

        out = out.reshape(-1, self.n_embd)
        return self.Wo(out) # [total_tokens, n_embd]
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.mlp_mul, bias=False)
        self.c_proj = nn.Linear(config.n_embd * config.mlp_mul, config.n_embd, bias=False)

    def forward(self, x):
        # x: [total_tokens, n_embd]
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.attn = CausalVarlenSelfAttention(config, cos_cache, sin_cache)
        self.mlp = MLP(config)

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        # We do both Pre and Post RMSNorm (without affecting the residual stream)
        # In addition, we also do QK Norm inside the attention layer.
        # Same normalizations as Gemma 3 (https://arxiv.org/pdf/2503.19786)
        x = x + norm(self.attn(norm(x), cu_seqlens, max_seqlen, position_ids))
        x = x + norm(self.mlp(norm(x)))
        return x

class RecursiveGPT(nn.Module):
    @staticmethod
    def build_rope_cache(max_seqlen, head_dim, device=None):
        """
        Build RoPE cache.

        Returns:
        cos: [max_seqlen, head_dim//2]
        sin: [max_seqlen, head_dim//2]
        """
        assert head_dim % 2 == 0
        half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        t = torch.arange(max_seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seqlen, half]
        return freqs.cos(), freqs.sin()

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Assert config is correct
        assert config.n_embd % config.n_head == 0

        self.cos_cache, self.sin_cache = RecursiveGPT.build_rope_cache(config.sequence_len, config.n_headdim)

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.recursive_block = Block(config, self.cos_cache, self.sin_cache)

    def forward(self, input_ids, cu_seqlens, max_seqlen, position_ids):
        # input_ids: [total_tokens] (flattened)
        x = self.embedding(input_ids)  # [total_tokens, n_embd]
        for _ in range(self.config.rec_depth):
            x = self.recursive_block(x, cu_seqlens, max_seqlen, position_ids)
        return self.lm_head(norm(x))   # [total_tokens, vocab_size]


