"""Definition of the recursive model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from flash_attn import flash_attn_varlen_qkvpacked_func # type: ignore

from dataclasses import dataclass

@dataclass
class ModelConfig:
    sequence_len: int = 256
    vocab_size: int = 32768
    n_head: int = 16 # number of attention heads
    n_hidden: int = 1024
    n_wembed: int = 128
    mlp_mul: int = 8
    rec_depth: int = 4
    tie_embed: bool = True

    # Standard gpt experimental mode to compare with non-recursive models
    standard_gpt: bool = False

    @property
    def n_headdim(self) -> int:
        return self.n_hidden // self.n_head

    @property
    def total_param_size(self) -> int:
        if self.standard_gpt:
            return self.total_unrolled_param_size
        embed = (self.vocab_size * self.n_wembed) + (2 * self.n_hidden * self.n_wembed)
        attn = (self.n_hidden * 4 * self.n_hidden)
        mlp = (self.n_hidden * self.n_hidden * self.mlp_mul) * 2
        if self.tie_embed:
            return embed + attn + mlp
        else:
            return embed + (self.vocab_size * self.n_wembed) + attn + mlp
        
    @property
    def total_unrolled_param_size(self) -> int:
        embed = (self.vocab_size * self.n_wembed) + (2 * self.n_hidden * self.n_wembed)
        attn = (self.n_hidden * 4 * self.n_hidden)
        mlp = (self.n_hidden * self.n_hidden * self.mlp_mul) * 2
        if self.tie_embed:
            return embed + ((attn + mlp) * self.rec_depth)
        else:
            return embed + (self.vocab_size * self.n_wembed) + ((attn + mlp) * self.rec_depth)
        
    # Layer embeddings not calculated as they are negligible (around 10k-20k)
    # TODO: Add layer embeddings too

class RMSNorm(nn.Module):
    # Replaced initial unparameterized norm with this to enable more flexibility for the model.

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [..., dim]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

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
        super().__init__()
        # One fused projection for QKV
        self.Wqkv = nn.Linear(config.n_hidden, 3 * config.n_hidden, bias=False)
        self.Wo   = nn.Linear(config.n_hidden, config.n_hidden, bias=False)
        with torch.no_grad():
            self.Wo.weight.mul_((2.0 * config.rec_depth) ** -0.5)
        self.n_hidden = config.n_hidden
        self.n_head = config.n_head
        self.head_dim = config.n_headdim
        self.norm_qk = RMSNorm(self.head_dim)
        # We register it as a buffer to ensure it gets moved to device together with the model
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
    
    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        """
        x: [total_tokens, n_hidden] (flattened packed tokens)
        cu_seqlens: [n_seqs+1] int32
        max_seqlen: int
        position_ids:[N]

        Returns: [total_tokens, n_hidden]
        """
        # Project to QKV and reshape to [total, 3, n_heads, head_dim]
        qkv = self.Wqkv(x)
        qkv = qkv.view(-1, 3, self.n_head, self.head_dim).contiguous()

        # pick cos/sin for each token position, then broadcast over heads
        cos = self.cos_cache[position_ids].unsqueeze(1)  # [N, 1, D//2]
        sin = self.sin_cache[position_ids].unsqueeze(1)  # [N, 1, D//2]

        # qkv[:, 0] is Q: [N, H, D], qkv[:, 1] is K: [N, H, D]
        # We also apply QK norm
        qkv[:, 0] = self.norm_qk(apply_rotary_emb(qkv[:, 0], cos, sin))
        qkv[:, 1] = self.norm_qk(apply_rotary_emb(qkv[:, 1], cos, sin))

        out = flash_attn_varlen_qkvpacked_func(
            qkv=qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0, # 0 dropout, not something i expect to use, can be added as hyperparam later
            causal=True,
        )  # out: [total_tokens, n_heads, head_dim] 

        out = out.reshape(-1, self.n_hidden)
        return self.Wo(out) # [total_tokens, n_hidden]
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_hidden, config.n_hidden * config.mlp_mul, bias=False)
        self.c_gate = nn.Linear(config.n_hidden, config.n_hidden * config.mlp_mul, bias=False)
        self.c_proj = nn.Linear(config.n_hidden * config.mlp_mul, config.n_hidden, bias=False)
        with torch.no_grad():
            self.c_proj.weight.mul_((2.0 * config.rec_depth) ** -0.5)

    def forward(self, x):
        # x: [total_tokens, n_hidden]
        x_fc = self.c_fc(x)
        x_gate = self.c_gate(x)
        x = F.silu(x_gate) * x_fc # SwiGLU, improved training speed a lot from ReLU^2, so we keep it
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.attn = CausalVarlenSelfAttention(config, cos_cache, sin_cache)
        self.mlp = MLP(config)
        self.norm_attn = RMSNorm(config.n_hidden)
        self.norm_mlp = RMSNorm(config.n_hidden)

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        # We do both Pre and Post RMSNorm (without affecting the residual stream)
        # In addition, we also do QK Norm inside the attention layer.
        # Same normalizations as Gemma 3 (https://arxiv.org/pdf/2503.19786)

        # TEMPORARILY REMOVED POST NORM
        # TODO: Update this comment to reflect post norm removal results and why we kept it.
        x = x + self.attn(self.norm_attn(x), cu_seqlens, max_seqlen, position_ids)
        x = x + self.mlp(self.norm_mlp(x))
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

    def __init__(self, config: ModelConfig, grad_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.grad_checkpointing = grad_checkpointing

        # Assert config is correct
        assert config.n_hidden % config.n_head == 0

        # We build cache then register it as a buffer later to ensure it gets moved to device together with the model
        cos_cache, sin_cache = RecursiveGPT.build_rope_cache(config.sequence_len, config.n_headdim)

        # Factorized Embeddings (https://arxiv.org/pdf/1909.11942)
        self.embedding = nn.Embedding(config.vocab_size, config.n_wembed)
        if not self.config.tie_embed:
            self.lm_head = nn.Linear(config.n_wembed, config.vocab_size, bias=False)

        self.e_to_h = nn.Linear(config.n_wembed, config.n_hidden, bias=False)
        self.h_to_e = nn.Linear(config.n_hidden, config.n_wembed, bias=False)
        self.norm_out = RMSNorm(config.n_wembed)
        
        if config.standard_gpt:
            self.blocks = nn.ModuleList([Block(config, cos_cache, sin_cache) for _ in range(config.rec_depth)])
        else:
            self.recursive_block = Block(config, cos_cache, sin_cache)

            # Layer embeddings (https://arxiv.org/pdf/2502.13181)
            # Note: Our layer embeddings are simpler than RingFormers, but the idea came from there.
            self.rec_layer_embedding = nn.Embedding(config.rec_depth, config.n_hidden)
            nn.init.zeros_(self.rec_layer_embedding.weight)

    def forward(self, input_ids, cu_seqlens, position_ids):
        # input_ids: [total_tokens] (flattened)
        x = self.e_to_h(self.embedding(input_ids))  # [total_tokens, n_hidden]
        if self.config.standard_gpt:
            for i in range(self.config.rec_depth):
                x = self.blocks[i](x, cu_seqlens, self.config.sequence_len, position_ids)
        else:
            for i in range(self.config.rec_depth):
                if self.grad_checkpointing:
                    def recursive_step(x, cu_seqlens, position_ids, i=i):
                        x = x + self.rec_layer_embedding.weight[i]
                        return self.recursive_block(x, cu_seqlens, self.config.sequence_len, position_ids)

                    x = checkpoint.checkpoint(recursive_step, x, cu_seqlens, position_ids, use_reentrant=False)
                else:
                    x = x + self.rec_layer_embedding.weight[i]
                    x = self.recursive_block(x, cu_seqlens, self.config.sequence_len, position_ids)
        if not self.config.tie_embed:
            return self.lm_head(self.norm_out(self.h_to_e(x))) # [total_tokens, vocab_size]
        else:
            return F.linear(self.norm_out(self.h_to_e(x)), self.embedding.weight) # [total_tokens, vocab_size]
