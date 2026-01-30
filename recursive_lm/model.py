"""Definition of the recursive model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import GLUMLP # fast MoE (https://arxiv.org/pdf/2403.08245)
from dataclasses import dataclass

# Moved from FA2 to torch.attention.varlen (introduced in torch 2.10) to simplify dependencies. Should be the same backend.
# TODO: Change dependencies when torch 2.10 stable is released (currently using nightly builds).
import torch.nn.attention.varlen as attention 

# TODO: Add backup attention implementation for non-cuda GPUs, could be useful for local inference and analysis.

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_head: int = 16 # number of attention heads
    n_hidden: int = 384
    n_wembed: int = 128
    moe: bool = False
    n_expert: int = 4
    top_k: int = 2
    n_mlp_intermediate: int = 1536
    rec_depth: int = 24
    tie_embed: bool = False # Tied embeddings greatly hurt performance on recursive mode
    rope_cache_len: int = 2048

    # Standard gpt experimental mode to compare with non-recursive models
    standard_gpt: bool = False

    @property
    def n_headdim(self) -> int:
        return self.n_hidden // self.n_head


def apply_rotary_emb(x, cos, sin):
    """
    x:   [N, H, D]
    cos: [N, 1, D//2]
    sin: [N, 1, D//2]
    """
    assert x.ndim == 3
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    # Removed torch.concat
    out = torch.empty_like(x)
    out[..., :d] = y1
    out[..., d:] = y2
    return out

class CausalVarlenSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        super().__init__()

        # One fused projection for QKV
        self.Wqkv = nn.Linear(config.n_hidden, 3 * config.n_hidden, bias=False)
        self.Wo   = nn.Linear(config.n_hidden, config.n_hidden, bias=False)
        nn.init.zeros_(self.Wo.weight) # Zero init (Idea from modded-nanogpt speedrun, empirically seems to work well)
        self.n_hidden = config.n_hidden
        self.n_head = config.n_head
        self.head_dim = config.n_headdim

        # Gated Attention (https://arxiv.org/pdf/2505.06708)
        # SDPAHeadwiseGate: per-head sigmoid gate applied to SDPA output.
        # Empirically, attention gating should benefit us since we don't use any <|BOS|> token during training,
        # meaning the model has no attention sinks. GA should reduce the need for attention sinks.
        self.gate = nn.Linear(self.n_hidden, self.n_head, bias=True)
        nn.init.zeros_(self.gate.weight)
        
        # Since the gated attention paper finds that the model converges toward a more sparse gate,
        # we initialize the gate bias with 0.0 (so the sigmoid of the bias is 0.5).
        # 0.5 is right in the middle, not too high to start with default behavior, not too low to enforce sparsity early on.
        # TODO: Rewrite this comment more clearly
        # TODO: Re-consider if bias is even necessary
        nn.init.constant_(self.gate.bias, 0.0) 

        # We register it as a buffer to ensure it gets moved to device together with the model
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
    
    def forward(self, x, cu_seqlens, max_seqlen, position_ids, norm_qk):
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
        qkv[:, 0] = norm_qk(apply_rotary_emb(qkv[:, 0], cos, sin))
        qkv[:, 1] = norm_qk(apply_rotary_emb(qkv[:, 1], cos, sin))

        # We split qkv as torch varlen-attn does not support packed qkv.
        attn_out = attention.varlen_attn(
            query=qkv[:, 0],
            key=qkv[:, 1],
            value=qkv[:, 2],
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            is_causal=True,
        )  # [total_tokens, n_heads, head_dim]

        gate = torch.sigmoid(self.gate(x)).view(-1, self.n_head, 1) # [N, H, 1]
        out = attn_out * gate # [N, H, D]

        out = out.reshape(-1, self.n_hidden)
        return self.Wo(out) # [total_tokens, n_hidden]
    

class MoE(nn.Module):
    # Standard MoE layer (for now), no shared experts.
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.n_expert = config.n_expert

        # Since we use SimBal (https://arxiv.org/pdf/2506.14038v2) for loss balancing,
        # Also initialize as orthogonal
        self.router = nn.Linear(config.n_hidden, self.n_expert, bias=False)
        nn.init.orthogonal_(self.router.weight)

        # Modified to zero init output (Idea from modded-nanogpt speedrun, empirically seems to work well)
        # SwiGLU by default
        self.experts = GLUMLP( 
            input_size=config.n_hidden,
            hidden_size=config.n_mlp_intermediate,
            num_experts=self.n_expert,
            top_k=self.top_k,
        )

    def forward(self, x):
        # x: [total_tokens, n_hidden]
        router_logits = self.router(x) # [N, n_expert]
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, sorted=False) # [N, k]
        topk_idx = topk_idx.to(torch.int32)

        # Sigmoid then normalize, faster than softmax
        topk_gates = torch.sigmoid(topk_vals)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-10)

        return self.experts(x, topk_gates, topk_idx)
    
class DenseMLP(nn.Module):
    # Standard dense SwiGLU MLP with zero init output (Idea from modded-nanogpt speedrun, empirically seems to work well)
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_hidden, 2 * config.n_mlp_intermediate, bias=False) # 2x for gating
        self.c_proj = nn.Linear(config.n_mlp_intermediate, config.n_hidden, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        h, gates = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(F.silu(gates) * h)
    
class Block(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        super().__init__()
        self.use_moe = config.moe
        self.attn = CausalVarlenSelfAttention(config, cos_cache, sin_cache)
        if self.use_moe:
            self.moe = MoE(config)
        else:
            self.mlp = DenseMLP(config)

    def forward(self, x, cu_seqlens, max_seqlen, position_ids, norm_attn, norm_mlp, norm_qk):
        # We do pre-norm and QK norm. 
        # We used to do a Gemma 3 style post-norm, but removed it to improve stability 
        # and keep the residual stream norm in check. Seems to work fine.
        x = x + self.attn(norm_attn(x), cu_seqlens, max_seqlen, position_ids, norm_qk)
        if self.use_moe:
            x = x + self.moe(norm_mlp(x))
        else:
            x = x + self.mlp(norm_mlp(x))
        return x


class StandardBlocks(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        super().__init__()
        self.depth = config.rec_depth

        # Independent norms for each depth
        self.attn_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.mlp_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.qk_norms = nn.ModuleList([nn.RMSNorm(config.n_headdim, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])

        self.blocks = nn.ModuleList([Block(config, cos_cache, sin_cache) for _ in range(self.depth)])

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        for i in range(self.depth):
            x = self.blocks[i](x, cu_seqlens, max_seqlen, position_ids, self.attn_norms[i], self.mlp_norms[i], self.qk_norms[i])
        return x

class RecursiveBlocks(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        super().__init__()
        self.depth = config.rec_depth

        # Independent norms for each depth
        self.attn_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.mlp_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.qk_norms = nn.ModuleList([nn.RMSNorm(config.n_headdim, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])

        self.recursive_block = Block(config, cos_cache, sin_cache)

        # Per layer embeddings
        # TODO: Explain the idea in more detail and find a new name. 
        # We initialize at zero to let model start without any depth specific information and learn it gradually.
        # Bf16, shouldnt hurt stability, prevents casting issues
        self.rec_layer_embedding = nn.Embedding(self.depth, config.n_hidden, dtype=torch.bfloat16) 
        nn.init.zeros_(self.rec_layer_embedding.weight) 

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        for i in range(self.depth):
            x = x + self.rec_layer_embedding.weight[i]
            x = self.recursive_block(x, cu_seqlens, max_seqlen, position_ids, self.attn_norms[i], self.mlp_norms[i], self.qk_norms[i])
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
        cos, sin = freqs.cos(), freqs.sin()
        return cos.bfloat16(), sin.bfloat16() # Keep in bf16

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_factorized = config.n_wembed != config.n_hidden # Made FE optional

        assert config.n_hidden % config.n_head == 0

        # We build cache then register it as a buffer later to ensure it gets moved to device together with the model
        cos_cache, sin_cache = RecursiveGPT.build_rope_cache(config.rope_cache_len, config.n_headdim)

        # Factorized Embeddings (https://arxiv.org/pdf/1909.11942)
        if self.use_factorized:
            self.embedding = nn.Embedding(config.vocab_size, config.n_wembed)
            if not self.config.tie_embed:
                self.lm_head = nn.Linear(config.n_wembed, config.vocab_size, bias=False)
            self.e_to_h = nn.Linear(config.n_wembed, config.n_hidden, bias=False)
            self.h_to_e = nn.Linear(config.n_hidden, config.n_wembed, bias=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.n_hidden)
            if not self.config.tie_embed:
                self.lm_head = nn.Linear(config.n_hidden, config.vocab_size, bias=False)
        self.norm_out = nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16)
        
        if config.standard_gpt:
            self.blocks = StandardBlocks(config, cos_cache, sin_cache)
        else:
            self.blocks = RecursiveBlocks(config, cos_cache, sin_cache)

    def forward_hidden(self, input_ids, cu_seqlens, position_ids):
        # TODO: Think about whether passing rope_cache_len as max_seqlen to flash-attn makes sense for inference

        # input_ids: [total_tokens] (flattened)
        x = self.e_to_h(self.embedding(input_ids)) if self.use_factorized else self.embedding(input_ids)  # [total_tokens, n_hidden]
        return self.blocks(x, cu_seqlens, self.config.rope_cache_len, position_ids)

    def forward(self, input_ids, cu_seqlens, position_ids):
        x = self.forward_hidden(input_ids, cu_seqlens, position_ids)
        x = self.norm_out(x)
        if self.use_factorized:
            x = self.h_to_e(x)
        if not self.config.tie_embed:
            return self.lm_head(x) # [total_tokens, vocab_size]
        return F.linear(x, self.embedding.weight) # [total_tokens, vocab_size]
        
    @property
    def total_param_size(self) -> int:
        return sum(p.numel() for p in self.parameters())
