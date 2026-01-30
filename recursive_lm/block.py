import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import GLUMLP # fast MoE (https://arxiv.org/pdf/2403.08245)

# Moved from FA2 to torch.attention.varlen (introduced in torch 2.10) to simplify dependencies. Should be the same backend.
# TODO: Change dependencies when torch 2.10 stable is released (currently using nightly builds).
import torch.nn.attention.varlen as attention 

# TODO: Add backup attention implementation for non-cuda GPUs, could be useful for local inference and analysis.

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
    def __init__(self, config, cos_cache, sin_cache):
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
    def __init__(self, config, cos_cache, sin_cache):
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
