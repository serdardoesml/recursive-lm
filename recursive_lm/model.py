"""Definition of the recursive model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
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
        self.norm_qk = nn.RMSNorm(self.head_dim, eps=1e-6, dtype=torch.bfloat16)

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
        self.router = nn.Linear(config.n_hidden, self.n_expert, bias=True)

        # Modified to zero init output (Idea from modded-nanogpt speedrun, empirically seems to work well)
        # SwiGLU by default
        self.experts = GLUMLP( 
            input_size=config.n_hidden,
            hidden_size=config.n_mlp_intermediate,
            num_experts=self.n_expert,
            top_k=self.top_k,
        )

        # Routing balance metrics
        self.register_buffer("balance_entropy", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("balance_eff", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("balance_count", torch.zeros((), dtype=torch.float32), persistent=False)

        # Init router bias as 0
        nn.init.zeros_(self.router.bias)

    def forward(self, x, training):
        # x: [total_tokens, n_hidden]
        router_logits = self.router(x) # [N, n_expert]
        topk_vals, topk_idx = torch.topk(router_logits, k=self.top_k, sorted=False) # [N, k]
        topk_idx = topk_idx.to(torch.int32)

        # Sigmoid then normalize, faster than softmax
        topk_gates = torch.sigmoid(topk_vals)
        topk_gates = topk_gates / (topk_gates.sum(dim=-1, keepdim=True) + 1e-10)

        if training and not torch.compiler.is_compiling():
            # Measure balance with load entropy
            load = torch.bincount(topk_idx.reshape(-1), minlength=self.n_expert)
            load = load.to(dtype=torch.float32)
            load = load / load.sum()
            entropy = -(load * (load + 1e-9).log()).sum()
            self.balance_entropy += entropy
            self.balance_eff += entropy.exp()
            self.balance_count += 1.0

        return self.experts(x, topk_gates, topk_idx)
    
class Block(nn.Module):
    def __init__(self, config, cos_cache, sin_cache):
        super().__init__()
        self.attn = CausalVarlenSelfAttention(config, cos_cache, sin_cache)
        self.moe = MoE(config)
        self.norm_attn = nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16)
        self.norm_mlp = nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16)

    def forward(self, x, cu_seqlens, max_seqlen, position_ids, training):
        # We do pre-norm and QK norm. 
        # We used to do a Gemma 3 style post-norm, but removed it to improve stability 
        # and keep the residual stream norm in check. Seems to work fine.
        x = x + self.attn(self.norm_attn(x), cu_seqlens, max_seqlen, position_ids)
        x = x + self.moe(self.norm_mlp(x), training)
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

    def __init__(self, config: ModelConfig, grad_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.grad_checkpointing = grad_checkpointing

        # Assert config is correct
        assert config.n_hidden % config.n_head == 0

        # We build cache then register it as a buffer later to ensure it gets moved to device together with the model
        cos_cache, sin_cache = RecursiveGPT.build_rope_cache(config.rope_cache_len, config.n_headdim)

        # Factorized Embeddings (https://arxiv.org/pdf/1909.11942)
        self.embedding = nn.Embedding(config.vocab_size, config.n_wembed)
        if not self.config.tie_embed:
            self.lm_head = nn.Linear(config.n_wembed, config.vocab_size, bias=False)

        self.e_to_h = nn.Linear(config.n_wembed, config.n_hidden, bias=False)
        self.h_to_e = nn.Linear(config.n_hidden, config.n_wembed, bias=False)
        self.norm_out = nn.RMSNorm(config.n_wembed, eps=1e-6, dtype=torch.bfloat16)
        
        if config.standard_gpt:
            self.blocks = nn.ModuleList([Block(config, cos_cache, sin_cache) for _ in range(config.rec_depth)])
        else:
            self.recursive_block = Block(config, cos_cache, sin_cache)

            # Per layer embeddings
            # TODO: Explain the idea in more detail. 
            # (Removed reference to RingFormers as this is fundamentally different and not dependent on input)
            # We initialize at zero to let model start without any depth specific information and learn it gradually.
            # Bf16, shouldnt hurt stability, prevents casting issues
            self.rec_layer_embedding = nn.Embedding(config.rec_depth, config.n_hidden, dtype=torch.bfloat16) 
            nn.init.zeros_(self.rec_layer_embedding.weight) 

    def forward_hidden(self, input_ids, cu_seqlens, position_ids, training=False):
        # TODO: Think about whether passing rope_cache_len as max_seqlen to flash-attn makes sense for inference

        # input_ids: [total_tokens] (flattened)
        x = self.e_to_h(self.embedding(input_ids))  # [total_tokens, n_hidden]
        if self.config.standard_gpt:
            for i in range(self.config.rec_depth):
                x = self.blocks[i](x, cu_seqlens, self.config.rope_cache_len, position_ids, training)
        else:
            for i in range(self.config.rec_depth):
                if self.grad_checkpointing:
                    def recursive_step(x, cu_seqlens, position_ids, training, i=i):
                        x = x + self.rec_layer_embedding.weight[i]
                        return self.recursive_block(x, cu_seqlens, self.config.rope_cache_len, position_ids, training)

                    x = checkpoint.checkpoint(recursive_step, x, cu_seqlens, position_ids, training, use_reentrant=False)
                else:
                    x = x + self.rec_layer_embedding.weight[i]
                    x = self.recursive_block(x, cu_seqlens, self.config.rope_cache_len, position_ids, training)
        return x

    def forward(self, input_ids, cu_seqlens, position_ids, training=False):
        x = self.forward_hidden(input_ids, cu_seqlens, position_ids, training)
        if not self.config.tie_embed:
            return self.lm_head(self.norm_out(self.h_to_e(x))) # [total_tokens, vocab_size]
        else:
            return F.linear(self.norm_out(self.h_to_e(x)), self.embedding.weight) # [total_tokens, vocab_size]
        
    @property
    def total_param_size(self) -> int:
        return sum(p.numel() for p in self.parameters())
