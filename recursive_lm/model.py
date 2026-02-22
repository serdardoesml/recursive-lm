"""Definition of the recursive model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block # Moved smaller pieces to seperate file to improve readability
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_head: int = 16 # Number of attention heads (16 seems ideal, 8 can be better for standard model)
    n_hidden: int = 512
    n_wembed: int = 128 # Allows for factorized embeddings, only makes sense at babylm scale.
    moe: bool = False # Free lunch when flop constrained and not data constrained
    n_expert: int = 1
    top_k: int = 1
    n_mlp_intermediate: int = 8192
    std_depth: int = 0
    rec_depth: int = 24
    tie_embed: bool = False # Tied embeddings greatly hurt performance on recursive mode
    rope_cache_len: int = 2048

    @property
    def n_headdim(self) -> int:
        return self.n_hidden // self.n_head
    

class StandardBlocks(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        super().__init__()
        self.depth = config.std_depth
        self.moe = config.moe

        # Independent norms for each depth
        self.attn_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.mlp_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.qk_norms = nn.ModuleList([nn.RMSNorm(config.n_headdim, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])

        if self.moe:
            # Independent routers for each depth
            self.routers = nn.ModuleList([nn.Linear(config.n_hidden, config.n_expert, bias=False) for _ in range(self.depth)])
            
            # Orthogonal init for routers (since we use SimBal loss)
            for rt in self.routers:
                nn.init.orthogonal_(rt.weight)

        self.blocks = nn.ModuleList([Block(config, cos_cache, sin_cache) for _ in range(self.depth)])

    def forward(self, x, cu_seqlens, max_seqlen, position_ids):
        for i in range(self.depth):
            if self.moe:
                x = self.blocks[i](x, cu_seqlens, max_seqlen, position_ids, self.attn_norms[i], self.mlp_norms[i], self.qk_norms[i], self.routers[i])
            else:
                x = self.blocks[i](x, cu_seqlens, max_seqlen, position_ids, self.attn_norms[i], self.mlp_norms[i], self.qk_norms[i])
        return x
    
    def get_param_groups(self):
        muon_params = []
        if self.moe:
            muon_params += list(self.routers.parameters())
        for block in self.blocks:
            muon_params += list(block.attn.Wqkv.parameters())
            muon_params += list(block.attn.Wo.parameters())
            if block.use_moe:
                muon_params += list(block.moe.parameters())
            else:
                muon_params += list(block.mlp.parameters())
            muon_params.append(block.attn.gate.weight)
        muon_param_ids = {id(p) for p in muon_params}
        adam_params = [p for p in self.parameters() if id(p) not in muon_param_ids]
        return adam_params, muon_params

class RecursiveBlocks(nn.Module):
    def __init__(self, config: ModelConfig, cos_cache, sin_cache):
        super().__init__()
        self.depth = config.rec_depth
        self.moe = config.moe

        # Independent norms for each depth
        self.attn_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.mlp_norms = nn.ModuleList([nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])
        self.qk_norms = nn.ModuleList([nn.RMSNorm(config.n_headdim, eps=1e-6, dtype=torch.bfloat16) for _ in range(self.depth)])

        if self.moe:
            # Independent routers for each depth
            self.routers = nn.ModuleList([nn.Linear(config.n_hidden, config.n_expert, bias=False) for _ in range(self.depth)])

            # Orthogonal init for routers (since we use SimBal loss)
            for rt in self.routers:
                nn.init.orthogonal_(rt.weight)

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
            if self.moe:
                x = self.recursive_block(x, cu_seqlens, max_seqlen, position_ids, self.attn_norms[i], self.mlp_norms[i], self.qk_norms[i], self.routers[i])
            else:
                x = self.recursive_block(x, cu_seqlens, max_seqlen, position_ids, self.attn_norms[i], self.mlp_norms[i], self.qk_norms[i])
        return x
    
    def get_param_groups(self):
        muon_params = []
        if self.moe:
            muon_params += list(self.routers.parameters())
        muon_params += list(self.recursive_block.attn.Wqkv.parameters())
        muon_params += list(self.recursive_block.attn.Wo.parameters())
        if self.recursive_block.use_moe:
            muon_params += list(self.recursive_block.moe.parameters())
        else:
            muon_params += list(self.recursive_block.mlp.parameters())
        muon_params.append(self.recursive_block.attn.gate.weight)
        muon_param_ids = {id(p) for p in muon_params}
        adam_params = [p for p in self.parameters() if id(p) not in muon_param_ids]
        return adam_params, muon_params
    
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
        self.adam_params = []
        self.muon_params = []
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
            
            # Param group
            self.adam_params += list(self.e_to_h.parameters())
            self.adam_params += list(self.h_to_e.parameters())
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.n_hidden)
            if not self.config.tie_embed:
                self.lm_head = nn.Linear(config.n_hidden, config.vocab_size, bias=False)

        self.norm_out = nn.RMSNorm(config.n_hidden, eps=1e-6, dtype=torch.bfloat16)

        # Param group
        self.adam_params += list(self.norm_out.parameters())
        self.adam_params += list(self.embedding.parameters())
        
        if config.std_depth != 0:
            self.std_blocks = StandardBlocks(config, cos_cache, sin_cache)

            # Param group
            block_adam_params, block_muon_params = self.std_blocks.get_param_groups()
            self.adam_params += block_adam_params
            self.muon_params += block_muon_params
        if config.rec_depth != 0:
            self.rec_blocks = RecursiveBlocks(config, cos_cache, sin_cache)

            # Param group
            block_adam_params, block_muon_params = self.rec_blocks.get_param_groups()
            self.adam_params += block_adam_params
            self.muon_params += block_muon_params
      
        if hasattr(self, "lm_head"): # Param group
            self.adam_params += list(self.lm_head.parameters())

    def forward_hidden(self, input_ids, cu_seqlens, position_ids):
        # TODO: Think about whether passing rope_cache_len as max_seqlen to flash-attn makes sense for inference

        # input_ids: [total_tokens] (flattened)
        x = self.e_to_h(self.embedding(input_ids)) if self.use_factorized else self.embedding(input_ids)  # [total_tokens, n_hidden]
        if self.config.std_depth != 0:
            x = self.std_blocks(x, cu_seqlens, self.config.rope_cache_len, position_ids)
        if self.config.rec_depth != 0:
            x = self.rec_blocks(x, cu_seqlens, self.config.rope_cache_len, position_ids)
        return x

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

    @property
    def block_param_size(self) -> int:
        return sum(p.numel() for p in self.muon_params)

    @property
    def embed_param_size(self) -> int:
        return sum(p.numel() for p in self.adam_params)
