"""
Hugging Face wrapper and checkpoint conversion for RecursiveGPT.
Requires uv group 'hf'.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel # type: ignore
from transformers.modeling_outputs import CausalLMOutputWithPast # type: ignore

from recursive_lm.model import ModelConfig, RecursiveGPT


class RecursiveLMConfig(PretrainedConfig):
    model_type = "recursive_lm"

    def __init__(
        self,
        sequence_len: int = ModelConfig.sequence_len,
        vocab_size: int = ModelConfig.vocab_size,
        n_head: int = ModelConfig.n_head,
        n_hidden: int = ModelConfig.n_hidden,
        n_wembed: int = ModelConfig.n_wembed,
        mlp_mul: int = ModelConfig.mlp_mul,
        rec_depth: int = ModelConfig.rec_depth,
        tie_embed: bool = ModelConfig.tie_embed,
        rope_cache_len: int = ModelConfig.rope_cache_len,
        standard_gpt: bool = ModelConfig.standard_gpt,
        **kwargs,
    ):
        auto_map = kwargs.pop(
            "auto_map",
            {
                "AutoConfig": "hf_wrapper.RecursiveLMConfig",
                "AutoModelForCausalLM": "hf_wrapper.RecursiveLMForCausalLM",
                "AutoTokenizer": "hf_tokenizer.RecursiveLMTokenizer",
                "AutoProcessor": "hf_tokenizer.RecursiveLMTokenizer",
            },
        )
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.n_wembed = n_wembed
        self.mlp_mul = mlp_mul
        self.rec_depth = rec_depth
        self.tie_embed = tie_embed
        self.rope_cache_len = rope_cache_len
        self.standard_gpt = standard_gpt
        self.auto_map = auto_map
        self.tie_word_embeddings = tie_embed

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            sequence_len=self.sequence_len,
            vocab_size=self.vocab_size,
            n_head=self.n_head,
            n_hidden=self.n_hidden,
            n_wembed=self.n_wembed,
            mlp_mul=self.mlp_mul,
            rec_depth=self.rec_depth,
            tie_embed=self.tie_embed,
            rope_cache_len=self.rope_cache_len,
            standard_gpt=self.standard_gpt,
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "RecursiveLMConfig":
        return cls(**asdict(config))


class RecursiveLMForCausalLM(PreTrainedModel):
    config_class = RecursiveLMConfig
    base_model_prefix = "model"

    def __init__(self, config: RecursiveLMConfig):
        super().__init__(config)
        self.model = RecursiveGPT(config.to_model_config())

    def init_weights(self):
        return

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            flat_input = input_ids.reshape(-1)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).repeat(batch_size)
        else:
            lengths = attention_mask.sum(dim=1).to(dtype=torch.int32, device=device)
            flat_chunks = []
            pos_chunks = []
            for b in range(batch_size):
                seq_len_b = int(lengths[b].item())
                if seq_len_b == 0:
                    continue
                flat_chunks.append(input_ids[b, :seq_len_b])
                pos_chunks.append(torch.arange(seq_len_b, device=device, dtype=torch.long))
            if flat_chunks:
                flat_input = torch.cat(flat_chunks, dim=0)
                position_ids = torch.cat(pos_chunks, dim=0)
            else:
                flat_input = input_ids.new_empty((0,), dtype=input_ids.dtype)
                position_ids = torch.empty((0,), device=device, dtype=torch.long)

        cu_seqlens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)

        if int(cu_seqlens[-1].item()) == 0:
            logits = torch.zeros(
                (batch_size, seq_len, self.config.vocab_size),
                device=device,
                dtype=self.model.embedding.weight.dtype,
            )
        else:
            logits_flat = self.model(flat_input, cu_seqlens, position_ids)
            logits = torch.zeros(
                (batch_size, seq_len, logits_flat.shape[-1]),
                device=device,
                dtype=logits_flat.dtype,
            )
            offset = 0
            for b in range(batch_size):
                seq_len_b = int(lengths[b].item())
                if seq_len_b == 0:
                    continue
                logits[b, :seq_len_b] = logits_flat[offset : offset + seq_len_b]
                offset += seq_len_b

        loss = None
        if labels is not None:
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)


def _map_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("model.") for key in state_dict):
        return state_dict
    return {f"model.{key}": value for key, value in state_dict.items()}


def convert_checkpoint(pth_path: str, out_dir: str) -> str:
    checkpoint = torch.load(pth_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain 'state_dict' and 'config'.")
    config = RecursiveLMConfig(**checkpoint["config"])
    model = RecursiveLMForCausalLM(config)
    state_dict = _map_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(out_dir)
    return out_dir


try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer # type: ignore

    from recursive_lm.huggingface.hf_tokenizer import RecursiveLMTokenizer

    AutoConfig.register(RecursiveLMConfig.model_type, RecursiveLMConfig)
    AutoModelForCausalLM.register(RecursiveLMConfig, RecursiveLMForCausalLM)
    AutoTokenizer.register(RecursiveLMConfig, RecursiveLMTokenizer)
except Exception:
    pass
