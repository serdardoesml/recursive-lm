"""
Hugging Face wrapper and checkpoint conversion for RecursiveGPT.
Requires uv group 'hf'.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel # type: ignore
from transformers.modeling_outputs import ( # type: ignore
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)
from torch.nn.utils.rnn import pad_sequence

from recursive_lm.model import ModelConfig, RecursiveGPT


class RecursiveLMConfig(PretrainedConfig):
    model_type = "recursive_lm"

    def __init__(
        self,
        vocab_size: int = ModelConfig.vocab_size,
        n_head: int = ModelConfig.n_head,
        n_hidden: int = ModelConfig.n_hidden,
        n_wembed: int = ModelConfig.n_wembed,
        moe: bool = ModelConfig.moe,
        n_expert: int = ModelConfig.n_expert,
        top_k: int = ModelConfig.top_k,
        n_mlp_intermediate: int = ModelConfig.n_mlp_intermediate,
        std_depth: int = ModelConfig.std_depth,
        rec_depth: int = ModelConfig.rec_depth,
        tie_embed: bool = ModelConfig.tie_embed,
        rope_cache_len: int = ModelConfig.rope_cache_len,
        **kwargs,
    ):
        auto_map = kwargs.pop(
            "auto_map",
            {
                "AutoConfig": "hf_wrapper.RecursiveLMConfig",
                "AutoModel": "hf_wrapper.RecursiveLMModel",
                "AutoModelForCausalLM": "hf_wrapper.RecursiveLMForCausalLM",
                "AutoModelForSequenceClassification": "hf_wrapper.RecursiveLMForSequenceClassification",
                "AutoTokenizer": "hf_tokenizer.RecursiveLMTokenizer",
                "AutoProcessor": "hf_tokenizer.RecursiveLMTokenizer",
            },
        )
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.n_wembed = n_wembed
        self.moe = moe
        self.n_expert = n_expert
        self.top_k = top_k
        self.n_mlp_intermediate = n_mlp_intermediate
        self.std_depth = std_depth
        self.rec_depth = rec_depth
        self.tie_embed = tie_embed
        self.rope_cache_len = rope_cache_len
        self.auto_map = auto_map
        self.tie_word_embeddings = tie_embed
        self.hidden_size = self.n_hidden
        self.num_attention_heads = self.n_head
        self.intermediate_size = self.n_mlp_intermediate
        self.max_position_embeddings = self.rope_cache_len
        self.is_decoder = True

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            n_head=self.n_head,
            n_hidden=self.n_hidden,
            n_wembed=self.n_wembed,
            moe=self.moe,
            n_expert=self.n_expert,
            top_k=self.top_k,
            n_mlp_intermediate=self.n_mlp_intermediate,
            std_depth=self.std_depth,
            rec_depth=self.rec_depth,
            tie_embed=self.tie_embed,
            rope_cache_len=self.rope_cache_len,
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "RecursiveLMConfig":
        return cls(**asdict(config))


class RecursiveLMPreTrainedModel(PreTrainedModel):
    config_class = RecursiveLMConfig
    base_model_prefix = "model"

    def init_weights(self):
        return

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    @staticmethod
    def _normalize_inputs(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, -1]
        return input_ids, attention_mask, labels

    @staticmethod
    def _flatten_varlen(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
            flat_input = input_ids.reshape(-1)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).repeat(batch_size)
            return flat_input, position_ids, lengths

        mask = attention_mask.to(dtype=torch.bool, device=device)
        lengths = mask.sum(dim=1).to(dtype=torch.int32, device=device)

        flat_chunks = []
        pos_chunks = []
        for b in range(batch_size):
            seq_len_b = int(lengths[b].item())
            if seq_len_b == 0:
                continue
            tokens_b = input_ids[b][mask[b]]
            flat_chunks.append(tokens_b)
            pos_chunks.append(torch.arange(seq_len_b, device=device, dtype=torch.long))
        if flat_chunks:
            flat_input = torch.cat(flat_chunks, dim=0)
            position_ids = torch.cat(pos_chunks, dim=0)
        else:
            flat_input = input_ids.new_empty((0,), dtype=input_ids.dtype)
            position_ids = torch.empty((0,), device=device, dtype=torch.long)
        return flat_input, position_ids, lengths

    @staticmethod
    def _pad_packed(packed: torch.Tensor, lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        splits = torch.split(packed, lengths.tolist(), dim=0)
        padded = pad_sequence(splits, batch_first=True)
        if padded.size(1) < seq_len:
            pad_len = seq_len - padded.size(1)
            padded = F.pad(padded, (0, 0, 0, pad_len))
        return padded

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        flat_input, position_ids, lengths = self._flatten_varlen(input_ids, attention_mask)

        cu_seqlens = torch.zeros(batch_size + 1, device=input_ids.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)

        if int(cu_seqlens[-1].item()) == 0:
            hidden = torch.zeros(
                (batch_size, seq_len, self.config.hidden_size),
                device=input_ids.device,
                dtype=self.model.embedding.weight.dtype,
            )
        else:
            hidden_flat = self.model.forward_hidden(flat_input, cu_seqlens, position_ids)
            if attention_mask is None:
                hidden = self._pad_packed(hidden_flat, lengths, seq_len)
            else:
                hidden = torch.zeros(
                    (batch_size, seq_len, hidden_flat.shape[-1]),
                    device=input_ids.device,
                    dtype=hidden_flat.dtype,
                )
                mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
                offset = 0
                for b in range(batch_size):
                    seq_len_b = int(lengths[b].item())
                    if seq_len_b == 0:
                        continue
                    idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
                    hidden[b, idx] = hidden_flat[offset : offset + seq_len_b]
                    offset += seq_len_b
        return hidden, lengths


class RecursiveLMForCausalLM(RecursiveLMPreTrainedModel):
    config_class = RecursiveLMConfig
    base_model_prefix = "model"

    def __init__(self, config: RecursiveLMConfig):
        super().__init__(config)
        self.model = RecursiveGPT(config.to_model_config())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        input_ids, attention_mask, labels = self._normalize_inputs(input_ids, attention_mask, labels)
        batch_size, seq_len = input_ids.shape
        hidden, _ = self._forward_hidden(input_ids, attention_mask)
        hidden = self.model.norm_out(hidden)
        if self.model.use_factorized:
            hidden = self.model.h_to_e(hidden)
        if not self.config.tie_embed:
            logits = self.model.lm_head(hidden)
        else:
            logits = F.linear(hidden, self.model.embedding.weight)

        if logits.size(1) < seq_len:
            logits = torch.zeros(
                (batch_size, seq_len, self.config.vocab_size),
                device=input_ids.device,
                dtype=self.model.embedding.weight.dtype,
            )

        loss = None
        if labels is not None:
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_logits.numel() == 0:
                loss = logits.sum() * 0.0
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits)


class RecursiveLMModel(RecursiveLMPreTrainedModel):
    config_class = RecursiveLMConfig
    base_model_prefix = "model"

    def __init__(self, config: RecursiveLMConfig):
        super().__init__(config)
        self.model = RecursiveGPT(config.to_model_config())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        input_ids, attention_mask, _ = self._normalize_inputs(input_ids, attention_mask, None)
        hidden, _ = self._forward_hidden(input_ids, attention_mask)
        hidden = self.model.norm_out(hidden)

        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        if not return_dict:
            return (hidden,)
        return BaseModelOutputWithPast(last_hidden_state=hidden)


class RecursiveLMForSequenceClassification(RecursiveLMPreTrainedModel):
    config_class = RecursiveLMConfig
    base_model_prefix = "model"

    def __init__(self, config: RecursiveLMConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RecursiveGPT(config.to_model_config())
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        input_ids, attention_mask, labels = self._normalize_inputs(input_ids, attention_mask, labels)
        hidden, lengths = self._forward_hidden(input_ids, attention_mask)
        hidden = self.model.norm_out(hidden)

        if attention_mask is not None:
            pooled_idx = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
        else:
            pooled_idx = lengths.to(dtype=torch.long) - 1
        pooled_idx = pooled_idx.clamp(min=0)

        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), pooled_idx]
        logits = self.score(pooled)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        return_dict = kwargs.get("return_dict", self.config.use_return_dict)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(loss=loss, logits=logits)


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
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer # type: ignore

    from recursive_lm.huggingface.hf_tokenizer import RecursiveLMTokenizer

    AutoConfig.register(RecursiveLMConfig.model_type, RecursiveLMConfig)
    AutoModel.register(RecursiveLMConfig, RecursiveLMModel)
    AutoModelForCausalLM.register(RecursiveLMConfig, RecursiveLMForCausalLM)
    AutoModelForSequenceClassification.register(RecursiveLMConfig, RecursiveLMForSequenceClassification)
    AutoTokenizer.register(RecursiveLMConfig, RecursiveLMTokenizer)
except Exception:
    pass
