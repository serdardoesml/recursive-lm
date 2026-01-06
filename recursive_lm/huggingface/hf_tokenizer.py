"""
Hugging Face tokenizer wrapper for RustBPETokenizer.
Requires uv group 'hf'.
"""

from __future__ import annotations

from typing import Optional
import os

from transformers import PreTrainedTokenizer # type: ignore
from transformers.tokenization_utils_base import BatchEncoding # type: ignore
from transformers.utils.hub import cached_file # type: ignore

from recursive_lm.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


class RecursiveLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"tokenizer_file": "tokenizer.pkl"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file: str, **kwargs):
        if tokenizer_file is None:
            raise ValueError("tokenizer_file is required.")
        self._tokenizer = RustBPETokenizer.load_from_dir(tokenizer_file)
        self._vocab: Optional[dict[str, int]] = None

        bos_token = kwargs.pop("bos_token", SPECIAL_TOKENS[0])
        additional_special_tokens = kwargs.pop("additional_special_tokens", SPECIAL_TOKENS[1:])
        pad_token = kwargs.pop("pad_token", bos_token)
        auto_map = kwargs.pop(
            "auto_map",
            {
                "AutoTokenizer": "hf_tokenizer.RecursiveLMTokenizer",
                "AutoProcessor": "hf_tokenizer.RecursiveLMTokenizer",
            },
        )
        super().__init__(
            bos_token=bos_token,
            additional_special_tokens=additional_special_tokens,
            pad_token=pad_token,
            auto_map=auto_map,
            **kwargs,
        )

    @property
    def tokenizer(self):
        return self

    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoProcessor"):
        cls._auto_class = auto_class
        return cls

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *init_inputs, **kwargs):
        if kwargs.get("tokenizer_file") is None:
            kwargs["tokenizer_file"] = cached_file(
                pretrained_model_name_or_path,
                cls.vocab_files_names["tokenizer_file"],
                cache_dir=kwargs.get("cache_dir"),
                revision=kwargs.get("revision"),
                token=kwargs.get("token") or kwargs.get("use_auth_token"),
                local_files_only=kwargs.get("local_files_only", False),
                subfolder=kwargs.get("subfolder", ""),
            )
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

    def __call__(self, text, text_pair=None, **kwargs):
        return_offsets_mapping = kwargs.pop("return_offsets_mapping", False)
        return_offsets_mapping = kwargs.pop("return_offset_mapping", return_offsets_mapping)
        return_tensors = kwargs.pop("return_tensors", None)
        padding = kwargs.pop("padding", False)
        truncation = kwargs.pop("truncation", False)
        max_length = kwargs.pop("max_length", None)
        return_attention_mask = kwargs.pop("return_attention_mask", True)
        return_token_type_ids = kwargs.pop("return_token_type_ids", False)

        pairs: list[tuple[str, str | None]]
        if isinstance(text, (list, tuple)):
            if text_pair is None and text and isinstance(text[0], (list, tuple)):
                pairs = [(a, b) for a, b in text]
            elif text_pair is not None:
                pairs = list(zip(text, text_pair))
            else:
                pairs = [(t, None) for t in text]
        else:
            pairs = [(text, text_pair)]

        input_ids_batch = []
        token_type_ids_batch = []
        offsets_batch = []

        for a, b in pairs:
            ids_a = self._tokenizer.encode(a) if a is not None else []
            ids_b = self._tokenizer.encode(b) if b is not None else []
            input_ids = ids_a + ids_b
            token_type_ids = [0] * len(ids_a) + [1] * len(ids_b)

            offsets = None
            if return_offsets_mapping:
                offsets_a = self._offsets_for_text(a) if a is not None else []
                offsets_b = self._offsets_for_text(b) if b is not None else []
                offsets = offsets_a + offsets_b

            if truncation and max_length is not None and len(input_ids) > max_length:
                if self.truncation_side == "left":
                    start = len(input_ids) - max_length
                    input_ids = input_ids[start:]
                    token_type_ids = token_type_ids[start:]
                    if offsets is not None:
                        offsets = offsets[start:]
                else:
                    input_ids = input_ids[:max_length]
                    token_type_ids = token_type_ids[:max_length]
                    if offsets is not None:
                        offsets = offsets[:max_length]

            input_ids_batch.append(input_ids)
            token_type_ids_batch.append(token_type_ids)
            if return_offsets_mapping:
                offsets_batch.append(offsets if offsets is not None else [])

        pad_to_length = None
        if padding is True or padding == "longest":
            pad_to_length = max(len(ids) for ids in input_ids_batch) if input_ids_batch else 0
        elif padding == "max_length":
            pad_to_length = max_length

        attention_mask_batch = []
        if pad_to_length is not None:
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            for i, ids in enumerate(input_ids_batch):
                pad_len = pad_to_length - len(ids)
                if pad_len < 0:
                    pad_len = 0
                if self.padding_side == "left":
                    input_ids_batch[i] = [pad_id] * pad_len + ids
                    token_type_ids_batch[i] = [0] * pad_len + token_type_ids_batch[i]
                    if return_offsets_mapping:
                        offsets_batch[i] = [(0, 0)] * pad_len + offsets_batch[i]
                    attn = [0] * pad_len + [1] * len(ids)
                else:
                    input_ids_batch[i] = ids + [pad_id] * pad_len
                    token_type_ids_batch[i] = token_type_ids_batch[i] + [0] * pad_len
                    if return_offsets_mapping:
                        offsets_batch[i] = offsets_batch[i] + [(0, 0)] * pad_len
                    attn = [1] * len(ids) + [0] * pad_len
                attention_mask_batch.append(attn)
        else:
            attention_mask_batch = [[1] * len(ids) for ids in input_ids_batch]

        enc = {"input_ids": input_ids_batch}
        if return_attention_mask:
            enc["attention_mask"] = attention_mask_batch
        if return_token_type_ids:
            enc["token_type_ids"] = token_type_ids_batch
        if return_offsets_mapping:
            enc["offset_mapping"] = offsets_batch

        return BatchEncoding(enc, tensor_type=return_tensors)

    def _offsets_for_text(self, text: str) -> list[tuple[int, int]]:
        token_ids = self._tokenizer.enc.encode_ordinary(text)
        offsets: list[tuple[int, int]] = []
        idx = 0
        for token_id in token_ids:
            token_text = self._tokenizer.enc.decode([token_id])
            start = idx
            end = start + len(token_text)
            offsets.append((start, end))
            idx = end
        return offsets

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        if self._vocab is None:
            self._vocab = {self._tokenizer.id_to_token(i): i for i in range(self.vocab_size)}
        return dict(self._vocab)

    def _tokenize(self, text: str) -> list[str]:
        ids = self._tokenizer.encode(text)
        return [self._tokenizer.id_to_token(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._tokenizer.get_special_tokens():
            return self._tokenizer.encode_special(token)
        return self._tokenizer.enc.encode_single_token(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self._tokenizer.decode(ids)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            raise FileNotFoundError(f"Directory does not exist: {save_directory}")
        filename = "tokenizer.pkl"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        out_path = os.path.join(save_directory, filename)
        self._tokenizer.save(out_path)
        return (out_path,)
