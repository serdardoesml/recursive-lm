"""
Hugging Face tokenizer wrapper for RustBPETokenizer.
Requires uv group 'hf'.
"""

from __future__ import annotations

from typing import Optional
import os

from transformers import PreTrainedTokenizer # type: ignore
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
