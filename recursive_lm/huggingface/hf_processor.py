"""
Hugging Face processor wrapper for RecursiveLMTokenizer.
Requires uv group 'hf'.
"""

from __future__ import annotations

import json
import os
from importlib import import_module


class RecursiveLMProcessor:
    attributes = ["tokenizer"]
    tokenizer_class = "RecursiveLMTokenizer"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoProcessor"):
        cls._auto_class = auto_class
        return cls

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tokenizer_module = import_module("hf_tokenizer")
        tokenizer = tokenizer_module.RecursiveLMTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        return cls(tokenizer)

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        processor_config = {
            "processor_class": "RecursiveLMProcessor",
            "tokenizer_class": "RecursiveLMTokenizer",
            "auto_map": {"AutoProcessor": "hf_processor.RecursiveLMProcessor"},
        }
        config_path = os.path.join(save_directory, "processor_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(processor_config, f, indent=2)
        return (save_directory,)
