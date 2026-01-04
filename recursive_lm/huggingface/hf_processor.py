"""
Hugging Face processor wrapper for RecursiveLMTokenizer.
Requires uv group 'hf'.
"""

from __future__ import annotations

from transformers import ProcessorMixin # type: ignore


class RecursiveLMProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "RecursiveLMTokenizer"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
