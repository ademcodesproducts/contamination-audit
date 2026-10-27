"""The tokenizers the published filters are defined against.

Papers rarely state the tokenizer, yet it changes what a filter means. An
8-gram over words spans eight words; an 8-gram over Qwen2 tokens spans however
many words those tokens happen to cover. Model tokenizers load on first use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from functools import cached_property


class Tokenizer(ABC):
    @abstractmethod
    def __call__(self, text: str) -> Sequence[Hashable]:
        """Split text into words or token ids; filters treat both alike."""


@dataclass(frozen=True)
class LowercaseWords(Tokenizer):
    """s1's normalisation: lowercase, then split on whitespace.

    Mirrors simplescaling/s1 data/decontaminate_util.py. Punctuation is kept,
    so LaTeX stays attached to the word it is written against.
    """

    def __call__(self, text: str) -> list[str]:
        return text.lower().split()


@dataclass(frozen=True)
class HuggingFaceTokenizer(Tokenizer):
    model: str

    @cached_property
    def _encoder(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model)

    def __call__(self, text: str) -> list[int]:
        return self._encoder.encode(text, add_special_tokens=False)


@dataclass(frozen=True)
class TiktokenTokenizer(Tokenizer):
    encoding: str

    @cached_property
    def _encoder(self):
        import tiktoken

        return tiktoken.get_encoding(self.encoding)

    def __call__(self, text: str) -> list[int]:
        return self._encoder.encode(text)


@dataclass(frozen=True)
class SpacyTokenizer(Tokenizer):
    """Tulu 3's tokenizer (allenai/open-instruct decontamination/search.py).

    Only the tokenizer is loaded. open-instruct runs the full pipeline and reads
    token.text, which the tagger, parser and NER do not change; the tokenizer
    alone gives identical tokens and is about 150 times faster.
    """

    model: str

    @cached_property
    def _tokenizer(self):
        import spacy

        return spacy.load(self.model).tokenizer

    def __call__(self, text: str) -> list[str]:
        return [token.text for token in self._tokenizer(text)]


# Shared instances, so each model is loaded once however many filters use it.
S1_WORDS = LowercaseWords()
QWEN2 = HuggingFaceTokenizer("Qwen/Qwen2-7B-Instruct")
CL100K = TiktokenTokenizer("cl100k_base")
SPACY_LG = SpacyTokenizer("en_core_web_lg")
