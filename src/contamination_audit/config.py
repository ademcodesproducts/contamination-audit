"""Typed configuration dataclasses + YAML loader.

Every numeric threshold or hyperparameter in the paper lives here. Scripts read
YAML from ``configs/`` and never define magic numbers inline.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


@dataclass(frozen=True)
class NgramConfig:
    """N-gram filter parameters per project.

    ``threshold_mode``:
        ``any``     — flag pairs with any shared n-gram (s1, OpenThoughts)
        ``percent`` — flag pairs whose test-side token coverage exceeds ``coverage`` (Tülu)
    """

    n: int
    threshold_mode: Literal["any", "percent"]
    coverage: float = 0.5
    tokenizer: str = "Qwen/Qwen2-7B-Instruct"


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "all-mpnet-base-v2"
    sim_threshold: float = 0.70
    top_k: int = 5
    batch_size: int = 32
    chunk_size: int = 5000
    embed_dim: int = 768


@dataclass(frozen=True)
class JudgeConfig:
    model: str = "gpt-4o-mini"
    provider: Literal["openai", "google", "vertex", "openai_compat"] = "openai"
    base_url: str = ""                 # required when provider == "openai_compat"
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = 3
    timeout_seconds: int = 30
    max_tokens: int = 300
    temperature: float = 0.0
    truncate_chars: int = 1500
    rate_limit_sleep: float = 0.1


@dataclass(frozen=True)
class RetrievalConfig:
    """Per-project retrieval method (paper §5.1).

    s1, openthoughts → ``dense`` (all-mpnet-base-v2)
    tulu → ``tfidf`` (char-ngram TF-IDF with density-aware top-K)
    """

    method: Literal["dense", "tfidf"] = "dense"


@dataclass(frozen=True)
class StatsConfig:
    n_bootstrap: int = 10_000
    alpha: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class AppConfig:
    ngram: dict[str, NgramConfig]
    retrieval: dict[str, RetrievalConfig]
    judges: dict[str, JudgeConfig]              # keyed by prompt name (e.g. "default", "strict", "tulu")
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    seed: int = 42

    # Backwards-compat shim: ``app_config.judge`` returns the default judge so
    # older call sites keep working.
    @property
    def judge(self) -> JudgeConfig:
        return self.judges.get("default", JudgeConfig())


def default_config() -> AppConfig:
    """Return the configuration that produced the paper's headline numbers."""
    return AppConfig(
        ngram={
            "s1": NgramConfig(n=8, threshold_mode="any"),
            "tulu": NgramConfig(n=8, threshold_mode="percent", coverage=0.5),
            "openthoughts": NgramConfig(n=13, threshold_mode="any"),
            "openthoughts_full": NgramConfig(n=13, threshold_mode="any"),
        },
        retrieval={
            "s1": RetrievalConfig(method="dense"),
            "tulu": RetrievalConfig(method="tfidf"),
            "openthoughts": RetrievalConfig(method="dense"),
            "openthoughts_full": RetrievalConfig(method="dense"),
        },
        judges={
            "default": JudgeConfig(model="gpt-4o-mini", provider="openai"),
            "strict": JudgeConfig(model="gpt-4o-mini", provider="openai"),
            "tulu": JudgeConfig(model="gemini-2.5-flash", provider="vertex"),
        },
    )


def load_config(path: str | Path) -> AppConfig:
    """Load configuration from a YAML file. Falls back to defaults for missing fields."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = default_config()
    ngram_raw = raw.get("ngram", {})
    ngram = {name: NgramConfig(**cfg) for name, cfg in ngram_raw.items()} if ngram_raw else defaults.ngram

    retrieval_raw = raw.get("retrieval", {})
    retrieval = (
        {name: RetrievalConfig(**cfg) for name, cfg in retrieval_raw.items()}
        if retrieval_raw else defaults.retrieval
    )

    judges_raw = raw.get("judges", {})
    judges = (
        {name: JudgeConfig(**cfg) for name, cfg in judges_raw.items()}
        if judges_raw else defaults.judges
    )

    return AppConfig(
        ngram=ngram,
        retrieval=retrieval,
        judges=judges,
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        stats=StatsConfig(**raw.get("stats", {})),
        seed=raw.get("seed", 42),
    )


def seed_everything(seed: int) -> None:
    """Set Python, NumPy, and (if available) PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: WPS433
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_logging(level: int = logging.INFO) -> None:
    """Standard logger config used by every script entry point."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
