"""JSONL I/O and canonical path resolution for all pipeline stages.

All scripts use ``paths(project)`` to resolve their canonical inputs/outputs.
Override with explicit paths only in tests or one-off ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import jsonlines


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical input/output paths for one audited project."""

    project: str
    train: Path
    math500: Path
    ngram_hits: Path
    embedding_candidates: Path
    train_embs: Path
    math500_embs: Path
    judge_results: Path
    c_lex: Path
    c_sem: Path


def paths(project: str, root: Path | None = None) -> ProjectPaths:
    """Return canonical paths for a project.

    Supports the four audited projects from the paper plus the ``openthoughts_full``
    variant used to compute the headline 113,957-item OpenThoughts row in Table 2.
    """
    root = root or REPO_ROOT
    train_files = {
        "s1": "s1k.jsonl",
        "tulu": "tulu_math.jsonl",
        "openthoughts": "openthoughts.jsonl",
        "openthoughts_full": "openthoughts_full.jsonl",
    }
    if project not in train_files:
        raise ValueError(f"Unknown project: {project!r}. Expected one of {list(train_files)}.")

    return ProjectPaths(
        project=project,
        train=root / "data" / "raw" / train_files[project],
        math500=root / "data" / "raw" / "math500.jsonl",
        ngram_hits=root / "results" / "ngram" / f"{project}_hits.jsonl",
        embedding_candidates=root / "results" / "embeddings" / f"{project}_candidates.jsonl",
        train_embs=root / "results" / "embeddings" / f"{project}_train_embs.npy",
        math500_embs=root / "results" / "embeddings" / f"{project}_math500_embs.npy",
        judge_results=root / "results" / "judge" / f"{project}_results.jsonl",
        c_lex=root / "data" / "processed" / f"{project}_c_lex.jsonl",
        c_sem=root / "data" / "processed" / f"{project}_c_sem.jsonl",
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with jsonlines.open(path) as reader:
        return list(reader)


def save_jsonl(items: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(items)
