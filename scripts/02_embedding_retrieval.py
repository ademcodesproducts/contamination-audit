"""Stage 2: dense semantic retrieval (Type 2 contamination candidates).

Embeds train + MATH-500 problems with all-mpnet-base-v2, searches the top-K
neighbors of each MATH-500 item, filters by similarity threshold, and excludes
any pair already flagged by the n-gram stage.

Reads:  data/raw/{project}.jsonl, data/raw/math500.jsonl, results/ngram/{project}_hits.jsonl
Writes: results/embeddings/{project}_candidates.jsonl + cached .npy embeddings
"""

from __future__ import annotations

import argparse
import logging

import _common  # noqa: F401

from contamination_audit.config import configure_logging, load_config, seed_everything
from contamination_audit.embedding import Encoder, search_candidates
from contamination_audit.io import load_jsonl, paths, save_jsonl

_log = logging.getLogger("embedding_retrieval")

DEFAULT_PROJECTS = ["s1", "tulu", "openthoughts"]


def _load_ngram_pairs(path) -> set[tuple[str, str]]:
    if not path.exists():
        _log.warning("no n-gram hits at %s — proceeding without exclusion", path)
        return set()
    return {(item["math500_id"], item["train_id"]) for item in load_jsonl(path)}


def run_project(project: str, encoder: Encoder) -> int:
    p = paths(project)
    train = load_jsonl(p.train)
    math500 = load_jsonl(p.math500)
    ngram_pairs = _load_ngram_pairs(p.ngram_hits)

    candidates = search_candidates(
        project=project,
        train_items=train,
        math500_items=math500,
        ngram_hit_pairs=ngram_pairs,
        encoder=encoder,
        train_embs_cache=p.train_embs,
        math500_embs_cache=p.math500_embs,
    )
    save_jsonl(candidates, p.embedding_candidates)
    _log.info("  -> %s", p.embedding_candidates)
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)
    seed_everything(app_config.seed)
    encoder = Encoder(app_config.embedding)

    for project in args.projects:
        _log.info("=== %s ===", project)
        run_project(project, encoder)


if __name__ == "__main__":
    main()
