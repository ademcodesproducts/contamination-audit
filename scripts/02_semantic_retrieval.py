"""Stage 2: semantic retrieval — dense embeddings OR TF-IDF, per project.

Paper §5.1 specifies different retrieval methods per project:
  s1, OpenThoughts — dense embeddings (FAISS, all-mpnet-base-v2, ≥0.70 sim)
  Tülu 3           — TF-IDF cosine similarity (character n-grams)

The method is selected via ``configs/thresholds.yaml`` (``retrieval.<project>.method``).

Reads:  data/raw/{project}.jsonl, data/raw/math500.jsonl, results/ngram/{project}_hits.jsonl
Writes: results/embeddings/{project}_candidates.jsonl + cached .npy embeddings (dense only)
"""

from __future__ import annotations

import argparse
import logging

import _common  # noqa: F401

from contamination_audit.config import configure_logging, load_config, seed_everything
from contamination_audit.embedding import Encoder, search_candidates
from contamination_audit.io import load_jsonl, paths, save_jsonl
from contamination_audit.tfidf import TfidfConfig, retrieve as tfidf_retrieve

_log = logging.getLogger("semantic_retrieval")

DEFAULT_PROJECTS = ["s1", "tulu", "openthoughts"]


def _load_ngram_pairs(path) -> set[tuple[str, str]]:
    if not path.exists():
        _log.warning("no n-gram hits at %s — proceeding without exclusion", path)
        return set()
    return {(item["math500_id"], item["train_id"]) for item in load_jsonl(path)}


def run_dense(project: str, encoder: Encoder) -> int:
    p = paths(project)
    train = load_jsonl(p.train)
    math500 = load_jsonl(p.math500)
    ngram_pairs = _load_ngram_pairs(p.ngram_hits)

    candidates = search_candidates(
        project=project, train_items=train, math500_items=math500,
        ngram_hit_pairs=ngram_pairs, encoder=encoder,
        train_embs_cache=p.train_embs, math500_embs_cache=p.math500_embs,
    )
    save_jsonl(candidates, p.embedding_candidates)
    _log.info("  -> %s", p.embedding_candidates)
    return len(candidates)


def run_tfidf(project: str) -> int:
    p = paths(project)
    train = load_jsonl(p.train)
    math500 = load_jsonl(p.math500)

    candidates = tfidf_retrieve(
        project=project, train_items=train, math500_items=math500,
        config=TfidfConfig(),
    )
    # Exclude pairs already flagged at the n-gram stage (Type 1 contamination).
    ngram_pairs = _load_ngram_pairs(p.ngram_hits)
    if ngram_pairs:
        before = len(candidates)
        candidates = [
            c for c in candidates
            if (c["math500_id"], c["train_id"]) not in ngram_pairs
        ]
        _log.info("  excluded %d ngram-hit pairs", before - len(candidates))

    save_jsonl(candidates, p.embedding_candidates)
    _log.info("  -> %s", p.embedding_candidates)
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    parser.add_argument("--method", choices=("auto", "dense", "tfidf"), default="auto",
                        help="Override the per-project method from config (default: auto from yaml)")
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)
    seed_everything(app_config.seed)

    encoder: Encoder | None = None  # lazy — only created if any dense run is needed
    for project in args.projects:
        method = args.method
        if method == "auto":
            method = app_config.retrieval.get(project).method if project in app_config.retrieval else "dense"
        _log.info("=== %s (%s) ===", project, method)
        if method == "tfidf":
            run_tfidf(project)
        else:
            if encoder is None:
                encoder = Encoder(app_config.embedding)
            run_dense(project, encoder)


if __name__ == "__main__":
    main()
