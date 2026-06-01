"""Stage 1: replicate each project's n-gram filter and find items that survived.

Identifies *Type 1 contamination* — pairs that should have been removed by the
project's claimed decontamination filter but were not.

Reads:  data/raw/{project}.jsonl, data/raw/math500.jsonl
Writes: results/ngram/{project}_hits.jsonl
"""

from __future__ import annotations

import argparse
import logging

import _common  # noqa: F401

from contamination_audit.config import configure_logging, load_config, seed_everything
from contamination_audit.io import load_jsonl, paths, save_jsonl
from contamination_audit.ngram import find_hits

_log = logging.getLogger("ngram_filter")

DEFAULT_PROJECTS = ["s1", "tulu", "openthoughts"]


def run_project(project: str, app_config) -> int:
    p = paths(project)
    if project not in app_config.ngram:
        _log.error("no n-gram config for project=%r", project)
        return 0
    ngram_cfg = app_config.ngram[project]

    train = load_jsonl(p.train)
    math500 = load_jsonl(p.math500)
    hits = find_hits(project, train, math500, ngram_cfg)
    save_jsonl(hits, p.ngram_hits)
    _log.info("  -> %s", p.ngram_hits)
    return len(hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS,
                        help="One or more of: s1, tulu, openthoughts, openthoughts_full")
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)
    seed_everything(app_config.seed)

    summary = {}
    for project in args.projects:
        _log.info("=== %s ===", project)
        summary[project] = run_project(project, app_config)

    _log.info("=== n-gram audit complete ===")
    for project, n_hits in summary.items():
        _log.info("  %-20s %d raw hits", project, n_hits)


if __name__ == "__main__":
    main()
