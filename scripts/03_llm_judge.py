"""Stage 3: LLM-as-judge verifies embedding candidates.

Classifies each candidate pair as CONTAMINATED / RELATED / CLEAN. Default prompt
is the lenient version used in the paper for s1 and OpenThoughts; ``--strict``
swaps to the step-for-step prompt used for the Tulu re-judge (Section 7).

Resumes automatically from the existing output file.

Reads:  results/embeddings/{project}_candidates.jsonl
Writes: results/judge/{project}_results.jsonl
"""

from __future__ import annotations

import argparse
import logging
import time

import jsonlines
from tqdm import tqdm

import _common  # noqa: F401

from contamination_audit.config import configure_logging, load_config, seed_everything
from contamination_audit.io import load_jsonl, paths
from contamination_audit.judge import Judge

_log = logging.getLogger("llm_judge")

DEFAULT_PROJECTS = ["s1", "tulu", "openthoughts"]


def _classification_to_type(cls: str) -> str:
    if cls == "CONTAMINATED":
        return "c_sem"
    if cls == "RELATED":
        return "related"
    if cls == "CLEAN":
        return "clean_candidate"
    return "error"


def run_project(
    project: str,
    judge: Judge,
    *,
    max_candidates: int | None,
    sim_threshold: float | None,
    rate_limit_sleep: float,
) -> dict:
    p = paths(project)
    candidates = load_jsonl(p.embedding_candidates)
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    if sim_threshold is not None:
        candidates = [c for c in candidates if c["similarity_score"] >= sim_threshold]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    already_judged: dict[tuple[str, str], dict] = {}
    if p.judge_results.exists():
        for item in load_jsonl(p.judge_results):
            already_judged[(item["math500_id"], item["train_id"])] = item
        _log.info("  resuming: %d already judged", len(already_judged))

    results = list(already_judged.values())
    to_judge = [c for c in candidates if (c["math500_id"], c["train_id"]) not in already_judged]
    _log.info("  remaining: %d", len(to_judge))

    p.judge_results.parent.mkdir(parents=True, exist_ok=True)

    for candidate in tqdm(to_judge, desc=f"judging {project}"):
        judgment = judge.judge(candidate["train_problem"], candidate["math500_problem"])
        record = {
            **candidate,
            **judgment.to_dict(),
            "contamination_type": _classification_to_type(judgment.classification),
        }
        results.append(record)
        # Incremental save — safe to interrupt.
        with jsonlines.open(p.judge_results, mode="w") as writer:
            writer.write_all(results)
        time.sleep(rate_limit_sleep)

    confirmed = sum(1 for r in results if r["classification"] == "CONTAMINATED")
    total = sum(1 for r in results if r["classification"] != "ERROR")
    precision = confirmed / max(total, 1)
    _log.info("  CONTAMINATED %d / %d valid  (precision %.1f%%)",
              confirmed, total, precision * 100)
    return {"contaminated": confirmed, "total_valid": total, "precision": precision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Cap candidates per project (default: judge all)")
    parser.add_argument("--sim-threshold", type=float, default=None,
                        help="Only judge candidates with similarity >= this (e.g. 0.80 for Tulu strict)")
    parser.add_argument("--strict", action="store_true",
                        help="Use the step-for-step prompt (judge_strict.txt)")
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)
    seed_everything(app_config.seed)

    prompt = "judge_strict" if args.strict else "judge_default"
    judge = Judge(app_config.judge, prompt_name=prompt)

    for project in args.projects:
        _log.info("=== %s (prompt=%s) ===", project, prompt)
        run_project(
            project,
            judge,
            max_candidates=args.max_candidates,
            sim_threshold=args.sim_threshold,
            rate_limit_sleep=app_config.judge.rate_limit_sleep,
        )


if __name__ == "__main__":
    main()
