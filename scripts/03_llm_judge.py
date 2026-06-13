"""Stage 3: LLM-as-judge verifies retrieval candidates.

Each project uses its paper-specified judge by default (configurable per project
under ``judges`` in ``configs/thresholds.yaml``):

  s1, OpenThoughts → ``judges.default`` (GPT-4o-mini + judge_default prompt)
  Tülu 3           → ``judges.tulu``   (Gemini 2.5 Flash via Vertex + judge_tulu prompt)

Pass ``--judge <name>`` to override (e.g. ``--judge strict`` for the
step-for-step Tülu re-judge described in paper §7).

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

# Map project → (judge config key, prompt name).
PROJECT_JUDGE = {
    "s1": ("default", "judge_default"),
    "openthoughts": ("default", "judge_default"),
    "openthoughts_full": ("default", "judge_default"),
    "tulu": ("tulu", "judge_tulu"),
}


def _classification_to_type(cls: str) -> str:
    """Map any of the supported classification schemes to a contamination_type."""
    if cls in ("CONTAMINATED", "INSTANCE_CONTAMINATED", "TEMPLATE_CONTAMINATED"):
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
        with jsonlines.open(p.judge_results, mode="w") as writer:
            writer.write_all(results)
        time.sleep(rate_limit_sleep)

    confirmed = sum(
        1 for r in results
        if r["classification"] in ("CONTAMINATED", "INSTANCE_CONTAMINATED", "TEMPLATE_CONTAMINATED")
    )
    total = sum(1 for r in results if r["classification"] != "ERROR")
    precision = confirmed / max(total, 1)
    _log.info("  confirmed contaminated %d / %d valid  (rate %.1f%%)",
              confirmed, total, precision * 100)
    return {"contaminated": confirmed, "total_valid": total, "precision": precision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    parser.add_argument("--projects", nargs="+", default=DEFAULT_PROJECTS)
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Cap candidates per project (default: judge all)")
    parser.add_argument("--sim-threshold", type=float, default=None,
                        help="Only judge candidates with similarity >= this")
    parser.add_argument("--judge", default=None,
                        help="Override the judge config key (default: per-project mapping)")
    parser.add_argument("--prompt", default=None,
                        help="Override the prompt name (default: per-project mapping)")
    args = parser.parse_args()

    configure_logging()
    app_config = load_config(args.config)
    seed_everything(app_config.seed)

    for project in args.projects:
        default_judge_key, default_prompt = PROJECT_JUDGE.get(
            project, ("default", "judge_default")
        )
        judge_key = args.judge or default_judge_key
        prompt = args.prompt or default_prompt
        judge_cfg = app_config.judges[judge_key]
        _log.info("=== %s  judge=%s  provider=%s  model=%s  prompt=%s ===",
                  project, judge_key, judge_cfg.provider, judge_cfg.model, prompt)
        judge = Judge(judge_cfg, prompt_name=prompt)
        run_project(
            project, judge,
            max_candidates=args.max_candidates,
            sim_threshold=args.sim_threshold,
            rate_limit_sleep=judge_cfg.rate_limit_sleep,
        )


if __name__ == "__main__":
    main()
