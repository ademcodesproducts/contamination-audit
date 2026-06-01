"""Stage 0: download and normalize all training and benchmark datasets.

Reads HF IDs from configs/datasets.yaml. Writes one jsonl per dataset to data/raw/.
Idempotent: skips datasets whose output file already exists unless --force is passed.

Schema (uniform across all train datasets):
  train_id, problem, solution, source, dataset

MATH-500 schema:
  math500_id, problem, solution, answer, subject, level
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import jsonlines
import yaml
from datasets import load_dataset

import _common  # noqa: F401  (sys.path setup)

from contamination_audit.config import configure_logging, seed_everything
from contamination_audit.io import save_jsonl

_log = logging.getLogger("load_datasets")


TULU_MATH_KEYWORDS = ("math", "numina", "gsm", "platypus", "wizard", "orca", "camel")


def _try_load(hf_ids: list[str], split: str):
    last_err = None
    for hf_id in hf_ids:
        try:
            _log.info("  trying %s ...", hf_id)
            ds = load_dataset(hf_id, split=split)
            _log.info("  loaded %d items (%s)", len(ds), hf_id)
            return ds, hf_id
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log.warning("  failed: %s", e)
    raise RuntimeError(f"All HF IDs failed. Last error: {last_err}")


def _normalize_messages(messages: list[dict]) -> tuple[str, str]:
    """Extract (problem, solution) from a conversation list, handling both schemas."""
    problem = solution = ""
    for msg in messages:
        role = msg.get("role") or msg.get("from", "")
        content = msg.get("content") or msg.get("value", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        if role in ("user", "human") and not problem:
            problem = content
        elif role in ("assistant", "gpt") and not solution:
            solution = content
    return problem, solution


def _parse_level(raw) -> int:
    try:
        return int(str(raw).replace("Level ", ""))
    except (ValueError, TypeError):
        return -1


def load_math500(cfg: dict, out: Path) -> None:
    ds, _ = _try_load(cfg["hf_ids"], cfg["split"])
    items = []
    for i, row in enumerate(ds):
        items.append({
            "math500_id": f"math500_{i:04d}",
            "problem": row.get("problem") or row.get("question", ""),
            "solution": row.get("solution", ""),
            "answer": row.get("answer", ""),
            "subject": row.get("subject") or row.get("type", ""),
            "level": _parse_level(row.get("level", "")),
        })
    save_jsonl(items[:500], out)
    _log.info("  saved %d MATH-500 items -> %s", len(items[:500]), out)


def load_s1(cfg: dict, out: Path) -> None:
    ds, _ = _try_load(cfg["hf_ids"], cfg["split"])
    items = []
    for i, row in enumerate(ds):
        items.append({
            "train_id": f"s1k_{i:04d}",
            "problem": row.get("problem") or row.get("question") or row.get("prompt", ""),
            "solution": row.get("solution") or row.get("response") or row.get("cot", ""),
            "source": row.get("source") or row.get("source_type", "unknown"),
            "dataset": "s1k",
        })
    save_jsonl(items, out)
    _log.info("  saved %d s1K items -> %s", len(items), out)


def load_tulu(cfg: dict, out: Path) -> None:
    ds, hf_id = _try_load(cfg["hf_ids"], cfg["split"])

    items = []
    for i, row in enumerate(ds):
        source = str(row.get("source", ""))
        messages = row.get("messages", [])

        if messages:
            # Full mixture format — filter to math sources only.
            if not any(kw in source.lower() for kw in TULU_MATH_KEYWORDS):
                continue
            problem, solution = _normalize_messages(messages)
        else:
            problem = row.get("problem") or row.get("question") or row.get("input", "")
            solution = row.get("solution") or row.get("output") or row.get("response", "")

        if not problem.strip():
            continue

        items.append({
            "train_id": f"tulu_{i:06d}",
            "problem": problem,
            "solution": solution,
            "source": source or "numinamath",
            "dataset": "tulu",
        })

    save_jsonl(items, out)
    _log.info("  saved %d Tulu math items -> %s", len(items), out)


def load_openthoughts(cfg: dict, out: Path, *, full: bool = False) -> None:
    ds, _ = _try_load(cfg["hf_ids"], cfg["split"])

    sample_size = cfg.get("sample_size")
    if not full and sample_size and sample_size < len(ds):
        random.seed(cfg.get("sample_seed", 42))
        indices = random.sample(range(len(ds)), sample_size)
        sample = ds.select(indices)
    else:
        indices = list(range(len(ds)))
        sample = ds

    items = []
    for i, row in enumerate(sample):
        messages = row.get("conversations") or row.get("messages", [])
        if messages:
            problem, solution = _normalize_messages(messages)
        else:
            problem = row.get("problem") or row.get("question", "")
            solution = row.get("solution") or row.get("response", "")

        if not problem.strip():
            continue

        original_idx = indices[i] if not full else i
        items.append({
            "train_id": f"ot_{original_idx:06d}",
            "problem": problem,
            "solution": (solution or "")[:1000],  # truncate long CoT
            "source": row.get("source", "openthoughts"),
            "dataset": "openthoughts",
            "original_index": original_idx,
        })

    save_jsonl(items, out)
    _log.info("  saved %d OpenThoughts items -> %s (full=%s)", len(items), out, full)


LOADERS = {
    "math500": load_math500,
    "s1": load_s1,
    "tulu": load_tulu,
    "openthoughts": lambda cfg, out: load_openthoughts(cfg, out, full=False),
    "openthoughts_full": lambda cfg, out: load_openthoughts(cfg, out, full=True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-config", default=str(_common.DATASETS_CONFIG))
    parser.add_argument("--only", nargs="*", help="Subset of dataset names to load")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    seed_everything(args.seed)

    with open(args.datasets_config, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    targets = args.only or list(registry.keys())
    for name in targets:
        if name not in LOADERS:
            _log.warning("unknown dataset %r, skipping", name)
            continue
        cfg = registry[name]
        out = _common.REPO_ROOT / cfg["local"]
        if out.exists() and not args.force:
            _log.info("%s already cached, skipping (use --force to re-download)", name)
            continue
        _log.info("=== loading %s ===", name)
        out.parent.mkdir(parents=True, exist_ok=True)
        LOADERS[name](cfg, out)

    _log.info("done.")


if __name__ == "__main__":
    main()
