"""Stage 7: run OpenThinker-7B / Tulu-3-8B-SFT / s1.1-7B on the DiD prompt set.

Linearised from ``downloads/19_did_inference (2).ipynb``. Loads the two
perturbation CSVs (contaminated + clean), routes each prompt to its target
model via the ``dataset`` column, and writes per-model traces with resume
support.

Hardware: a single NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM, bf16) per
paper Table 3. Use ``--4bit`` to run on smaller GPUs.

Reads:  data/processed/contaminated_and_perturbations.csv
        data/processed/clean_baseline_prompts.csv
Writes: results/traces/{model}_traces.jsonl
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import yaml

import _common  # noqa: F401

from contamination_audit.config import configure_logging, seed_everything
from contamination_audit.inference import InferenceConfig, run
from contamination_audit.io import REPO_ROOT

_log = logging.getLogger("run_inference")


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_problem_record(row: dict, *, split: str) -> dict:
    return {
        "math500_id": row["math500_id"],
        "split": split,
        "subject": row["math500_subject"],
        "level": row["math500_level"],
        "perturbation_type": row["perturbation_type"],
        "problem": row["problem"],
        "answer": row["answer"],
        "answer_confidence": row.get("answer_confidence", ""),
        "dataset": row.get("dataset"),
    }


def load_problems(contam_csv: Path, clean_csv: Path, models: list[str]) -> dict[str, list[dict]]:
    """Route each problem to the model whose ``dataset`` tag matches."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for src_csv, split in ((contam_csv, "contaminated"), (clean_csv, "clean")):
        if not src_csv.exists():
            _log.warning("missing %s", src_csv)
            continue
        for row in _load_csv(src_csv):
            ds = row.get("dataset")
            if ds in models:
                by_model[ds].append(_build_problem_record(row, split=split))
    return by_model


def _load_inference_models(yaml_path: Path) -> list[InferenceConfig]:
    """Read inference-model settings from configs/models.yaml."""
    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out = []
    for entry in raw.get("inference_models", []):
        out.append(InferenceConfig(
            model_id=entry["hf_id"],
            name=entry["name"],
            n_samples=entry.get("n_samples", 1),
            temperature=entry.get("temperature", 0.8),
            max_new_tokens=entry.get("max_new_tokens", 4096),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", default=str(_common.MODELS_CONFIG))
    parser.add_argument("--only", nargs="*",
                        help="Subset of model names to run (default: all in models.yaml)")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true",
                        help="Use 4-bit BitsAndBytes quantization (for low-VRAM GPUs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    seed_everything(args.seed)

    all_models = _load_inference_models(Path(args.models_config))
    if args.only:
        all_models = [m for m in all_models if m.name in args.only]
    if not all_models:
        _log.error("no inference models selected")
        return

    contam_csv = REPO_ROOT / "data" / "processed" / "contaminated_and_perturbations.csv"
    clean_csv = REPO_ROOT / "data" / "processed" / "clean_baseline_prompts.csv"
    problems_by_model = load_problems(contam_csv, clean_csv, [m.name for m in all_models])

    out_dir = REPO_ROOT / "results" / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in all_models:
        if args.use_4bit:
            cfg = InferenceConfig(**{**cfg.__dict__, "use_4bit": True})
        problems = problems_by_model.get(cfg.name, [])
        _log.info("=== %s — %d prompts ===", cfg.name, len(problems))
        if not problems:
            continue
        out_path = out_dir / f"{cfg.name}_traces.jsonl"
        run(cfg, problems, out_path)


if __name__ == "__main__":
    main()
