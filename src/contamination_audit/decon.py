"""Run Ai2's decon, the decontamination tool behind Olmo 3, over the study's inputs.

decon is a Rust binary and is deliberately not reimplemented. Its first stage
alone -- any shared 5-gram of cl100k tokens -- flags 199 of the 200 OlymMATH
items where the real tool flags none: the cluster expansion and weighted
scoring applied on top are what turn candidates into a verdict. Build it with

    git clone https://github.com/allenai/decon
    cd decon && cargo build --release

and pass the binary at target/release/decon to `contamination-audit decon`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .data import DOLCI, BenchmarkItem, TrainingDocument
from .experiments.base import Experiment, Study
from .experiments.scaling import CORPUS_SIZES
from .jsonio import write_jsonl

log = logging.getLogger(__name__)

SCORE_THRESHOLDS = (0.8, 0.6, 0.4, 0.2)  # 0.8 is decon's default


@dataclass(frozen=True)
class DeconBinary:
    path: Path

    def contaminated_documents(
        self, training: Path, evals: Path, report: Path, threshold: float | None = None
    ) -> int:
        """How many training documents decon reports as contaminated."""
        command = [
            str(self.path), "detect",
            "--training-dir", str(training),
            "--content-key", "text",
            "--evals-dir", str(evals),
            "--report-output-dir", str(report),
        ]  # fmt: skip
        if threshold is not None:
            command += ["--contamination-score-threshold", str(threshold)]

        # decon draws box characters, which Windows' default codec cannot decode.
        finished = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )
        if finished.returncode != 0:
            raise RuntimeError(
                f"decon exited {finished.returncode}:\n{(finished.stderr or '').strip()[-800:]}"
            )
        # The report layout is not a stable interface, so fail loudly if it moves.
        reported = re.search(r"Contaminated documents\s*│?\s*([\d,]+)", finished.stdout)
        if not reported:
            raise RuntimeError(f"decon's report did not parse:\n{finished.stdout[-800:]}")
        return int(reported.group(1).replace(",", ""))


def write_training_set(documents: Iterable[TrainingDocument], directory: Path) -> int:
    records = ({"text": d.problem, "id": d.train_id, "source": d.source} for d in documents)
    return write_jsonl(records, directory / "dolci.jsonl")


def write_eval_set(items: Iterable[BenchmarkItem], group: str, directory: Path) -> int:
    """decon's own eval schema; doc_id and eval_instance_index must be integers."""
    records = (
        {
            "eval_key": item.benchmark,
            "eval_instance_index": i,
            "split": group,
            "question": item.problem,
            "answer": item.answer,
            "config": item.benchmark,
            "doc_id": i,
            "is_correct": True,
            "fingerprint": hashlib.md5(item.problem.encode(), usedforsecurity=False).hexdigest()[
                :16
            ],
        }
        for i, item in enumerate(items)
    )
    return write_jsonl(records, directory / "bench.jsonl")


class DeconRun(Experiment):
    """decon at its default setting, a sweep of its score threshold, and optionally
    the corpus sizes of the scaling experiment."""

    name = "decon"
    title = "Ai2 decon on the main corpus"

    def __init__(self, study: Study, binary: Path, scaling: bool = False) -> None:
        super().__init__(study)
        self.decon = DeconBinary(binary)
        self.scaling = scaling
        self.work = study.paths.work / "decon"

    def run(self):
        split = self.study.split(DOLCI)
        training = self.work / "training"
        result = {
            "corpus_rows": write_training_set(self.study.documents(DOLCI), training),
            "n_measure": write_eval_set(split.measurement, "measure", self.work / "evals_measure"),
            "n_control": write_eval_set(split.control, "control", self.work / "evals_control"),
        }

        for group in ("measure", "control"):
            result[group] = self.decon.contaminated_documents(
                training, self.work / f"evals_{group}", self.work / f"report_{group}"
            )

        result["threshold_sweep"] = {
            threshold: self.decon.contaminated_documents(
                training, self.work / "evals_measure", self.work / f"report_t{threshold}", threshold
            )
            for threshold in SCORE_THRESHOLDS
        }

        if self.scaling:
            # decon is not a rule in this package, so the claim that it stays at
            # zero across corpus sizes needs its own run at each size.
            result["scaling"] = {}
            for size in CORPUS_SIZES:
                sized = self.work / "training_scaling"
                write_training_set(self.study.documents(DOLCI, size), sized)
                result["scaling"][size] = self.decon.contaminated_documents(
                    sized, self.work / "evals_measure", self.work / f"report_size{size}"
                )
        return result

    def report(self, result):
        lines = [
            f"measurement {result['measure']}/{result['n_measure']}, "
            f"control {result['control']}/{result['n_control']} contaminated"
        ]
        for threshold, hits in result["threshold_sweep"].items():
            lines.append(f"score threshold {threshold}: {hits} contaminated documents")
        for size, hits in result.get("scaling", {}).items():
            lines.append(f"{size:>9,d} documents: {hits} contaminated")
        return lines
