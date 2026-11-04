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
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .data import DOLCI, NEMOTRON, TULU, BenchmarkItem, TrainingCorpus, TrainingDocument
from .experiments.base import Experiment, Study
from .experiments.scaling import CORPUS_SIZES
from .jsonio import write_jsonl

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.8
SCORE_THRESHOLDS = (DEFAULT_THRESHOLD, 0.6, 0.4, 0.2)


@dataclass(frozen=True)
class DeconFindings:
    documents: int  # training documents decon reports as contaminated
    items: list[int]  # positions, in the eval file, of the items they contaminate

    def as_json(self) -> dict:
        return {"documents": self.documents, "items": self.items}


@dataclass(frozen=True)
class DeconBinary:
    path: Path

    def detect(
        self,
        training: Path,
        evals: Path,
        report: Path,
        threshold: float | None = None,
        index_answers: bool = True,
    ) -> DeconFindings:
        """What decon reports as contaminated, at its defaults unless told otherwise."""
        command = [
            str(self.path), "detect",
            "--training-dir", str(training),
            "--content-key", "text",
            "--evals-dir", str(evals),
            "--report-output-dir", str(report),
        ]  # fmt: skip
        if threshold is not None:
            command += ["--contamination-score-threshold", str(threshold)]
        if not index_answers:
            command += ["--index-answers", "false"]
        # A report left by an earlier run would be read as this run's.
        shutil.rmtree(report, ignore_errors=True)

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
        # The report holds one line per contaminated (document, item) pair, above
        # the threshold only; eval_instance_index is the item's line in the eval file.
        items = {
            json.loads(line)["eval_instance_index"]
            for path in sorted(report.rglob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        return DeconFindings(int(reported.group(1).replace(",", "")), sorted(items))


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
    """decon at its default setting on one corpus, writing results/paper/<name>.json."""

    corpus: ClassVar[TrainingCorpus]

    def __init__(self, study: Study, binary: Path, scaling: bool = False) -> None:
        super().__init__(study)
        self.decon = DeconBinary(binary)
        self.scaling = scaling
        self.work = study.paths.work / self.name

    def training_documents(self) -> list[TrainingDocument]:
        return self.study.documents(self.corpus)

    def run(self):
        split = self.study.split(self.corpus)
        training = self.work / "training"
        result = {
            "corpus_rows": write_training_set(self.training_documents(), training),
            "n_measure": write_eval_set(split.measurement, "measure", self.work / "evals_measure"),
            "n_control": write_eval_set(split.control, "control", self.work / "evals_control"),
        }
        # Flagged items under the same keys as flag_counts uses for every other filter.
        result["documents"] = {}
        for group in ("measure", "control"):
            findings = self.decon.detect(
                training, self.work / f"evals_{group}", self.work / f"report_{group}"
            )
            result[group] = findings.items
            result["documents"][group] = findings.documents
        return result

    def report(self, result):
        lines = [
            f"measurement {len(result['measure'])}/{result['n_measure']} items, "
            f"control {len(result['control'])}/{result['n_control']} items flagged; "
            f"{result['documents']['measure']} and {result['documents']['control']} "
            "contaminated documents"
        ]

        def found(findings):
            return f"{len(findings['items'])} items, {findings['documents']} documents"

        for group, by_threshold in result.get("threshold_sweep", {}).items():
            for threshold, findings in by_threshold.items():
                lines.append(f"threshold {threshold}, {group}: {found(findings)}")
        for group, findings in result.get("without_answers", {}).items():
            lines.append(f"without answers, {group}: {found(findings)}")
        for size, findings in result.get("scaling", {}).items():
            # sizes are ints when just run, strings when read back from JSON
            lines.append(f"{int(size):>9,d} documents: {found(findings)}")
        return lines


class DolciDecon(DeconRun):
    """On all of Dolci, with a sweep of decon's score threshold and, optionally,
    every corpus size of the scaling experiment."""

    name = "decon"
    title = "Ai2 decon on all of Dolci"
    corpus = DOLCI

    def training_documents(self):
        return self.study.full_documents(DOLCI)

    def run(self):
        result = super().run()
        training = self.work / "training"
        # Swept on the controls too: what a lower threshold adds is only
        # contamination if it adds more measurement items than controls.
        result["threshold_sweep"] = {
            group: {
                threshold: (
                    {"documents": result["documents"][group], "items": result[group]}
                    if threshold == DEFAULT_THRESHOLD
                    else self.decon.detect(
                        training,
                        self.work / f"evals_{group}",
                        self.work / f"report_{group}_t{threshold}",
                        threshold,
                    ).as_json()
                )
                for threshold in SCORE_THRESHOLDS
            }
            for group in ("measure", "control")
        }
        # Our eval files carry the benchmark answer and the training documents do
        # not, so a question matched alone is scored on decon's question-and-answer
        # weights, where the question term counts 0.75: below the 0.8 default even
        # for a verbatim copy. Without indexed answers that ceiling is gone.
        result["without_answers"] = {
            group: self.decon.detect(
                training,
                self.work / f"evals_{group}",
                self.work / f"report_{group}_without_answers",
                index_answers=False,
            ).as_json()
            for group in ("measure", "control")
        }
        if self.scaling:
            # decon is not a rule in this package, so the claim that it stays at
            # zero across corpus sizes needs its own run at each size.
            documents = self.training_documents()
            result["scaling"] = {}
            for size in CORPUS_SIZES:
                if size is None:  # the whole corpus, already run above
                    result["scaling"][len(documents)] = result["threshold_sweep"]["measure"][
                        DEFAULT_THRESHOLD
                    ]
                    continue
                sized = self.work / "training_scaling"
                write_training_set(documents[:size], sized)
                result["scaling"][size] = self.decon.detect(
                    sized, self.work / "evals_measure", self.work / f"report_size{size}"
                ).as_json()
        return result


class TuluDecon(DeconRun):
    name = "decon_tulu"
    title = "Ai2 decon on the Tulu 3 math SFT data (100,000 documents)"
    corpus = TULU


class NemotronDecon(DeconRun):
    name = "decon_nemotron"
    title = "Ai2 decon on Nemotron-SFT-Math-v4 (100,000 documents)"
    corpus = NEMOTRON


DECON_RUNS: dict[str, type[DeconRun]] = {
    "dolci": DolciDecon,
    "tulu": TuluDecon,
    "nemotron": NemotronDecon,
}
