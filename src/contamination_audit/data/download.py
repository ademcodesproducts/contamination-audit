"""Download the benchmarks and training corpora from the Hugging Face Hub.

Benchmarks are small and tracked in git, so downloading them again is only
needed to check that the tracked copies match upstream. Corpora are large and
fetched on demand.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..jsonio import write_jsonl
from .benchmarks import (
    CONTROL_COMPETITIONS,
    CONTROL_FILE,
    LATE_CONTROL_COMPETITIONS,
    LATE_CONTROL_FILE,
    MEASUREMENT_COMPETITIONS,
    MEASUREMENT_FILE,
)
from .corpora import DOLCI, NEMOTRON, TULU, TrainingCorpus, TrainingDocument

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- benchmarks


def fetch_olymmath(directory: Path) -> None:
    from datasets import load_dataset

    for config in ("en-hard", "en-easy"):
        tag = config.replace("-", "_")
        rows = load_dataset("RUC-AIBOX/OlymMATH", config, split="test")
        records = (
            {
                "id": f"olym_{tag}_{i:04d}",
                "problem": row.get("problem") or row.get("question", ""),
                "answer": str(row.get("answer", "")),
            }
            for i, row in enumerate(rows)
        )
        count = write_jsonl(records, directory / f"olymmath_{tag}.jsonl")
        log.info("OlymMATH %s: %d items", config, count)


@dataclass(frozen=True)
class MathArenaGroup:
    """A set of MathArena competitions stored together in one file."""

    competitions: tuple[str, ...]
    group: str
    filename: str

    def fetch(self, directory: Path) -> None:
        from datasets import load_dataset

        records, failures = [], []
        for name in self.competitions:
            try:
                rows = load_dataset(f"MathArena/{name}", split="train")
            except Exception as error:  # noqa: BLE001  any failure is reported below
                failures.append(f"{name}: {str(error)[:80]}")
                continue
            if "problem" not in rows.column_names:
                failures.append(f"{name}: no problem field in {rows.column_names}")
                continue
            for i, row in enumerate(rows):
                records.append(
                    {
                        "id": f"{name}_{i:04d}",
                        "benchmark": name,
                        "group": self.group,
                        "problem": row["problem"],
                        "answer": str(row.get("answer", "")),
                    }
                )
            log.info("  %-22s %4d", name, len(rows))

        # Every number in the paper is a fraction of one of these groups, so a
        # group that is quietly short gives results that look fine and are not
        # the paper's.
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(self.competitions)} {self.group} sets did not "
                "download:\n  " + "\n  ".join(failures)
            )
        write_jsonl(records, directory / self.filename)
        log.info("%s: %d items", self.group, len(records))


MATHARENA_GROUPS = (
    MathArenaGroup(MEASUREMENT_COMPETITIONS, "measure", MEASUREMENT_FILE),
    MathArenaGroup(CONTROL_COMPETITIONS, "control", CONTROL_FILE),
    MathArenaGroup(LATE_CONTROL_COMPETITIONS, "control_post_may26", LATE_CONTROL_FILE),
)


def fetch_benchmarks(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fetch_olymmath(directory)
    for group in MATHARENA_GROUPS:
        group.fetch(directory)


# -------------------------------------------------------------------- corpora


def first_user_turn(messages: list[dict[str, Any]] | None) -> str:
    """The first user message, across the field names these datasets use.

    Content is sometimes a plain string and sometimes a list of typed parts.
    """
    for message in messages or []:
        if (message.get("role") or message.get("from")) in ("user", "human"):
            content = message.get("content") or message.get("value") or ""
            if isinstance(content, str):
                return content
            return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class CorpusDownload(ABC):
    """Stream a corpus from the Hub, keeping one question per training example.

    Long streaming downloads drop their connection, so this appends to what is
    already on disk and resumes. Skipping the rows already written gets close to
    where the last attempt stopped; the set of written ids makes any overlap
    harmless.
    """

    corpus: TrainingCorpus
    dataset: str
    default_rows: int

    @abstractmethod
    def document(self, row: dict[str, Any], index: int) -> TrainingDocument | None:
        """The training document for one dataset row, or None to skip it."""

    def fetch(self, directory: Path, rows: int | None = None, attempts: int = 30) -> None:
        from datasets import load_dataset

        target = rows or self.default_rows
        path = directory / self.corpus.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        written = self._already_written(path)

        with open(path, "a", encoding="utf-8", newline="\n") as out:
            for attempt in range(1, attempts + 1):
                if len(written) >= target:
                    break
                try:
                    start = len(written)
                    stream = load_dataset(self.dataset, split="train", streaming=True)
                    for index, row in enumerate(stream.skip(start), start=start):
                        doc = self.document(row, index)
                        if doc is None or not doc.problem.strip() or doc.train_id in written:
                            continue
                        record = {
                            "train_id": doc.train_id,
                            "problem": doc.problem,
                            "source": doc.source,
                        }
                        out.write(json.dumps(record) + "\n")
                        written.add(doc.train_id)
                        if len(written) % 25_000 == 0:
                            out.flush()
                            log.info("  %d documents", len(written))
                        if len(written) >= target:
                            break
                except Exception as error:  # noqa: BLE001  any transport error retries
                    log.warning("  stream dropped (%s), attempt %d", type(error).__name__, attempt)
                    time.sleep(5)

        if len(written) < target:
            raise RuntimeError(f"{self.dataset}: got {len(written)} of {target} documents")
        log.info("%s: %d documents -> %s", self.dataset, len(written), path)

    @staticmethod
    def _already_written(path: Path) -> set[str]:
        if not path.exists():
            return set()
        written = set()
        with open(path, encoding="utf-8") as lines:
            for line in lines:
                try:
                    written.add(json.loads(line)["train_id"])
                except json.JSONDecodeError:
                    log.warning("skipping a malformed line in %s", path)
        log.info("resuming with %d documents already on disk", len(written))
        return written


class DolciDownload(CorpusDownload):
    corpus = DOLCI
    dataset = "allenai/Dolci-Think-SFT-32B"
    default_rows = 390_000  # the largest corpus size in the scaling experiment

    def document(self, row, index):
        return TrainingDocument(
            row.get("id"), first_user_turn(row.get("messages")), row.get("source")
        )


class NemotronDownload(CorpusDownload):
    corpus = NEMOTRON
    dataset = "nvidia/Nemotron-SFT-Math-v4"
    default_rows = 100_000

    def document(self, row, index):
        return TrainingDocument(row.get("uuid"), row.get("problem") or "", row.get("source"))


class TuluMathDownload(CorpusDownload):
    """The mathematics components of the Tulu 3 SFT mixture, in mixture order.

    Documents are identified by their row in the mixture, which has no stable
    id of its own. The first 100,000 math rows all come from the PersonaHub
    component, so that is what the replication experiment measures.
    """

    corpus = TULU
    dataset = "allenai/tulu-3-sft-mixture"
    default_rows = 100_000

    MATH_SOURCES = frozenset(
        {
            "ai2-adapt-dev/personahub_math_v5_regen_149960",
            "allenai/tulu-3-sft-personas-math-grade",
            "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
            "ai2-adapt-dev/numinamath_tir_math_decontaminated",
            "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
        }
    )

    def document(self, row, index):
        if row.get("source") not in self.MATH_SOURCES:
            return None
        return TrainingDocument(
            f"tulu_{index}", first_user_turn(row.get("messages")), row["source"]
        )


CORPUS_DOWNLOADS: dict[str, CorpusDownload] = {
    "dolci": DolciDownload(),
    "tulu": TuluMathDownload(),
    "nemotron": NemotronDownload(),
}
