"""Download the benchmarks and training corpora from the Hugging Face Hub.

Benchmarks are small and tracked in git, so downloading them again is only
needed to check that the tracked copies match upstream. Corpora are large and
fetched on demand.
"""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..jsonio import write_json, write_jsonl
from .benchmarks import (
    CONTROL_COMPETITIONS,
    CONTROL_FILE,
    EARLIER_EDITIONS,
    EARLIER_EDITIONS_FILE,
    LATE_CONTROL_COMPETITIONS,
    LATE_CONTROL_FILE,
    MEASUREMENT_COMPETITIONS,
    MEASUREMENT_FILE,
    arxiv_paper,
)
from .corpora import DOLCI, NEMOTRON, TULU, TrainingCorpus, TrainingDocument

log = logging.getLogger(__name__)

ALL = math.inf  # as a row count: download the whole corpus

# Every download is pinned to a dataset revision, so the inputs cannot change
# under the results. MathArena sets are pinned to the revisions current when the
# committed benchmark files were taken (2026-10-06); three of them have since
# been corrected upstream, and fetching the latest would not reproduce the paper.
OLYMMATH_REVISION = "2c6532ea2cf929ac1c421532af5951553eaee727"
MATHARENA_REVISIONS = {
    "aime_2023_I": "4e55d1bf38f277ab89ba7fc987f76604a6c3877c",
    "aime_2023_II": "bcfb17ba39a7ac54545c749c9f57b836f23e070f",
    "hmmt_feb_2023": "43d9106e3ae5733bb57be0322b3e855e6b2cd4ea",
    "usamo_2024": "cfd53f87a053e0b414577a45bed5ab5e91b1c68e",
    "usamo_2025": "0a2c60f2249e07b8ee76c942bca4f5f87aa959df",
    "aime_2024_I": "ea5b061c3e8039dc9858defaafc407d04b995e9f",
    "aime_2024_II": "29d5d31e9b46e215fc24d9b2a3047506823dd101",
    "aime_2025": "c94da77eb22bbd6439e62a323bec18493a421302",
    "aime_2026": "d2de22f3c656b4f56cf8981212186377d1e23bc3",
    "apex-shortlist": "f3efdf224ef665f129ddaae37699f6098c65781b",
    "apex_2025": "ac8a641db12cc87be39e61ea89f2e04c80c5f2e7",
    "arxivlean-0626": "1ced46a1d45d57c6de242d51c0baf574b96022ea",
    "arxivmath-0126": "bac939ba603c93f905aa48072980ea681a2824cd",
    "arxivmath-0326": "14ff32983ae4c72b75fb64eba750f3c27f46607a",
    "arxivmath-0626": "dca0d771ab20c1b8e8c50d0921bf693f7bfe663d",
    "arxivmath-0826": "977a3a0083198e11ac4f20fe79288351cebd6281",
    "brokenarxiv-0326": "510ad632a1dfa157a1a50ca744b5cea3b0fe37b8",
    "brokenarxiv-0626": "73dd424784fbdeab599557fcba3d77559c89d1ee",
    "brokenarxiv-0826": "6445b9740c9f932c44a3c1c6019230d08e81a793",
    "brumo_2025": "6de9bc0987f5710c6376ee980e1483dbeb55cfda",
    "cmimc_2025": "8f27517ce4ad1d05e023c0e7eca4344754ba2a04",
    "hmmt_feb_2024": "b3cf7fb48814f76142d0847003a5aac0876f3bf4",
    "hmmt_feb_2025": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
    "hmmt_feb_2026": "02fba4f74d8e68e73e66a02d540fd979c05c274c",
    "smt_2025": "d8e99d83f1583842f46d4953cd960eea5d887e45",
    "usamo_2026": "57c61ae5f2a1cc085b6c49667dc320bd2d4187bc",
}


# ----------------------------------------------------------------- benchmarks


def fetch_olymmath(directory: Path) -> None:
    from datasets import load_dataset

    for config in ("en-hard", "en-easy"):
        tag = config.replace("-", "_")
        rows = load_dataset("RUC-AIBOX/OlymMATH", config, split="test", revision=OLYMMATH_REVISION)
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


def arxiv_id(source: str | float) -> str:
    """An arXiv identifier as text, repairing ids that were stored as numbers.

    Some MathArena sets store '2601.01950' as the float 2601.0195, which drops
    the trailing zero and names a different paper. New-style ids always have
    five digits after the point.
    """
    return f"{source:.5f}" if isinstance(source, float) else str(source)


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
                rows = load_dataset(
                    f"MathArena/{name}", split="train", revision=MATHARENA_REVISIONS[name]
                )
            except Exception as error:  # noqa: BLE001  any failure is reported below
                failures.append(f"{name}: {str(error)[:80]}")
                continue
            if "problem" not in rows.column_names:
                failures.append(f"{name}: no problem field in {rows.column_names}")
                continue
            for i, row in enumerate(rows):
                record = {
                    "id": f"{name}_{i:04d}",
                    "benchmark": name,
                    "group": self.group,
                    "problem": row["problem"],
                    "answer": str(row.get("answer", "")),
                }
                if row.get("source") is not None:
                    record["source"] = arxiv_id(row["source"])
                records.append(record)
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
    MathArenaGroup(EARLIER_EDITIONS, "earlier_edition", EARLIER_EDITIONS_FILE),
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


def parquet_shards(dataset: str, revision: str) -> list[tuple[str, int]]:
    """A dataset's Parquet files in order, with row counts read from their footers."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    root = f"datasets/{dataset}@{revision}/"
    shards = []
    for path in sorted(fs.glob(root + "**/*.parquet")):
        with fs.open(path) as handle:
            shards.append((path[len(root) :], pq.ParquetFile(handle).metadata.num_rows))
    return shards


def parquet_rows(dataset: str, revision: str, columns: tuple[str, ...], start: int):
    """(row index, row) for every row from `start` on, one downloaded shard at a time.

    Shards wholly before `start` are skipped without downloading them, and each
    shard is deleted once read, so resuming is cheap and disk use stays small.
    """
    import tempfile

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    offset = 0
    for filename, rows in parquet_shards(dataset, revision):
        if offset + rows <= start:
            offset += rows
            continue
        with tempfile.TemporaryDirectory() as scratch:
            local = hf_hub_download(
                dataset, filename, repo_type="dataset", revision=revision, local_dir=scratch
            )
            with open(local, "rb") as handle:
                for batch in pq.ParquetFile(handle).iter_batches(columns=list(columns)):
                    for row in batch.to_pylist():
                        if offset >= start:
                            yield offset, row
                        offset += 1


class CorpusDownload(ABC):
    """Download a corpus from the Hub, keeping one question per training example.

    Downloads append to what is already on disk and resume after a dropped
    connection. Resuming at the number of documents written lands at or before
    where the last attempt stopped, since examples with no question are read
    but not written; the set of written ids makes the overlap harmless.
    """

    corpus: TrainingCorpus
    dataset: str
    revision: str
    columns: tuple[str, ...]
    default_rows: float

    @abstractmethod
    def document(self, row: dict[str, Any], index: int) -> TrainingDocument | None:
        """The training document for one dataset row, or None to skip it."""

    def fetch(self, directory: Path, rows: float | None = None, attempts: int = 30) -> None:
        """Download the first `rows` documents, or the whole corpus if `rows` is ALL."""
        target = self.default_rows if rows is None else rows
        path = directory / self.corpus.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        written = self._already_written(path)
        exhausted = False

        with open(path, "a", encoding="utf-8", newline="\n") as out:
            for attempt in range(1, attempts + 1):
                if exhausted or len(written) >= target:
                    break
                try:
                    rows = parquet_rows(self.dataset, self.revision, self.columns, len(written))
                    for index, row in rows:
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
                    else:
                        exhausted = True  # every shard was read rather than the link dropping
                except Exception as error:  # noqa: BLE001  any transport error retries
                    log.warning("  download failed (%s), attempt %d", type(error).__name__, attempt)
                    time.sleep(min(60, 5 * attempt))

        if len(written) < target and not (exhausted and target == ALL):
            raise RuntimeError(f"{self.dataset}: got {len(written)} of {target} documents")
        if exhausted:
            # Experiments that read the whole corpus check this before trusting the file.
            write_json(
                {"dataset": self.dataset, "revision": self.revision, "documents": len(written)},
                self.corpus.completion_marker(directory),
            )
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
    columns = ("id", "messages", "source")
    # The March 2026 re-upload of the November 2025 release that came with Olmo 3.
    # It keeps every example, in the same order with the same question text, and
    # replaces a single placeholder source label with one per original dataset.
    revision = "7668c638cc84100951973b456069a1a462d6d915"
    default_rows = ALL  # the headline experiments read the whole corpus

    def document(self, row, index):
        return TrainingDocument(
            row.get("id"), first_user_turn(row.get("messages")), row.get("source")
        )


class NemotronDownload(CorpusDownload):
    corpus = NEMOTRON
    dataset = "nvidia/Nemotron-SFT-Math-v4"
    columns = ("uuid", "problem", "source")
    revision = "84d42ad0cb960f07f951b9baa9ed2b46a5a18c66"
    default_rows = ALL

    def document(self, row, index):
        return TrainingDocument(row.get("uuid"), row.get("problem") or "", row.get("source"))


class TuluMathDownload(CorpusDownload):
    """The mathematics components of the Tulu 3 SFT mixture, in mixture order.

    Documents are identified by their row in the mixture, which has no stable
    id of its own. The components are not shuffled together, so any prefix
    over-represents the first of them, PersonaHub.
    """

    corpus = TULU
    dataset = "allenai/tulu-3-sft-mixture"
    columns = ("messages", "source")
    revision = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
    default_rows = ALL

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


# --------------------------------------------------------------- arXiv dates

ARXIV_DATES_FILE = "arxiv_first_posted.json"


def fetch_arxiv_dates(directory: Path) -> None:
    """The date each research-derived control's source paper first appeared on arXiv.

    An identifier gives only the month. The arXiv API gives the day the first
    version was published, which matters when a corpus was released mid-month.
    """
    import urllib.request
    import xml.etree.ElementTree as ElementTree

    from ..jsonio import read_jsonl, write_json

    identifiers = sorted(
        {
            paper
            for name in (CONTROL_FILE, LATE_CONTROL_FILE)
            for row in read_jsonl(directory / name)
            if (paper := arxiv_paper(row.get("source")))
        }
    )
    atom = "{http://www.w3.org/2005/Atom}"
    dates = {}
    for start in range(0, len(identifiers), 100):
        batch = identifiers[start : start + 100]
        url = "http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(batch)
        with urllib.request.urlopen(url, timeout=60) as response:
            feed = ElementTree.fromstring(response.read())
        for entry in feed.iter(f"{atom}entry"):
            identifier = entry.find(f"{atom}id").text.rsplit("/", 1)[-1].split("v")[0]
            dates[identifier] = entry.find(f"{atom}published").text[:10]
        time.sleep(3)  # arXiv asks for at most one request every three seconds
    missing = sorted(set(identifiers) - set(dates))
    if missing:
        raise RuntimeError(f"arXiv returned no record for {len(missing)} ids: {missing[:5]}")
    write_json(dict(sorted(dates.items())), directory / ARXIV_DATES_FILE)
    log.info("first-posted dates for %d papers", len(dates))
