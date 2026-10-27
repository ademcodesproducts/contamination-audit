"""The training corpora whose decontamination the study measures.

Only the first user turn of each training example is kept: it is where a
benchmark problem would appear if it had leaked, and it is the field the
released decontamination code searches.

Corpus order matters. Experiments always read a prefix, never a sample, so a
larger corpus size contains every smaller one and the scaling curve measures
size and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..errors import MissingDataError
from ..jsonio import iter_jsonl
from .benchmarks import BenchmarkLibrary, TemporalSplit


@dataclass(frozen=True)
class TrainingDocument:
    train_id: str
    problem: str
    source: str | None


@dataclass(frozen=True)
class TrainingCorpus:
    name: str
    filename: str
    released: date
    split_rule: Callable[[BenchmarkLibrary], TemporalSplit]

    def split(self, library: BenchmarkLibrary) -> TemporalSplit:
        """Which benchmark items this corpus could contain, and which it cannot."""
        return self.split_rule(library)

    def read(self, directory: Path, limit: int) -> list[TrainingDocument]:
        """The first `limit` documents, refusing to silently return fewer."""
        path = directory / self.filename
        if not path.exists():
            raise MissingDataError(
                f"{path} is missing; run `contamination-audit fetch --help` to download it"
            )
        documents = [
            TrainingDocument(r["train_id"], r["problem"], r.get("source"))
            for r in iter_jsonl(path, limit)
        ]
        # A short file would otherwise be reported under the size that was asked
        # for, labelling a 100k run as 390k.
        if len(documents) < limit:
            raise MissingDataError(
                f"{path} holds {len(documents)} documents but {limit} are needed"
            )
        return documents


DOLCI = TrainingCorpus(
    name="Dolci-Think-SFT-32B",
    filename="dolci.jsonl",
    released=date(2025, 11, 20),
    split_rule=BenchmarkLibrary.dolci_split,
)

TULU = TrainingCorpus(
    name="tulu_math",
    filename="tulu_math.jsonl",
    released=date(2024, 11, 21),
    split_rule=BenchmarkLibrary.tulu_split,
)

NEMOTRON = TrainingCorpus(
    name="Nemotron-SFT-Math-v4",
    filename="nemotron.jsonl",
    released=date(2026, 5, 1),
    split_rule=BenchmarkLibrary.nemotron_split,
)
