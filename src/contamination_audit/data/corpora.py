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
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from ..errors import MissingDataError
from ..jsonio import iter_jsonl, read_json
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
        """Which benchmark items this corpus could contain, and which it cannot.

        A control question published after the corpus can still rest on a paper
        posted before it, whose content the corpus could hold. Such items are
        dropped: a control counts only if its source paper also postdates the
        corpus's release.
        """
        split = self.split_rule(library)
        return replace(
            split,
            control=tuple(
                item
                for item in split.control
                if item.first_posted is None or item.first_posted > self.released
            ),
        )

    def completion_marker(self, directory: Path) -> Path:
        """Written by a download that read every row of the dataset."""
        return directory / f"{Path(self.filename).stem}.complete.json"

    def read(self, directory: Path, limit: int | None) -> list[TrainingDocument]:
        """The first `limit` documents, or the whole corpus if `limit` is None.

        Refuses to silently return fewer than were asked for.
        """
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
        # for, labelling a 100k run as 390k, or a partial download as the corpus.
        if limit is not None and len(documents) < limit:
            raise MissingDataError(
                f"{path} holds {len(documents)} documents but {limit} are needed"
            )
        if limit is None:
            marker = self.completion_marker(directory)
            if not marker.exists():
                raise MissingDataError(
                    f"{path} is not a complete download; finish it with "
                    "`contamination-audit fetch <corpus> --rows all`"
                )
            expected = read_json(marker)["documents"]
            if len(documents) != expected:
                raise MissingDataError(
                    f"{path} holds {len(documents)} documents; its download wrote {expected}"
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
    # The day its data first appeared on the Hub; the repository itself is from
    # 2026-05-26. Later revisions rewrote the files without changing the
    # documents this study reads.
    released=date(2026, 6, 2),
    split_rule=BenchmarkLibrary.nemotron_split,
)
