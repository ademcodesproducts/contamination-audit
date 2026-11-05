"""What every experiment shares: the study's inputs, and how results are saved.

An experiment reproduces one result from the paper. It computes a
JSON-serialisable value in `run`, describes it for a terminal in `report`, and
the study saves it to results/paper/<name>.json.

The study loads each input once. A few experiments read the first 100,000
documents of Dolci. The headline experiments read the whole corpus through a scan:
one pass of a published filter over every document, against every benchmark
item at once, cached under work/scans/ and shared by every experiment that
needs it.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from ..data import (
    DOLCI,
    BenchmarkItem,
    BenchmarkLibrary,
    TemporalSplit,
    TrainingCorpus,
    TrainingDocument,
)
from ..errors import MissingDataError
from ..filters import PublishedFilter, published_filters
from ..filters.rules import Matches
from ..jsonio import read_json, write_json
from ..paths import ProjectPaths

log = logging.getLogger(__name__)

MAIN_CORPUS_SIZE = 100_000


@dataclass(frozen=True)
class Scan:
    """One filter's matches over a whole corpus, against a fixed list of items."""

    items: tuple[BenchmarkItem, ...]
    matches: Matches  # document index -> positions in `items`
    corpus_n: int

    @cached_property
    def _position(self) -> dict[str, int]:
        return {item.id: i for i, item in enumerate(self.items)}

    def flagged(self, items: Sequence[BenchmarkItem], within: int | None = None) -> set[int]:
        """Positions within `items` that some document matches.

        With `within`, only the first `within` documents count, which is the
        same as having filtered a corpus that long.
        """
        wanted = {self._position[item.id]: i for i, item in enumerate(items)}
        return {
            wanted[position]
            for document, positions in self.matches.items()
            if within is None or document < within
            for position in positions
            if position in wanted
        }

    def documents(self, items: Sequence[BenchmarkItem]) -> set[int]:
        """The documents that match at least one of `items`."""
        wanted = {self._position[item.id] for item in items}
        return {d for d, positions in self.matches.items() if positions & wanted}


@dataclass
class Study:
    paths: ProjectPaths
    filters: Sequence[PublishedFilter] = field(default_factory=published_filters)
    corpus_size: int = MAIN_CORPUS_SIZE
    _documents: dict[str, list[TrainingDocument]] = field(
        default_factory=dict, init=False, repr=False
    )
    _complete: set[str] = field(default_factory=set, init=False, repr=False)
    _scans: dict[str, Scan] = field(default_factory=dict, init=False, repr=False)

    @cached_property
    def benchmarks(self) -> BenchmarkLibrary:
        return BenchmarkLibrary.load(self.paths.benchmarks)

    def split(self, corpus: TrainingCorpus) -> TemporalSplit:
        return corpus.split(self.benchmarks)

    def documents(self, corpus: TrainingCorpus, size: int | None = None) -> list[TrainingDocument]:
        """The first `size` documents of a corpus, the main corpus size by default."""
        size = size or self.corpus_size
        cached = self._documents.get(corpus.name, [])
        if len(cached) < size and corpus.name not in self._complete:
            log.info("reading %s documents from %s", f"{size:,}", corpus.name)
            cached = corpus.read(self.paths.corpora, size)
            self._documents[corpus.name] = cached
        if len(cached) < size:
            raise MissingDataError(f"{corpus.name} has {len(cached)} documents, {size} asked for")
        return cached[:size]

    def questions(self, corpus: TrainingCorpus, size: int | None = None) -> list[str]:
        """Just the question texts, which is all a filter reads."""
        return [document.problem for document in self.documents(corpus, size)]

    def full_documents(self, corpus: TrainingCorpus) -> list[TrainingDocument]:
        """Every document in the corpus file."""
        if corpus.name not in self._complete:
            log.info("reading every document of %s", corpus.name)
            self._documents[corpus.name] = corpus.read(self.paths.corpora, None)
            self._complete.add(corpus.name)
        return self._documents[corpus.name]

    def full_questions(self, corpus: TrainingCorpus) -> list[str]:
        return [document.problem for document in self.full_documents(corpus)]

    # --------------------------------------------------------------- scans

    @cached_property
    def scan_items(self) -> tuple[BenchmarkItem, ...]:
        """Every benchmark item a Dolci headline experiment asks about."""
        split, library = self.split(DOLCI), self.benchmarks
        return split.measurement + split.control + library.earlier_editions

    def scan(self, published: PublishedFilter) -> Scan:
        """One pass of a published filter over all of Dolci, cached on disk."""
        if published.name in self._scans:
            return self._scans[published.name]
        corpus = self.full_questions(DOLCI)
        fingerprint = hashlib.sha256(
            "\n".join(
                [published.name, str(len(corpus))] + [i.problem for i in self.scan_items]
            ).encode()
        ).hexdigest()[:16]
        path = self.paths.work / "scans" / f"{published.name}-{fingerprint}.json"
        if path.exists():
            saved = read_json(path)
            matches = {int(d): set(positions) for d, positions in saved.items()}
        else:
            log.info("scanning %s documents with %s", f"{len(corpus):,}", published.name)
            matches = published.match(corpus, [item.problem for item in self.scan_items])
            write_json({d: sorted(p) for d, p in sorted(matches.items())}, path)
        self._scans[published.name] = Scan(self.scan_items, matches, len(corpus))
        return self._scans[published.name]

    # ------------------------------------------------------------ results

    def result_path(self, name: str) -> Path:
        return self.paths.paper_results / f"{name}.json"

    def has_result(self, name: str) -> bool:
        return self.result_path(name).exists()

    def load_result(self, name: str) -> Any:
        path = self.result_path(name)
        if not path.exists():
            hint = (
                "run `contamination-audit decon`"
                if name.startswith("decon")
                else f"run `contamination-audit run {name}`"
            )
            raise MissingDataError(f"{path} is missing; {hint} first")
        return read_json(path)

    def decon_result(self, name: str, corpus_n: int) -> dict | None:
        """decon's saved result, if it was run on a corpus of this many documents.

        decon is run separately, so its result can be left over from a run on a
        different number of documents; mixing the two would compare filters
        that saw different corpora.
        """
        if not self.has_result(name):
            return None
        result = self.load_result(name)
        if result["corpus_rows"] != corpus_n:
            log.warning(
                "%s was run on %s documents, not %s; leaving it out until it is rerun",
                name,
                f"{result['corpus_rows']:,}",
                f"{corpus_n:,}",
            )
            return None
        return result

    def run(self, experiment: type[Experiment], **options: Any) -> Any:
        """Run one experiment, save its result, and print its report."""
        log.info("%s: %s", experiment.name, experiment.title)
        instance = experiment(self, **options)
        result = instance.run()
        write_json(result, self.result_path(experiment.name))
        print(f"\n{experiment.title}")
        for line in instance.report(result):
            print(f"  {line}")
        return result


class Experiment(ABC):
    name: ClassVar[str]
    title: ClassVar[str]

    def __init__(self, study: Study) -> None:
        self.study = study

    @abstractmethod
    def run(self) -> Any:
        """Compute the result as plain JSON-serialisable data."""

    def report(self, result: Any) -> list[str]:
        """The result as a few lines of text, for a terminal."""
        return []
