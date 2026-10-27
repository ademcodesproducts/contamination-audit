"""What every experiment shares: the study's inputs, and how results are saved.

An experiment reproduces one result from the paper. It computes a
JSON-serialisable value in `run`, describes it for a terminal in `report`, and
the study saves it to results/paper/<name>.json.

The study loads each input once. Running every experiment in one process reads
the 100,000-document corpus once rather than once per experiment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from ..data import BenchmarkLibrary, TemporalSplit, TrainingCorpus, TrainingDocument
from ..errors import MissingDataError
from ..filters import PublishedFilter, published_filters
from ..jsonio import read_json, write_json
from ..paths import ProjectPaths

log = logging.getLogger(__name__)

MAIN_CORPUS_SIZE = 100_000


@dataclass
class Study:
    paths: ProjectPaths
    filters: Sequence[PublishedFilter] = field(default_factory=published_filters)
    corpus_size: int = MAIN_CORPUS_SIZE
    _documents: dict[str, list[TrainingDocument]] = field(
        default_factory=dict, init=False, repr=False
    )

    @cached_property
    def benchmarks(self) -> BenchmarkLibrary:
        return BenchmarkLibrary.load(self.paths.benchmarks)

    def split(self, corpus: TrainingCorpus) -> TemporalSplit:
        return corpus.split(self.benchmarks)

    def documents(self, corpus: TrainingCorpus, size: int | None = None) -> list[TrainingDocument]:
        """The first `size` documents of a corpus, the main corpus size by default."""
        size = size or self.corpus_size
        cached = self._documents.get(corpus.name, [])
        if len(cached) < size:
            log.info("reading %s documents from %s", f"{size:,}", corpus.name)
            cached = corpus.read(self.paths.corpora, size)
            self._documents[corpus.name] = cached
        return cached[:size]

    def questions(self, corpus: TrainingCorpus, size: int | None = None) -> list[str]:
        """Just the question texts, which is all a filter reads."""
        return [document.problem for document in self.documents(corpus, size)]

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
                if name == "decon"
                else f"run `contamination-audit run {name}`"
            )
            raise MissingDataError(f"{path} is missing; {hint} first")
        return read_json(path)

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
