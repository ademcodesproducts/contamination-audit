"""Where the project's inputs and outputs live on disk.

Everything hangs off a handful of directories so the whole pipeline can be
pointed at a scratch copy. That is how the committed results are checked: rerun
everything into an empty directory and compare.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    data: Path
    results: Path
    work: Path

    @classmethod
    def at(cls, root: Path | str = ".") -> ProjectPaths:
        """The standard layout of a checkout rooted at `root`."""
        root = Path(root)
        return cls(
            data=root / "data",
            results=root / "results",
            work=root / "work",
        )

    @property
    def benchmarks(self) -> Path:
        """Small benchmark files, tracked in git."""
        return self.data / "benchmarks"

    @property
    def corpora(self) -> Path:
        """Training corpora, fetched rather than tracked."""
        return self.data / "corpora"

    @property
    def paper_results(self) -> Path:
        """One JSON file per experiment, holding the numbers the paper reports."""
        return self.results / "paper"

    @property
    def judge_record(self) -> Path:
        """The frozen LLM-judge experiment: inputs, responses and manifests."""
        return self.results / "llm_judge"
