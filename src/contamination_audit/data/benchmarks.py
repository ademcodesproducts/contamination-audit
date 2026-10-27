"""Benchmark items, and the temporal split that makes filter error measurable.

A benchmark item published after a training corpus was assembled cannot be in
it, so a filter that flags such an item is wrong by construction. Items that
could be in the corpus form the measurement set; items that provably cannot
form the control set.

The split depends on the corpus. AIME 2026 is a control for Dolci, released in
November 2025, but a measurement item for Nemotron, released in May 2026. Each
corpus therefore names its own split rule; see `corpora.py`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import MissingDataError
from ..jsonio import read_jsonl

# MathArena competitions held on or before September 2025, which could be inside
# a corpus assembled by October 2025.
MEASUREMENT_COMPETITIONS = (
    "apex_2025",
    "apex-shortlist",
    "smt_2025",
    "cmimc_2025",
    "brumo_2025",
    "aime_2025",
    "hmmt_feb_2025",
    "aime_2024_I",
    "aime_2024_II",
    "hmmt_feb_2024",
)

# Sets published from January 2026 onward. Olmo 3 was announced on 2025-11-20,
# so the band from October to December 2025 is left out of both groups.
CONTROL_COMPETITIONS = (
    "aime_2026",
    "hmmt_feb_2026",
    "usamo_2026",
    "arxivmath-0126",
    "arxivmath-0326",
    "arxivmath-0626",
    "brokenarxiv-0326",
    "brokenarxiv-0626",
)

# Sets published from June 2026 onward, the only ones that postdate Nemotron.
LATE_CONTROL_COMPETITIONS = (
    "arxivmath-0626",
    "arxivmath-0826",
    "brokenarxiv-0626",
    "brokenarxiv-0826",
    "arxivlean-0626",
)

# Tulu 3 shipped in November 2024, so only the 2024 competitions predate it.
COMPETITIONS_2024 = frozenset({"aime_2024_I", "aime_2024_II", "hmmt_feb_2024"})

# Dolci controls published before Nemotron's release, which Nemotron could contain.
PUBLISHED_BEFORE_MAY_2026 = frozenset(
    {
        "aime_2026",
        "hmmt_feb_2026",
        "usamo_2026",
        "arxivmath-0126",
        "arxivmath-0326",
        "brokenarxiv-0326",
    }
)

OLYMMATH_FILES = ("olymmath_en_hard.jsonl", "olymmath_en_easy.jsonl")
MEASUREMENT_FILE = "matharena_measurement.jsonl"
CONTROL_FILE = "matharena_control.jsonl"
LATE_CONTROL_FILE = "matharena_control_post_may2026.jsonl"


@dataclass(frozen=True)
class BenchmarkItem:
    id: str
    benchmark: str
    problem: str
    answer: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any], benchmark: str | None = None):
        return cls(
            id=record["id"],
            benchmark=benchmark or record["benchmark"],
            problem=record["problem"],
            answer=str(record.get("answer", "")),
        )


@dataclass(frozen=True)
class TemporalSplit:
    """Items a corpus could contain, and items it provably cannot."""

    measurement: tuple[BenchmarkItem, ...]
    control: tuple[BenchmarkItem, ...]

    @property
    def measurement_texts(self) -> list[str]:
        """The problem statements, which is all a filter reads."""
        return [item.problem for item in self.measurement]

    @property
    def control_texts(self) -> list[str]:
        return [item.problem for item in self.control]


@dataclass(frozen=True)
class BenchmarkLibrary:
    """Every benchmark item the study uses, grouped as it was downloaded."""

    olymmath: tuple[BenchmarkItem, ...]
    measurement: tuple[BenchmarkItem, ...]
    control: tuple[BenchmarkItem, ...]
    late_control: tuple[BenchmarkItem, ...]

    @classmethod
    def load(cls, directory: Path) -> BenchmarkLibrary:
        def items(filename: str, benchmark: str | None = None):
            path = directory / filename
            if not path.exists():
                raise MissingDataError(
                    f"{path} is missing; run `contamination-audit fetch` to download it"
                )
            return tuple(BenchmarkItem.from_record(r, benchmark) for r in read_jsonl(path))

        return cls(
            olymmath=sum((items(f, "olymmath") for f in OLYMMATH_FILES), ()),
            measurement=items(MEASUREMENT_FILE),
            control=items(CONTROL_FILE),
            late_control=items(LATE_CONTROL_FILE),
        )

    def matharena_by_competition(self) -> dict[str, list[BenchmarkItem]]:
        """The Dolci-era MathArena items, keyed by competition."""
        grouped: dict[str, list[BenchmarkItem]] = defaultdict(list)
        for item in self.measurement + self.control:
            grouped[item.benchmark].append(item)
        return dict(grouped)

    # ------------------------------------------------------- split rules

    def dolci_split(self) -> TemporalSplit:
        return TemporalSplit(
            measurement=self.olymmath + self.measurement,
            control=self.control,
        )

    def tulu_split(self) -> TemporalSplit:
        """Everything from 2025 on postdates Tulu 3, so it all joins the control."""
        dolci = self.dolci_split()
        measurement = tuple(i for i in dolci.measurement if i.benchmark in COMPETITIONS_2024)
        later = tuple(i for i in dolci.measurement if i.benchmark not in COMPETITIONS_2024)
        return TemporalSplit(measurement=measurement, control=later + dolci.control)

    def nemotron_split(self) -> TemporalSplit:
        """Nemotron could contain early 2026; only June 2026 onward is control."""
        dolci = self.dolci_split()
        early_2026 = tuple(i for i in dolci.control if i.benchmark in PUBLISHED_BEFORE_MAY_2026)
        return TemporalSplit(
            measurement=dolci.measurement + early_2026,
            control=self.late_control,
        )
