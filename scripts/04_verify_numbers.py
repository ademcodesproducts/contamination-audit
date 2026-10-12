"""Check every headline number in the paper against the JSON that produced it.

A transcription error between a run and the manuscript is invisible to the
compiler and to every reviewer who does not re-run the pipeline. This reads
results/paper/*.json, derives the figures the paper states, and reports any that
are missing from or inconsistent with paper/arr_decontamination.tex.

It is not a substitute for regenerating from a clean checkout, which also catches
a stale JSON. It catches the step after that one.

    python scripts/04_verify_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

PAPER = Path("paper/arr_decontamination.tex")
OUT = Path("results/paper")

# A label for the report, and the string the paper is expected to contain.
Check = tuple[str, str]


# ------------------------------------------------------------------ formatting

# How the paper prints each kind of quantity. A correctly computed number in the
# wrong format will not be found, so the formatting is defined once here rather
# than inline at each use.


def pct(value: float) -> str:
    return f"{value:.1f}%"


def pct_of(count: int, total: int) -> str:
    return pct(100 * count / total)


def interval(ci: list[float]) -> str:
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]"


# ---------------------------------------------------------------------- checks

# One generator per results file. Each takes that file's contents and yields the
# strings the paper should contain; nothing here touches the paper itself.


def check_main(data) -> Iterator[Check]:
    """Table 2: per-filter flag counts on both benchmark groups."""
    n_measure, n_control = data["n_measure"], data["n_control"]
    yield "measurement set size", str(n_measure)
    yield "control set size", str(n_control)
    for name, flagged in data["filters"].items():
        yield f"{name} measurement count", pct_of(len(flagged["measure"]), n_measure)
        yield f"{name} control count", pct_of(len(flagged["control"]), n_control)


def check_consensus(data) -> Iterator[Check]:
    """Sec. 6: how far the filters agree, with bootstrap intervals."""
    yield "items flagged by >=1 filter", str(data["flagged_any"])

    exactly_one = data["consensus"].get("1")
    if exactly_one:
        yield "flagged by exactly one filter", pct(exactly_one["pct"])
        yield "exactly-one interval", interval(exactly_one["ci"])

    for name, rates in data.get("rates", {}).items():
        control = rates["control"]
        if control["pct"] == 0:
            continue  # the paper does not print a [0.0, 0.0] interval
        yield f"{name} false-positive interval", interval(control["ci"])


def check_scaling(data) -> Iterator[Check]:
    """Fig. 1: the ends of the corpus-size sweep, and the error fraction at each."""
    smallest, largest = data[0], data[-1]
    yield "smallest corpus", str(smallest["corpus"])
    yield "largest corpus", str(largest["corpus"])
    for key in ("s1_frac", "ot3_frac"):
        if smallest.get(key) and largest.get(key):
            yield f"{key} at the smallest corpus", f"{smallest[key]:.2f}"
            yield f"{key} at the largest corpus", f"{largest[key]:.2f}"


def check_mechanism(data) -> Iterator[Check]:
    """Sec. 9: how few distinct n-grams the false positives trace to."""
    yield "distinct false-positive n-grams", str(data["distinct_ngrams"])
    yield "false positives traced", str(data["false_positives"])


def check_matched(data) -> Iterator[Check]:
    """Sec. 13: the same competition one year apart."""
    for competition, rates in data.items():
        yield f"matched pair {competition}, s1", pct(rates["s1"])


def check_nemotron(data) -> Iterator[Check]:
    """Sec. 14: the third corpus, which gets its own control split."""
    yield "nemotron measurement size", str(data["n_measure"])
    yield "nemotron control size", str(data["n_control"])
    for name, rates in data["filters"].items():
        if rates["fp"] or rates["detect"]:
            yield f"nemotron {name} false-positive rate", pct(rates["fp"])


def check_phrases(data) -> Iterator[Check]:
    """Table 5: the phrase-agreement sweep.

    The sweep runs one step looser than the table reports. That last point is
    plotted in Figure 2 but has no table row to check against.
    """
    for row in data:
        required = row["required"]
        if required < 0.10:
            continue
        yield f"phrase agreement >={required} detection", pct(row["detect"])
        yield f"phrase agreement >={required} false positives", pct(row["fp"])
        if row["frac"] is not None:
            yield f"phrase agreement >={required} FD share", f"{row['frac']:.2f}"


def check_decon(data) -> Iterator[Check]:
    """Sec. 7: decon's Table 2 row, and its own contamination-score sweep.

    decon is run by a separate script, so its Table 2 row is the one place a
    stale result could reach the paper unnoticed. "0 (0.0%)" on its own matches
    several other filters, so match the whole row rather than the counts.
    """
    measure = f"{data['measure']} ({pct_of(data['measure'], data['n_measure'])})"
    control = f"{data['control']} ({pct_of(data['control'], data['n_control'])})"
    yield "decon Table 2 row", f"decon}} & {measure} & {control}"

    sweep = data.get("threshold_sweep", {})
    if "0.2" in sweep:
        yield "decon at threshold 0.2", str(sweep["0.2"])


RESULTS = "python scripts/02_results.py"

# Each results file: the checks it supports, and what writes it.
SOURCES: dict[str, tuple] = {
    "main": (check_main, f"{RESULTS} --stage main"),
    "consensus": (check_consensus, f"{RESULTS} --stage consensus"),
    "scaling": (check_scaling, f"{RESULTS} --stage scaling"),
    "mechanism": (check_mechanism, f"{RESULTS} --stage mechanism"),
    "matched": (check_matched, f"{RESULTS} --stage matched"),
    "nemotron": (check_nemotron, f"{RESULTS} --stage nemotron"),
    "phrases": (check_phrases, f"{RESULTS} --stage phrases"),
    "decon": (check_decon, "python scripts/03_run_decon.py --decon-bin ..."),
}


# ---------------------------------------------------------------------- report


def normalise(tex: str) -> str:
    """Strip LaTeX number formatting so '100{,}000' matches '100000'."""
    plain = tex.replace("{,}", "").replace("\\%", "%").replace("~", " ")
    return re.sub(r"\s+", " ", plain)


def load(name: str):
    path = OUT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    if not PAPER.exists():
        print(f"missing {PAPER}")
        return 1
    tex = normalise(PAPER.read_text(encoding="utf-8"))

    n_found = 0
    missing: list[Check] = []
    skipped: list[tuple[str, str]] = []

    for name, (build, command) in SOURCES.items():
        data = load(name)
        if data is None:
            skipped.append((name, command))
            continue
        for label, needle in build(data):
            if normalise(needle) in tex:
                n_found += 1
            else:
                missing.append((label, needle))

    total = n_found + len(missing)
    print(f"{n_found}/{total} figures found verbatim in the paper")

    # A file that was never generated contributes no checks, so without this its
    # numbers would pass unexamined and the count above would still read clean.
    if skipped:
        print("\nNOT CHECKED -- no results file, so these numbers are unverified:")
        for name, command in skipped:
            print(f"  {name:22s} {command}")

    if missing:
        print("\nNOT FOUND (stale, reworded, or genuinely wrong):")
        for label, needle in missing:
            print(f"  {label:44s} expected: {needle}")
        print("\nA miss is not automatically an error: the paper may phrase a number")
        print("differently, or omit it deliberately. Each one needs a human look.")

    # An unrun stage verifies nothing, so a clean count over an empty results
    # directory would otherwise exit 0 and read as a pass.
    return 1 if missing or skipped else 0


if __name__ == "__main__":
    sys.exit(main())
