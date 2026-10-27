"""Build the anonymised supplementary-material archive.

A hand-rolled zip of the working tree is how .git, a cached corpus, or an author
name reaches a double-blind submission. This builds from an explicit allowlist
instead, refuses to write if it finds an identifying string, and records a
checksum for every file it ships.

    python scripts/07_package.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "submission.zip"

# Only these reach the archive. A glob that matches nothing is an error below,
# so a renamed directory fails loudly rather than shipping a smaller archive.
INCLUDE = [
    "README.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "src/contamination_audit/*.py",
    "scripts/*.py",
    "tests/*.py",
    "paper/paper.tex",
    "paper/figures/*.pdf",
    "results/paper/*.json",
    "results/llm_judge/*",
    # The benchmark groups are small and are what the temporal control is made
    # of. The training corpora are multi-gigabyte downloads and are not shipped.
    "data/raw/bench_*.jsonl",
    "data/raw/olymmath_en_*.jsonl",
]

# This script is build tooling rather than part of the artifact, and its own
# scanner list spells out the names it exists to catch, so packaging it would
# ship exactly what it is meant to prevent.
EXCLUDE = {"scripts/07_package.py"}

# Anything that would deanonymise the submission. Checked against every text
# file actually being written, not against the repository as a whole.
IDENTIFIERS = re.compile(
    r"ademarteau|ademcodesproducts|demarteau|rishbha|ngetich|ischool\.berkeley",
    re.IGNORECASE,
)

TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".tex", ".json", ".jsonl"}

# Hosting links are checked by owner rather than by string. A link to our own
# repository deanonymises the submission as surely as a name in a file does,
# because the commits behind it carry one, and IDENTIFIERS cannot see through a
# URL. Only the upstream projects the filters are reimplemented from belong here.
UPSTREAM_OWNERS = {"allenai"}

REPO_URL = re.compile(
    r"(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/([A-Za-z0-9_.-]+)"
)


def collect() -> list[Path]:
    paths: list[Path] = []
    for pattern in INCLUDE:
        matched = sorted(
            p
            for p in ROOT.glob(pattern)
            if p.is_file() and p.relative_to(ROOT).as_posix() not in EXCLUDE
        )
        if not matched:
            raise SystemExit(f"nothing matched {pattern!r}; fix the allowlist")
        paths.extend(matched)
    return paths


def scan(paths: list[Path]) -> list[str]:
    """Identifying strings found in files bound for the archive."""
    hits = []
    for path in paths:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in set(IDENTIFIERS.findall(text)):
            hits.append(f"{path.relative_to(ROOT)}: {match}")
        for owner in set(REPO_URL.findall(text)) - UPSTREAM_OWNERS:
            hits.append(f"{path.relative_to(ROOT)}: links to repository owner {owner!r}")
    return hits


def main() -> int:
    paths = collect()

    hits = scan(paths)
    if hits:
        print("refusing to package, identifying strings found:")
        for hit in hits:
            print(f"  {hit}")
        return 1

    lines = []
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {name}")
        archive.writestr("MANIFEST.sha256", "\n".join(lines) + "\n")

    size = ARCHIVE.stat().st_size
    print(f"wrote {ARCHIVE.name}  {len(paths) + 1} files  {size / 1e6:.1f} MB")
    print("no identifying strings in any packaged file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
