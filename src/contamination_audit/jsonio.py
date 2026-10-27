"""Reading and writing JSON and JSON Lines.

Files are always written with LF line endings, whatever the platform, because
the judge record verifies its files against SHA-256 hashes and a CRLF rewrite
would break them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Records from a JSON Lines file, in order, stopping after `limit`."""
    with open(path, encoding="utf-8") as lines:
        records = (json.loads(line) for line in lines if line.strip())
        yield from islice(records, limit)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, limit))


def write_jsonl(records: Iterable[dict[str, Any]], path: Path, *, ensure_ascii: bool = True) -> int:
    """Write one record per line and return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    return count


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(value: Any, path: Path, *, indent: int = 1) -> None:
    """Pretty-printed JSON, the format every file under results/paper/ uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=indent), encoding="utf-8", newline="\n")
