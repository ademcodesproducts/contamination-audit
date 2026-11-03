"""Human labels for a sample of the judged pairs, and how far they agree.

The model's repeat run measures its consistency, not its accuracy. This draws a
stratified sample of the judged pairs for two people to label independently,
with the same definitions the model was given, and reports agreement between
the annotators and between each annotator and the model.

The annotation sheet shows only the two question texts. The model's label, the
pair's temporal group, and which filter retrieved it are kept in a separate key.
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

from .record import JUDGE_SCRIPT, JudgeRecord

LABELS = ("same_problem", "related_but_distinct", "generic_overlap", "unrelated", "uncertain")
SHEET_COLUMNS = ("pair_id", "problem_a", "problem_b", "label", "note")


def label_definitions(record: JudgeRecord) -> str:
    """The label definitions from the prompt the model was given, word for word."""
    script = (record.folder / JUDGE_SCRIPT).read_text(encoding="utf-8")
    prompt = re.search(r'PROMPT = """(.*?)"""', script, flags=re.S).group(1)
    return prompt[prompt.index("Choose one label:") : prompt.index("Also classify")].strip()


def draw_sample(record: JudgeRecord, size: int, seed: int = 7) -> list[dict]:
    """An equal number of pairs per model label, split by group in proportion."""
    rng = random.Random(seed)
    groups = {
        pair["pair_id"]: "control"
        if all(i["group"] == "control" for i in pair["benchmark_items"])
        else "measure"
        for pair in record.pairs
    }
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in record.labels:
        by_cell.setdefault((row["judgment"]["label"], groups[row["pair_id"]]), []).append(row)

    labels = sorted({label for label, _ in by_cell})
    per_label = size // len(labels)
    sample = []
    for label in labels:
        cells = {group: rows for (lab, group), rows in by_cell.items() if lab == label}
        total = sum(len(rows) for rows in cells.values())
        for group, rows in sorted(cells.items()):
            take = round(per_label * len(rows) / total)
            sample += [(row, group) for row in rng.sample(rows, min(take, len(rows)))]
    rng.shuffle(sample)
    return [
        {
            "pair_id": row["pair_id"],
            "model_label": row["judgment"]["label"],
            "group": group,
            "row": row,
        }
        for row, group in sample
    ]


def write_sheets(record: JudgeRecord, directory: Path, size: int = 150) -> None:
    """One blank sheet per annotator, a codebook, and the key, kept apart."""
    directory.mkdir(parents=True, exist_ok=True)
    sample = draw_sample(record, size)
    rng = random.Random(11)
    for annotator in ("annotator_a", "annotator_b"):
        with open(directory / f"{annotator}.csv", "x", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(SHEET_COLUMNS)
            for entry in sample:
                texts = [entry["row"]["benchmark_problem"], entry["row"]["training_problem"]]
                rng.shuffle(texts)  # which text came from the benchmark is not shown
                writer.writerow([entry["pair_id"], *texts, "", ""])
    with open(directory / "key.csv", "x", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(("pair_id", "model_label", "group"))
        writer.writerows((e["pair_id"], e["model_label"], e["group"]) for e in sample)
    (directory / "codebook.md").write_text(
        "# Labelling pairs of questions\n\n"
        "Each row holds two problem statements. Choose the label that describes how they are "
        "related, using only the two texts. Write it in the `label` column exactly as spelled "
        f"below: {', '.join(LABELS)}. Use `note` for anything unclear.\n\n"
        "Label alone, without discussing rows with the other annotator.\n\n"
        f"```\n{label_definitions(record)}\n```\n",
        encoding="utf-8",
        newline="\n",
    )


def cohen_kappa(first: list[str], second: list[str]) -> float:
    """Agreement beyond what the two raters' label frequencies predict by chance."""
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    counts_a, counts_b = Counter(first), Counter(second)
    expected = sum(counts_a[label] * counts_b[label] for label in counts_a) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def read_labels(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as sheet:
        rows = {row["pair_id"]: row["label"].strip() for row in csv.DictReader(sheet)}
    unknown = sorted({label for label in rows.values() if label not in LABELS})
    if unknown:
        raise ValueError(f"{path}: labels not in the codebook: {unknown}")
    return rows


def agreement(directory: Path) -> dict:
    """Kappa between the annotators, and between each annotator and the model."""
    a = read_labels(directory / "annotator_a.csv")
    b = read_labels(directory / "annotator_b.csv")
    with open(directory / "key.csv", encoding="utf-8", newline="") as sheet:
        model = {row["pair_id"]: row["model_label"] for row in csv.DictReader(sheet)}
    ids = sorted(model)
    columns = {
        "annotator_a": [a[i] for i in ids],
        "annotator_b": [b[i] for i in ids],
        "model": [model[i] for i in ids],
    }
    agreed = [i for i in ids if a[i] == b[i]]
    return {
        "pairs": len(ids),
        "kappa": {
            "annotators": cohen_kappa(columns["annotator_a"], columns["annotator_b"]),
            "annotator_a_vs_model": cohen_kappa(columns["annotator_a"], columns["model"]),
            "annotator_b_vs_model": cohen_kappa(columns["annotator_b"], columns["model"]),
        },
        # Where the annotators agree, how often the model gives the same label.
        "model_matches_agreed_label": sum(model[i] == a[i] for i in agreed) / len(agreed)
        if agreed
        else None,
        "human_same_problem": sum(a[i] == b[i] == "same_problem" for i in ids),
        "labels": {name: dict(Counter(values)) for name, values in columns.items()},
    }
