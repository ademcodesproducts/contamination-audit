"""
Step 1: Download MATH-500 and s1K datasets, save locally as JSON.
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _row_to_problem(i, row, subject_field="subject"):
    return {
        "id": i,
        "problem": row.get("problem", row.get("question", "")),
        "solution": row.get("solution", row.get("answer", "")),
        "answer": row.get("answer", ""),
        "subject": row.get(subject_field, ""),
        "level": row.get("level", ""),
    }


def download_math500():
    out_path = DATA_DIR / "math500.json"
    if out_path.exists():
        print(f"MATH-500 already exists at {out_path}, skipping.")
        return

    print("Downloading MATH-500...")

    # Try known working dataset identifiers in order
    attempts = [
        # (dataset_id, split, subject_field, trim_to_500)
        ("HuggingFaceH4/MATH-500",       "test",  "subject", False),
        ("TIGER-Lab/MATH-500",            "test",  "subject", False),
        ("EleutherAI/hendrycks_math",     "test",  "subject", True),
        ("competition_math",              "test",  "type",    True),
        ("math",                          "test",  "type",    True),
    ]

    ds = None
    subject_field = "subject"
    trim = False
    for dataset_id, split, sf, should_trim in attempts:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split=split)
            subject_field = sf
            trim = should_trim
            print(f"  Success: {dataset_id} ({len(ds)} items)")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError(
            "Could not download MATH-500 from any known source.\n"
            "Please manually download from https://huggingface.co/datasets/HuggingFaceH4/MATH-500\n"
            "and save the 'test' split as data/math500.json"
        )

    problems = [_row_to_problem(i, row, subject_field) for i, row in enumerate(ds)]

    if trim and len(problems) > 500:
        print(f"  Trimming {len(problems)} → 500 items for pilot.")
        problems = problems[:500]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(problems)} MATH-500 items to {out_path}")


def download_s1k():
    out_path = DATA_DIR / "s1k.json"
    if out_path.exists():
        print(f"s1K already exists at {out_path}, skipping.")
        return

    print("Downloading s1K...")
    s1k_attempts = ["simplescaling/s1K", "simplescaling/s1k"]
    ds = None
    for dataset_id in s1k_attempts:
        try:
            print(f"  Trying {dataset_id} ...")
            ds = load_dataset(dataset_id, split="train")
            print(f"  Success ({len(ds)} items). Columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not download s1K from HuggingFace.")

    problems = [
        {
            "id": i,
            "problem": row.get("problem", row.get("question", "")),
            "solution": row.get("solution", row.get("answer", "")),
            "thinking": row.get("thinking", row.get("reasoning", "")),
            "source": row.get("source", row.get("dataset", "")),
            "subject": row.get("subject", row.get("category", "")),
        }
        for i, row in enumerate(ds)
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(problems)} s1K items to {out_path}")


if __name__ == "__main__":
    download_math500()
    download_s1k()
    print("\nDone. Data saved to ./data/")
