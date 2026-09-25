"""Run Ai2's decon (Olmo 3's decontamination tool) over the paper's inputs.

decon is a Rust binary, not a Python filter, and it is deliberately NOT
reimplemented here. An earlier attempt reproduced only its candidate-generation
stage -- any shared 5-gram over cl100k tokens -- which flagged 199/200 items where
the real tool flags 0. The cluster expansion and weighted scoring it applies on
top are what turn candidates into a verdict, so a partial reimplementation
inverts the result. Build the real thing:

    git clone https://github.com/allenai/decon
    cd decon && cargo build --release

then point --decon-bin at target/release/decon (or decon.exe on Windows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
from pathlib import Path

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import load_jsonl

_log = logging.getLogger("run_decon")

THRESHOLDS = [0.8, 0.6, 0.4, 0.2]   # 0.8 is decon's own default


def write_evals(rows: list[dict], group: str, out_dir: Path) -> int:
    """decon expects its own schema; doc_id and eval_instance_index must be ints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bench.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "eval_key": r.get("benchmark", "olymmath"),
                "eval_instance_index": i,
                "split": group,
                "question": r["problem"],
                "answer": str(r.get("answer", "")),
                "config": r.get("benchmark", "olymmath"),
                "doc_id": i,
                "is_correct": True,
                "fingerprint": hashlib.md5(r["problem"].encode()).hexdigest()[:16],
            }) + "\n")
    return len(rows)


def write_training(corpus_rows: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out_dir / "dolci.jsonl").open("w", encoding="utf-8") as out, \
         open("data/raw/dolci_questions.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= corpus_rows:
                break
            r = json.loads(line)
            out.write(json.dumps({"text": r["problem"], "id": r["train_id"],
                                  "source": r["source"]}) + "\n")
            n += 1
    return n


def run(binary: str, train_dir: Path, evals_dir: Path, report_dir: Path,
        threshold: float | None) -> int:
    cmd = [binary, "detect", "--training-dir", str(train_dir),
           "--content-key", "text", "--evals-dir", str(evals_dir),
           "--report-output-dir", str(report_dir)]
    if threshold is not None:
        cmd += ["--contamination-score-threshold", str(threshold)]
    # decon draws box characters; Windows' default codec cannot decode them.
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    m = re.search(r"Contaminated documents\s*│?\s*([\d,]+)", res.stdout)
    if not m:
        _log.error("could not parse decon output:\n%s", res.stdout[-800:])
        raise SystemExit(1)
    return int(m.group(1).replace(",", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decon-bin", required=True,
                    help="path to decon's release binary")
    ap.add_argument("--corpus-rows", type=int, default=100_000)
    ap.add_argument("--work", default="results/paper/decon_run")
    args = ap.parse_args()
    configure_logging()

    work = Path(args.work)
    olym = [{"id": r["id"], "benchmark": "olymmath", "problem": r["problem"],
             "answer": r.get("answer", "")}
            for r in load_jsonl("data/raw/olymmath_en_hard.jsonl")
            + load_jsonl("data/raw/olymmath_en_easy.jsonl")]
    measure = olym + load_jsonl("data/raw/bench_measure.jsonl")
    control = load_jsonl("data/raw/bench_control.jsonl")

    n_train = write_training(args.corpus_rows, work / "training")
    n_m = write_evals(measure, "measure", work / "evals_measure")
    n_c = write_evals(control, "control", work / "evals_control")
    _log.info("training=%d  measurement=%d  control=%d", n_train, n_m, n_c)

    out = {"corpus_rows": n_train, "n_measure": n_m, "n_control": n_c}

    print("\ndecon at its own defaults")
    for group, n in (("measure", n_m), ("control", n_c)):
        hits = run(args.decon_bin, work / "training", work / f"evals_{group}",
                   work / f"report_{group}", None)
        out[group] = hits
        print(f"  {group:12s} {hits:5d}/{n} contaminated documents")

    print("\ndecon score-threshold sweep (measurement set; default is 0.8)")
    sweep = {}
    for t in THRESHOLDS:
        hits = run(args.decon_bin, work / "training", work / "evals_measure",
                   work / f"report_t{t}", t)
        sweep[t] = hits
        print(f"  threshold {t:.1f}  {hits:6d} contaminated documents")
    out["threshold_sweep"] = sweep

    dest = Path("results/paper/decon.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    _log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
