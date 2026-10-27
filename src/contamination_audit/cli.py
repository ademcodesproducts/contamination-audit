"""The `contamination-audit` command.

    contamination-audit fetch benchmarks dolci     download inputs
    contamination-audit run                        every experiment, in paper order
    contamination-audit run scaling tuning         just these
    contamination-audit decon --binary PATH        run Ai2's decon
    contamination-audit judge-verify               check the frozen LLM-judge record

Run from the repository root, or pass --root.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from .errors import IntegrityError, MissingDataError
from .paths import ProjectPaths

DATASETS = ("benchmarks", "dolci", "tulu", "nemotron")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contamination-audit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument(
        "--results-dir", type=Path, help="read and write results here instead of <root>/results"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="log warnings only")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    fetch = commands.add_parser("fetch", help="download benchmarks and training corpora")
    fetch.add_argument(
        "datasets", nargs="+", metavar="DATASET", help=f"any of: {', '.join(DATASETS)}"
    )
    fetch.add_argument(
        "--rows", type=int, help="corpus documents to download (default: what the experiments read)"
    )

    run = commands.add_parser("run", help="run experiments and save their results")
    run.add_argument("experiments", nargs="*", metavar="EXPERIMENT", help="default: all of them")
    run.add_argument("--list", action="store_true", help="list the experiments and exit")

    decon = commands.add_parser("decon", help="run Ai2's decon binary over the main corpus")
    decon.add_argument("--binary", type=Path, required=True, help="path to decon's release build")
    decon.add_argument(
        "--scaling", action="store_true", help="also run at every scaling corpus size"
    )

    pairs = commands.add_parser("judge-pairs", help="rebuild the LLM judge's input pairs")
    pairs.add_argument(
        "--output", type=Path, required=True, help="where to write the pairs (must not exist)"
    )

    verify = commands.add_parser("judge-verify", help="verify the frozen LLM-judge record")
    verify.add_argument("--output-dir", type=Path, help="also write the rebuilt summaries here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = ProjectPaths.at(args.root)
    if args.results_dir:
        paths = dataclasses.replace(paths, results=args.results_dir)

    try:
        return COMMANDS[args.command](args, paths)
    except (MissingDataError, IntegrityError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------- commands


def fetch(args, paths: ProjectPaths) -> int:
    from .data.download import CORPUS_DOWNLOADS, fetch_benchmarks

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        print(f"error: unknown dataset {', '.join(unknown)}; choose from {', '.join(DATASETS)}")
        return 2
    for dataset in args.datasets:
        if dataset == "benchmarks":
            fetch_benchmarks(paths.benchmarks)
        else:
            CORPUS_DOWNLOADS[dataset].fetch(paths.corpora, rows=args.rows)
    return 0


def run(args, paths: ProjectPaths) -> int:
    from .experiments import EXPERIMENTS, Study

    if args.list:
        for name, experiment in EXPERIMENTS.items():
            print(f"  {name:12s} {experiment.title}")
        return 0
    unknown = [name for name in args.experiments if name not in EXPERIMENTS]
    if unknown:
        print(f"error: unknown experiment {', '.join(unknown)}; see `run --list`")
        return 2

    study = Study(paths)
    for name in args.experiments or EXPERIMENTS:
        study.run(EXPERIMENTS[name])
    return 0


def decon(args, paths: ProjectPaths) -> int:
    from .decon import DeconRun
    from .experiments import Study

    Study(paths).run(DeconRun, binary=args.binary, scaling=args.scaling)
    return 0


def judge_pairs(args, paths: ProjectPaths) -> int:
    from .jsonio import read_json
    from .judge import CandidatePairBuilder

    # The pairs carry decon's verdict as a fixed negative, which is only true
    # while the saved decon run found nothing.
    decon = read_json(paths.paper_results / "decon.json")
    if decon["measure"] or decon["control"]:
        raise IntegrityError(
            "the judge snapshot assumes decon flagged nothing, but decon.json disagrees"
        )

    matches = read_json(paths.judge_record / "candidate_matches.json")
    pairs = CandidatePairBuilder(matches).build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x", encoding="utf-8", newline="\n") as out:
        for pair in pairs:
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {args.output}")
    return 0


def judge_verify(args, paths: ProjectPaths) -> int:
    from .judge import JudgeRecord

    record = JudgeRecord(paths.judge_record)
    record.verify()
    summaries = record.summaries()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, summary in summaries.items():
            with open(args.output_dir / name, "x", encoding="utf-8", newline="\n") as out:
                out.write(json.dumps(summary, indent=2) + "\n")
    print(f"verified {len(record.labels)} judgments and {len(record.review_labels)} repeats")
    print("labels:", json.dumps(summaries["summary.json"]["labels"]))
    print("repeat agreement:", json.dumps(summaries["review_summary.json"]["random_sample"]))
    return 0


COMMANDS = {
    "fetch": fetch,
    "run": run,
    "decon": decon,
    "judge-pairs": judge_pairs,
    "judge-verify": judge_verify,
}
