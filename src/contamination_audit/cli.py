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

DATASETS = ("benchmarks", "arxiv-dates", "dolci", "tulu", "nemotron")


def row_count(text: str) -> float:
    from .data.download import ALL

    return ALL if text == "all" else int(text)


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
        "--rows",
        type=row_count,
        help="corpus documents to download, or 'all' (default: what the experiments read)",
    )

    run = commands.add_parser("run", help="run experiments and save their results")
    run.add_argument("experiments", nargs="*", metavar="EXPERIMENT", help="default: all of them")
    run.add_argument("--list", action="store_true", help="list the experiments and exit")

    decon = commands.add_parser("decon", help="run Ai2's decon binary over a training corpus")
    decon.add_argument("--binary", type=Path, required=True, help="path to decon's release build")
    decon.add_argument(
        "--corpus", choices=("dolci", "tulu", "nemotron"), default="dolci", help="which corpus"
    )
    decon.add_argument(
        "--scaling", action="store_true", help="Dolci only: also run at every scaling size"
    )

    rewrite = commands.add_parser(
        "paraphrase-controls", help="restate the control items with an LLM (paid API calls)"
    )
    rewrite.add_argument("--model", required=True, help="OpenAI model to restate with")

    pairs = commands.add_parser("judge-pairs", help="rebuild the LLM judge's input pairs")
    pairs.add_argument(
        "--output", type=Path, required=True, help="where to write the pairs (must not exist)"
    )

    annotate = commands.add_parser("judge-annotate", help="write blind sheets for human labels")
    annotate.add_argument("--output", type=Path, required=True, help="directory (must not exist)")
    annotate.add_argument("--size", type=int, default=150, help="pairs to sample")

    agree = commands.add_parser("judge-agreement", help="kappa for completed human labels")
    agree.add_argument("directory", type=Path, help="the directory judge-annotate wrote")

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
    from .data.download import CORPUS_DOWNLOADS, fetch_arxiv_dates, fetch_benchmarks

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        print(f"error: unknown dataset {', '.join(unknown)}; choose from {', '.join(DATASETS)}")
        return 2
    for dataset in args.datasets:
        if dataset == "benchmarks":
            fetch_benchmarks(paths.benchmarks)
        elif dataset == "arxiv-dates":
            fetch_arxiv_dates(paths.benchmarks)
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
    from .decon import DECON_RUNS
    from .experiments import Study

    Study(paths).run(DECON_RUNS[args.corpus], binary=args.binary, scaling=args.scaling)
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


def paraphrase_controls(args, paths: ProjectPaths) -> int:
    from .data import BenchmarkLibrary
    from .experiments.planted import PARAPHRASES_FILE
    from .paraphrase import paraphrase_controls as run

    output = paths.benchmarks / PARAPHRASES_FILE
    run(BenchmarkLibrary.load(paths.benchmarks), output, args.model)
    print(f"wrote {output}")
    return 0


def judge_annotate(args, paths: ProjectPaths) -> int:
    from .judge import JudgeRecord
    from .judge.annotation import write_sheets

    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists")
    write_sheets(JudgeRecord(paths.judge_record), args.output, args.size)
    print(f"wrote annotator_a.csv, annotator_b.csv, codebook.md and key.csv to {args.output}")
    print("give each annotator the codebook and their own sheet; keep key.csv from both")
    return 0


def judge_agreement(args, paths: ProjectPaths) -> int:
    from .judge.annotation import agreement

    print(json.dumps(agreement(args.directory), indent=2))
    return 0


COMMANDS = {
    "fetch": fetch,
    "run": run,
    "decon": decon,
    "paraphrase-controls": paraphrase_controls,
    "judge-pairs": judge_pairs,
    "judge-annotate": judge_annotate,
    "judge-agreement": judge_agreement,
    "judge-verify": judge_verify,
}
