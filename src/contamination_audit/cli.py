"""The `contamination-audit` command.

    contamination-audit fetch benchmarks dolci     download inputs
    contamination-audit run                        every experiment, in paper order
    contamination-audit run scaling tuning         just these
    contamination-audit decon --binary PATH        run Ai2's decon

Run from the repository root, or pass --root.
"""

from __future__ import annotations

import argparse
import dataclasses
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
        "--corpus",
        choices=("dolci", "tulu", "nemotron", "planted"),
        default="dolci",
        help="which corpus, or 'planted' for recall on the planted control items",
    )
    decon.add_argument(
        "--scaling", action="store_true", help="Dolci only: also run at every scaling size"
    )

    rewrite = commands.add_parser(
        "paraphrase-controls", help="restate the control items with an LLM (paid API calls)"
    )
    rewrite.add_argument("--model", required=True, help="OpenAI model to restate with")
    rewrite.add_argument(
        "--style",
        choices=("faithful", "free"),
        default="faithful",
        help="'free' tells the model not to copy any formula",
    )
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


def paraphrase_controls(args, paths: ProjectPaths) -> int:
    from .data import BenchmarkLibrary
    from .experiments.planted import PARAPHRASE_FILES
    from .paraphrase import paraphrase_controls as run

    form = "paraphrase_free" if args.style == "free" else "paraphrase"
    output = paths.benchmarks / PARAPHRASE_FILES[form]
    run(BenchmarkLibrary.load(paths.benchmarks), output, args.model, args.style)
    print(f"wrote {output}")
    return 0


COMMANDS = {
    "fetch": fetch,
    "run": run,
    "decon": decon,
    "paraphrase-controls": paraphrase_controls,
}
