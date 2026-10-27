# Decontaminated by Whose Standard?

Anonymous research artifact for *Decontaminated by Whose Standard? A Label-Free
Evaluation of Decontamination Filters*.

This repository contains published decontamination filter implementations,
benchmark questions, saved results, and analysis scripts. Temporal controls use
questions published after a training corpus was assembled to measure false
positives under the paper's temporal assumption.

## Contents

| Path | Contents |
|---|---|
| `src/contamination_audit/` | Filter implementations and utilities |
| `scripts/01_prepare_data.py` | Downloads benchmarks, Dolci, and Nemotron |
| `scripts/02_results.py` | Runs analyses and writes result JSON files |
| `scripts/03_run_decon.py` | Runs the separately installed Ai2 decon binary |
| `scripts/04_verify_numbers.py` | Checks an expected manuscript source |
| `scripts/05_figures.py` | Generates scaling and tuning plots |
| `scripts/06_prepare_judge_pairs.py` | Samples candidate pairs for the LLM judge |
| `scripts/06_llm_judge.py` | Labels those pairs; needs `OPENAI_API_KEY` |
| `scripts/06_summarize_judge.py` | Aggregates the labels reported in the paper |
| `data/raw/bench_*.jsonl` | Benchmark groups, including later Nemotron controls |
| `data/raw/olymmath_en_*.jsonl` | OlymMATH benchmark questions |
| `results/paper/*.json` | Saved experiment outputs |
| `results/llm_judge/` | Saved judge pairs, labels and run manifest |
| `paper/paper.tex` | Manuscript source, checked by `04_verify_numbers.py` |
| `paper/figures/*.pdf` | Saved scaling and tuning plots |
| `tests/` | Tests of filter behavior |

Large downloaded training corpora are local caches and are excluded from the
upload ZIP. Third-party names and dataset identifiers are retained for
attribution and reproducibility.

## Installation

Use Python 3.10 or newer. The included scripts run on CPU and do not require GPU
inference or API keys. Dataset and tokenizer downloads require network access.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m spacy download en_core_web_lg
python -m pytest tests/ -q
```

The requirements file includes spaCy for the Tulu tokenizer. Installing only the
editable package does not install that dependency or its language model.

## Data and analyses

Small benchmark files are included. Fetch the large Dolci and Nemotron corpora:

```bash
python scripts/01_prepare_data.py --nemotron
```

Run individual analyses:

```bash
python scripts/02_results.py --stage main
python scripts/02_results.py --stage consensus
python scripts/02_results.py --stage validate
python scripts/02_results.py --stage scaling
python scripts/02_results.py --stage tuning
python scripts/02_results.py --stage phrases
python scripts/02_results.py --stage matched
python scripts/02_results.py --stage nemotron
python scripts/05_figures.py
```

`python scripts/02_results.py --help` lists all twelve stages. Some stages read
earlier results; preserve the included JSON files when running a subset. Running
without `--stage` executes every stage and requires all input corpora.

The existing `second` stage also expects `data/raw/tulu_math.jsonl`, containing a
`problem` field on each record. The preparation script does not download this
file. Its saved results are included in `results/paper/second_corpus.json`;
regeneration requires supplying the corresponding Tulu mathematics corpus in
its original order.

The included CPU pipeline does not reproduce the manuscript's supplementary LLM
judge annotations. The scripts are provided unchanged for subsequent code review.

## decon

Ai2's decon is an external Rust program:

```bash
git clone https://github.com/allenai/decon
cd decon
cargo build --release
cd ..
python scripts/03_run_decon.py --decon-bin decon/target/release/decon
```

Saved outputs are in `results/paper/decon.json`. Working copies of corpora and
reports under `results/paper/decon_run/` are generated locally and excluded.

## Reproduction notes

- Corpus order matters: scaling uses nested prefixes, with up to 390,000 Dolci
  records. A different ordering can change the results.
- Downloads are resumable. Complete the requested download before comparing
  numbers.
- Filters are deterministic and bootstrap calculations use fixed seeds.
  Dataset and tokenizer versions can still affect reproducibility.
- Figures read saved JSON outputs and write to `paper/figures/`.
- `scripts/04_verify_numbers.py` expects `paper/paper.tex`.
  That source is absent from the current artifact; `make verify` requires it.
  The checker does not inspect the separately submitted PDF.
- Makefile targets are conveniences for Unix-like environments; the Python
  commands can also be run directly.

## Saved candidate-pair judge (October 2026)

The optional `scripts/06_llm_judge.py` reviews saved benchmark/training question
pairs through the OpenAI API. The completed experiment includes 565 judgments,
20 repeat checks, and the frozen input and output files. See
[the judge experiment](results/llm_judge/README.md) for results, scope, and API
instructions. Install the optional dependencies with `pip install -e '.[judge]'`.

Rebuild and verify the saved experiment without API calls:

```bash
python scripts/06_prepare_judge_pairs.py --output results/judge/rebuilt_pairs.jsonl
cmp results/judge/rebuilt_pairs.jsonl results/llm_judge/pairs.jsonl
python scripts/06_summarize_judge.py
```

## License

The code is licensed under MIT; see `LICENSE`. Dataset and benchmark contents
remain subject to their upstream licenses and terms.
