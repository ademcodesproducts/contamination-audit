# Artifact: *Decontaminated by Whose Standard?*

Everything needed to reproduce the paper. All results run on CPU; no GPU and no
API keys are required.

## What is here

| Component | Location |
|---|---|
| The five filter reimplementations | `src/contamination_audit/filters.py` |
| Temporal control benchmark sets | `data/raw/bench_measure.jsonl`, `data/raw/bench_control.jsonl` |
| OlymMATH subsets | `data/raw/olymmath_en_{hard,easy}.jsonl` |
| Data acquisition | `scripts/19_prepare_paper_data.py` |
| Every paper number | `scripts/20_paper_results.py` |
| decon wrapper | `scripts/21_run_decon.py` |
| Figures | `scripts/98_make_paper_figures.py` |
| Regenerated results | `results/paper/*.json` |

The benchmark groups are committed because they are the paper's own contribution
and total under 400 KB. The training corpus is not: it is roughly 300 MB and
regenerates from HuggingFace.

## The two benchmark groups

This split is the paper's method and is defined in
`scripts/19_prepare_paper_data.py`.

- **Measurement** (502 items): OlymMATH plus ten MathArena competitions held on or
  before September 2025. These could plausibly be inside a corpus assembled by
  October 2025, so a flag may be correct or incorrect.
- **Control** (280 items): eight MathArena sets from January to August 2026. These
  postdate the corpus, so they cannot be in it and **every flag against them is a
  false positive by construction** — no annotation required.

Olmo 3 was announced 2025-11-20. We treat Dolci as closed by roughly October 2025
and leave the ambiguous October–December 2025 band out of both groups.

## Reproducing

```bash
pip install -e .
make paper-data        # ~30 min: benchmarks are fast; the corpus streams from HF
make paper-results     # ~50 min single-threaded; writes results/paper/*.json
make paper-figures     # seconds
```

`make paper-results` prints each number in the form the paper states it, so the
two can be diffed directly. Individual sections:

```bash
python scripts/20_paper_results.py --stage main         # Table 2
python scripts/20_paper_results.py --stage consensus    # Sec. 6
python scripts/20_paper_results.py --stage scaling      # Fig. 1  (slowest, ~25 min)
python scripts/20_paper_results.py --stage tuning       # Fig. 2  (~12 min)
python scripts/20_paper_results.py --stage mechanism    # Sec. 9
python scripts/20_paper_results.py --stage composition  # Sec. 11
```

### decon

decon is Ai2's Rust tool and is deliberately **not** reimplemented. Build the real
one:

```bash
git clone https://github.com/allenai/decon && cd decon && cargo build --release
cd - && make paper-decon DECON_BIN=decon/target/release/decon
```

We stress this because approximating it fails badly. Reproducing only its
candidate-generation stage — any shared 5-gram over `cl100k` tokens, without the
cluster expansion and weighted scoring — flags 199 of 200 items where the real
binary flags 0. A two-stage filter cannot be approximated by its first stage. The
registry marks partial implementations `faithful=False` and excludes them from
comparisons by default.

## Notes on reproduction

- **Streaming is flaky.** `scripts/19` resumes by skipping rows already on disk;
  rerun it if a pull drops.
- **Corpus order matters.** All scaling points take the first *N* rows of
  `data/raw/dolci_questions.jsonl`, so a partial pull shifts every number. Pull the
  full 390,000 before comparing against the paper.
- **Runtimes are single-threaded Python.** The Qwen2 tokenizer dominates; the 390k
  scaling point alone takes roughly 25 minutes.
- **Determinism.** All sampling and bootstrapping is seeded (42). The filters are
  deterministic; no LLM is called anywhere in the pipeline.

## Relationship to the rest of the repository

This repository also contains a superseded draft (`paper/iclr2026_conference.tex`,
`paper/Final_submission_ANLP_Sp26.pdf`) and its pipeline, reachable through the
`legacy-*` make targets. None of it feeds the current paper, and several of its
claims did not survive re-analysis — see the README. Artifact reviewers should
ignore everything outside the table above.
