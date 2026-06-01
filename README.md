# Contamination Audit

Code for **"Are Reasoning Model Benchmark Gains Real? Measuring the Inflationary Effect of Decontamination Failures"** (Demarteau, Jain, Ngetich · ANLP Sp26 / ICLR 2026 submission).

> Open reasoning model post-training has produced striking gains on established mathematical benchmarks, yet the validity of these gains depends critically on the integrity of decontamination pipelines that are meant to prevent training-test overlap. We conduct a systematic audit of three prominent open post-training projects — s1, Tülu 3, and OpenThoughts — each of which claimed decontamination against MATH-500 before release. Using a multi-stage detection pipeline combining n-gram replication, dense semantic embeddings, and LLM-as-judge verification, we identify benchmark items that survived each project's decontamination despite meaningful overlap with training data. We then conduct behavioral analyses via perturbation testing and chain-of-thought structural comparison, finding that contamination manifests as solution-schema brittleness rather than verbatim recitation: contaminated problems produce a 37.5% answer-abandonment rate under numerical perturbation versus 20.8% for clean problems, alongside measurably fewer self-corrections and higher math token density in reasoning traces.

The full PDF is at [`paper/Final_submission_ANLP_Sp26.pdf`](paper/Final_submission_ANLP_Sp26.pdf).

## Repository layout

```
.
├── src/contamination_audit/   # importable library (no CLI logic)
├── scripts/                   # thin numbered entry points, one per pipeline stage
├── configs/                   # YAML — datasets, thresholds, models
├── data/
│   ├── raw/                   # downloaded datasets (gitignored — regenerable from HF)
│   └── processed/             # C_lex / C_sem sets, clean baseline, annotation worksheets
├── results/
│   ├── ngram/                 # stage-1 hits per project (gitignored bulk)
│   ├── embeddings/            # stage-2 candidates + .npy caches (gitignored bulk)
│   ├── judge/                 # stage-3 LLM-judge outputs (gitignored bulk)
│   ├── traces/                # model inference traces (gitignored — produced by notebooks)
│   ├── tables/                # paper tables as CSV (tracked)
│   ├── figures/               # paper figures (tracked)
│   └── annotations/           # manual-validation worksheets (tracked)
├── notebooks/                 # external notebooks for stages 6,7,9,10,11 (see notebooks/README.md)
├── paper/                     # LaTeX source + final PDF
└── tests/                     # pytest suite
```

## Setup

Requires Python ≥ 3.10. The full pipeline needs a single NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM) for the inference stage; detection stages run on CPU.

```bash
python -m venv venv
source venv/bin/activate          # or: venv\Scripts\activate on Windows
pip install -e .                  # installs the contamination_audit package + deps
cp .env.example .env              # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

## Reproducing the paper

```bash
make all                          # full pipeline; stops at stub stages with [stub] notice
```

Or step-by-step:

```bash
python scripts/00_load_datasets.py
python scripts/01_ngram_filter.py
python scripts/02_embedding_retrieval.py
python scripts/03_llm_judge.py
python scripts/04_build_clean_set.py
python scripts/05_validate_and_report.py --crosscheck   # writes Table 2 + crosscheck
python scripts/12_robustness_checks.py                  # writes failure_mode + tulu_sources + n-gram sweep
python scripts/13_build_annotation_csv.py
python scripts/99_make_pipeline_figure.py
```

Stages 6, 7, 9, 10, 11 currently raise `NotImplementedError` — see [`notebooks/README.md`](notebooks/README.md) for the contract.

## Paper-to-script map

| Paper artifact | Producing script | Output file |
|---|---|---|
| Table 1 (eval set composition) | `scripts/05_validate_and_report.py` | `results/tables/table2_contamination_counts.csv` (derived) |
| Table 2 (contamination counts) | `scripts/05_validate_and_report.py` | `results/tables/table2_contamination_counts.csv` |
| Table 3 (model configs) | static | `configs/models.yaml` |
| Table 4 (accuracy by split/perturbation) | `scripts/09_compute_did.py` (stub) | `results/tables/table4_accuracy.csv` |
| Table 5 (DiD estimates) | `scripts/09_compute_did.py` (stub) | `results/tables/table5_did.csv` |
| Table 6 (null rates) | `scripts/10_compute_null_rate.py` (stub) | `results/tables/table6_null_rate.csv` |
| Table 7 (CoT features) | `scripts/11_cot_features.py` (stub) | `results/tables/table7_cot_features.csv` |
| Failure-mode comparison | `scripts/12_robustness_checks.py --check failure_mode` | `results/tables/failure_mode_comparison.csv` |
| Tülu source breakdown | `scripts/12_robustness_checks.py --check tulu_sources` | `results/tables/tulu_source_breakdown.csv` |
| n-gram robustness sweep (n=15, n=20) | `scripts/12_robustness_checks.py --check ngram_sweep` | `results/tables/robustness_ngram_sweep.csv` |
| Figure 1 (null rate bars) | `scripts/10_compute_null_rate.py` (stub) | `results/figures/figure1_null_rates.png` |
| Figure 2 (CoT distributions) | `scripts/11_cot_features.py` (stub) | `results/figures/figure2_cot_distributions.png` |
| Pipeline figure | `scripts/99_make_pipeline_figure.py` | `results/figures/pipeline_figure.png` |
| Cross-check (caught-by-both vs semantic-only) | `scripts/05_validate_and_report.py --crosscheck` | `results/tables/crosscheck.csv` |
| Annotation worksheet (C_sem) | `scripts/13_build_annotation_csv.py` | `results/annotations/csem_annotation.csv` |

(Stub) scripts will be filled in as `notebooks/*.ipynb` are linearized — see `notebooks/README.md`.

## Configuration

All hyperparameters live in `configs/thresholds.yaml`:

- N-gram size and threshold mode per project (s1: 8/any, Tülu: 8/percent50, OpenThoughts: 13/any)
- Embedding model, similarity threshold (0.70), top-K (5)
- Judge model (gpt-4o-mini), retries, temperature
- Bootstrap iterations (10,000) and seed (42)

Override by editing the YAML and re-running. The dataset registry lives in `configs/datasets.yaml`; the model registry in `configs/models.yaml`.

## Tests

```bash
pytest tests/
```

Four files cover the silent-failure paths most likely to break Table 5 / Figure 1: n-gram extraction + coverage, FAISS retrieval determinism, judge JSON parsing (incl. regex fallback), and `\boxed{}` answer extraction + equivalence.

## License

MIT — see [LICENSE](LICENSE).
