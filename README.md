# Contamination Audit

Code for **"Are Reasoning Model Benchmark Gains Real? Measuring the Inflationary Effect of Decontamination Failures"** (Demarteau, Jain, Ngetich · ANLP Sp26 / ICLR 2026 submission).

> Open reasoning model post-training has produced striking gains on established mathematical benchmarks, yet the validity of these gains depends critically on the integrity of decontamination pipelines that are meant to prevent training-test overlap. We conduct a systematic audit of three prominent open post-training projects — s1, Tülu 3, and OpenThoughts — each of which claimed decontamination against MATH-500 before release. Using a multi-stage detection pipeline combining n-gram replication, dense semantic embeddings, and LLM-as-judge verification, we identify benchmark items that survived each project's decontamination despite meaningful overlap with training data. We then conduct behavioral analyses via perturbation testing and chain-of-thought structural comparison, finding that contamination manifests as solution-schema brittleness rather than verbatim recitation: contaminated problems produce a 37.5% answer-abandonment rate under numerical perturbation versus 20.8% for clean problems, alongside measurably fewer self-corrections and higher math token density in reasoning traces.

The full PDF is at [`paper/Final_submission_ANLP_Sp26.pdf`](paper/Final_submission_ANLP_Sp26.pdf).

## Repository layout

```
.
├── src/contamination_audit/   # importable library (no CLI logic)
│   ├── ngram.py, tfidf.py, embedding.py   # stage-2 retrieval (paper §5.1 — per-project method)
│   ├── judge.py                            # multi-provider LLM judge (OpenAI, Vertex/Gemini, ...)
│   ├── inference.py                        # HF transformers DiD driver (paper §5.3)
│   ├── did.py, null_rate.py,               # paper Tables 4-7 + Figures 1-2
│   ├── cot_features.py, recitation.py,
│   ├── stats.py, answer_extract.py
│   └── prompts/                            # judge_default, judge_strict, judge_tulu, ...
├── scripts/                   # numbered entry points (one per pipeline stage)
├── configs/                   # datasets.yaml, thresholds.yaml, models.yaml
├── data/
│   ├── raw/                   # downloaded datasets (gitignored — regenerable from HF)
│   └── processed/             # C_lex / C_sem sets, perturbations, clean baseline, annotations
├── results/
│   ├── ngram/, embeddings/,   # stage-1/2/3 bulk outputs (gitignored)
│   │   judge/, traces/
│   ├── tables/                # paper tables 2-7 as CSV (tracked)
│   ├── figures/               # paper figures (tracked)
│   └── annotations/           # manual-validation worksheets (tracked)
├── notebooks/                 # canonical analysis + inference notebooks + exploratory probes
├── paper/                     # LaTeX source + final PDF
└── tests/                     # pytest suite (22 tests)
```

## Setup

Requires Python ≥ 3.10. Inference uses a single NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM, bf16); detection stages run on CPU.

```bash
python -m venv venv
venv\Scripts\activate                 # on Windows; or: source venv/bin/activate
pip install -e .                      # installs contamination_audit + deps
cp .env.example .env                  # fill in OPENAI_API_KEY (and optionally GOOGLE_CLOUD_PROJECT for Vertex)
```

## Reproducing the paper

```bash
make all
```

Or step-by-step:

```bash
python scripts/00_load_datasets.py
python scripts/01_ngram_filter.py
python scripts/02_semantic_retrieval.py            # dense (s1/OT) or TF-IDF (Tulu), auto per config
python scripts/03_llm_judge.py                     # OpenAI or Vertex/Gemini per project
python scripts/04_build_clean_set.py
python scripts/05_validate_and_report.py --crosscheck

# Behavioral stages (require GPU)
python scripts/06_generate_perturbations.py        # stub — see notebooks/temperature_variance.ipynb
python scripts/07_run_inference.py                 # OpenThinker-7B, Tulu-3-8B-SFT, s1.1-7B
python scripts/08_score_answers.py results/traces/openthoughts_traces.jsonl
python scripts/08_score_answers.py results/traces/tulu_traces.jsonl
python scripts/08_score_answers.py results/traces/s1_traces.jsonl

# Analysis (no GPU)
python scripts/09_compute_did.py                   # Tables 4 + 5
python scripts/10_compute_null_rate.py             # Table 6 + Figure 1
python scripts/11_cot_features.py                  # Table 7 + Figure 2
python scripts/12_robustness_checks.py             # failure_mode + tulu_sources + n-gram sweep
python scripts/13_build_annotation_csv.py
python scripts/14_recitation_analysis.py           # §7.2 shortcut-vs-recall breakdown
python scripts/99_make_pipeline_figure.py
```

## Paper-to-script map

| Paper artifact | Producing script | Output file |
|---|---|---|
| Table 1 (eval set composition) | `scripts/05_validate_and_report.py` | `results/tables/table2_contamination_counts.csv` (derived) |
| Table 2 (contamination counts) | `scripts/05_validate_and_report.py` | `results/tables/table2_contamination_counts.csv` |
| Table 3 (model configs) | static | `configs/models.yaml` |
| Table 4 (accuracy by split/perturbation) | `scripts/09_compute_did.py` | `results/tables/table4_accuracy.csv` |
| Table 5 (DiD estimates) | `scripts/09_compute_did.py` | `results/tables/table5_did.csv` |
| Table 6 (null rates) | `scripts/10_compute_null_rate.py` | `results/tables/table6_null_rate.csv` |
| Table 7 (CoT features) | `scripts/11_cot_features.py` | `results/tables/table7_cot_features.csv` |
| §7.2 recitation breakdown | `scripts/14_recitation_analysis.py` | `results/tables/table_recitation.csv` |
| Failure-mode comparison | `scripts/12_robustness_checks.py --check failure_mode` | `results/tables/failure_mode_comparison.csv` |
| Tülu source breakdown | `scripts/12_robustness_checks.py --check tulu_sources` | `results/tables/tulu_source_breakdown.csv` |
| n-gram robustness sweep (n=15, n=20) | `scripts/12_robustness_checks.py --check ngram_sweep` | `results/tables/robustness_ngram_sweep.csv` |
| Figure 1 (null rate bars) | `scripts/10_compute_null_rate.py` | `results/figures/figure1_null_rates.png` |
| Figure 2 (CoT distributions) | `scripts/11_cot_features.py` | `results/figures/figure2_cot_distributions.png` |
| Cohen's d heatmap | `scripts/11_cot_features.py` | `results/figures/figure2b_cohens_d.png` |
| Pipeline figure | `scripts/99_make_pipeline_figure.py` | `results/figures/pipeline_figure.png` |
| Cross-check (caught-by-both vs semantic-only) | `scripts/05_validate_and_report.py --crosscheck` | `results/tables/crosscheck.csv` |
| Annotation worksheet (C_sem) | `scripts/13_build_annotation_csv.py` | `results/annotations/csem_annotation.csv` |

Every numbered script has a matching notebook in `notebooks/` — see [`notebooks/README.md`](notebooks/README.md). When the script and notebook disagree, the notebook is canonical.

## Stage-2 retrieval (per paper §5.1)

The retrieval method is selected per project in `configs/thresholds.yaml`:

| Project | Method | Rationale |
|---|---|---|
| s1, OpenThoughts | Dense (`all-mpnet-base-v2` + FAISS, sim ≥ 0.70) | Captures paraphrase + topical similarity |
| Tülu 3 | TF-IDF (char-ngrams 3–5, density-aware top-K) | Sharper on LaTeX-heavy competition math; teammate src/02 baseline |

## LLM judges (per paper §5.1)

| Project | Default judge | Prompt | Provider |
|---|---|---|---|
| s1, OpenThoughts | GPT-4o-mini | `judge_default` | OpenAI |
| Tülu 3 | Gemini 2.5 Flash | `judge_tulu` (instance/template/clean) | Vertex AI |

Other providers wired up under `judges:` and `JudgeConfig.provider`: `openai`, `google` (AI Studio), `vertex`, `openai_compat` (Groq, Cerebras, DeepInfra, NVIDIA NIM).

## ⚠ Open question for co-authors

The paper §5.4 specifies **GPT-4o-mini** as the answer-equivalence re-judge that produced Tables 3–4. The teammate `src/06_llm_judge_correctness.py` (canonical source) uses **Vertex/Gemini 2.5 Flash** by default. Before publication, confirm with Rishbha + Victor which judge actually produced the numbers in the submitted PDF. Both are wired up under `configs/models.yaml > answer_judges`.

## Configuration

All hyperparameters live in `configs/thresholds.yaml`:

- N-gram size + threshold mode per project (s1: 8/any, Tülu: 8/percent50, OpenThoughts: 13/any)
- Retrieval method per project (dense vs TF-IDF)
- Embedding model, similarity threshold (0.70), top-K (5)
- Per-judge config (provider, model, retries, temperature, rate limit) — `judges.default`, `judges.strict`, `judges.tulu`
- Bootstrap iterations (10,000) and seed (42)

The dataset registry lives in `configs/datasets.yaml`; the model registry in `configs/models.yaml`.

## Tests

```bash
pytest tests/
```

22 tests cover the silent-failure paths most likely to break a paper number: n-gram extraction + coverage, FAISS retrieval determinism, judge JSON parsing (incl. regex fallback for malformed LaTeX-laden responses), and `\boxed{}` answer extraction + equivalence.

## License

MIT — see [LICENSE](LICENSE).
