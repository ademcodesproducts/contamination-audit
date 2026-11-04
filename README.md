# Decontaminated by Whose Standard?

Code and data for *Decontaminated by Whose Standard? A Label-Free Evaluation of
Decontamination Filters* ([arXiv](https://arxiv.org/abs/XXXX.XXXXX)).

Open post-training datasets routinely report that they were decontaminated
against public benchmarks. We take the filters behind five of those claims,
reimplementing four and running Ai2's decon as released, and apply them to the
same training documents and benchmark questions. They disagree widely: on all
2.25 million Dolci documents and 502 benchmark questions, the four
reimplemented filters flag anywhere from 2 to 231 items.

To tell which flags are wrong without annotating anything, we add benchmark
questions published *after* each corpus was assembled. They cannot be in the
corpus, so a filter that flags one is wrong by construction. The two filters
that flag the most have false-positive rates of 20.0% and 22.9% on all of
Dolci, and over half on competition problems. To measure what the filters
catch, we plant some of those same questions into the corpus, in several
forms, and count how many come back.

## Quick start

```bash
git clone <this repository> && cd <it>
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
```

Everything runs on CPU. Without downloading any corpus, `pytest` already checks
the behaviour of every filter and experiment.

To reproduce the results from scratch, download the training corpora and run
every experiment:

```bash
contamination-audit fetch dolci tulu nemotron
contamination-audit run                 # writes results/paper/*.json
```

All of Dolci is a 36 GB download, read one Parquet shard at a time, from which
about 1.8 GB of questions is kept; Tülu 3 and Nemotron default to their first
100,000 documents. A download that drops resumes where it stopped when run
again.

`contamination-audit run --list` shows the experiments, and
`contamination-audit run scaling tuning` runs a subset. The experiments on all
of Dolci share one pass of each filter over the corpus, cached under
`work/scans/`. Those passes take about an hour on a laptop, almost all of it in
the OpenThoughts3 and Tülu 3 filters, and need about 6 GB of memory; the
experiments that read them then take minutes.

## Where each result comes from

Each experiment writes one file, `results/paper/<name>.json`.

| Result | Experiment |
|---|---|
| Longest span each control shares with the corpus | `validation` |
| When the papers behind the research controls first appeared | `control_dates` |
| Items flagged by each filter on all of Dolci | `flag_counts`, plus `decon` |
| Filter agreement and bootstrap intervals | `agreement` |
| Flag rates by kind of question, on both sides of the split | `strata` |
| Each competition against its own 2026 edition | `matched` |
| Training documents each filter removes, and how many only for controls | `collateral` |
| False positives against corpus size, up to all of Dolci | `scaling` |
| Replication on Tülu 3 and Nemotron | `tulu`, `nemotron`, plus `decon_tulu`, `decon_nemotron` |
| Recall on planted contamination | `planted` |
| Phrases behind s1's false positives | `mechanism` |
| Parameter sweeps | `tuning` |
| Phrase-agreement filtering | `phrases` |
| Filtering on rare n-grams, tuned and tested on separate items | `rare_ngrams` |
| Corpus composition, and the sources the flags come from | `composition` |

`flag_counts`, `agreement`, `strata`, `matched`, `collateral`, `rare_ngrams`
and the last point of `scaling` use all of Dolci. The parameter sweeps,
`mechanism`, `phrases` and `validation` use its first 100,000 documents, as do
the replications on the other two corpora.

## The filters

| Filter | Rule | Parameters from |
|---|---|---|
| s1 | any shared 8-gram of lowercased words | released code |
| OpenThoughts3 | any shared 13-gram of Qwen2 tokens, or `fuzz.ratio` ≥ 75 | released code |
| OpenThoughts-114K | `fuzz.ratio` ≥ 95 on raw strings | released code |
| Tülu 3 | 8-grams of spaCy tokens covering > 50% of an item within one document | paper |
| Olmo 3's decon | two-stage cluster scoring, threshold 0.8 | released binary |

Each is defined in `src/contamination_audit/filters/published.py`, built from
the matching rules in `rules.py`. The Tülu 3 code declares no default for its
n-gram size or threshold, so those come from its paper. open-instruct finds
candidate documents through Elasticsearch, capped at 100 hits per n-gram, where
our version scans every document; seeing more documents can only raise an
item's best score, so ours is an upper bound on the original.

decon is a Rust binary and is run as one rather than reimplemented. Its first
stage alone flags 199 of the 200 OlymMATH items where the full tool flags none.
On Dolci the run also sweeps decon's score threshold, on the controls as well
as the measurement items, and repeats the default without indexing answers. Our eval files give each benchmark item its answer
and training questions have none, so a question matched on its own is scored
with decon's question-and-answer weights, where the question counts 0.75 of
the score, below the default threshold of 0.8 even for a verbatim copy.

```bash
git clone https://github.com/allenai/decon && cd decon
git checkout 4cdcc5696ed047bf7d0d48b8d86a274f5a06c0a6 && cargo build --release && cd ..
contamination-audit decon --binary decon/target/release/decon --scaling
contamination-audit decon --binary decon/target/release/decon --corpus tulu
contamination-audit decon --binary decon/target/release/decon --corpus nemotron
contamination-audit run tulu nemotron   # adds decon to the replications
```

## Planted contamination

`planted` measures recall by adding control questions to the corpus as new
training documents: verbatim, followed by the answer, retypeset with other
LaTeX delimiters, cut to their first half, and inside a longer conversation.
Half the controls, chosen within each kind of question, are planted and half
stay clean, so recall and false positives come from the same items.

A further form, the question restated by a language model, is committed in
`data/benchmarks/planted_paraphrases.jsonl`, made with `gpt-6-astra`; each
line records the model, the response id and the SHA-256 of the prompt. Both
experiments add the form whenever that file exists, so they never call an API.
A new run words them differently, so the committed file is what reproduces
the paper. To make it again, which needs an OpenAI key in `OPENAI_API_KEY`,

```bash
pip install -e ".[paraphrase]"
contamination-audit paraphrase-controls --model gpt-6-astra
```

## Data

**Benchmarks** are small and committed under `data/benchmarks/`.

| File | Contents |
|---|---|
| `olymmath_en_{easy,hard}.jsonl` | OlymMATH, 200 items |
| `matharena_measurement.jsonl` | ten MathArena competitions held before November 2025, 302 items |
| `matharena_control.jsonl` | eight MathArena sets published from January 2026 on, 280 items |
| `matharena_control_post_may2026.jsonl` | five sets from June 2026 on, the controls for Nemotron, 261 items |
| `matharena_earlier_editions.jsonl` | AIME and HMMT February 2023 and USAMO 2024–25, 69 items |
| `arxiv_first_posted.json` | when the paper behind each research-derived control first appeared |

`contamination-audit fetch benchmarks arxiv-dates` downloads them again. Every
MathArena set is pinned to the revision current on 2026-10-06, when the
committed files were taken. Three of them, AIME 2026, HMMT February 2026 and
CMIMC 2025, were corrected upstream on 2026-10-19; fetching their latest
version would not reproduce the paper.

A question published after a corpus can still come from a paper posted before
it. Research-derived controls therefore count only when their source paper
first appeared on arXiv after the corpus's release. This drops none of Dolci's
or Tülu 3's controls and 20 of Nemotron's.

**Training corpora** are downloaded into `data/corpora/`, keeping only the
first user turn of each example, which is where a leaked benchmark question
would appear and what the released filter code searches.

| Corpus | Source | Revision | Released | Documents used |
|---|---|---|---|---|
| Dolci | `allenai/Dolci-Think-SFT-32B` | `7668c638` | Nov 2025 | all 2,253,676 |
| Tülu 3 math | `allenai/tulu-3-sft-mixture`, math sources | `b14afda6` | Nov 2024 | the first 100,000 |
| Nemotron | `nvidia/Nemotron-SFT-Math-v4` | `84d42ad0` | June 2026 | the first 100,000 |

Dolci's revision is the March 2026 re-upload of the November 2025 release. It
has the same examples in the same order with the same questions, and replaces
one placeholder source label with one per original dataset. Eight of its
2,253,684 examples have an empty first user turn and are skipped. Nemotron's
revision rewrote the files of its first upload without changing any document
read here. Its release date is the day its data first appeared on the Hub.

Experiments always read a prefix of a corpus, never a sample, so a larger
corpus contains every smaller one and download order matters. The first
100,000 math rows of the Tülu 3 mixture all come from one component,
`ai2-adapt-dev/personahub_math_v5_regen_149960`.

Each corpus has its own temporal split. AIME 2026 is a control for Dolci, which
it postdates, but a measurement item for Nemotron, which postdates it. The
splits are defined in `src/contamination_audit/data/benchmarks.py`.

## Repository layout

```
src/contamination_audit/
    filters/        tokenizers, matching rules, and the published filters
    data/           benchmarks, corpora, their temporal splits, downloaders
    experiments/    one class per result in the paper
    decon.py        running Ai2's decon binary
    paraphrase.py   LLM restatements of the control items
    cli.py          the contamination-audit command
data/benchmarks/    measurement and control questions
results/paper/      one JSON file per experiment
tests/
```

## Reproducibility

Every filter is deterministic, and bootstrap intervals and samples use fixed
seeds, so rerunning reproduces the committed results exactly. They were
checked by regenerating every file into an empty directory and comparing
bytes. The committed results were produced with Python 3.10, transformers 5.6,
tokenizers 0.22, spaCy 3.8 with `en_core_web_lg` 3.8.0, and rapidfuzz 3.14.

transformers, datasets and tokenizers all require `huggingface-hub` below 2.0.
If an unrelated package has upgraded it, imports fail; reinstall it with
`pip install "huggingface-hub<2"`.

## Citation

```bibtex
@article{decontaminated-by-whose-standard,
  title  = {Decontaminated by Whose Standard? A Label-Free Evaluation of Decontamination Filters},
  author = {Demarteau, Arno and Jain, Rishbha},
  year   = {2026},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
}
```

## License

The code is released under the MIT License. Benchmark questions and corpora
remain under their original licenses and terms.
