# Decontaminated by Whose Standard?

Code and data for *Decontaminated by Whose Standard? A Label-Free Evaluation of
Decontamination Filters* ([arXiv](https://arxiv.org/abs/XXXX.XXXXX)).

Open post-training datasets routinely report that they were decontaminated
against public benchmarks. We reimplement five of the filters behind those
claims and apply them to the same training documents and benchmark questions.
They disagree widely: on 100,000 Dolci documents and 502 benchmark questions
they flag anywhere from 0 to 141 items.

To tell which flags are wrong without annotating anything, we add benchmark
questions published *after* each corpus was assembled. They cannot be in the
corpus, so a filter that flags one is wrong by construction. The two filters
that flag the most have false-positive rates of 12.9% and 11.1% on Dolci.

## Quick start

```bash
git clone <this repository> && cd <it>
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
```

Everything runs on CPU and needs no API key. Without downloading any corpus,
you can already check what is committed here:

```bash
pytest                                  # behaviour of every filter and experiment
contamination-audit judge-verify        # integrity of the LLM-judge record
```

To reproduce the results from scratch, download the training corpora (about
1.5 GB) and run every experiment:

```bash
contamination-audit fetch dolci tulu nemotron
contamination-audit run                 # writes results/paper/*.json
```

`contamination-audit run --list` shows the experiments, and
`contamination-audit run scaling tuning` runs a subset. The whole set takes
about an hour on a laptop, most of it in `scaling`, which tokenizes 390,000
documents with the Qwen2 tokenizer.

## Where each result comes from

Each experiment writes one file, `results/paper/<name>.json`.

| In the paper | Experiment |
|---|---|
| Longest span shared with the corpus (§4.1, Table 1) | `validation` |
| Items flagged by each filter (Table 2) | `flag_counts`, plus `decon` |
| Filter agreement and bootstrap intervals (§5.1, §5.2) | `agreement` |
| False positives against corpus size (§6.1, Figure 3, Table 6) | `scaling` |
| Replication on Tülu 3 and Nemotron (§6.2, Table 3) | `tulu`, `nemotron` |
| The same competitions one year apart (§6.3, Table 4) | `matched` |
| Phrases behind s1's false positives (Appendix D, Table 9) | `mechanism` |
| Parameter sweeps (§7.2, Figure 4, Table 8) | `tuning` |
| Phrase-agreement filtering (§7.3, Table 5) | `phrases` |
| Control set by question type (Appendix E, Table 10) | `strata` |
| LLM judge of candidate pairs (Appendix B, Table 7) | `results/llm_judge/` |
| Corpus composition (Figure 1), and the sources the flags come from | `composition` |

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

```bash
git clone https://github.com/allenai/decon && (cd decon && cargo build --release)
contamination-audit decon --binary decon/target/release/decon --scaling
```

Its committed result is `results/paper/decon.json`.

## Data

**Benchmarks** are small and committed under `data/benchmarks/`. The
measurement set holds OlymMATH and ten MathArena competitions held before
November 2025; the control set holds eight MathArena sets published from
January 2026 on. `contamination-audit fetch benchmarks` downloads them again.

**Training corpora** are downloaded into `data/corpora/`, keeping only the
first user turn of each example, which is where a leaked benchmark question
would appear and what the released filter code searches.

| Corpus | Source | Released | Documents used |
|---|---|---|---|
| Dolci | `allenai/Dolci-Think-SFT-32B` | Nov 2025 | the first 100,000, and 390,000 for scaling |
| Tülu 3 math | `allenai/tulu-3-sft-mixture`, math sources | Nov 2024 | the first 100,000 |
| Nemotron | `nvidia/Nemotron-SFT-Math-v4` | May 2026 | the first 100,000 |

Experiments always read a prefix of a corpus, never a sample, so a larger
corpus contains every smaller one and download order matters. The first
100,000 math rows of the Tülu 3 mixture all come from one component,
`ai2-adapt-dev/personahub_math_v5_regen_149960`.

Each corpus has its own temporal split. AIME 2026 is a control for Dolci, which
it postdates, but a measurement item for Nemotron, which postdates it. The
splits are defined in `src/contamination_audit/data/benchmarks.py`.

## The LLM-judge record

`results/llm_judge/` holds a completed run in which a language model labelled
565 benchmark/training pairs that the filters matched. It is frozen: inputs,
responses, and manifests recording the SHA-256 of each, of the prompt, and of
`run_judge.py`, the script that made the calls, kept byte for byte as it ran.
`contamination-audit judge-verify` recomputes every hash and summary.
See [its README](results/llm_judge/README.md) for what the labels do and do
not show.

## Repository layout

```
src/contamination_audit/
    filters/        tokenizers, matching rules, and the published filters
    data/           benchmarks, corpora, their temporal splits, downloaders
    experiments/    one class per result in the paper
    judge/          rebuilding and verifying the LLM-judge record
    decon.py        running Ai2's decon binary
    cli.py          the contamination-audit command
data/benchmarks/    measurement and control questions
results/paper/      one JSON file per experiment
results/llm_judge/  the frozen LLM-judge run
tests/
```

## Reproducibility

Every filter is deterministic, and bootstrap intervals use a fixed seed, so
rerunning reproduces the committed results exactly. They were checked by
regenerating every file into an empty directory and comparing bytes. The
committed results were produced with Python 3.10, transformers 5.6, tokenizers
0.22, spaCy 3.8 with `en_core_web_lg` 3.8.0, and rapidfuzz 3.14.

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
