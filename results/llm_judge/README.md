# LLM review of saved candidate pairs

This package contains the completed October 11, 2026 judge experiment: 565
distinct benchmark/training text pairs and a same-model repeat of 20 pairs.
The model received only the two question texts, without answers, filter names,
scores, dates, or first-pass judgments. The exact prompt is in
`run_judge.py` and every saved response. The run used `gpt-6-astra`
with high reasoning effort; Python and SDK versions are in `run_manifest.json`.

## Results and scope

| Relationship | Measurement pairs | Control pairs | Total |
|---|---:|---:|---:|
| Related but distinct | 187 | 28 | 215 |
| Generic overlap | 164 | 50 | 214 |
| Unrelated | 100 | 36 | 136 |
| Same problem | 0 | 0 | 0 |
| Uncertain | 0 | 0 | 0 |
| Total | 451 | 114 | 565 |

The pairs cover 198 distinct measurement questions and 51 controls. Supporting
quotes occur in the supplied texts for all 565 judgments. That checks quotation
accuracy, not whether the semantic judgment is correct. Relationship labels
agree on 19/20 random repeats; same-problem versus other agrees on 20/20.
These are model judgments and consistency checks, not completed human review.

Under the study's temporal assumption, controls are negatives. The judge
describes what their matches share; it does not change those temporal labels.
Zero `same_problem` judgments does not establish that the full corpus is free
of contamination, or give filter precision or recall.

## How the pairs were selected

`candidate_matches.json` freezes the original audit's saved retrieval evidence
for 502 measurement entries and 280 controls against 100,000 Dolci questions.
It stores up to three candidate records per branch, not every corpus match.
The input-building script selects:

- Saved s1 and Qwen 13-gram candidates for items flagged by those branches.
- Saved coverage candidates exceeding 50% within one training document.
- Saved raw-string fuzzy candidates with a score of at least 75.

Identical benchmark/training text pairs are merged while preserving benchmark
memberships, training IDs, and retrieval branches. Branch counts overlap.
The original and postprocessed item flags are metadata for interpretation;
neither is sent to the judge. Missing pair evidence is not a negative decision.
Per-document whitespace coverage is an adaptation, not the released Tulu
pipeline. These inputs describe the saved local audit, not later author-reported
filter corrections. This package does not update the main filter implementations
or `results/paper/`.

The 20 repeat pairs were sampled with seed 42 before first-pass completion.
Both passes used the same model and prompt, with the first judgment withheld
from the second pass. Primary labels are preserved after disagreements.

## Files

| File | Contents |
|---|---|
| `run_judge.py` | The script that made the API calls, byte for byte as it ran |
| `candidate_matches.json` | Frozen retrieval snapshot used to construct the inputs |
| `pairs.jsonl` | All 565 inputs, pair IDs, memberships, scores, and matched spans |
| `labels.jsonl` | Original responses, judgments, quotes, model, prompt, and usage |
| `summary.json` | Relationship and overlap counts by group and retrieval branch |
| `input_manifest.json` | Candidate selection rules and source/input hashes |
| `run_manifest.json` | Original run versions, hashes, model, and call counts |
| `review_random_sample.json` | Seed and the 20 randomly selected pair IDs |
| `review_pairs.jsonl` | Inputs for the repeat pass |
| `review_labels.jsonl` | Original repeat-pass responses |
| `review_summary.json` | Agreement and disagreement counts |

The original labels, inputs, and judge script are byte-preserved. The input
manifest's source path is now relative to this directory. Its historical
postprocessing hash identifies the earlier audit output; the standalone builder
reconstructs those metadata directly from the shipped retrieval snapshot. The
control-pairs hash refers to the control-only subset of `pairs.jsonl`.

## Reproduce without API calls

From the repository root, with no API calls:

```bash
contamination-audit judge-pairs --output work/judge/rebuilt_pairs.jsonl
cmp work/judge/rebuilt_pairs.jsonl results/llm_judge/pairs.jsonl
contamination-audit judge-verify
```

The first two commands reconstruct the 565 inputs byte-for-byte. The third
verifies the source, input and output hashes, the judge script and prompt, pair
alignment, quotes, the seeded repeat sample, and both saved summaries. To also
write the rebuilt summaries to a separate directory:

```bash
contamination-audit judge-verify --output-dir work/judge/rebuilt_summaries
```

Use fresh output paths; both commands refuse to overwrite existing files.
The pair builder checks the existing zero-result decon summary before attaching
its historical negative item flags. It does not rerun any filter. This package
starts from saved retrieval evidence, not from a re-download of the full corpus.

## Make new judge calls (optional)

```bash
pip install -e '.[judge]'
python results/llm_judge/run_judge.py results/llm_judge/pairs.jsonl --output work/judge/new_labels.jsonl
python results/llm_judge/run_judge.py results/llm_judge/review_pairs.jsonl --output work/judge/new_review_labels.jsonl
```

Set `OPENAI_API_KEY` in your environment first. These commands make paid API
calls and require access to the selected model. `--model` overrides the recorded
default; a different model is a new experiment. `--limit 50 --seed 42` selects a
smaller input sample, not deterministic model generation. New responses may
differ even with the same model. The verification command above checks the
frozen published responses; it does not expect new responses to match them.
Write new outputs under the ignored `work/` directory so they do not replace
the published evidence.
