# Notebooks

Two of these notebooks are the **canonical source of truth** for the paper's behavioral-analysis numbers; the other two are exploratory probes from §5.3 future work. Each canonical notebook also has a matching linearised script under [scripts/](../scripts/) — the script is what `make all` runs, the notebook is what the paper analyses.

## Canonical (linearised into scripts)

| Notebook | Maps to script | Paper artifact |
|---|---|---|
| [05_did_inference.ipynb](05_did_inference.ipynb) | [`scripts/07_run_inference.py`](../scripts/07_run_inference.py) | §5.3 inference traces, Table 3 |
| [07_analysis.ipynb](07_analysis.ipynb) | [`scripts/09_compute_did.py`](../scripts/09_compute_did.py) (Tables 4 + 5)<br>[`scripts/10_compute_null_rate.py`](../scripts/10_compute_null_rate.py) (Table 6 + Fig 1)<br>[`scripts/11_cot_features.py`](../scripts/11_cot_features.py) (Table 7 + Fig 2)<br>[`scripts/14_recitation_analysis.py`](../scripts/14_recitation_analysis.py) (§7.2) | Tables 4–7, Figures 1–2, §7.2 recitation |

When the notebook and script disagree on a number, **the notebook is canonical** — file a bug against the script.

## Exploratory (no script equivalent)

| Notebook | Status | What it does |
|---|---|---|
| [temperature_variance.ipynb](temperature_variance.ipynb) | §5.3 future work | Generates N=10 solutions per problem at T=1.0 to measure reasoning-path variance. Not completed at the paper's sample size — listed as future work in §5.3. |
| [cot_consistency.ipynb](cot_consistency.ipynb) | Appendix probe | Tests whether contaminated problems produce less-diverse chain-of-thought across repeated runs. Smaller, focused probe related to the variance hypothesis. |

## Trace record schema

Every record in `results/traces/{model}_traces.jsonl` has these fields:

```jsonc
{
  "math500_id":        "math500_0193",
  "model":             "openthoughts" | "tulu" | "s1",
  "split":             "contaminated" | "clean",
  "perturbation_type": "original" | "number_swap" | "surface_noise",
  "subject":           "Algebra" | "Counting & Probability" | ...,
  "level":             1 | 2 | 3 | 4 | 5,
  "answer_confidence": "high" | "medium" | "low" | "",
  "temperature":       0.8,
  "sample_index":      0,
  "full_trace":        "<raw model output, includes \\boxed{} if present>",
  "final_answer":      "<extracted+normalized answer, or null>",
  "ground_truth":      "<raw answer from MATH-500 / perturbation CSV>",
  "correct":           true | false | null
}
```

The judged trace files (`*_traces_judged.jsonl`) add:

```jsonc
{
  ...,
  "llm_correct":       true | false,
  "llm_model_answer":  "<extracted answer the judge saw>",
  "extracted":         true | false
}
```

`extracted=true` means the answer came from the natural-language fallback in `answer_extract.extract_from_trace`; `extracted=false` means it came from a `\boxed{}` regex hit.
