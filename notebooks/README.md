# Notebooks

Six notebooks (to be added) produce the paper's behavioral-analysis artifacts. Each one corresponds to a stub script under [scripts/](../scripts/). When you drop the notebook here, follow the linearization contract below — keep the input and output schemas consistent so the existing scripts can replace the notebook downstream.

| Notebook | Produces | Maps to script | Paper artifact |
|---|---|---|---|
| `generate_perturbations.ipynb` | `data/processed/contaminated_and_perturbations.csv` | [`scripts/06_generate_perturbations.py`](../scripts/06_generate_perturbations.py) | §5.5 (perturbation generation) |
| `run_inference.ipynb` | `results/traces/{model}_traces.jsonl` | [`scripts/07_run_inference.py`](../scripts/07_run_inference.py) | Table 3 inference |
| `compute_did.ipynb` | `results/tables/table4_accuracy.csv`, `results/tables/table5_did.csv` | [`scripts/09_compute_did.py`](../scripts/09_compute_did.py) | Tables 4 + 5 |
| `null_rate.ipynb` | `results/tables/table6_null_rate.csv`, `results/figures/figure1_null_rates.png` | [`scripts/10_compute_null_rate.py`](../scripts/10_compute_null_rate.py) | Table 6 + Figure 1 |
| `cot_features.ipynb` | `results/tables/table7_cot_features.csv`, `results/figures/figure2_cot_distributions.png` | [`scripts/11_cot_features.py`](../scripts/11_cot_features.py) | Table 7 + Figure 2 |
| `mech_probing.ipynb` | Appendix C summary CSV | _(no script — exploratory only)_ | Appendix C |

## Linearization contract

When converting a notebook to its matching script:

1. Move the heavy logic into the library module (e.g. perturbation, did, cot_features) — pure functions, no I/O.
2. Make the script a CLI entry point: `argparse`, load config, call the library, write to the canonical path above.
3. Set seeds at the top of the script from `configs/thresholds.yaml` via `contamination_audit.config.seed_everything`.
4. Use `contamination_audit.config.configure_logging()` instead of `print`.
5. Replace the stub's `NotImplementedError` with the real implementation in one PR per notebook.

## Trace record schema

Every record in `results/traces/{model}_traces.jsonl` (currently checked in for `openthoughts` and `tulu`) must have:

```jsonc
{
  "math500_id":        "math500_0193",
  "split":             "contaminated" | "clean",
  "perturbation_type": "original" | "number_swap" | "surface_noise",
  "prompt":            "<exact text sent to the model>",
  "full_trace":        "<raw model output including any \\boxed{}>",
  "ground_truth":      "<expected answer for the perturbed problem>",
  "final_answer":      "<extracted answer or null>",
  "correct":           true | false
}
```

The judged trace files (`*_traces_judged.jsonl`) extend this with `llm_correct`, `llm_model_answer`, and `extracted` (boolean — true if the answer came from `answer_extract.extract_from_trace`).
