"""Stage 11: CoT feature extraction + Mann-Whitney U (paper Table 7 + Figure 2).

Computes five per-trace features (word count, uncertainty, hedging,
self-correction, math density), runs Mann-Whitney U + Cohen's d clean vs
contaminated on the **original** perturbation only (so any signal is intrinsic
to the trace, not the perturbation), and saves:

  - Table 7 — per-feature mean / U / p / Cohen's d  (CSV)
  - Figure 2 — box-plot grid + Cohen's d heatmap    (PNG)

Reads:  results/traces/{model}_traces_judged.jsonl
Writes: results/tables/table7_cot_features.csv
        results/figures/figure2_cot_distributions.png
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns            # noqa: E402

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.cot_features import features_dataframe, mann_whitney_table
from contamination_audit.io import REPO_ROOT, load_jsonl

_log = logging.getLogger("cot_features")

DEFAULT_MODELS = ["openthoughts", "tulu", "s1"]
METRICS = ("word_count", "uncertainty", "hedging", "self_correction", "math_density")
METRIC_LABELS = ("Word Count", "Uncertainty", "Hedging", "Self-Correction", "Math Density")


def _load(models: list[str]) -> list[dict]:
    base = REPO_ROOT / "results" / "traces"
    out: list[dict] = []
    for model in models:
        for name in (f"{model}_traces_judged.jsonl", f"{model}_traces.jsonl"):
            path = base / name
            if path.exists():
                records = load_jsonl(path)
                for r in records:
                    r.setdefault("model", model)
                out.extend(records)
                _log.info("loaded %d %s records from %s", len(records), model, path.name)
                break
    return out


def render_figure(cot_df, mwu_df, out_path) -> None:
    models = list(mwu_df["model"].unique())
    orig = cot_df[cot_df["perturbation_type"] == "original"]

    fig, axes = plt.subplots(len(models), len(METRICS), figsize=(4 * len(METRICS), 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for row, model in enumerate(models):
        sub = orig[orig["model"] == model]
        for col, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
            ax = axes[row][col]
            clean_vals = sub[sub["split"] == "clean"][metric].dropna()
            contam_vals = sub[sub["split"] == "contaminated"][metric].dropna()
            if clean_vals.empty or contam_vals.empty:
                ax.set_visible(False)
                continue
            ax.boxplot(
                [clean_vals, contam_vals], labels=["clean", "contaminated"],
                patch_artist=True,
                boxprops=dict(facecolor="#5b8db8" if col % 2 == 0 else "#e07b54", alpha=0.6),
                medianprops=dict(color="black", linewidth=2),
            )
            ax.set_title(f"{model} — {label}", fontsize=9, fontweight="bold")
            ax.set_ylabel(label if col == 0 else "")
            row_match = mwu_df[(mwu_df["model"] == model) & (mwu_df["metric"] == metric)]
            if not row_match.empty:
                p = float(row_match["p"].iloc[0])
                sig = row_match["sig"].iloc[0]
                ax.set_xlabel(f"p={p:.3f} {sig}", fontsize=8)

    plt.suptitle("CoT Features: Clean vs Contaminated (original problems only)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def render_cohens_d_heatmap(mwu_df, out_path) -> None:
    models = list(mwu_df["model"].unique())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = mwu_df[mwu_df["model"] == model].set_index("metric")[["cohens_d"]]
        sns.heatmap(sub, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax, cbar=ax == axes[-1],
                    linewidths=0.5, linecolor="grey")
        ax.set_title(f"{model} — Cohen's d\n(clean − contaminated)", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.suptitle("CoT Feature Effect Sizes (Cohen's d): Clean vs Contaminated",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    configure_logging()
    records = _load(args.models)
    if not records:
        _log.error("no trace data found — run scripts/07_run_inference.py first")
        return

    cot_df = features_dataframe(records)
    mwu_df = mann_whitney_table(cot_df)

    out_dir = REPO_ROOT / "results" / "tables"
    fig_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    mwu_df.to_csv(out_dir / "table7_cot_features.csv", index=False)
    cot_df.to_csv(out_dir / "table7_cot_features_per_trace.csv", index=False)
    render_figure(cot_df, mwu_df, fig_dir / "figure2_cot_distributions.png")
    render_cohens_d_heatmap(mwu_df, fig_dir / "figure2b_cohens_d.png")

    print(mwu_df.to_string(index=False))
    _log.info("wrote %s and figure2_cot_distributions.png",
              out_dir / "table7_cot_features.csv")


if __name__ == "__main__":
    main()
