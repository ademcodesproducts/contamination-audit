"""Stage 10: null-rate analysis (paper Table 6 + Figure 1).

A null trace is one whose ``final_answer`` is ``None`` (no extractable
``\\boxed{}`` answer). The paper's clearest contamination signal is the 37.5%
null rate on OpenThoughts contaminated × number_swap, nearly double the 20.8%
clean baseline.

Reads:  results/traces/{model}_traces_judged.jsonl
Writes: results/tables/table6_null_rate.csv
        results/figures/figure1_null_rates.png
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT, load_jsonl
from contamination_audit.null_rate import per_cell_null_rate

_log = logging.getLogger("null_rate")

DEFAULT_MODELS = ["openthoughts", "tulu", "s1"]
PTYPE_ORDER = ["original", "surface_noise", "number_swap"]


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


def render_figure(df, out_path) -> None:
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        pivot = sub.pivot_table(
            index="split", columns="perturbation_type", values="null_pct",
            aggfunc="first",
        )
        # Re-order to (clean, contaminated) on x and (orig, surf, num) on bars.
        present = [p for p in PTYPE_ORDER if p in pivot.columns]
        pivot = pivot.reindex(index=["contaminated", "clean"])[present]
        pivot.plot(
            kind="bar", ax=ax,
            color=["#4c7fb5", "#e8a838", "#c0504d"][: len(present)],
            edgecolor="black", width=0.6,
        )
        ax.set_title(f"{model} — Null rate by split × perturbation", fontweight="bold")
        ax.set_ylabel("Null %")
        ax.set_xlabel("")
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(title="Perturbation")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f%%", fontsize=8, padding=2)

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

    df = per_cell_null_rate(records)
    out_dir = REPO_ROOT / "results" / "tables"
    fig_dir = REPO_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "table6_null_rate.csv", index=False)
    print(df.to_string(index=False))
    render_figure(df, fig_dir / "figure1_null_rates.png")
    _log.info("wrote %s and %s",
              out_dir / "table6_null_rate.csv",
              fig_dir / "figure1_null_rates.png")


if __name__ == "__main__":
    main()
