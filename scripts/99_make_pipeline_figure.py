"""Render the pipeline overview figure (paper figure on page 3).

Reads results/tables/table2_contamination_counts.csv so the per-dataset counts
shown on the diagram stay in sync with the underlying audit numbers. The
previous version had every count hardcoded.

Outputs: results/figures/pipeline_figure.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import _common  # noqa: F401

from contamination_audit.config import configure_logging
from contamination_audit.io import REPO_ROOT

_log = logging.getLogger("pipeline_figure")

PALETTE = {
    "input": "#1e3a5f",
    "clex": "#6b2580",
    "csem": "#1a6b4a",
    "human": "#5a4a10",
    "output": "#7a1515",
    "text": "#f0f0f0",
    "subtext": "#999999",
    "bg": "#0f1117",
}


def _box(ax, x, y, w, h, label, sublabel, color, *, fontsize=9.5, subsize=7.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#ffffff18", linewidth=1.0, zorder=3,
    ))
    cy = y + h / 2 + (0.15 if sublabel else 0)
    ax.text(x + w / 2, cy, label, ha="center", va="center",
            fontsize=fontsize, color=PALETTE["text"], fontweight="bold",
            zorder=4, multialignment="center")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel, ha="center", va="center",
                fontsize=subsize, color=PALETTE["subtext"], zorder=4,
                multialignment="center")


def _arr(ax, x1, y1, x2, y2, color, lw=1.6, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)


def _badge(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color + "22",
                      edgecolor=color, linewidth=1.2), zorder=5)


def _summarize_counts(table_path: Path) -> tuple[str, str, str]:
    """Pull C_lex / C_sem / total strings from table2_contamination_counts.csv."""
    if not table_path.exists():
        return ("C_lex: see Table 2", "C_sem: see Table 2", "Confirmed: see Table 2")

    df = pd.read_csv(table_path)
    lex = df[df["contamination_type"] == "C_lex"].set_index("project")
    sem = df[df["contamination_type"] == "C_sem"].set_index("project")
    fmt_lex = "  ".join(
        f"{name}: {int(lex.loc[name, 'n_unique_math500'])}"
        for name in ("s1", "tulu", "openthoughts") if name in lex.index
    )
    fmt_sem = "  ".join(
        f"{name}: {int(sem.loc[name, 'n_unique_math500'])}"
        for name in ("s1", "tulu", "openthoughts") if name in sem.index
    )
    confirmed_union = 0
    for name in ("s1", "tulu", "openthoughts"):
        if name in lex.index:
            confirmed_union += int(lex.loc[name, "n_unique_math500"])
        if name in sem.index:
            confirmed_union += int(sem.loc[name, "n_unique_math500"])
    return fmt_lex, fmt_sem, f"Union: {confirmed_union} unique MATH-500 items"


def render(out_path: Path, table_path: Path) -> None:
    lex_str, sem_str, summary_str = _summarize_counts(table_path)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.text(10, 9.6, "Contamination Detection Pipeline", ha="center",
            fontsize=15, color=PALETTE["text"], fontweight="bold")
    ax.text(10, 9.25,
            "MATH-500 vs. Training Corpora · Two-track detection · Human validation",
            ha="center", fontsize=9, color=PALETTE["subtext"])

    _box(ax, 0.3, 6.9, 2.4, 1.0, "MATH-500", "500 problems", PALETTE["input"])
    _box(ax, 0.3, 4.4, 2.4, 2.1, "Training Corpora",
         "s1K · 1,000 items\nTülu 3 · 84,312 items\nOpenThoughts · 113,957 items",
         PALETTE["input"], fontsize=9, subsize=7.5)

    _arr(ax, 2.7, 7.4, 3.8, 7.4, PALETTE["subtext"])
    _arr(ax, 2.7, 5.5, 3.8, 7.4, PALETTE["subtext"], rad=-0.25)
    _arr(ax, 2.7, 5.5, 3.8, 4.15, PALETTE["subtext"], rad=0.25)

    _badge(ax, 7.5, 8.75, "  C_lex  —  Lexical contamination  ", PALETTE["clex"])
    _badge(ax, 7.5, 2.9, "  C_sem  —  Semantic contamination  ", PALETTE["csem"])

    bw, bh, gap = 2.4, 1.0, 0.4
    xs = [3.8 + i * (bw + gap) for i in range(4)]

    _box(ax, xs[0], 6.9, bw, bh, "N-gram Tokenizer",
         "Qwen2-7B tokenizer\nn = 8 or 13", PALETTE["clex"])
    _arr(ax, xs[0] + bw, 7.4, xs[1], 7.4, PALETTE["clex"])
    _box(ax, xs[1], 6.9, bw, bh, "Shared N-gram\nDetection",
         "All train × test pairs", PALETTE["clex"])
    _arr(ax, xs[1] + bw, 7.4, xs[2], 7.4, PALETTE["clex"])
    _box(ax, xs[2], 6.9, bw, bh, "Threshold Filter",
         "≥5 shared n-grams", PALETTE["clex"])
    _arr(ax, xs[2] + bw, 7.4, xs[3], 7.4, PALETTE["clex"])
    _box(ax, xs[3], 6.9, bw, bh, "C_lex Candidates", lex_str, PALETTE["clex"], fontsize=8.5)

    ys = 3.65
    _box(ax, xs[0], ys, bw, bh, "Sentence Embeddings",
         "all-mpnet-base-v2\n+ FAISS index", PALETTE["csem"])
    _arr(ax, xs[0] + bw, ys + 0.5, xs[1], ys + 0.5, PALETTE["csem"])
    _box(ax, xs[1], ys, bw, bh, "Cosine Similarity\nSearch",
         "Top-K · threshold ≥ 0.70", PALETTE["csem"])
    _arr(ax, xs[1] + bw, ys + 0.5, xs[2], ys + 0.5, PALETTE["csem"])
    _box(ax, xs[2], ys, bw, bh, "LLM Judge",
         "GPT-4o-mini / Gemini\nCONTAM / RELATED / CLEAN", PALETTE["csem"])
    _arr(ax, xs[2] + bw, ys + 0.5, xs[3], ys + 0.5, PALETTE["csem"])
    _box(ax, xs[3], ys, bw, bh, "C_sem Candidates", sem_str, PALETTE["csem"], fontsize=8.5)

    hx = xs[3] + bw + gap
    _arr(ax, xs[3] + bw, 7.4, hx, 6.65, color="#aaaaaa", lw=1.4, rad=-0.2)
    _arr(ax, xs[3] + bw, ys + 0.5, hx, 5.85, color="#aaaaaa", lw=1.4, rad=0.2)
    _box(ax, hx, 5.6, 2.5, 1.3, "Human Validation",
         "C_sem precision\nvalidated on stratified samples\nκ = 0.83 vs human",
         PALETTE["human"], fontsize=9)
    _arr(ax, hx + 1.25, 5.6, hx + 1.25, 4.1, color="#dddddd", lw=1.8)
    _box(ax, hx, 2.7, 2.5, 1.3, "Confirmed Contaminated", summary_str,
         PALETTE["output"], fontsize=9)

    legend_items = [
        mpatches.Patch(color=PALETTE["input"], label="Input data"),
        mpatches.Patch(color=PALETTE["clex"], label="C_lex track"),
        mpatches.Patch(color=PALETTE["csem"], label="C_sem track"),
        mpatches.Patch(color=PALETTE["human"], label="Human validation"),
        mpatches.Patch(color=PALETTE["output"], label="Confirmed output"),
    ]
    ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.0, 0.0),
              fontsize=8, framealpha=0.25, facecolor="#1a1a2e", edgecolor="#444455",
              labelcolor=PALETTE["text"], ncol=5)

    plt.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    _log.info("saved %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        default=str(REPO_ROOT / "results" / "figures" / "pipeline_figure.png"))
    parser.add_argument("--table",
                        default=str(REPO_ROOT / "results" / "tables" / "table2_contamination_counts.csv"))
    args = parser.parse_args()

    configure_logging()
    render(Path(args.out), Path(args.table))


if __name__ == "__main__":
    main()
