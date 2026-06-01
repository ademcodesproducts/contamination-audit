"""Generate contamination detection pipeline figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

C_INPUT  = '#1e3a5f'
C_CLEX   = '#6b2580'
C_CSEM   = '#1a6b4a'
C_HUMAN  = '#5a4a10'
C_OUTPUT = '#7a1515'
TEXT     = '#f0f0f0'
SUBTEXT  = '#999999'

def box(ax, x, y, w, h, label, sublabel='', color=C_INPUT,
        fontsize=9.5, subsize=7.5):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.1',
                          facecolor=color, edgecolor='#ffffff18',
                          linewidth=1.0, zorder=3)
    ax.add_patch(rect)
    cy = y + h/2 + (0.15 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha='center', va='center',
            fontsize=fontsize, color=TEXT, fontweight='bold', zorder=4,
            multialignment='center')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha='center', va='center',
                fontsize=subsize, color=SUBTEXT, zorder=4,
                multialignment='center')

def arr(ax, x1, y1, x2, y2, color='#556677', lw=1.6, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)

def track_badge(ax, x, y, text, color):
    ax.text(x, y, text, ha='center', va='center', fontsize=8,
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color+'22',
                      edgecolor=color, linewidth=1.2), zorder=5)

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(10, 9.6, 'Contamination Detection Pipeline', ha='center',
        fontsize=15, color=TEXT, fontweight='bold')
ax.text(10, 9.25, 'MATH-500 vs. Training Corpora  ·  Two-track detection  ·  Human validation',
        ha='center', fontsize=9, color=SUBTEXT)

# ── Inputs ─────────────────────────────────────────────────────────────────
box(ax, 0.3, 6.9, 2.4, 1.0, 'MATH-500', '500 problems', C_INPUT)
box(ax, 0.3, 4.4, 2.4, 2.1,
    'Training Corpora',
    's1K  ·  1,000 items\nTülu 3  ·  334k items\nOpenThoughts  ·  114k items',
    C_INPUT, fontsize=9, subsize=7.5)

# arrows from inputs to both tracks
arr(ax, 2.7, 7.4,  3.8, 7.4,  color='#667799')   # MATH-500 → C_lex
arr(ax, 2.7, 5.5,  3.8, 7.4,  color='#667799', rad=-0.25)  # corpora → C_lex
arr(ax, 2.7, 5.5,  3.8, 4.15, color='#667799', rad=0.25)   # corpora → C_sem

# ── Track badges ───────────────────────────────────────────────────────────
track_badge(ax, 7.5, 8.75, '  C_lex  Track  —  Lexical Contamination  ', C_CLEX)
track_badge(ax, 7.5, 2.9,  '  C_sem  Track  —  Semantic Contamination  ', C_CSEM)

# ── C_lex track  (y ≈ 6.9) ─────────────────────────────────────────────────
bw, bh, gap = 2.4, 1.0, 0.4
xs = [3.8, 3.8+bw+gap, 3.8+2*(bw+gap), 3.8+3*(bw+gap)]

box(ax, xs[0], 6.9, bw, bh, 'N-gram Tokenizer',
    'Qwen2-7B tokenizer\nn = 8 or 13', C_CLEX)
arr(ax, xs[0]+bw, 7.4, xs[1], 7.4, color=C_CLEX)

box(ax, xs[1], 6.9, bw, bh, 'Shared N-gram\nDetection',
    'All train × test pairs', C_CLEX)
arr(ax, xs[1]+bw, 7.4, xs[2], 7.4, color=C_CLEX)

box(ax, xs[2], 6.9, bw, bh, 'Threshold Filter',
    '≥5 shared n-grams', C_CLEX)
arr(ax, xs[2]+bw, 7.4, xs[3], 7.4, color=C_CLEX)

box(ax, xs[3], 6.9, bw, bh, 'C_lex Candidates',
    's1: 36 · Tülu: 15 · OT: 93', C_CLEX, fontsize=9)

# ── C_sem track  (y ≈ 3.65) ────────────────────────────────────────────────
ys = 3.65
box(ax, xs[0], ys, bw, bh, 'Sentence Embeddings',
    'all-mpnet-base-v2\n+ FAISS index', C_CSEM)
arr(ax, xs[0]+bw, ys+0.5, xs[1], ys+0.5, color=C_CSEM)

box(ax, xs[1], ys, bw, bh, 'Cosine Similarity\nSearch',
    'Top-k neighbors\nthreshold ≥ 0.70', C_CSEM)
arr(ax, xs[1]+bw, ys+0.5, xs[2], ys+0.5, color=C_CSEM)

box(ax, xs[2], ys, bw, bh, 'LLM Judge',
    'GPT-4o-mini\nCONTAM / RELATED / CLEAN', C_CSEM)
arr(ax, xs[2]+bw, ys+0.5, xs[3], ys+0.5, color=C_CSEM)

box(ax, xs[3], ys, bw, bh, 'C_sem Candidates',
    'Tülu: 29 · OT: 42 · s1: 1', C_CSEM, fontsize=9)

# ── Human validation ───────────────────────────────────────────────────────
hx = xs[3] + bw + gap
arr(ax, xs[3]+bw, 7.4,   hx, 6.65, color='#aaaaaa', lw=1.4, rad=-0.2)
arr(ax, xs[3]+bw, ys+0.5, hx, 5.85, color='#aaaaaa', lw=1.4, rad=0.2)

box(ax, hx, 5.6, 2.5, 1.3,
    'Human Validation',
    '72 pairs reviewed\nTülu: 79% precision\nOT: 36% precision',
    C_HUMAN, fontsize=9, subsize=7.5)

# ── Final output ───────────────────────────────────────────────────────────
arr(ax, hx+1.25, 5.6, hx+1.25, 4.1, color='#dddddd', lw=1.8)

box(ax, hx, 2.7, 2.5, 1.3,
    'Confirmed Contaminated',
    's1: 4 items  (0.8%)\nTülu: 32 items  (6.4%)\nOT: 35 items  (7.0%)',
    C_OUTPUT, fontsize=9, subsize=7.5)

ax.text(hx+1.25, 2.5, '64 unique MATH-500 items  (12.8%)',
        ha='center', va='top', fontsize=8, color='#ffaaaa', fontweight='bold')

# ── Robustness callout ─────────────────────────────────────────────────────
ax.text(3.8, 2.2,
        'Robustness check (OpenThoughts):  n=13 → 93 items  ·  n=15 → 65  ·  n=20 → 32',
        ha='left', va='center', fontsize=7.8, color=SUBTEXT, style='italic')

# ── NuminaMath callout ─────────────────────────────────────────────────────
ax.text(3.8, 1.65,
        'Tülu source:  NuminaMath-TIR drives 93% of C_lex and 97% of C_sem',
        ha='left', va='center', fontsize=7.8, color=SUBTEXT, style='italic')

# ── Legend ─────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_INPUT,  label='Input data'),
    mpatches.Patch(color=C_CLEX,   label='C_lex — lexical track'),
    mpatches.Patch(color=C_CSEM,   label='C_sem — semantic track'),
    mpatches.Patch(color=C_HUMAN,  label='Human validation'),
    mpatches.Patch(color=C_OUTPUT, label='Confirmed output'),
]
ax.legend(handles=legend_items, loc='lower left',
          bbox_to_anchor=(0.0, 0.0),
          fontsize=8, framealpha=0.25,
          facecolor='#1a1a2e', edgecolor='#444455',
          labelcolor=TEXT, ncol=5)

plt.tight_layout(pad=0.4)
out = 'results/pipeline_figure.png'
plt.savefig(out, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f'Saved to {out}')
