import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8.5, 'legend.fontsize': 7, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.6,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

sizes  = [12500, 25000, 50000, 100000, 150000, 250000, 390000]
s1_det = [13.7, 15.7, 18.1, 25.3, 27.1, 30.3, 32.9]
s1_fp  = [6.8, 8.2, 11.1, 12.9, 14.3, 16.1, 17.9]
o3_det = [12.9, 16.9, 21.9, 27.5, 32.9, 35.7, 37.6]
o3_fp  = [5.4, 6.1, 8.2, 11.1, 13.2, 15.4, 17.1]
s1_fr  = [0.49, 0.52, 0.61, 0.51, 0.53, 0.53, 0.54]
o3_fr  = [0.41, 0.36, 0.37, 0.40, 0.40, 0.43, 0.46]

# ---------------- Figure 1: scaling ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.4))

ax1.plot(sizes, s1_det, '-o', color=BLUE, lw=1.6, ms=3.5, label='s1, flagged')
ax1.plot(sizes, s1_fp, '--o', color=BLUE, lw=1.4, ms=3.5, alpha=.55, label='s1, false positive')
ax1.plot(sizes, o3_det, '-s', color=ORANGE, lw=1.6, ms=3.5, label='OpenThoughts3, flagged')
ax1.plot(sizes, o3_fp, '--s', color=ORANGE, lw=1.4, ms=3.5, alpha=.55, label='OpenThoughts3, false positive')
ax1.set_xscale('log')
ax1.set_xlabel('Training documents')
ax1.set_ylabel('Benchmark items (%)')
ax1.set_title('(a) Both rates rise with corpus size', loc='left')
ax1.set_ylim(0, 45)
ax1.grid(True, alpha=.7, lw=.5)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc='upper left', handlelength=1.8, borderpad=0.1)

ax2.plot(sizes, s1_fr, '-o', color=BLUE, lw=1.8, ms=4, label='s1')
ax2.plot(sizes, o3_fr, '-s', color=ORANGE, lw=1.8, ms=4, label='OpenThoughts3')
ax2.set_xscale('log')
ax2.set_ylim(0, 0.8)
ax2.set_xlabel('Training documents')
ax2.set_ylabel('False positives / flagged')
ax2.set_title('(b) The error fraction does not', loc='left')
ax2.grid(True, alpha=.7, lw=.5)
ax2.set_axisbelow(True)
ax2.annotate('s1', (250000, 0.53), xytext=(0, 8), textcoords='offset points',
             color=BLUE, fontsize=7.5, ha='center')
ax2.annotate('OpenThoughts3', (100000, 0.40), xytext=(0, -13), textcoords='offset points',
             color=ORANGE, fontsize=7.5, ha='center')
for a in (ax1, ax2):
    a.set_xticks([12500, 25000, 50000, 100000, 250000, 390000])
    a.set_xticklabels(['12.5k', '25k', '50k', '100k', '250k', '390k'])
    a.minorticks_off()
fig.savefig('paper/figures/scaling.pdf')
fig.savefig('paper/figures/scaling.png', dpi=220)
print('wrote paper/figures/scaling.{pdf,png}')

# ---------------- Figure 2: precision/recall frontier ----------------
word_d = [93.0, 25.3, 7.2, 1.4, 0.0]
word_f = [72.5, 12.9, 3.9, 0.7, 0.4]
qwen_d = [99.8, 86.3, 62.0, 27.5, 14.3, 4.2, 1.2, 0.4]
qwen_f = [99.6, 68.2, 36.4, 11.1, 6.8, 2.5, 0.4, 0.0]
fz_d   = [0, 0, 0, 0, 0.8, 2.8, 6.2, 15.3]
fz_f   = [0, 0, 0, 0, 0, 0, 0.7, 4.6]

fig2, ax = plt.subplots(figsize=(3.3, 3.0))
ax.plot(word_f, word_d, '-o', color=BLUE, lw=1.6, ms=3.5, label='$n$-gram, whitespace')
ax.plot(qwen_f, qwen_d, '-s', color=ORANGE, lw=1.6, ms=3.5, label='$n$-gram, Qwen2 tokens')
ax.plot(fz_f, fz_d, '-^', color=AQUA, lw=1.9, ms=4.5, label='fuzz.ratio (whole string)')

ax.plot(12.9, 25.3, 'o', mfc='none', mec=BLUE, ms=10, mew=1.4)
ax.plot(11.1, 27.5, 's', mfc='none', mec=ORANGE, ms=10, mew=1.4)
ax.plot(0.0, 0.0, '^', mfc='none', mec=AQUA, ms=10, mew=1.4)

ax.annotate('s1 ships here ($n$=8)', (12.9, 25.3), xytext=(9.2, 17.0),
            fontsize=6.5, color=BLUE,
            arrowprops=dict(arrowstyle='-', color=BLUE, lw=.6))
ax.annotate('OpenThoughts3 ships\nhere ($n$=13)', (11.1, 27.5), xytext=(12.2, 31.5),
            fontsize=6.5, color=ORANGE,
            arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.6))
ax.annotate('OpenThoughts-114K ships here\n(cutoff 95): finds nothing',
            (0.0, 0.0), xytext=(7.6, 1.2), fontsize=6.5, color=AQUA,
            arrowprops=dict(arrowstyle='-', color=AQUA, lw=.6))
ax.annotate('same method,\ncutoff 65', (0.7, 6.2), xytext=(4.4, 17.0),
            fontsize=6.5, color=AQUA,
            arrowprops=dict(arrowstyle='-', color=AQUA, lw=.6))

ax.set_xlabel('False positive rate (%)')
ax.set_ylabel('Detection rate (%)')
ax.set_xlim(-0.8, 20)
ax.set_ylim(-1.5, 40)
ax.grid(True, alpha=.7, lw=.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper left', handlelength=1.8, borderpad=0.1)
fig2.savefig('paper/figures/frontier.pdf')
fig2.savefig('paper/figures/frontier.png', dpi=220)
print('wrote paper/figures/frontier.{pdf,png}')
