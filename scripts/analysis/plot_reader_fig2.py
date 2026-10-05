#!/usr/bin/env python3
"""Reader-study main figure (Fig 2): mean Likert score by setting, per step and axis,
plus Global Coherence. Values/SEM from S2_reader_full.csv, stars from Holm p in
S2_pvalues.csv (* p<.05, ** p<.01, *** p<.001). Pretendard, editable vector output.
Renders Fig2_reader_scores.{svg,pdf,png}.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import xml.etree.ElementTree as ET

for p in ['/Users/kayoung/Library/Fonts/Pretendard-Regular.otf', '/Users/kayoung/Library/Fonts/Pretendard-Medium.otf',
          '/Users/kayoung/Library/Fonts/Pretendard-SemiBold.otf', '/Users/kayoung/Library/Fonts/Pretendard-Bold.otf']:
    try: fm.fontManager.addfont(p)
    except Exception: pass
plt.rcParams['font.family'] = 'Pretendard'; plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # SVG keeps editable text (system Pretendard)
plt.rcParams['pdf.fonttype'] = 3       # Type3 = glyph outlines; renders even though Pretendard is OTF/CFF

# ---- authoritative values: (mean, sem, stars) at Image / +Pred / +GT ----
STEPS = [
    ('Step 1  Observation', {
        'acc': [(2.88, 0.08, ''), (3.22, 0.07, '***'), (3.41, 0.07, '***')],
        'det': [(3.12, 0.06, ''), (3.36, 0.05, '**'),  (3.50, 0.06, '***')]}),
    ('Step 2  Insight', {
        'acc': [(2.99, 0.07, ''), (3.27, 0.07, '***'), (3.47, 0.06, '***')],
        'det': [(3.07, 0.06, ''), (3.38, 0.06, '***'), (3.50, 0.05, '***')]}),
    ('Step 3  Plan', {
        'acc': [(3.43, 0.06, ''), (3.54, 0.05, '*'),   (3.67, 0.05, '**')],
        'det': [(3.25, 0.06, ''), (3.44, 0.05, '**'),  (3.57, 0.05, '***')]}),
]
GC = [(3.34, 0.04, ''), (3.50, 0.05, '**'), (3.62, 0.05, '***')]
XLAB = ['Image', '+Pred', '+GT']

CACC, CDET, CGC = '#9db8d8', '#e8b48a', '#a9c68c'
TACC, TDET, TGC = '#3a5f8a', '#b5702f', '#5d7d38'
INK, GRID = '#2b2b2b', '#dfe3e3'
YLIM = (2.72, 3.98); YT = [2.8, 3.0, 3.2, 3.4, 3.6, 3.8]


def flatten_svg(path):
    NS = 'http://www.w3.org/2000/svg'; ET.register_namespace('', NS)
    def fl(e):
        new = []
        for c in list(e):
            fl(c); tag = c.tag.split('}')[-1]; extra = {k for k in c.attrib if k != 'id'}
            new.extend(list(c)) if (tag == 'g' and not extra) else new.append(c)
        for c in list(e): e.remove(c)
        for c in new: e.append(c)
    t = ET.parse(path); r = t.getroot(); fl(r)
    for el in r.iter(): el.attrib.pop('clip-path', None)
    for defs in r.iter('{%s}defs' % NS):
        for cp in list(defs.findall('{%s}clipPath' % NS)): defs.remove(cp)
    def prune(e):
        for c in list(e):
            prune(c)
            if c.tag.split('}')[-1] == 'g' and len(list(c)) == 0 and not {k for k in c.attrib if k != 'id'}:
                e.remove(c)
    prune(r); t.write(path, xml_declaration=True, encoding='utf-8')


def bars(ax, x, vals, col, tcol, width):
    m = [v[0] for v in vals]; se = [v[1] for v in vals]; st = [v[2] for v in vals]
    ax.bar(x, m, width, color=col, zorder=3, linewidth=0)
    ax.errorbar(x, m, yerr=se, fmt='none', ecolor=tcol, elinewidth=1.3, capsize=3.5, zorder=4)
    for xi, mi, si, sti in zip(x, m, se, st):
        top = mi + si
        ax.text(xi, top + 0.018, f'{mi:.2f}', ha='center', va='bottom', fontsize=9, color=tcol)
        if sti:
            ax.text(xi, top + 0.075, sti, ha='center', va='bottom', fontsize=11, color=tcol, fontweight='bold')


def style(ax, title, show_y):
    ax.set_ylim(*YLIM); ax.set_yticks(YT)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c8cccc'); ax.spines[s].set_linewidth(1.1)
    ax.tick_params(length=0, labelsize=10, colors=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(XLAB, fontsize=10.5, color=INK)
    ax.set_xlabel(title, fontsize=12, color=INK, fontweight='bold', labelpad=9)
    if not show_y:
        ax.tick_params(labelleft=False)


fig = plt.figure(figsize=(13, 4.4), dpi=200)
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.10)
axes = [fig.add_subplot(gs[i]) for i in range(4)]
for a in axes[1:]: a.sharey(axes[0])

w = 0.38
for i, (title, d) in enumerate(STEPS):
    ax = axes[i]; x = np.arange(3)
    bars(ax, x - w / 2, d['acc'], CACC, TACC, w)
    bars(ax, x + w / 2, d['det'], CDET, TDET, w)
    style(ax, title, show_y=(i == 0))
    ax.set_xlim(-0.7, 2.7)

axg = axes[3]
bars(axg, np.arange(3), GC, CGC, TGC, 0.5)
style(axg, 'Global Coherence', show_y=False)
axg.set_xlim(-0.7, 2.7)

axes[0].set_ylabel('Mean Likert Score', fontsize=12.5, color=INK, fontweight='bold', labelpad=6)

leg = [mpatches.Patch(facecolor=CACC, label='Accuracy'),
       mpatches.Patch(facecolor=CDET, label='Detail'),
       mpatches.Patch(facecolor=CGC, label='Global Coherence')]
fig.legend(handles=leg, ncol=3, frameon=False, fontsize=12.5, loc='upper center',
           bbox_to_anchor=(0.5, 1.06), handlelength=1.5, columnspacing=2.6)
fig.subplots_adjust(top=0.88, bottom=0.15, left=0.055, right=0.992)
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(f'../../Fig2_reader_scores.{ext}', facecolor='white', bbox_inches='tight')
flatten_svg('../../Fig2_reader_scores.svg')
print('wrote Fig2_reader_scores.{svg,pdf,png}')
