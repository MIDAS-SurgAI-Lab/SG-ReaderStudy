#!/usr/bin/env python3
"""Automatic-evaluation figure (Fig 3): Action Triplet F1, Anatomy List F1, CVS mAP (Step 1)
and BDI Risk MAE (Step 2), each across Image / +Pred / +GT. Triplet/Anatomy/CVS/BDI values
verified against auto_eval_quadruple.py, cvs_ap_unified, and bdi_mae_per_case.csv
(MAE vs human median; error bars = SEM). Pretendard, editable vector output.
Renders Fig3_autoeval.{svg,pdf,png}.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.gridspec as gridspec
import xml.etree.ElementTree as ET

for p in ['/Users/kayoung/Library/Fonts/Pretendard-Regular.otf', '/Users/kayoung/Library/Fonts/Pretendard-Medium.otf',
          '/Users/kayoung/Library/Fonts/Pretendard-SemiBold.otf', '/Users/kayoung/Library/Fonts/Pretendard-Bold.otf']:
    try: fm.fontManager.addfont(p)
    except Exception: pass
plt.rcParams['font.family'] = 'Pretendard'; plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # SVG keeps editable text (system Pretendard)
plt.rcParams['pdf.fonttype'] = 3       # Type3 = glyph outlines; renders even though Pretendard is OTF/CFF

INK, GRID = '#2b2b2b', '#e3e6e6'
XLAB = ['Image', '+Pred', '+GT']
PANELS = [
    dict(name='Action Triplet', step=1, ylabel='F1-Score', ylim=(0, 1.08), yt=[.2, .4, .6, .8, 1.0],
         vals=[0.30, 0.39, 0.52], err=None, fill='#a8a3d4', text='#5a4fa0', kind='bar', showy=True),
    dict(name='Anatomy List', step=1, ylabel=None, ylim=(0, 1.08), yt=[.2, .4, .6, .8, 1.0],
         vals=[0.58, 0.68, 0.86], err=None, fill='#eaadad', text='#b04a4e', kind='bar', showy=False),
    dict(name='CVS', step=1, ylabel='mAP', ylim=(0, 0.43), yt=[.1, .2, .3, .4],
         vals=[0.32, 0.35, 0.39], err=None, fill='#7fbdb0', text='#2f7a6a', kind='bar', showy=True),
    dict(name='BDI Risk', step=2, ylabel='MAE', ylim=(0.80, 1.17), yt=[.9, 1.0, 1.1],
         vals=[0.99, 1.04, 0.90], err=[0.075, 0.076, 0.069], fill='#d3c74a', text='#897d17', kind='dot', showy=True),
]


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


fig = plt.figure(figsize=(13, 3.7), dpi=200)
gs = gridspec.GridSpec(1, 4, wspace=0.28)
x = np.arange(3)
for i, P in enumerate(PANELS):
    ax = fig.add_subplot(gs[i])
    rng = P['ylim'][1] - P['ylim'][0]; gap = rng * 0.022
    if P['kind'] == 'bar':
        ax.bar(x, P['vals'], 0.62, color=P['fill'], zorder=3, linewidth=0)
        for xi, v in zip(x, P['vals']):
            ax.text(xi, v + gap, f'{v:.2f}', ha='center', va='bottom', fontsize=10, color=P['text'], fontweight='medium')
    else:
        ax.errorbar(x, P['vals'], yerr=P['err'], fmt='o', color=P['fill'], ecolor=P['fill'],
                    markersize=8, elinewidth=1.7, capsize=4.5, capthick=1.7, zorder=3)
        for xi, v in zip(x, P['vals']):
            ax.text(xi + 0.14, v, f'{v:.2f}', ha='left', va='center', fontsize=10, color=P['text'], fontweight='medium')
    ax.set_ylim(*P['ylim']); ax.set_yticks(P['yt'])
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c8cccc'); ax.spines[s].set_linewidth(1.1)
    ax.tick_params(length=0, labelsize=10, colors=INK)
    ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=10.5, color=INK)
    ax.set_xlim(-0.62, 2.62 if P['kind'] == 'bar' else 2.78)
    if not P['showy']:
        ax.tick_params(labelleft=False)
    if P['ylabel']:
        ax.set_ylabel(P['ylabel'], fontsize=12, color=INK, fontweight='bold', labelpad=5)
    ax.set_xlabel(P['name'], fontsize=12.5, color=INK, fontweight='bold', labelpad=8)
    ax.annotate(f'(Step {P["step"]})', xy=(0.5, -0.235), xycoords='axes fraction',
                ha='center', va='top', fontsize=10, color='#6b6b6b')

fig.subplots_adjust(top=0.95, bottom=0.24, left=0.05, right=0.99)
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(f'../../Fig3_autoeval.{ext}', facecolor='white', bbox_inches='tight')
flatten_svg('../../Fig3_autoeval.svg')
print('wrote Fig3_autoeval.{svg,pdf,png}')
