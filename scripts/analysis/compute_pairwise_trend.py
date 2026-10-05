#!/usr/bin/env python3
"""Does an automatic metric reproduce the expert's RELATIVE trend across settings and steps?

Rationale: Likert scores are treated as relative (setting-to-setting and step-to-step) rather
than absolute. So we ask whether, per frame, the metric's difference between two conditions
tracks the expert's difference. Ties are kept (a difference of 0 is a valid value).

For each pair (A, B) and metric and score-type:
  per frame f:  x_f = expert_score(A) - expert_score(B)   (case-mean; 0 if tie)
                y_f = metric(A) - metric(B)
  trend = Spearman correlation across frames (tie-inclusive), with a frame bootstrap 95% CI.

Setting pairs (within a step, averaged over steps): GT-Image, Pred-Image, GT-Pred.
Step pairs   (within a setting): S2-S1, S3-S2, S3-S1.
Metrics: BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr, BERTScore (all higher = better).

Writes csv/pairwise_trend.json.
"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = '..'
M = pd.read_csv(f'{OUT}/csv/metric_corr_perframe.csv'); M['frame'] = M.frame.astype(str)
long = pd.read_csv(f'{OUT}/csv/long_ratings.csv'); long['frame'] = long.frame.astype(str)
METS = {'BLEU-1': 'BLEU-1_mean', 'BLEU-4': 'BLEU-4_mean', 'METEOR': 'METEOR_mean',
        'ROUGE-L': 'ROUGE-L_mean', 'CIDEr': 'CIDEr', 'BERTScore': 'BERTScore_mean'}
MNAMES = list(METS)
frames = sorted(M.frame.unique())
rng = np.random.default_rng(42)

# pre-index expert case-means and metric values for speed
exp = long.groupby(['setting', 'step', 'score_type', 'frame']).rating.mean()
met = M.set_index(['setting', 'step', 'frame'])

def boot_ci(x, y, nb=1000):
    n = len(x); v = []
    for _ in range(nb):
        ix = rng.integers(0, n, n)
        r = spearmanr(x[ix], y[ix]).correlation
        if not np.isnan(r): v.append(r)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def setting_trend(cA, cB, col, axis):
    """step-averaged Spearman of (expert diff, metric diff), tie-inclusive."""
    rs = []; xs_all = []; ys_all = []
    for step in [1, 2, 3]:
        x, y = [], []
        for f in frames:
            try:
                ea = exp[(cA, step, axis, f)]; eb = exp[(cB, step, axis, f)]
                ma = met.loc[(cA, step, f), col]; mb = met.loc[(cB, step, f), col]
            except KeyError:
                continue
            x.append(ea - eb); y.append(ma - mb)
        x, y = np.array(x), np.array(y)
        rs.append(spearmanr(x, y).correlation)
        xs_all.append(x); ys_all.append(y)
    # bootstrap on the pooled step-stacked diffs (frame-level resample within each step)
    return float(np.nanmean(rs)), np.concatenate(xs_all), np.concatenate(ys_all)

def step_trend(sA, sB, c, col, axis):
    x, y = [], []
    for f in frames:
        try:
            ea = exp[(c, sA, axis, f)]; eb = exp[(c, sB, axis, f)]
            ma = met.loc[(c, sA, f), col]; mb = met.loc[(c, sB, f), col]
        except KeyError:
            continue
        x.append(ea - eb); y.append(ma - mb)
    x, y = np.array(x), np.array(y)
    return float(spearmanr(x, y).correlation), x, y

res = {'meta': {'method': 'Spearman of per-frame (expert diff, metric diff), tie-inclusive',
                'setting_pairs': ['GT-Image', 'Pred-Image', 'GT-Pred'],
                'step_pairs': ['S2-S1', 'S3-S2', 'S3-S1'], 'metrics': MNAMES},
       'setting': {}, 'step': {}}
SETPAIRS = [('GT-Image', 'gt', 'image'), ('Pred-Image', 'pred', 'image'), ('GT-Pred', 'gt', 'pred')]
STEPPAIRS = [('S2-S1', 2, 1), ('S3-S2', 3, 2), ('S3-S1', 3, 1)]

for axis in ['accuracy', 'detail']:
    for m, col in METS.items():
        for lab, cA, cB in SETPAIRS:
            r, x, y = setting_trend(cA, cB, col, axis)
            res['setting'][f'{axis.title()}|{lab}|{m}'] = dict(rho=round(r, 3), ci=[round(v, 3) for v in boot_ci(x, y)], n=int(len(x)))
        for lab, sA, sB in STEPPAIRS:
            r, x, y = step_trend(sA, sB, 'gt', col, axis)   # step trend under GT setting
            res['step'][f'{axis.title()}|{lab}|{m}'] = dict(rho=round(r, 3), ci=[round(v, 3) for v in boot_ci(x, y)], n=int(len(x)))

json.dump(res, open(f'{OUT}/csv/pairwise_trend.json', 'w'), indent=2)
print('wrote pairwise_trend.json')
print('\nSetting trend (Accuracy):')
for lab, _, _ in SETPAIRS:
    print(' ', lab, {m: res['setting'][f'Accuracy|{lab}|{m}']['rho'] for m in MNAMES})
print('\nStep trend under GT (Accuracy):')
for lab, _, _ in STEPPAIRS:
    print(' ', lab, {m: res['step'][f'Accuracy|{lab}|{m}']['rho'] for m in MNAMES})
