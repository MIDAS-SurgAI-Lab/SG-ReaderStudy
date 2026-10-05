#!/usr/bin/env python3
"""
UNIFIED CVS metric = threshold-free Average Precision (AP).
  - prediction score: continuous in [0,1], used as ranking (NO threshold)
  - ground truth: gt_refined CVS, binarized to {0,1} (required to define positives for AP)
Reports per-criterion AP + macro mAP for img / pred / gt, with paired bootstrap 95% CI and
pairwise contrasts. Replaces the contradictory cvs_map_summary.json (threshold-0.7) numbers.
"""
import json, numpy as np, csv
from pathlib import Path
from sklearn.metrics import average_precision_score

ROOT = Path('/Users/kayoung/Downloads/judgement')
JSON_PATH = ROOT / 'ALL_THING.json'
KEY_FILTER = ROOT / 'sg-userstudy/GPT5_gt_graph_task_text_keypoints.json'
PRED = ['i_refined', 'i_pred_refined', 'i_gt_refined']
LABEL = {'i_refined': 'img', 'i_pred_refined': 'pred', 'i_gt_refined': 'gt'}
GT_BIN = 0.5   # GT binarization threshold (>=0.5 -> positive)

def cvsvec(item, key):
    v = item.get(key, {}).get('cvs')
    if not isinstance(v, (list, tuple)) or len(v) != 3: return None
    try: return [min(1.0, max(0.0, float(x))) for x in v]
    except Exception: return None

data = json.loads(JSON_PATH.read_text())
allowed = set(json.loads(KEY_FILTER.read_text()).keys())
yt, ys = [], {p: [] for p in PRED}
for item in data:
    iid = str(item.get('image_id') or item.get('id') or item.get('file_name', '')).split('.')[0]
    # match allowed ids by any id-like field
    ids = {str(item.get(k)) for k in ('image_id', 'id', 'file_name') if item.get(k) is not None}
    ids |= {s.split('.')[0] for s in ids}
    if allowed and not (ids & allowed): continue
    gt = cvsvec(item, 'gt_refined')
    preds = {p: cvsvec(item, p) for p in PRED}
    if gt is None or any(v is None for v in preds.values()): continue
    yt.append(gt)
    for p in PRED: ys[p].append(preds[p])
yt = np.array(yt); ys = {p: np.array(ys[p]) for p in PRED}
N = len(yt)
print(f'N samples = {N}')
# GT value check
uniq = sorted(set(yt.flatten().tolist()))
print(f'GT distinct values: {uniq[:12]}{" ..." if len(uniq)>12 else ""}')
ytb = (yt >= GT_BIN).astype(int)
print(f'GT binarized at >={GT_BIN}: positive rate per criterion = {ytb.mean(axis=0).round(3).tolist()}')
if set(uniq) <= {0.0, 1.0}:
    print('  (GT already binary — threshold is a no-op)')

def macro_map(idx, ytb, yscore):
    aps = []
    for j in range(3):
        t = ytb[idx, j]
        if len(set(t.tolist())) < 2:      # degenerate column in this resample
            return np.nan
        aps.append(average_precision_score(t, yscore[idx, j]))
    return float(np.mean(aps))

def per_crit(ytb, yscore):
    return [average_precision_score(ytb[:, j], yscore[:, j]) for j in range(3)]

# point estimates
print('\n' + '=' * 62)
print('UNIFIED CVS — threshold-free AP  (GT binarized @0.5, preds continuous)')
print('=' * 62)
pts = {}
print(f'{"setting":8}{"C1_AP":>9}{"C2_AP":>9}{"C3_AP":>9}{"macro_mAP":>11}')
for p in PRED:
    pc = per_crit(ytb, ys[p]); mm = float(np.mean(pc)); pts[p] = mm
    print(f'{LABEL[p]:8}{pc[0]:9.3f}{pc[1]:9.3f}{pc[2]:9.3f}{mm:11.3f}')

# paired bootstrap (same resampled indices across settings)
B = 5000
rng = np.random.default_rng(0)
boot = {p: [] for p in PRED}
contrasts = [('i_refined', 'i_pred_refined'), ('i_refined', 'i_gt_refined'), ('i_pred_refined', 'i_gt_refined')]
bd = {c: [] for c in contrasts}
for _ in range(B):
    idx = rng.integers(0, N, N)
    mm = {p: macro_map(idx, ytb, ys[p]) for p in PRED}
    if any(np.isnan(v) for v in mm.values()): continue
    for p in PRED: boot[p].append(mm[p])
    for a, b in contrasts: bd[(a, b)].append(mm[a] - mm[b])

def ci(a): a = np.array(a); return np.percentile(a, 2.5), np.percentile(a, 97.5)
print('\nmacro mAP with 95% bootstrap CI:')
for p in PRED:
    lo, hi = ci(boot[p]); print(f'  {LABEL[p]:5} {pts[p]:.3f}  [{lo:.3f}, {hi:.3f}]')
print('\nPaired contrasts  ΔmAP = A - B  (95% CI, two-sided boot p):')
rows = [['contrast', 'A', 'B', 'dmAP', 'ci_lo', 'ci_hi', 'p_boot']]
for a, b in contrasts:
    d = np.array(bd[(a, b)]); dm = float(d.mean()); lo, hi = ci(d)
    p_two = 2 * min((d > 0).mean(), (d < 0).mean())
    star = '  *' if (lo > 0 or hi < 0) else '  ns'
    print(f'  {LABEL[a]:4} vs {LABEL[b]:4}: Δ={dm:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p_two:.3f}{star}')
    rows.append([f'{LABEL[a]}_vs_{LABEL[b]}', LABEL[a], LABEL[b], f'{dm:.4f}', f'{lo:.4f}', f'{hi:.4f}', f'{p_two:.4f}'])

# save
with open(ROOT / 'cvs_ap_unified.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['setting', 'C1_AP', 'C2_AP', 'C3_AP', 'macro_mAP', 'ci_lo', 'ci_hi'])
    for p in PRED:
        pc = per_crit(ytb, ys[p]); lo, hi = ci(boot[p])
        w.writerow([LABEL[p], f'{pc[0]:.4f}', f'{pc[1]:.4f}', f'{pc[2]:.4f}', f'{pts[p]:.4f}', f'{lo:.4f}', f'{hi:.4f}'])
    w.writerow([])
    w.writerows(rows)
print('\nSaved: cvs_ap_unified.csv')
