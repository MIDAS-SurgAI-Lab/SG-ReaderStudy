#!/usr/bin/env python3
"""
Consolidated automatic evaluation of all 5 dimensions vs EXISTING GT only
(no Anatomy-Update / parked data). Bootstrap 95% CI.
  Anatomy List : 5-class (CVS-critical) multilabel micro-F1   [GT = gt_refined.structures, n=312]
  Quadruple    : action-triplet (instr,verb,target) micro-F1  [GT = gt_refined.quadruple, n=312]
  CVS          : threshold-free macro-AP over 3 criteria       [GT = gt_refined.cvs, n=312]
  BDI risk     : MAE vs 3-expert mean                          [GT = translated_all.bdi, n=100]
  Text         : BERTScore-F1 vs expert annotation             [GT = Annotation_En, n=100]
"""
import json, csv, re, numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import bert_score

ROOT = '/Users/kayoung/Downloads/judgement'
data = json.load(open(f'{ROOT}/ALL_THING.json'))
allowed = set(json.load(open(f'{ROOT}/sg-userstudy/GPT5_gt_graph_task_text_keypoints.json')).keys())
COND = ['i_refined', 'i_pred_refined', 'i_gt_refined']; LAB = ['img', 'pred', 'gt']
K2L = dict(zip(COND, LAB))
rng = np.random.default_rng(0)

def ids(it):
    s = {str(it.get(k)) for k in ('id', 'image_id', 'file_name') if it.get(k) is not None}
    return s | {x.split('.')[0] for x in s}

# ---------- structures (5-class) ----------
def canon(tokens):
    out = set()
    for t in tokens or []:
        s = str(t).lower()
        if 'plate' in s: out.add('cystic_plate')
        if 'duct' in s: out.add('cystic_duct')
        if 'arter' in s: out.add('cystic_artery')
        if 'calot' in s or 'hepatocystic' in s: out.add('calot_triangle')
        if 'bladder' in s or 'infundibul' in s or 'fundus' in s or 'hartmann' in s or s.startswith('gb'): out.add('gallbladder')
    return out

# ---------- quadruple triplet ----------
VERB = {'retract': 'retract', 'retracting': 'retract', 'traction': 'retract',
        'dissect': 'dissect', 'dissecting': 'dissect', 'dissection': 'dissect',
        'clip': 'clip', 'clipping': 'clip', 'grasp': 'grasp', 'grasping': 'grasp',
        'coagulate': 'coagulate', 'coagulating': 'coagulate', 'position': 'position', 'positioning': 'position'}
def ctarget(s):
    s = str(s).lower()
    if 'plate' in s: return 'cystic_plate'
    if 'duct' in s: return 'cystic_duct'
    if 'arter' in s: return 'cystic_artery'
    if 'calot' in s or 'hepatocystic' in s: return 'calot_triangle'
    if 'bladder' in s or 'infundibul' in s or 'fundus' in s or 'hartmann' in s or s.startswith('gb'): return 'gallbladder'
    if 'null' in s: return 'null'
    return s.strip()
def triplet(q):
    if not isinstance(q, (list, tuple)) or len(q) < 4: return None
    v = str(q[2]).lower().strip()
    return (str(q[1]).lower().strip(), VERB.get(v, v), ctarget(q[3]))

# ---------- build 312-frame records ----------
items = []
for it in data:
    if allowed and not (ids(it) & allowed): continue
    gt = it.get('gt_refined') or {}
    gcvs = gt.get('cvs')
    if not (isinstance(gcvs, list) and len(gcvs) == 3): continue
    if any(not isinstance((it.get(k) or {}).get('cvs'), list) for k in COND): continue
    rec = {'gt_struct': canon(gt.get('structures')),
           'gt_trip': set(filter(None, (triplet(q) for q in (gt.get('quadruple') or [])))),
           'gt_cvs': [1 if float(x) >= 0.5 else 0 for x in gcvs]}
    for k in COND:
        sub = it.get(k) or {}
        rec[f'{k}_struct'] = canon(sub.get('structures'))
        rec[f'{k}_trip'] = set(filter(None, (triplet(q) for q in (sub.get('quadruple') or []))))
        rec[f'{k}_cvs'] = [min(1.0, max(0.0, float(x))) for x in sub.get('cvs')]
    items.append(rec)
N = len(items)

def micro_f1(idx, k, fld):
    tp = fp = fn = 0
    for i in idx:
        g = items[i][f'gt_{fld}']; p = items[i][f'{k}_{fld}']
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return 2 * P * R / (P + R) if P + R else 0
def cvs_ap(idx, k):
    yt = np.array([items[i]['gt_cvs'] for i in idx]); ys = np.array([items[i][f'{k}_cvs'] for i in idx])
    aps = []
    for j in range(3):
        if len(set(yt[:, j].tolist())) < 2: return np.nan
        aps.append(average_precision_score(yt[:, j], ys[:, j]))
    return float(np.mean(aps))

boot312 = [rng.integers(0, N, N) for _ in range(5000)]
def ci312(fn):
    pt = fn(range(N)); bs = [fn(b) for b in boot312]; bs = [x for x in bs if not np.isnan(x)]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------- BDI (n=100) ----------
rows = [{kk.strip(): vv.strip() for kk, vv in r.items()} for r in csv.DictReader(open(f'{ROOT}/translated_all.csv', encoding='utf-8-sig'))]
eb = defaultdict(list)
for r in rows:
    if r['taskKey'] == 'Task2' and r.get('bdi', ''):
        try: eb[r['caseId']].append(float(r['bdi']))
        except: pass
mods = {k: json.load(open(f'{ROOT}/sg-userstudy/GPT5_{f}_task_text_keypoints_100.json'))
        for k, f in [('i_refined', 'img_only'), ('i_pred_refined', 'pred_graph'), ('i_gt_refined', 'gt_graph')]}
bcases = [c for c in mods['i_refined'] if c in eb]
bdi_err = {k: np.array([abs(float(mods[k][c]['Task2_BDI']) - np.median(eb[c])) for c in bcases]) for k in COND}
bootB = [rng.integers(0, len(bcases), len(bcases)) for _ in range(5000)]
def bdi_ci(k):
    e = bdi_err[k]; pt = e.mean(); bs = [e[b].mean() for b in bootB]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------- Text BERTScore (n=100) ----------
TF = {'Task1': 'Task1_Anatomy', 'Task2': 'Task2_Final', 'Task3': 'Task3_Final'}
norm = lambda s: re.sub(r'\s+', ' ', str(s or '')).strip()
refs = defaultdict(lambda: defaultdict(list))
for r in rows:
    t = r['taskKey']; c = r['caseId']; a = norm(r['Annotation_En'])
    if t in TF and a: refs[t][c].append(a)
tcases = sorted(refs['Task1'].keys())
pairs = []
for k in COND:
    for t, f in TF.items():
        for c in tcases:
            h = norm(mods[k][c].get(f, ''))
            for rf in refs[t][c]:
                if h: pairs.append((k, c, h, rf))
print(f'BERTScore pairs: {len(pairs)}')
_, _, F = bert_score.score([p[2] for p in pairs], [p[3] for p in pairs], lang='en', rescale_with_baseline=True, verbose=False)
tagg = defaultdict(list)
for i, (k, c, h, rf) in enumerate(pairs): tagg[(k, c)].append(float(F[i]))
tcasev = {kc: np.mean(v) for kc, v in tagg.items()}
def text_vals(k): return np.array([tcasev[(k, c)] for c in tcases if (k, c) in tcasev])
bootT = [rng.integers(0, len(tcases), len(tcases)) for _ in range(5000)]
def text_ci(k):
    v = text_vals(k); pt = v.mean(); bs = [v[b].mean() for b in bootT]
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------- report ----------
def line(name, direction, fn):
    cells = []
    for k in COND:
        pt, lo, hi = fn(k)
        cells.append(f'{pt:.3f} [{lo:.3f},{hi:.3f}]')
    print(f'{name:26}{direction:4}' + ''.join(f'{c:>24}' for c in cells))

print('\n' + '=' * 100)
print('AUTOMATIC EVALUATION vs EXISTING GT (bootstrap 95% CI)')
print('=' * 100)
print(f'{"dimension":26}{"dir":4}' + ''.join(f'{l:>24}' for l in LAB))
print('-' * 100)
line(f'Anatomy list (5-cls F1)', 'up', lambda k: ci312(lambda ix: micro_f1(ix, k, 'struct')))
line(f'Quadruple (triplet F1)', 'up', lambda k: ci312(lambda ix: micro_f1(ix, k, 'trip')))
line(f'CVS (macro-AP)', 'up', lambda k: ci312(lambda ix: cvs_ap(ix, k)))
line(f'BDI risk (MAE)', 'DOWN', bdi_ci)
line(f'Text (BERTScore-F1)', 'up', text_ci)
print('-' * 100)
print(f'n: Anatomy/Quadruple/CVS = {N} (312-set) ; BDI = {len(bcases)} ; Text = {len(tcases)} (x3 steps)')

# save CSV
with open(f'{ROOT}/auto_eval_all_summary.csv', 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['dimension', 'metric', 'direction', 'n', 'img', 'pred', 'gt'])
    def savef(name, metric, direction, n, fn):
        w.writerow([name, metric, direction, n] + [f'{fn(k)[0]:.4f}' for k in COND])
    savef('Anatomy list', 'micro-F1 (5-class)', 'higher', N, lambda k: ci312(lambda ix: micro_f1(ix, k, 'struct')))
    savef('Quadruple', 'action-triplet micro-F1', 'higher', N, lambda k: ci312(lambda ix: micro_f1(ix, k, 'trip')))
    savef('CVS', 'threshold-free macro-AP', 'higher', N, lambda k: ci312(lambda ix: cvs_ap(ix, k)))
    savef('BDI risk', 'MAE vs expert median', 'lower', len(bcases), bdi_ci)
    savef('Text', 'BERTScore-F1 vs expert', 'higher', len(tcases), text_ci)
print('Saved: auto_eval_all_summary.csv')
