#!/usr/bin/env python3
"""Quadruple = action-triplet (instrument, verb, target) micro-F1, with a documented
synonym/canonicalization map. Only CLEAR same-device / same-action synonyms are merged;
ambiguous or GT-absent tokens are left unmatched (strict)."""
import json, numpy as np
ROOT = '/Users/kayoung/Downloads/judgement'
data = json.load(open(f'{ROOT}/ALL_THING.json'))
allowed = set(json.load(open(f'{ROOT}/sg-userstudy/GPT5_gt_graph_task_text_keypoints.json')).keys())
def ids(it):
    s = {str(it.get(k)) for k in ('id', 'image_id', 'file_name') if it.get(k) is not None}
    return s | {x.split('.')[0] for x in s}
COND = ['i_refined', 'i_pred_refined', 'i_gt_refined']; LAB = {'i_refined': 'img', 'i_pred_refined': 'pred', 'i_gt_refined': 'gt'}

# ---- VERB map: morphological variants + clear synonyms ----
VERB = {'retract': 'retract', 'retracting': 'retract', 'retraction': 'retract', 'traction': 'retract',
        'dissect': 'dissect', 'dissecting': 'dissect', 'dissection': 'dissect',
        'blunt dissection': 'dissect', 'blunt_dissection': 'dissect',
        'clip': 'clip', 'clipping': 'clip',
        'grasp': 'grasp', 'grasping': 'grasp',
        'coagulate': 'coagulate', 'coagulating': 'coagulate', 'coagulation': 'coagulate',
        'cauterize': 'coagulate', 'cautery': 'coagulate', 'cauterizing': 'coagulate', 'electrocautery': 'coagulate',
        'position': 'position', 'positioning': 'position'}
# ---- INSTRUMENT map: only suction-irrigator family merged; rest verbatim ----
INSTR = {'suction': 'irrigator', 'suctioning': 'irrigator', 'aspirator': 'irrigator',
         'suction-irrigator': 'irrigator', 'suction irrigator': 'irrigator'}
def vb(s): s = str(s).lower().strip(); return VERB.get(s, s)
def instr(s): s = str(s).lower().strip(); return INSTR.get(s, s)
def ct(s):  # target -> CVS-5 canonical, else verbatim
    s = str(s).lower()
    if 'plate' in s: return 'cystic_plate'
    if 'duct' in s: return 'cystic_duct'
    if 'arter' in s: return 'cystic_artery'
    if 'calot' in s or 'hepatocystic' in s: return 'calot_triangle'
    if 'bladder' in s or 'infundibul' in s or 'fundus' in s or 'hartmann' in s or s.startswith('gb'): return 'gallbladder'
    if 'null' in s: return 'null'
    return s.strip()
def trip(q):
    if not isinstance(q, (list, tuple)) or len(q) < 4: return None
    return (instr(q[1]), vb(q[2]), ct(q[3]))

items = []
for it in data:
    if allowed and not (ids(it) & allowed): continue
    if not (it.get('gt_refined') or {}).get('cvs'): continue
    rec = {'gt': set(filter(None, (trip(q) for q in (it.get('gt_refined') or {}).get('quadruple') or [])))}
    for k in COND:
        rec[k] = set(filter(None, (trip(q) for q in (it.get(k) or {}).get('quadruple') or [])))
    items.append(rec)
N = len(items); print('N =', N)
def f1(elemfn, k, idx):
    tp = fp = fn = 0
    for i in idx:
        r = items[i]; g = set(elemfn(t) for t in r['gt']); p = set(elemfn(t) for t in r[k])
        tp += len(p & g); fp += len(p - g); fn += len(g - p)
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return P, R, (2 * P * R / (P + R) if P + R else 0)
rng = np.random.default_rng(0); boots = [rng.integers(0, N, N) for _ in range(5000)]
ELEM = {'full triplet': lambda t: t, 'instrument': lambda t: t[0], 'verb': lambda t: t[1], 'target': lambda t: t[2]}
print(f'\n=== Quadruple F1 with expanded synonym map (95% CI on full triplet) ===')
print(f'{"element":14}'+''.join(f'{LAB[k]:>22}' for k in COND))
for name, fn in ELEM.items():
    cells = []
    for k in COND:
        pt = f1(fn, k, range(N))[2]
        if name == 'full triplet':
            bs = [f1(fn, k, b)[2] for b in boots]
            cells.append(f'{pt:.3f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]')
        else:
            cells.append(f'{pt:.3f}')
    print(f'{name:14}'+''.join(f'{c:>22}' for c in cells))
print('\n(strict-match baseline for full triplet was: img 0.295 / pred 0.354 / gt 0.498)')
