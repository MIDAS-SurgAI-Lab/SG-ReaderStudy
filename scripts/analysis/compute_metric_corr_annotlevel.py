#!/usr/bin/env python3
"""Annotation-level metric-vs-expert agreement with a crossed mixed model.

Instead of averaging the metric over a frame's up-to-3 references and correlating with the
case-mean expert score, we pair each rater's OWN annotation with that rater's OWN Likert score:
  for (rater j, frame f, step s, setting c)
    x = metric(model_text_{c,f,s}, annotation_{j,f,s})    # that rater's annotation, no averaging
    y = rater j's Likert score for setting c at step s
The association is estimated within each (setting, step, metric, score-type) cell with a linear
mixed model, standardizing x and y within rater and using crossed random intercepts for rater and
frame (same random-effect structure as the propagation analysis).

CIDEr is corpus-based (needs a reference set, not a single annotation), so it is omitted here and
kept only in the frame-level table. Metrics: BLEU-1, BLEU-4, METEOR, ROUGE-L, BERTScore.

Writes csv/metric_corr_annotlevel.json.
"""
import json, re, time
import numpy as np, pandas as pd
from collections import defaultdict
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

ROOT = '../..'; OUT = '..'
TF = {1: 'Task1_Anatomy', 2: 'Task2_Final', 3: 'Task3_Final'}
SETF = {'image': 'img_only', 'pred': 'pred_graph', 'gt': 'gt_graph'}
CONDS = ['image', 'pred', 'gt']; STEPS = [1, 2, 3]
STEPN = {'Task1': 1, 'Task2': 2, 'Task3': 3}
norm = lambda s: re.sub(r'\s+', ' ', str(s or '')).strip()

import nltk
for p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    try: nltk.download(p, quiet=True)
    except Exception: pass
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import bert_score, torch
DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
sm = SmoothingFunction().method1
rsc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def s_bleu(r, h, w): return sentence_bleu([word_tokenize(r.lower())], word_tokenize(h.lower()), weights=w, smoothing_function=sm)
def s_met(r, h): return meteor_score([word_tokenize(r.lower())], word_tokenize(h.lower()))
def s_rouge(r, h): return rsc.score(r, h)['rougeL'].fmeasure

# raw ratings: each row = (rater, frame, step) with that rater's annotation + scores for 3 settings
raw = pd.read_csv(f'{ROOT}/translated_all.csv'); raw.columns = [c.strip() for c in raw.columns]
cand = {c: json.load(open(f'{ROOT}/sg-userstudy/GPT5_{SETF[c]}_task_text_keypoints_100.json')) for c in CONDS}
SCCOL = {'accuracy': 'score1', 'detail': 'score2'}

# build annotation-level records: one per (rater, frame, step, setting)
recs = []
bpairs = []  # (idx, hyp, ref) for BERTScore batching
for _, r in raw.iterrows():
    s = STEPN[r['taskKey']]; f = str(r['caseId']); ann = norm(r['Annotation_En'])
    if not ann: continue
    for c in CONDS:
        hyp = norm(cand[c].get(f, {}).get(TF[s], ''))
        if not hyp: continue
        rec = dict(rater=int(r['expertId']), frame=f, step=s, setting=c,
                   accuracy=float(r[f'{c[:3] if c!="image" else "img"}_score1']) if False else float(r[f'{("img" if c=="image" else c)}_score1']),
                   detail=float(r[f'{("img" if c=="image" else c)}_score2']),
                   hyp=hyp, ref=ann)
        idx = len(recs); recs.append(rec); bpairs.append((idx, hyp, ann))
D = pd.DataFrame(recs)
print(f"annotation-level records: {len(D)}", flush=True)

# per-pair metrics
t0 = time.time()
D['BLEU-1'] = [s_bleu(r.ref, r.hyp, (1, 0, 0, 0)) for r in D.itertuples()]
D['BLEU-4'] = [s_bleu(r.ref, r.hyp, (.25, .25, .25, .25)) for r in D.itertuples()]
D['METEOR'] = [s_met(r.ref, r.hyp) for r in D.itertuples()]
D['ROUGE-L'] = [s_rouge(r.ref, r.hyp) for r in D.itertuples()]
print(f"  ngram/rouge done ({time.time()-t0:.0f}s)", flush=True)
_, _, F = bert_score.score([p[1] for p in bpairs], [p[2] for p in bpairs],
                           lang='en', rescale_with_baseline=True, verbose=False, device=DEV, batch_size=128)
bt = {p[0]: float(F[i]) for i, p in enumerate(bpairs)}
D['BERTScore'] = [bt[i] for i in D.index]
print(f"  bertscore done ({time.time()-t0:.0f}s)", flush=True)
D.to_csv(f'{OUT}/csv/metric_corr_annotlevel_perpair.csv', index=False)

METRICS = ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'BERTScore']
def star(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'

def zwithin(df, col):
    return df.groupby('rater')[col].transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))

results = {'meta': {'level': 'annotation (rater-own annotation vs rater-own score)',
                    'model': 'crossed mixed: db ~ da, (1|rater)+(1|frame), within-rater standardized',
                    'metrics': METRICS, 'note': 'CIDEr omitted (corpus-based)'},
           'cells': {}}
for axis in ['accuracy', 'detail']:
    for c in CONDS:
        for s in STEPS:
            for m in METRICS:
                d = D[(D.setting == c) & (D.step == s)][['rater', 'frame', m, axis]].dropna().copy()
                d['x'] = zwithin(d, m); d['y'] = zwithin(d, axis)
                d['rater'] = d.rater.astype(str); d['frame'] = d.frame.astype(str); d['grp'] = 1
                try:
                    mod = smf.mixedlm("y ~ x", d, groups=d['grp'],
                                      vc_formula={'rater': '0+C(rater)', 'frame': '0+C(frame)'},
                                      re_formula='0').fit(reml=False, method='lbfgs', maxiter=400)
                    b = mod.params['x']; se = mod.bse['x']; p = mod.pvalues['x']
                except Exception:
                    b = se = p = float('nan')
                results['cells'][f'{axis.title()}|{c}|{s}|{m}'] = dict(
                    beta=round(float(b), 3), se=round(float(se), 3),
                    ci=[round(float(b - 1.96 * se), 3), round(float(b + 1.96 * se), 3)],
                    p=round(float(p), 4), n=int(len(d)))
    # task-averaged beta (mean of 3 step betas) per (setting, metric)
for axis in ['accuracy', 'detail']:
    for c in CONDS:
        for m in METRICS:
            bs = [results['cells'][f'{axis.title()}|{c}|{s}|{m}']['beta'] for s in STEPS]
            results.setdefault('taskavg', {})[f'{axis.title()}|{c}|{m}'] = round(float(np.mean(bs)), 3)

json.dump(results, open(f'{OUT}/csv/metric_corr_annotlevel.json', 'w'), indent=2)
print(f"[done] {time.time()-t0:.0f}s -> csv/metric_corr_annotlevel.json", flush=True)
# quick print
print("\nBERTScore task-avg beta (annotation-level, crossed mixed):")
for axis in ['Accuracy', 'Detail']:
    print(f"  {axis}:", {c: results['taskavg'][f'{axis}|{c}|BERTScore'] for c in CONDS})
