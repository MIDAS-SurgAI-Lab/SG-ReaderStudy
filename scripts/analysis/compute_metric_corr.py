#!/usr/bin/env python3
"""Authoritative metric-vs-expert correlation, original definitions (matches Fig. 6 source).

Metrics per (setting, step, frame), using the SAME definitions as meta_corr_by_setting.py:
  BLEU-1, BLEU-4 : nltk sentence-BLEU per reference, aggregated over refs.
  METEOR         : nltk meteor per reference, aggregated over refs.
  BERTScore F1   : roberta-large (rescale, en) per reference, aggregated over refs.
  CIDEr          : pycocoevalcap corpus score per (setting, step) [100-doc corpus], per-image value.
Reference aggregation is computed BOTH ways: mean-over-refs (primary, matches the paper) and
max-over-refs (sensitivity). CIDEr has no aggregation choice (corpus).

Correlation: per (setting, step) Spearman with the case-mean expert score; the per-setting
task-averaged value (mean of the 3 step correlations) is the Fig. 6 quantity; frame-level
bootstrap 95% CI (1000) per step; Step 1 vs Step 3 difference (shared resample).

Writes csv/metric_corr_results.json + csv/metric_corr_perframe.csv.
"""
import json, re, time
import numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import spearmanr

ROOT = '../..'; OUT = '..'
TF = {1: 'Task1_Anatomy', 2: 'Task2_Final', 3: 'Task3_Final'}
SETF = {'image': 'img_only', 'pred': 'pred_graph', 'gt': 'gt_graph'}
CONDS = ['image', 'pred', 'gt']; STEPS = [1, 2, 3]
BLEU_METEOR = ['BLEU-1', 'BLEU-4', 'METEOR']

import nltk
for p in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    try: nltk.download(p, quiet=True)
    except Exception: pass
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer
import bert_score, torch
DEV = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
sm = SmoothingFunction().method1
rscorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
norm = lambda s: re.sub(r'\s+', ' ', str(s or '')).strip()

# references (up to 3 per step,frame) and candidates
raw = pd.read_csv(f'{ROOT}/translated_all.csv'); raw.columns = [c.strip() for c in raw.columns]
STEPN = {'Task1': 1, 'Task2': 2, 'Task3': 3}
refs = defaultdict(lambda: defaultdict(list))
for _, r in raw.iterrows():
    a = norm(r['Annotation_En'])
    if a: refs[STEPN[r['taskKey']]][str(r['caseId'])].append(a)
cand = {c: json.load(open(f'{ROOT}/sg-userstudy/GPT5_{SETF[c]}_task_text_keypoints_100.json')) for c in CONDS}
cases = sorted(refs[1].keys())

def s_bleu(ref, hyp, w): return sentence_bleu([word_tokenize(ref.lower())], word_tokenize(hyp.lower()), weights=w, smoothing_function=sm)
def s_met(ref, hyp): return meteor_score([word_tokenize(ref.lower())], word_tokenize(hyp.lower()))
def s_rouge(ref, hyp): return rscorer.score(ref, hyp)['rougeL'].fmeasure

t0 = time.time()
rows = []
for c in CONDS:
    for s in STEPS:
        frames = [f for f in cases if norm(cand[c][f].get(TF[s], '')) and refs[s][f]]
        hyps = {f: norm(cand[c][f][TF[s]]) for f in frames}
        # BERTScore per (frame, ref)
        bpairs = [(f, k) for f in frames for k in range(len(refs[s][f]))]
        ch = [hyps[f] for f, k in bpairs]; rr = [refs[s][f][k] for f, k in bpairs]
        _, _, F = bert_score.score(ch, rr, lang='en', rescale_with_baseline=True, verbose=False, device=DEV, batch_size=128)
        bert = defaultdict(list)
        for (f, k), v in zip(bpairs, F.tolist()): bert[f].append(v)
        # CIDEr corpus per (setting, step)
        gts = {f: refs[s][f] for f in frames}; res = {f: [hyps[f]] for f in frames}
        _, cider_per = Cider().compute_score(gts, res)
        cider = {f: float(cider_per[i]) for i, f in enumerate(frames)}
        for f in frames:
            rl = refs[s][f]; h = hyps[f]
            b1 = [s_bleu(r, h, (1, 0, 0, 0)) for r in rl]
            b4 = [s_bleu(r, h, (.25, .25, .25, .25)) for r in rl]
            me = [s_met(r, h) for r in rl]
            ro = [s_rouge(r, h) for r in rl]
            rows.append(dict(setting=c, step=s, frame=f,
                             **{'BLEU-1_mean': np.mean(b1), 'BLEU-1_max': np.max(b1),
                                'BLEU-4_mean': np.mean(b4), 'BLEU-4_max': np.max(b4),
                                'METEOR_mean': np.mean(me), 'METEOR_max': np.max(me),
                                'ROUGE-L_mean': np.mean(ro), 'ROUGE-L_max': np.max(ro),
                                'BERTScore_mean': np.mean(bert[f]), 'BERTScore_max': np.max(bert[f]),
                                'CIDEr': cider[f]}))
        print(f"  {c} S{s}: {len(frames)} frames ({time.time()-t0:.0f}s)", flush=True)
M = pd.DataFrame(rows)
M.to_csv(f'{OUT}/csv/metric_corr_perframe.csv', index=False)

# expert case-means
long = pd.read_csv(f'{OUT}/csv/long_ratings.csv'); long['frame'] = long.frame.astype(str)
cm = long.groupby(['setting', 'step', 'frame', 'score_type'])['rating'].mean().reset_index()
axmap = {'Accuracy': 'accuracy', 'Detail': 'detail'}
METRICS = ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'BERTScore']

def mcol(metric, agg):
    return 'CIDEr' if metric == 'CIDEr' else f'{metric}_{agg}'

rng = np.random.default_rng(42)
def boot(x, y, n=1000):
    N = len(x); v = []
    for _ in range(n):
        ix = rng.integers(0, N, N); r = spearmanr(x[ix], y[ix]).correlation
        if not np.isnan(r): v.append(r)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

res = {'meta': {'defs': 'sentence-BLEU/METEOR/BERTScore over refs; CIDEr per-setting corpus',
                'ref_counts': {s: len(refs[s]) for s in STEPS}, 'device': DEV}}
for agg in ['mean', 'max']:
    R = {'cells': {}, 'taskavg': {}, 'step_diff': {}}
    for axis in ['Accuracy', 'Detail']:
        for c in CONDS:
            rho_steps = {}
            for s in STEPS:
                sc = cm[(cm.setting == c) & (cm.step == s) & (cm.score_type == axmap[axis])][['frame', 'rating']]
                mg = M[(M.setting == c) & (M.step == s)].merge(sc, on='frame')
                for metric in METRICS:
                    x = mg[mcol(metric, agg)].values.astype(float); y = mg['rating'].values.astype(float)
                    rho = spearmanr(x, y).correlation
                    R['cells'][f'{axis}|{c}|{s}|{metric}'] = dict(rho=float(rho), ci=boot(x, y), n=int(len(mg)))
                    rho_steps.setdefault(metric, []).append(float(rho))
            for metric in METRICS:
                R['taskavg'][f'{axis}|{c}|{metric}'] = float(np.mean(rho_steps[metric]))
        # step1 vs step3 (shared resample)
        for c in CONDS:
            for metric in METRICS:
                d1 = M[(M.setting == c) & (M.step == 1)][['frame', mcol(metric, agg)]].merge(
                    cm[(cm.setting == c) & (cm.step == 1) & (cm.score_type == axmap[axis])][['frame', 'rating']], on='frame').set_index('frame')
                d3 = M[(M.setting == c) & (M.step == 3)][['frame', mcol(metric, agg)]].merge(
                    cm[(cm.setting == c) & (cm.step == 3) & (cm.score_type == axmap[axis])][['frame', 'rating']], on='frame').set_index('frame')
                common = sorted(set(d1.index) & set(d3.index)); n = len(common)
                d1 = d1.loc[common]; d3 = d3.loc[common]; diffs = []
                for _ in range(1000):
                    ix = rng.integers(0, n, n)
                    r1 = spearmanr(d1.iloc[:, 0].values[ix], d1['rating'].values[ix]).correlation
                    r3 = spearmanr(d3.iloc[:, 0].values[ix], d3['rating'].values[ix]).correlation
                    if not (np.isnan(r1) or np.isnan(r3)): diffs.append(r3 - r1)
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                R['step_diff'][f'{axis}|{c}|{metric}'] = dict(diff=float(np.mean(diffs)), ci=[float(lo), float(hi)], excl0=bool(lo > 0 or hi < 0))
    res[agg] = R
json.dump(res, open(f'{OUT}/csv/metric_corr_results.json', 'w'), indent=2)
print(f"[done] {time.time()-t0:.0f}s -> csv/metric_corr_results.json", flush=True)
