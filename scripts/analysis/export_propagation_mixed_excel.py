#!/usr/bin/env python3
"""Propagation analysis — MIXED MODEL with CROSSED random effects (rater + frame) as the primary
method — exported step by step to Excel.

Reader-study data are doubly clustered: each rating belongs to a rater (repeated over ~30 frames)
and to a frame (repeated over its 3 raters). The primary model therefore uses crossed random
intercepts for rater and frame.

Primary model, per (step pair, setting, axis):
  response  = Delta at step b, standardized within rater
  predictor = Delta at step a, standardized within rater
  db_z ~ da_z,  crossed random intercepts: (1|rater) + (1|frame)
  The fixed-effect slope beta is the within-rater association; its p is the significance.
  No Fisher-z, no rank transform.

Secondary (robustness only): per-rater Spearman aggregated by Fisher-z + one-sample t-test.

Sheets (computation order):
  0_README
  STEP1_delta            per (rater, frame) Delta = score(setting) - score(Image), all steps/settings/axes
  STEP2_within_z         Delta standardized within each rater -> model inputs
  STEP3_model_inputs     one worked cell (Acc, GT, dS1->dS2): long table fed to the model
  STEP4_random_effects   worked cell: fixed slope + rater/frame variance components + per-rater & per-frame intercepts
  STEP5_results_primary  ALL 12 cells: crossed-mixed beta, SE, z, p, sig  <-- report these
  AUX_twostage           robustness: per-rater Spearman + Fisher-z + t-test
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr, ttest_1samp
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('../../raw_data/01_reader_study/expert_ratings.csv')
df.columns = [c.strip() for c in df.columns]
SC = {'Accuracy': 'score1', 'Detail': 'score2'}
SETTINGS = ['gt', 'pred']
PAIRS = {'dS1-dS2': (1, 2), 'dS2-dS3': (2, 3), 'dS1-dS3': (1, 3)}
def star(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'

# ---------- STEP 1 ----------
rows = []
for axis, axcol in SC.items():
    for setting in SETTINGS:
        ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
        for step in [1, 2, 3]:
            for _, r in df[df.taskKey == f'Task{step}'].iterrows():
                rows.append(dict(axis=axis, setting=setting, step=step,
                                 rater=int(r.expertId), frame=int(r.caseId),
                                 delta=int(r[col] - r[ic])))
step1 = pd.DataFrame(rows)

# ---------- STEP 2 ----------
step2 = step1.copy()
step2['delta_z_within_rater'] = step2.groupby(['axis', 'setting', 'step', 'rater'])['delta']\
    .transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))

# ---------- model table for one cell ----------
def model_table(axis, setting, a, b):
    axcol = SC[axis]; ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
    da = df[df.taskKey == f'Task{a}'].set_index(['expertId', 'caseId']).eval(f"{col}-{ic}").rename('da_raw')
    db = df[df.taskKey == f'Task{b}'].set_index(['expertId', 'caseId']).eval(f"{col}-{ic}").rename('db_raw')
    d = pd.concat([da, db], axis=1).dropna().reset_index().rename(columns={'expertId': 'rater', 'caseId': 'frame'})
    d['da_z'] = d.groupby('rater').da_raw.transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))
    d['db_z'] = d.groupby('rater').db_raw.transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))
    d['grp'] = 1; d['rater'] = d.rater.astype(str); d['frame'] = d.frame.astype(str)
    return d

def fit_crossed(d):
    # crossed random intercepts: rater + frame, via variance components on a single group
    return smf.mixedlm("db_z ~ da_z", d, groups=d['grp'],
                       vc_formula={'rater': '0 + C(rater)', 'frame': '0 + C(frame)'},
                       re_formula='0').fit(reml=False, method='lbfgs', maxiter=500)

# ---------- STEP 3 ----------
worked = model_table('Accuracy', 'gt', 1, 2)
step3 = worked[['rater', 'frame', 'da_raw', 'db_raw', 'da_z', 'db_z']].copy()
step3.insert(0, 'axis', 'Accuracy'); step3.insert(1, 'setting', 'gt (GT graph)')
step3 = step3.rename(columns={'da_raw': 'dS1_raw', 'db_raw': 'dS2_raw',
                              'da_z': 'dS1_z(=predictor)', 'db_z': 'dS2_z(=response)'})

# ---------- STEP 4 ----------
mw = fit_crossed(worked)
vc = mw.vcomp   # variance components [rater, frame]
step4_meta = pd.DataFrame({
    'quantity': ['fixed intercept', 'fixed slope (da_z) = PRIMARY beta', 'slope SE', 'slope z', 'slope p',
                 'Var(rater intercept)', 'Var(frame intercept)', 'residual Var'],
    'value': [round(mw.params['Intercept'], 4), round(mw.params['da_z'], 4), round(mw.bse['da_z'], 4),
              round(mw.tvalues['da_z'], 3), round(mw.pvalues['da_z'], 5),
              round(float(vc[0]), 4), round(float(vc[1]), 4), round(mw.scale, 4)]
})
# per-rater and per-frame random intercepts (BLUPs)
re = mw.random_effects[1]  # Series indexed by 'rater[..]' and 'frame[..]'
blup = []
for name, val in re.items():
    kind = 'rater' if 'rater' in name else ('frame' if 'frame' in name else 'other')
    lab = name.split('[')[-1].rstrip(']').replace('C(rater)', '').replace('C(frame)', '').strip('. ')
    blup.append(dict(effect=kind, id=lab, random_intercept=round(val, 4)))
step4_blup = pd.DataFrame(blup)
step4_rater = step4_blup[step4_blup.effect == 'rater'].drop(columns='effect').reset_index(drop=True)

# ---------- STEP 5: all 12 cells ----------
prim = []
for axis in SC:
    for setting in SETTINGS:
        for pk, (a, b) in PAIRS.items():
            d = model_table(axis, setting, a, b)
            m = fit_crossed(d)
            b = m.params['da_z']; se = m.bse['da_z']
            prim.append(dict(axis=axis, setting=setting, pair=pk,
                             beta=round(b, 3), SE=round(se, 3),
                             CI95_low=round(b - 1.96 * se, 3), CI95_high=round(b + 1.96 * se, 3),
                             CI95=f"[{b-1.96*se:+.2f}, {b+1.96*se:+.2f}]",
                             z=round(m.tvalues['da_z'], 2), p=round(m.pvalues['da_z'], 4),
                             var_rater=round(float(m.vcomp[0]), 3), var_frame=round(float(m.vcomp[1]), 3),
                             n_obs=len(d), n_raters=d.rater.nunique(), n_frames=d.frame.nunique()))
step5 = pd.DataFrame(prim)

# ---------- AUX two-stage ----------
aux = []
for axis in SC:
    for setting in SETTINGS:
        for pk, (a, b) in PAIRS.items():
            axcol = SC[axis]; ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
            zs = []; npos = 0
            for rid, g in df.groupby('expertId'):
                da = g[g.taskKey == f'Task{a}'].set_index('caseId').eval(f"{col}-{ic}")
                db = g[g.taskKey == f'Task{b}'].set_index('caseId').eval(f"{col}-{ic}")
                j = pd.concat([da, db], axis=1, keys=['a', 'b']).dropna()
                r = spearmanr(j.a, j.b).correlation
                if not np.isnan(r): zs.append(np.arctanh(np.clip(r, -.999999, .999999))); npos += int(r > 0)
            p = ttest_1samp(zs, 0).pvalue
            aux.append(dict(axis=axis, setting=setting, pair=pk,
                            spearman_rbar=round(np.tanh(np.mean(zs)), 3), raters_r_gt0=f"{npos}/10",
                            fisher_t_p=round(p, 4), sig=star(p)))
auxdf = pd.DataFrame(aux)

readme = pd.DataFrame({
    'sheet': ['STEP1_delta', 'STEP2_within_z', 'STEP3_model_inputs', 'STEP4_random_effects',
              'STEP5_results_primary', 'AUX_twostage'],
    'description': [
        'STEP 1. Per (rater, frame) graph-induced change Delta = score(setting) - score(Image), for every step/setting/axis.',
        'STEP 2. Delta standardized WITHIN each rater (per axis/setting/step: mean 0, sd 1). These z-values feed the model.',
        'STEP 3. Worked cell (Accuracy, GT, dS1->dS2): the long table given to the model. predictor=dS1_z, response=dS2_z.',
        'STEP 4. CROSSED mixed model for that cell: fixed slope (PRIMARY beta) + rater & frame variance components + per-rater/per-frame random intercepts (BLUPs).',
        'STEP 5. PRIMARY RESULTS (report these): crossed mixed model (random intercepts for rater AND frame). beta with 95% CI (= beta +/- 1.96*SE) for all 12 cells. Report effect sizes with intervals; p and z are shown for reference but the figure/text use beta + CI, not significance stars.',
        'AUX (robustness). Two-stage per-rater Spearman + Fisher-z + one-sample t-test. Confirms the primary conclusion; not the reported test.'
    ]
})

out = '../../propagation_mixed_model.xlsx'
with pd.ExcelWriter(out, engine='openpyxl') as xl:
    readme.to_excel(xl, sheet_name='0_README', index=False)
    step1.to_excel(xl, sheet_name='STEP1_delta', index=False)
    step2.to_excel(xl, sheet_name='STEP2_within_z', index=False)
    step3.to_excel(xl, sheet_name='STEP3_model_inputs', index=False)
    step4_meta.to_excel(xl, sheet_name='STEP4_random_effects', index=False, startrow=0)
    step4_rater.to_excel(xl, sheet_name='STEP4_random_effects', index=False, startrow=len(step4_meta) + 2)
    step5.to_excel(xl, sheet_name='STEP5_results_primary', index=False)
    auxdf.to_excel(xl, sheet_name='AUX_twostage', index=False)
    for ws in xl.book.worksheets:
        for cc in ws.columns:
            w = max((len(str(c.value)) if c.value is not None else 0) for c in cc)
            ws.column_dimensions[cc[0].column_letter].width = min(max(w + 2, 11), 58)
print("wrote propagation_mixed_model.xlsx (crossed rater+frame)")
print("\nSTEP5 primary (crossed rater+frame) with 95% CI:")
print(step5[['axis', 'setting', 'pair', 'beta', 'CI95', 'p']].to_string(index=False))
