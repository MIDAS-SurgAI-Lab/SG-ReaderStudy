#!/usr/bin/env python3
"""Export every step of the mixed-model propagation computation to an Excel workbook.
Mixed model uses the RAW graph-induced changes (no rank transform, so integer-Delta ties do not
discretize the predictor), standardized WITHIN each rater (per-rater mean 0, sd 1) so the fixed
effect reflects the within-rater association, consistent with the per-rater two-stage Spearman.

Sheets:
  0_README            - what each sheet is
  1_delta_long        - STEP 1: per (rater, frame) Delta at each step (all steps, both settings)
  2_example_S1S2      - STEP 1-2 for the worked example (Acc, GT, dS1 vs dS2): raw Delta + z-score
  3_perrater_corr     - two-stage: per-rater Spearman r for all 12 cells
  4_mixed_vs_twostage - final comparison table (r-bar, mixed beta on raw Delta, p, significance) for 12 cells
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr, ttest_1samp
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('../../raw_data/01_reader_study/expert_ratings.csv')
df.columns = [c.strip() for c in df.columns]
SC = {'Accuracy': 'score1', 'Detail': 'score2'}
SETTINGS = {'gt': 'GT graph', 'pred': 'Predicted graph'}
PAIRS = {'dS1-dS2': (1, 2), 'dS2-dS3': (2, 3), 'dS1-dS3': (1, 3)}

def delta(axcol, setting, step):
    ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
    return df[df.taskKey == f'Task{step}'].assign(d=lambda x: x[f'{col}'] - x[ic])[['expertId', 'caseId', 'd']]

# ---------- Sheet 1: delta_long (all steps, both settings, both axes) ----------
long_rows = []
for axis, axcol in SC.items():
    for setting in ['gt', 'pred']:
        for step in [1, 2, 3]:
            d = delta(axcol, setting, step)
            for _, r in d.iterrows():
                long_rows.append(dict(axis=axis, setting=setting, step=step,
                                      rater=int(r.expertId), frame=int(r.caseId),
                                      delta=r.d))
delta_long = pd.DataFrame(long_rows)

# ---------- Sheet 2: worked example (Accuracy, GT graph) with all three steps' Delta ----------
axcol, setting = 'score1', 'gt'
ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
d1 = df[df.taskKey == 'Task1'].set_index(['expertId', 'caseId']).eval(f"{col}-{ic}").rename('dS1_raw')
d2 = df[df.taskKey == 'Task2'].set_index(['expertId', 'caseId']).eval(f"{col}-{ic}").rename('dS2_raw')
d3 = df[df.taskKey == 'Task3'].set_index(['expertId', 'caseId']).eval(f"{col}-{ic}").rename('dS3_raw')
example = pd.concat([d1, d2, d3], axis=1).dropna().reset_index()
example = example.rename(columns={'expertId': 'rater', 'caseId': 'frame'})
# within-rater z-score for each step's Delta (per rater: mean 0, sd 1)
for c in ['dS1_raw', 'dS2_raw', 'dS3_raw']:
    example[c.replace('_raw', '_z')] = example.groupby('rater')[c].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))
example.insert(0, 'axis', 'Accuracy')
example.insert(1, 'setting', 'gt (GT graph)')
example = example.rename(columns={
    'dS1_raw': 'dS1_raw=gt-img@Task1', 'dS2_raw': 'dS2_raw=gt-img@Task2', 'dS3_raw': 'dS3_raw=gt-img@Task3'})
example = example[['axis', 'setting', 'rater', 'frame',
                   'dS1_raw=gt-img@Task1', 'dS2_raw=gt-img@Task2', 'dS3_raw=gt-img@Task3',
                   'dS1_z', 'dS2_z', 'dS3_z']]

# ---------- Sheet 3: per-rater Spearman for all 12 cells ----------
pr_rows = []
for axis, axcol in SC.items():
    for setting in ['gt', 'pred']:
        for pk, (a, b) in PAIRS.items():
            ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
            for rid, g in df.groupby('expertId'):
                da = g[g.taskKey == f'Task{a}'].set_index('caseId').eval(f"{col}-{ic}")
                db = g[g.taskKey == f'Task{b}'].set_index('caseId').eval(f"{col}-{ic}")
                jj = pd.concat([da, db], axis=1, keys=['a', 'b']).dropna()
                r = spearmanr(jj.a, jj.b).correlation
                pr_rows.append(dict(axis=axis, setting=setting, pair=pk, rater=int(rid),
                                    n_frames=len(jj), spearman_r=round(r, 4),
                                    fisher_z=round(np.arctanh(np.clip(r, -.999999, .999999)), 4)))
perrater = pd.DataFrame(pr_rows)

# ---------- Sheet 4: final comparison (two-stage vs mixed) ----------
def star(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.'
comp_rows = []
for axis, axcol in SC.items():
    for setting in ['gt', 'pred']:
        for pk, (a, b) in PAIRS.items():
            ic = f"img_{axcol}"; col = f"{setting}_{axcol}"
            # two-stage
            zs = []
            recs = []
            for rid, g in df.groupby('expertId'):
                da = g[g.taskKey == f'Task{a}'].set_index('caseId').eval(f"{col}-{ic}")
                db = g[g.taskKey == f'Task{b}'].set_index('caseId').eval(f"{col}-{ic}")
                jj = pd.concat([da, db], axis=1, keys=['da', 'db']).dropna()
                r = spearmanr(jj.da, jj.db).correlation
                if not np.isnan(r): zs.append(np.arctanh(np.clip(r, -.999999, .999999)))
                for _, row in jj.iterrows(): recs.append((rid, row.da, row.db))
            rbar = np.tanh(np.mean(zs)); p_ts = ttest_1samp(zs, 0).pvalue
            npos = sum(1 for z in zs if z > 0)
            # mixed: on RAW Delta (no rank transform), standardized WITHIN each rater
            d = pd.DataFrame(recs, columns=['rater', 'da', 'db'])
            d['da'] = d.groupby('rater').da.transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))
            d['db'] = d.groupby('rater').db.transform(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1))
            try:
                m = smf.mixedlm("db ~ da", d, groups=d['rater'], re_formula="~da").fit(reml=False, method='lbfgs', maxiter=400)
            except Exception:
                m = smf.mixedlm("db ~ da", d, groups=d['rater']).fit(reml=False, method='lbfgs', maxiter=400)
            beta = m.params['da']; p_mix = m.pvalues['da']
            comp_rows.append(dict(axis=axis, setting=setting, pair=pk,
                                  twostage_rbar=round(rbar, 3), twostage_p=round(p_ts, 4), twostage_sig=star(p_ts),
                                  raters_r_gt0=f"{npos}/10",
                                  mixed_beta=round(beta, 3), mixed_p=round(p_mix, 4), mixed_sig=star(p_mix)))
comp = pd.DataFrame(comp_rows)

# ---------- README ----------
readme = pd.DataFrame({
    'sheet': ['1_delta_long', '2_example_S1S2', '3_perrater_corr', '4_mixed_vs_twostage'],
    'description': [
        'STEP 1. Per (rater, frame) graph-induced change Delta = score(setting) - score(Image), for every step/setting/axis.',
        'Worked example (Accuracy, GT graph): all three steps\' raw Delta (dS1,dS2,dS3) and their WITHIN-rater z-scores. Any pair (S1-S2, S2-S3, S1-S3) is read from these columns.',
        'TWO-STAGE intermediate: each rater\'s Spearman r (and Fisher z) between the two steps\' Delta, for all 12 cells (10 raters each).',
        'FINAL. For all 12 cells: two-stage r-bar (=tanh mean z of per-rater Spearman) with one-sample t-test p, and mixed-model standardized beta on RAW Delta standardized WITHIN rater (per-rater random intercept + slope) with its p.'
    ]
})

out = '../../propagation_computation.xlsx'
with pd.ExcelWriter(out, engine='openpyxl') as xl:
    readme.to_excel(xl, sheet_name='0_README', index=False)
    delta_long.to_excel(xl, sheet_name='1_delta_long', index=False)
    example.to_excel(xl, sheet_name='2_example_S1S2', index=False)
    perrater.to_excel(xl, sheet_name='3_perrater_corr', index=False)
    comp.to_excel(xl, sheet_name='4_mixed_vs_twostage', index=False)
    # widen columns
    for ws in xl.book.worksheets:
        for col_cells in ws.columns:
            w = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(w + 2, 10), 60)
print("wrote propagation_computation.xlsx")
print(f"  1_delta_long rows: {len(delta_long)}")
print(f"  2_example rows: {len(example)}")
print(f"  3_perrater rows: {len(perrater)}")
print(f"  4_comparison rows: {len(comp)}")
