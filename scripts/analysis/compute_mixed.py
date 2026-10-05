#!/usr/bin/env python3
"""Linear mixed-effects re-analysis of the reader ratings (primary model).

Rationale: the main analysis is a step-wise paired t-test on 3-rater case-means, which
discards rater variance, ignores the incomplete cross of raters and frames, and does not
directly test whether the setting effect varies by step. Here we model the raw 1-5 ratings.

R's ordinal::clmm is not available in this environment, so per the plan a LINEAR mixed model
(statsmodels MixedLM) is used as the primary model. The 1-5 responses are well spread (no
severe floor/ceiling), so the linear approximation is reasonable; this substitution is stated
in the results and Methods.

Random effects: crossed variance components for frame, rater, and rater:frame.
The rater:frame term is the side-by-side block (one rater judging one frame's three settings).
Fixed effects: setting * step. Accuracy and Detail are modelled separately.
Global Coherence is excluded because it has no step factor (one score per sample).

Outputs csv/mixed_results.json (consumed by make_mixed_outputs.py). Prints a full report.
"""
import json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')

OUT = '..'
long = pd.read_csv(f'{OUT}/csv/long_ratings.csv')
long['frame'] = long.frame.astype(str)
long['rater'] = long.rater.astype(str)
long['rater_frame'] = long.rater + ':' + long.frame
long['grp'] = 1
long['setting'] = pd.Categorical(long.setting, categories=['image', 'pred', 'gt'])
long['step'] = pd.Categorical(long.step, categories=[1, 2, 3])

# case-mean paired t-test reference (Holm over 2 within step), recomputed for the comparison table
def casemean_ttest(df, axis):
    sub = df[df.score_type == axis]
    out = {}
    for step in [1, 2, 3]:
        s = sub[sub.step == step]
        piv = s.pivot_table(index='frame', columns='setting', values='rating', aggfunc='mean')
        ref = piv['image']
        ps = {}
        for cond in ['pred', 'gt']:
            d = piv[cond] - ref
            t, p = stats.ttest_rel(piv[cond], ref)
            ps[cond] = (d.mean(), p)
        # Holm over the 2 within step
        raw = np.array([ps['pred'][1], ps['gt'][1]])
        order = np.argsort(raw); adj = np.empty(2); run = 0
        for i, idx in enumerate(order):
            run = max(run, (2 - i) * raw[idx]); adj[idx] = min(run, 1)
        out[step] = {'pred': (ps['pred'][0], adj[list(order).index(0)] if False else adj[0]),
                     'gt': (ps['gt'][0], adj[1])}
    return out

def fit(df, with_rf, reml):
    vc = {'frame': '0 + C(frame)', 'rater': '0 + C(rater)'}
    if with_rf:
        vc['rater_frame'] = '0 + C(rater_frame)'
    md = smf.mixedlm("rating ~ C(setting) * C(step)", df, groups=df['grp'],
                     vc_formula=vc, re_formula='0')
    return md.fit(reml=reml, method='lbfgs', maxiter=200)

def contrast(res, weights):
    names = list(res.params.index)
    L = np.array([weights.get(n, 0.0) for n in names])
    est = float(L @ res.params.values)
    se = float(np.sqrt(L @ res.cov_params().values @ L))
    z = est / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return est, se, est - 1.96 * se, est + 1.96 * se, p

def holm2(ps):
    raw = np.array(ps); order = np.argsort(raw); adj = np.empty(len(raw)); run = 0
    for i, idx in enumerate(order):
        run = max(run, (len(raw) - i) * raw[idx]); adj[idx] = min(run, 1.0)
    return adj

results = {'meta': {'model': 'linear MixedLM (statsmodels); R ordinal::clmm unavailable',
                    'random_effects': 'crossed VC: frame, rater, rater:frame (side-by-side block)'}}

for axis in ['accuracy', 'detail']:
    df = long[long.score_type == axis].copy()
    print(f"\n{'='*70}\n{axis.upper()}  (n={len(df)} ratings)\n{'='*70}")
    A = {}

    # --- fixed-effect stability: with vs without rater:frame (REML) ---
    res_rf = fit(df, True, reml=True)
    res_norf = fit(df, False, reml=True)
    print(f"converged: with_rf={res_rf.converged}  without_rf={res_norf.converged}")
    fe = [n for n in res_rf.params.index if 'C(' in n or n == 'Intercept']
    print(f"\n{'fixed effect':38}{'with rf est(SE)':>20}{'without rf est(SE)':>20}")
    festab = []
    for n in fe:
        e1, s1 = res_rf.params[n], res_rf.bse[n]
        e2, s2 = res_norf.params[n], res_norf.bse[n]
        festab.append(dict(name=n, est_rf=float(e1), se_rf=float(s1), est_norf=float(e2), se_norf=float(s2)))
        print(f"{n:38}{f'{e1:+.3f} ({s1:.3f})':>20}{f'{e2:+.3f} ({s2:.3f})':>20}")
    # variance components
    vcvar = {k: float(v) for k, v in res_rf.params.items() if 'Var' in k}
    print("variance components (with rf):", {k: round(v, 4) for k, v in vcvar.items()})
    zero_vc = [k for k, v in vcvar.items() if v < 1e-6]
    if zero_vc:
        print("  ZERO/near-zero variance components:", zero_vc)

    # --- LRT for setting*step interaction (ML) ---
    full = fit(df, True, reml=False)
    md_red = smf.mixedlm("rating ~ C(setting) + C(step)", df, groups=df['grp'],
                         vc_formula={'frame': '0 + C(frame)', 'rater': '0 + C(rater)', 'rater_frame': '0 + C(rater_frame)'},
                         re_formula='0')
    red = md_red.fit(reml=False, method='lbfgs', maxiter=200)
    lr = 2 * (full.llf - red.llf)
    dfd = 4
    plrt = 1 - stats.chi2.cdf(lr, dfd)
    print(f"\nINTERACTION LRT (setting x step): chi2({dfd})={lr:.3f}  p={plrt:.4f}")

    # --- residual checks (primary = with rf, REML) ---
    resid = res_rf.resid.values
    fitted = res_rf.fittedvalues.values
    sh_p = stats.shapiro(resid[:4999]).pvalue if len(resid) > 3 else np.nan
    # homoscedasticity: Levene across settings
    lev_p = stats.levene(*[resid[df['setting'].values == s] for s in ['image', 'pred', 'gt']]).pvalue
    print(f"residuals: Shapiro p={sh_p:.4f} (normality)  Levene-by-setting p={lev_p:.4f} (equal var)  skew={stats.skew(resid):.2f} kurt={stats.kurtosis(resid):.2f}")

    # --- post-hoc contrasts from full model (with rf, REML for CI) ---
    # parameter names
    def pn(setting, stepint=None):
        # main setting term
        base = f"C(setting)[T.{setting}]"
        if stepint is None:
            return base
        return f"C(setting)[T.{setting}]:C(step)[T.{stepint}]"
    contrasts = {}
    ph_ps = {}
    for step in [1, 2, 3]:
        for cond in ['pred', 'gt']:
            w = {pn(cond): 1.0}
            if step in (2, 3):
                w[pn(cond, step)] = 1.0
            est, se, lo, hi, p = contrast(res_rf, w)
            contrasts[f'{cond}-image@S{step}'] = dict(est=est, se=se, ci=[lo, hi], p=p)
            ph_ps.setdefault(step, {})[cond] = p
    # Holm over 2 within each step
    for step in [1, 2, 3]:
        adj = holm2([ph_ps[step]['pred'], ph_ps[step]['gt']])
        contrasts[f'pred-image@S{step}']['p_holm'] = float(adj[0])
        contrasts[f'gt-image@S{step}']['p_holm'] = float(adj[1])
    # step-difference: (gt-image at S1) - (gt-image at S3) = -interaction gt:step3
    est, se, lo, hi, p = contrast(res_rf, {pn('gt', 3): -1.0})
    contrasts['gt-image: S1-S3'] = dict(est=est, se=se, ci=[lo, hi], p=p)
    est, se, lo, hi, p = contrast(res_rf, {pn('pred', 3): -1.0})
    contrasts['pred-image: S1-S3'] = dict(est=est, se=se, ci=[lo, hi], p=p)

    print("\npost-hoc contrasts (mixed model):")
    for k, v in contrasts.items():
        ph = f" p_holm={v['p_holm']:.4f}" if 'p_holm' in v else ''
        print(f"  {k:22}: {v['est']:+.3f}  95%CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  p={v['p']:.4f}{ph}")

    # case-mean t-test comparison
    ct = casemean_ttest(long, axis)
    print("\n vs case-mean paired t-test (Holm/2):")
    cmp = {}
    for step in [1, 2, 3]:
        for cond in ['pred', 'gt']:
            mm = contrasts[f'{cond}-image@S{step}']
            cm_est, cm_p = ct[step][cond]
            cmp[f'{cond}@S{step}'] = dict(mm_est=mm['est'], mm_p=mm['p_holm'], cm_est=cm_est, cm_p=float(cm_p))
            flip = ' <== SIG FLIP' if (mm['p_holm'] < 0.05) != (cm_p < 0.05) else ''
            print(f"  {cond}-image S{step}: mixed {mm['est']:+.2f}(p_holm {mm['p_holm']:.3f}) | case-mean {cm_est:+.2f}(p {cm_p:.3f}){flip}")

    A['fixed'] = festab
    A['vc'] = vcvar
    A['zero_vc'] = zero_vc
    A['lrt'] = dict(chi2=float(lr), df=dfd, p=float(plrt))
    A['resid'] = dict(shapiro_p=float(sh_p), levene_p=float(lev_p), skew=float(stats.skew(resid)), kurt=float(stats.kurtosis(resid)))
    A['contrasts'] = contrasts
    A['compare'] = cmp
    A['converged'] = dict(with_rf=bool(res_rf.converged), without_rf=bool(res_norf.converged), full_ml=bool(full.converged), reduced_ml=bool(red.converged))
    results[axis] = A

json.dump(results, open(f'{OUT}/csv/mixed_results.json', 'w'), indent=2)
print("\n-> csv/mixed_results.json")
