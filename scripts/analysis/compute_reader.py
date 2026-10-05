#!/usr/bin/env python3
"""Compute Supplementary Tables S1-S4 from the raw reader-study export.

Usage:
    python compute_reader.py --raw ../../translated_all.csv \
        --alpha ../../official_krippendorff_alpha.csv \
        --coverage ../../bdi_experience_group_case_coverage.csv \
        --common ../../bdi_experience_common_cases.csv \
        --outdir ..

All values are computed from raw per-rater ratings. seed fixed where randomness applies.
score1 = Accuracy axis, score2 = Detail axis. taskKey Task1/2/3 = Step 1/2/3.
Global Coherence lives on the Task3 rows; BDI on the Task2 rows.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, ttest_rel, ttest_1samp

np.random.seed(42)

def holm(pvals):
    """Holm step-down over the given p-values. Returns adjusted p in original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    return adj

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def fmt_p(p):
    if p < 0.001: return 'p < 0.001'
    return f'p = {p:.3f}'

def neg(x, d=2):
    s = f'{abs(x):.{d}f}'
    return (r'$-$' + s) if x < 0 else s

SC = {'Accuracy': 'score1', 'Detail': 'score2'}

def load(raw):
    df = pd.read_csv(raw); df.columns = [c.strip() for c in df.columns]
    return df

def case_mean(df, step, setting, axis):
    col = f"{setting}_{SC[axis]}"
    return df[df.taskKey == f'Task{step}'].groupby('caseId')[col].mean()

def gc_case_mean(df, setting):
    return df[df.taskKey == 'Task3'].groupby('caseId')[f'{setting}_gc'].mean()

# ----------------------------------------------------------------------
def S2(df, outdir):
    rows, pv = [], []
    spec = [('Step 1 Observation', 1, 'Accuracy'), ('', 1, 'Detail'),
            ('Step 2 Insight', 2, 'Accuracy'), ('', 2, 'Detail'),
            ('Step 3 Plan', 3, 'Accuracy'), ('', 3, 'Detail')]
    for label, step, axis in spec:
        im = case_mean(df, step, 'img', axis); pr = case_mean(df, step, 'pred', axis); gt = case_mean(df, step, 'gt', axis)
        idx = im.index
        p_pred = ttest_rel(pr.reindex(idx), im).pvalue
        p_gt = ttest_rel(gt.reindex(idx), im).pvalue
        adj = holm([p_pred, p_gt])
        pv.append(dict(step=step, axis=axis, comparison='img-pred', p_raw=p_pred, p_holm=adj[0]))
        pv.append(dict(step=step, axis=axis, comparison='img-gt', p_raw=p_gt, p_holm=adj[1]))
        gm = gt - im
        ci = 1.96 * gm.std(ddof=1) / np.sqrt(gm.count())
        rows.append(dict(label=label, axis=axis,
                         img=f"{im.mean():.2f} $\\pm$ {im.std(ddof=1)/np.sqrt(im.count()):.2f}",
                         pred=f"{pr.mean():.2f} $\\pm$ {pr.std(ddof=1)/np.sqrt(pr.count()):.2f}{stars(adj[0])}",
                         gt=f"{gt.mean():.2f} $\\pm$ {gt.std(ddof=1)/np.sqrt(gt.count()):.2f}{stars(adj[1])}",
                         gtdiff=f"{gm.mean():+.2f} ({gm.mean()-ci:+.2f}, {gm.mean()+ci:+.2f})"))
    # Global coherence
    im = gc_case_mean(df, 'img'); pr = gc_case_mean(df, 'pred'); gt = gc_case_mean(df, 'gt')
    p_pred = ttest_rel(pr, im).pvalue; p_gt = ttest_rel(gt, im).pvalue; adj = holm([p_pred, p_gt])
    pv.append(dict(step='GC', axis='Coherence', comparison='img-pred', p_raw=p_pred, p_holm=adj[0]))
    pv.append(dict(step='GC', axis='Coherence', comparison='img-gt', p_raw=p_gt, p_holm=adj[1]))
    gm = gt - im; ci = 1.96 * gm.std(ddof=1) / np.sqrt(gm.count())
    rows.append(dict(label='---', axis='Global Coherence',
                     img=f"{im.mean():.2f} $\\pm$ {im.std(ddof=1)/np.sqrt(im.count()):.2f}",
                     pred=f"{pr.mean():.2f} $\\pm$ {pr.std(ddof=1)/np.sqrt(pr.count()):.2f}{stars(adj[0])}",
                     gt=f"{gt.mean():.2f} $\\pm$ {gt.std(ddof=1)/np.sqrt(gt.count()):.2f}{stars(adj[1])}",
                     gtdiff=f"{gm.mean():+.2f} ({gm.mean()-ci:+.2f}, {gm.mean()+ci:+.2f})"))
    pd.DataFrame(pv).to_csv(f"{outdir}/csv/S2_pvalues.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{outdir}/csv/S2_reader_full.csv", index=False)

    body = ""
    for r in rows:
        lab = r['label'] if r['label'] != '---' else r'\,'
        body += f"{lab} & {r['axis']} & {r['img']} & {r['pred']} & {r['gt']} & {r['gtdiff']} \\\\\n"
        if r['axis'] == 'Detail':
            body += "\\addlinespace[2pt]\n"
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Expert reader study, full numerical results ($n=100$ cases).}
Per-frame condition means were formed by averaging the three raters, and each cell reports the mean of the 100 case-means with its standard error.
Significance is from paired $t$-tests on case-means against the Image baseline, Holm-corrected within each row over the two contrasts (Image vs Pred and Image vs GT).
The final column gives the GT$-$Image difference with its 95\% confidence interval.
Stars denote $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}
\label{tab:S2}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccc@{}}
\toprule
\textbf{Step} & \textbf{Score} & \textbf{Image} & \textbf{+Pred} & \textbf{+GT} & \textbf{GT$-$Image (95\% CI)} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    open(f"{outdir}/tables/S2_reader_full.tex", 'w').write(tex)
    print("[S2] written")

# ----------------------------------------------------------------------
def S3(df, outdir):
    pairs = {'$\\Delta$S1--$\\Delta$S2': (1, 2), '$\\Delta$S1--$\\Delta$S3': (1, 3), '$\\Delta$S2--$\\Delta$S3': (2, 3)}
    rows = []; varrows = []
    for axis in ['Accuracy', 'Detail']:
        cell = {}
        for setting in ['gt', 'pred']:
            col = f"{setting}_{SC[axis]}"; icol = f"img_{SC[axis]}"
            for pname, (a, b) in pairs.items():
                zs, npos = [], 0
                for rid, g in df.groupby('expertId'):
                    da = g[g.taskKey == f'Task{a}'].set_index('caseId').eval(f"{col}-{icol}")
                    db = g[g.taskKey == f'Task{b}'].set_index('caseId').eval(f"{col}-{icol}")
                    j = pd.concat([da, db], axis=1, keys=['a', 'b']).dropna()
                    if len(j) < 3 or j['a'].std() == 0 or j['b'].std() == 0:
                        continue
                    r = spearmanr(j['a'], j['b']).correlation
                    if np.isnan(r): continue
                    r = np.clip(r, -0.999999, 0.999999); zs.append(np.arctanh(r)); npos += int(r > 0)
                rbar = np.tanh(np.mean(zs)); p = ttest_1samp(zs, 0).pvalue
                cell[(setting, pname)] = (rbar, p, npos, len(zs))
        for pname in pairs:
            g = cell[('gt', pname)]; p = cell[('pred', pname)]
            rows.append(dict(axis=axis, link=pname,
                             gt=f"{neg(g[0],3)}{stars(g[1])}", pred=f"{neg(p[0],3)}{stars(p[1])}",
                             gtn=f"{g[2]}/10", predn=f"{p[2]}/10"))
        # delta variance (SD of delta per step, gt-img)
        for setting in ['gt', 'pred']:
            col = f"{setting}_{SC[axis]}"; icol = f"img_{SC[axis]}"
            sds = []
            for st in [1, 2, 3]:
                d = df[df.taskKey == f'Task{st}'].eval(f"{col}-{icol}")
                sds.append(d.std(ddof=1))
            varrows.append(dict(axis=axis, setting={'gt': 'GT$-$Image', 'pred': 'Pred$-$Image'}[setting],
                                s1=f"{sds[0]:.2f}", s2=f"{sds[1]:.2f}", s3=f"{sds[2]:.2f}"))
    pd.DataFrame(rows).to_csv(f"{outdir}/csv/S3_propagation.csv", index=False)
    pd.DataFrame(varrows).to_csv(f"{outdir}/csv/S3_delta_variance.csv", index=False)

    body = ""
    prev = None
    for r in rows:
        ax = r['axis'] if r['axis'] != prev else ''
        prev = r['axis']
        body += f"{ax} & {r['link']} & {r['gt']} & {r['pred']} & {r['gtn']} & {r['predn']} \\\\\n"
        if r['link'].endswith('S3}') or 'S2--$\\Delta$S3' in r['link']:
            body += "\\addlinespace[2pt]\n"
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Co-variation of the graph-induced improvement across reasoning steps.}
For each rater the graph-induced improvement at a step is $\Delta = \text{score(setting)} - \text{score(Image)}$, and the Spearman correlation between two steps' improvements was computed within that rater over the frames they rated.
The ten per-rater correlations were aggregated by Fisher-$z$, and the reported value is $\bar r = \tanh(\text{mean } z)$ with a one-sample $t$-test on the $z$ values.
The last two columns give the number of raters with a positive correlation.
Stars denote $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}
\label{tab:S3}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccc@{}}
\toprule
\textbf{Score} & \textbf{Link} & \textbf{GT$-$Image $\bar r$} & \textbf{Pred$-$Image $\bar r$} & \textbf{raters $r>0$ (GT)} & \textbf{(Pred)} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    open(f"{outdir}/tables/S3_propagation.tex", 'w').write(tex)

    vbody = ""
    prev = None
    for r in varrows:
        ax = r['axis'] if r['axis'] != prev else ''
        prev = r['axis']
        vbody += f"{ax} & {r['setting']} & {r['s1']} & {r['s2']} & {r['s3']} \\\\\n"
    vtex = r"""\begin{table}[t]
\centering
\caption{\textbf{Dispersion of the graph-induced improvement at each step.}
Standard deviation of the per-frame improvement $\Delta$ at each reasoning step, over all rated frames.
The Step 3 improvement retains substantial spread, so its weaker co-variation with the earlier steps is not an artefact of a compressed range.}
\label{tab:S3var}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccc@{}}
\toprule
\textbf{Score} & \textbf{Setting} & \textbf{SD of $\Delta$S1} & \textbf{SD of $\Delta$S2} & \textbf{SD of $\Delta$S3} \\
\midrule
""" + vbody + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    open(f"{outdir}/tables/S3_delta_variance.tex", 'w').write(vtex)
    print("[S3] written")

# ----------------------------------------------------------------------
def S4(df, outdir, coverage, common):
    cov = pd.read_csv(coverage); MORE = set(); LESS = set()
    for s in cov['more_expert_ids'].dropna():
        MORE |= {int(x) for x in str(s).split(';') if x.strip()}
    for s in cov['less_expert_ids'].dropna():
        LESS |= {int(x) for x in str(s).split(';') if x.strip()}
    common_cases = sorted(set(pd.read_csv(common)['caseId']))

    def gmean(axis, step, setting, grp):
        col = f"{setting}_{SC[axis]}"
        sub = df[(df.taskKey == f'Task{step}') & (df.expertId.isin(grp))]
        return sub.groupby('caseId')[col].mean().reindex(common_cases)

    rows = []; overall = {}
    for axis in ['Accuracy', 'Detail']:
        for step in [1, 2, 3]:
            diffs = {}
            for gname, grp in [('More', MORE), ('Less', LESS)]:
                im = gmean(axis, step, 'img', grp); pr = gmean(axis, step, 'pred', grp); gt = gmean(axis, step, 'gt', grp)
                d = (gt - im); diffs[gname] = d
                overall.setdefault((axis, gname), []).append(d)
                rows.append(dict(axis=axis, step=step, group=gname,
                                 img=f"{im.mean():.2f}", pred=f"{pr.mean():.2f}", gt=f"{gt.mean():.2f}",
                                 gtdiff=f"{d.mean():+.2f}"))
            # between-group Wilcoxon on per-case GT-Image diffs (paired on common cases)
            j = pd.concat([diffs['More'], diffs['Less']], axis=1, keys=['m', 'l']).dropna()
            try:
                p = wilcoxon(j['m'], j['l']).pvalue
            except Exception:
                p = float('nan')
            for r in rows[-2:]:
                r['p'] = fmt_p(p) if r['group'] == 'More' else ''
    # global coherence
    def gc_g(setting, grp):
        sub = df[(df.taskKey == 'Task3') & (df.expertId.isin(grp))]
        return sub.groupby('caseId')[f'{setting}_gc'].mean().reindex(common_cases)
    diffs = {}
    for gname, grp in [('More', MORE), ('Less', LESS)]:
        im = gc_g('img', grp); pr = gc_g('pred', grp); gt = gc_g('gt', grp); d = gt - im; diffs[gname] = d
        rows.append(dict(axis='Global Coherence', step='---', group=gname,
                         img=f"{im.mean():.2f}", pred=f"{pr.mean():.2f}", gt=f"{gt.mean():.2f}", gtdiff=f"{d.mean():+.2f}"))
    j = pd.concat([diffs['More'], diffs['Less']], axis=1, keys=['m', 'l']).dropna()
    p = wilcoxon(j['m'], j['l']).pvalue
    for r in rows[-2:]:
        r['p'] = fmt_p(p) if r['group'] == 'More' else ''

    # overall GT-Image (mean of per-step per-case diffs) + Wilcoxon
    ov = {}
    for axis in ['Accuracy', 'Detail']:
        m = pd.concat(overall[(axis, 'More')], axis=1).mean(axis=1)
        l = pd.concat(overall[(axis, 'Less')], axis=1).mean(axis=1)
        jj = pd.concat([m, l], axis=1, keys=['m', 'l']).dropna()
        pp = wilcoxon(jj['m'], jj['l']).pvalue
        ov[axis] = (m.mean(), l.mean(), pp)

    pd.DataFrame(rows).to_csv(f"{outdir}/csv/S4_experience.csv", index=False)
    body = ""; prev = None
    for r in rows:
        ax = r['axis'] if (r['axis'], r['step']) != prev else ''
        prev = (r['axis'], r['step'])
        st = r['step'] if r['group'] == 'More' else ''
        axcol = r['axis'] if r['group'] == 'More' and (prev != getattr(S4, '_last', None)) else ''
        body += f"{r['axis'] if r['group']=='More' else ''} & {r['step'] if r['group']=='More' else ''} & {r['group']} & {r['img']} & {r['pred']} & {r['gt']} & {r['gtdiff']} & {r.get('p','')} \\\\\n"
        if r['group'] == 'Less':
            body += "\\addlinespace[2pt]\n"
    note = (f"Overall GT$-$Image (mean over the three steps): Accuracy "
            f"{ov['Accuracy'][0]:+.2f} (More) vs {ov['Accuracy'][1]:+.2f} (Less), {fmt_p(ov['Accuracy'][2])}; "
            f"Detail {ov['Detail'][0]:+.2f} vs {ov['Detail'][1]:+.2f}, {fmt_p(ov['Detail'][2])}.").replace(';', ',')
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Experience-stratified reader results ($n=70$ common cases).}
Per-frame group means were formed within each experience group before averaging over the 70 frames rated by both groups.
The final column reports a per-case paired Wilcoxon signed-rank test comparing the two groups' GT$-$Image differences at that step.
""" + note + r"""}
\label{tab:S4}
\footnotesize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lllccccc@{}}
\toprule
\textbf{Score} & \textbf{Step} & \textbf{Group} & \textbf{Image} & \textbf{+Pred} & \textbf{+GT} & \textbf{GT$-$Image} & \textbf{Between-group $p$} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    open(f"{outdir}/tables/S4_experience.tex", 'w').write(tex)
    print(f"[S4] written  overall Acc {ov['Accuracy'][0]:+.2f}/{ov['Accuracy'][1]:+.2f} p={ov['Accuracy'][2]:.4f}  "
          f"Det {ov['Detail'][0]:+.2f}/{ov['Detail'][1]:+.2f} p={ov['Detail'][2]:.4f}")

# ----------------------------------------------------------------------
def S1(df, outdir, alpha_csv):
    a = pd.read_csv(alpha_csv)
    # index alpha_ordinal by (task, item)
    look = {(str(r['task']), str(r['item'])): r['alpha_ordinal'] for _, r in a.iterrows()}

    def get(item):
        # item may be tied to a specific task or task-agnostic; take first match
        for (tk, it), v in look.items():
            if it == item:
                return v
        return None

    def get_task(task, item):
        return look.get((task, item))

    order = [('Accuracy', 1, 'score1'), ('Accuracy', 2, 'score1'), ('Accuracy', 3, 'score1'),
             ('Detail', 1, 'score2'), ('Detail', 2, 'score2'), ('Detail', 3, 'score2')]
    rows = []
    for score, step, sc in order:
        vals = [get_task(f'Task{step}', f'{s}_{sc}') for s in ['img', 'pred', 'gt']]
        rows.append(dict(score=score, step=step, Image=vals[0], Pred=vals[1], GT=vals[2]))
    # Global Coherence: per-setting (img_gc/pred_gc/gt_gc), task-agnostic
    rows.append(dict(score='Global Coherence', step='---',
                     Image=get('img_gc'), Pred=get('pred_gc'), GT=get('gt_gc')))
    # BDI risk level: rated once per case, single alpha (not per setting)
    bdi_alpha = get('bdi')
    rows.append(dict(score='BDI risk level', step='---', Image=bdi_alpha, Pred='__SPAN__', GT='__SPAN__'))
    pd.DataFrame(rows).to_csv(f"{outdir}/csv/S1_irr.csv", index=False)
    body = ""
    for r in rows:
        def c(v): return neg(v, 2) if (v is not None and not (isinstance(v, float) and pd.isna(v))) else 'NOT AVAILABLE'
        if r['Pred'] == '__SPAN__':
            body += f"{r['score']} & {r['step']} & \\multicolumn{{3}}{{c}}{{{c(r['Image'])} (single rating, not per setting)}} \\\\\n"
        else:
            body += f"{r['score']} & {r['step']} & {c(r['Image'])} & {c(r['Pred'])} & {c(r['GT'])} \\\\\n"
        if r['step'] == 3:
            body += "\\addlinespace[2pt]\n"
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Inter-rater reliability (Krippendorff's ordinal $\alpha$).}
Krippendorff's ordinal alpha computed within each input setting, with frames as units and raters as coders.
The primary condition effect reported in the main text is a within-subject contrast on case-means from the blinded, side-by-side design and does not depend on absolute agreement between raters.}
\label{tab:S1}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccc@{}}
\toprule
\textbf{Score} & \textbf{Step} & \textbf{Image} & \textbf{+Pred} & \textbf{+GT} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    open(f"{outdir}/tables/S1_irr.tex", 'w').write(tex)
    print("[S1] written")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='../../translated_all.csv')
    ap.add_argument('--alpha', default='../../official_krippendorff_alpha.csv')
    ap.add_argument('--coverage', default='../../bdi_experience_group_case_coverage.csv')
    ap.add_argument('--common', default='../../bdi_experience_common_cases.csv')
    ap.add_argument('--outdir', default='..')
    args = ap.parse_args()
    os.makedirs(f"{args.outdir}/tables", exist_ok=True)
    os.makedirs(f"{args.outdir}/csv", exist_ok=True)
    df = load(args.raw)
    S1(df, args.outdir, args.alpha)
    S2(df, args.outdir)
    S3(df, args.outdir)
    S4(df, args.outdir, args.coverage, args.common)
    print("done")
