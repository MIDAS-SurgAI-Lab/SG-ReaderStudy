#!/usr/bin/env python3
"""Compute Supplementary Tables S7 (per-structure anatomy detection) and
S8 (action-triplet synonym normalization map) from the raw model outputs.

S7 reuses the exact 5-class canonicalization from auto_eval_structured.py.
S8 dumps the exact synonym maps used by auto_eval_quadruple.py.

Usage: python compute_structured.py --root ../.. --outdir ..
"""
import argparse, json, os
import numpy as np, pandas as pd

COND = ['i_refined', 'i_pred_refined', 'i_gt_refined']
LAB = {'i_refined': 'Image', 'i_pred_refined': '+Pred', 'i_gt_refined': '+GT'}
GT_CANON = ['cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct', 'gallbladder']
PRETTY = {'cystic_plate': 'Cystic plate', 'calot_triangle': "Calot's triangle",
          'cystic_artery': 'Cystic artery', 'cystic_duct': 'Cystic duct', 'gallbladder': 'Gallbladder'}

def canon_structures(tokens):
    out = set()
    for t in (tokens or []):
        s = str(t).lower()
        if 'plate' in s: out.add('cystic_plate')
        if 'duct' in s: out.add('cystic_duct')
        if 'arter' in s: out.add('cystic_artery')
        if 'calot' in s or 'hepatocystic' in s: out.add('calot_triangle')
        if 'gallbladder' in s or 'gb' == s.strip():
            out.add('gallbladder')
    return out

def build_records(root):
    data = json.load(open(f'{root}/ALL_THING.json'))
    allowed = set(json.load(open(f'{root}/sg-userstudy/GPT5_gt_graph_task_text_keypoints.json')).keys())
    def item_ids(it):
        s = {str(it.get(k)) for k in ('id', 'image_id', 'file_name') if it.get(k) is not None}
        return s | {x.split('.')[0] for x in s}
    recs = []
    for it in data:
        gt = it.get('gt_refined') or {}
        if not gt.get('cvs'):
            continue
        rec = {'gt': canon_structures(gt.get('structures'))}
        ok = True
        for k in COND:
            sub = it.get(k) or {}
            rec[k] = canon_structures(sub.get('structures'))
        recs.append(rec)
    return recs

def S7(root, outdir):
    recs = build_records(root)
    n = len(recs)
    rows = []
    for st in GT_CANON:
        cells = {}
        for k in COND:
            tp = sum(1 for r in recs if st in r['gt'] and st in r[k])
            fp = sum(1 for r in recs if st not in r['gt'] and st in r[k])
            fn = sum(1 for r in recs if st in r['gt'] and st not in r[k])
            rec = tp / (tp + fn) if (tp + fn) else float('nan')
            prec = tp / (tp + fp) if (tp + fp) else float('nan')
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float('nan')
            cells[k] = (rec, prec, f1)
        rows.append(dict(structure=PRETTY[st],
                         **{f'{LAB[k]}_R': cells[k][0] for k in COND},
                         **{f'{LAB[k]}_P': cells[k][1] for k in COND},
                         **{f'{LAB[k]}_F1': cells[k][2] for k in COND}))
    df = pd.DataFrame(rows)
    df.to_csv(f'{outdir}/csv/S7_per_structure.csv', index=False)
    # verify vs section 8: cystic duct R 0.24->0.75, cystic artery 0.07->0.70
    cd = df[df.structure == 'Cystic duct'].iloc[0]
    ca = df[df.structure == 'Cystic artery'].iloc[0]
    print(f"[S7 verify] Cystic duct R  {cd['Image_R']:.2f} -> {cd['+GT_R']:.2f} (target 0.24 -> 0.75)")
    print(f"[S7 verify] Cystic artery R {ca['Image_R']:.2f} -> {ca['+GT_R']:.2f} (target 0.07 -> 0.70)")

    def f(v): return f'{v:.2f}' if not pd.isna(v) else 'NA'
    body = ''
    for _, r in df.iterrows():
        body += (f"{r['structure']} & {f(r['Image_R'])} & {f(r['+Pred_R'])} & {f(r['+GT_R'])} & "
                 f"{f(r['Image_P'])} & {f(r['+Pred_P'])} & {f(r['+GT_P'])} & "
                 f"{f(r['Image_F1'])} & {f(r['+Pred_F1'])} & {f(r['+GT_F1'])} \\\\\n")
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Per-structure anatomy detection ($n=312$).}
Frame-level recall (R), precision (P), and F1 for each of the five annotated anatomical structures under the three input settings, using the same five-class mapping as the main anatomy metric.
Grounding most improves recall of the two smallest and most safety-relevant structures, the cystic duct and the cystic artery.}
\label{tab:S7}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{\textbf{Recall}} & \multicolumn{3}{c}{\textbf{Precision}} & \multicolumn{3}{c}{\textbf{F1}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
\textbf{Structure} & Img & Pred & GT & Img & Pred & GT & Img & Pred & GT \\
\midrule
""" + body + r"""\bottomrule
\end{tabular*}
\end{table}
"""
    open(f'{outdir}/tables/S7_per_structure.tex', 'w').write(tex)
    print("[S7] written")

def S8(outdir):
    VERB = [('retracting, retraction, traction', 'retract', 'same action'),
            ('dissecting, dissection, blunt dissection', 'dissect', 'same action'),
            ('grasping', 'grasp', 'morphological variant'),
            ('coagulating, coagulation', 'coagulate', 'morphological variant'),
            ('cauterize, cautery, cauterizing, electrocautery', 'coagulate', 'same action')]
    INSTR = [('suction, suctioning, aspirator', 'irrigator', 'same device'),
             ('suction-irrigator, suction irrigator', 'irrigator', 'same device')]
    strict = ['dissector', 'maryland', 'skeletonize', 'clip applier vs clipper',
              'hook vs L-hook (kept separate unless identical)']
    rows_v = ''.join(f"{g} & {c} & verb, {t} \\\\\n" for g, c, t in VERB)
    rows_i = ''.join(f"{g} & {c} & instrument, {t} \\\\\n" for g, c, t in INSTR)
    pd.DataFrame([dict(kind='verb', generated=g, normalized=c, type=t) for g, c, t in VERB] +
                 [dict(kind='instrument', generated=g, normalized=c, type=t) for g, c, t in INSTR]
                 ).to_csv(f'{outdir}/csv/S8_synonym_map.csv', index=False)
    tex = r"""\begin{table}[t]
\centering
\caption{\textbf{Action-triplet synonym normalization map.}
Only clear same-device or same-action variants were merged before the action-triplet F1 was computed; every other token was kept verbatim.
Ambiguous tokens were deliberately not merged, which keeps the normalization conservative.}
\label{tab:S8}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}p{0.5\linewidth}ll@{}}
\toprule
\textbf{Generated token(s)} & \textbf{Normalized to} & \textbf{Type} \\
\midrule
""" + rows_v + r"\addlinespace[2pt]" + "\n" + rows_i + r"""\bottomrule
\end{tabular*}
\smallskip

{\footnotesize\raggedright Ambiguous tokens kept strict (not merged): """ + ', '.join(strict) + r""".\par}
\end{table}
"""
    open(f'{outdir}/tables/S8_synonym_map.tex', 'w').write(tex)
    print("[S8] written")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='../..')
    ap.add_argument('--outdir', default='..')
    args = ap.parse_args()
    os.makedirs(f'{args.outdir}/tables', exist_ok=True)
    os.makedirs(f'{args.outdir}/csv', exist_ok=True)
    S7(args.root, args.outdir)
    S8(args.outdir)
    print("done")
