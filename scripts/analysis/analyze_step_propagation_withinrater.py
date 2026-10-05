#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WITHIN-RATER version of the Step1 -> Step2/3 propagation analysis.

Difference from analyze_step_propagation.py (which paired at CASE level, averaging
the 3 experts): here the unit is a single rater's own judgment.

For each (expert, case) we take THAT expert's own scores and form
    ΔStep_k = score_k(gt) - score_k(img)   [also pred-img],  k in {1,2,3}
then ask whether, within one person's evaluations, ΔStep1 tracks ΔStep2/ΔStep3.

Reported at two levels:
  (1) pooled annotation level (n=300 expert-case deltas) -- direct but NON-independent
      (10 experts & 100 cases repeat), so treat its p-values with caution;
  (2) per-expert correlations (10 experts x 30 cases each), aggregated with Fisher-z
      and a one-sample t-test on z -- this respects the expert as the independent unit.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr, pearsonr, ttest_1samp

for _fp in ["sg-userstudy/Pretendard-Medium.otf", "sg-userstudy/Pretendard-ExtraBold.otf"]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("./analysis_out"); OUT.mkdir(exist_ok=True)
CSV = "관리자페이지 (씬그래프 리더스터디)의 사본 (최종_가영) - ALL.csv"
SCORES = {"score1": "정확성", "score2": "타당성"}
CONTRASTS = {"gt-img": ("gt", "img"), "pred-img": ("pred", "img")}


def build_deltas():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    recs = []
    for (exp, case), g in df.groupby(["expertId", "caseId"]):
        gt = g.set_index("taskKey")
        if not {"Task1", "Task2", "Task3"}.issubset(gt.index):
            continue
        rec = {"expertId": exp, "caseId": case}
        for sc in SCORES:
            for cname, (hi, lo) in CONTRASTS.items():
                for task in ["Task1", "Task2", "Task3"]:
                    rec[f"{sc}|{cname}|{task}"] = (
                        gt.loc[task, f"{hi}_{sc}"] - gt.loc[task, f"{lo}_{sc}"])
        recs.append(rec)
    return pd.DataFrame(recs)


def fisher_aggregate(rs):
    rs = np.clip(np.asarray(rs, float), -0.999, 0.999)
    rs = rs[~np.isnan(rs)]
    z = np.arctanh(rs)
    mz = z.mean()
    t, p = ttest_1samp(z, 0.0)
    return float(np.tanh(mz)), float(t), float(p), len(rs)


def main():
    d = build_deltas()
    report = []

    def out(line=""):
        print(line); report.append(line)

    n_ann = len(d)
    out(f"# Within-rater Step1 → Step2/3 propagation  (annotation n={n_ann}, "
        f"{d.expertId.nunique()} experts × {d.caseId.nunique()} cases)\n")
    out("Unit = a single rater's own (gt−img / pred−img) score change per step.\n")

    # (1) pooled annotation level
    out("## (1) Pooled annotation level (n=300; NON-independent — p's optimistic)\n")
    out("| score | contrast | pair | Spearman r | p | Pearson r |")
    out("|---|---|---|---|---|---|")
    pooled = {}
    for sc in SCORES:
        for cname in CONTRASTS:
            d1 = d[f"{sc}|{cname}|Task1"].values
            for tgt in ["Task2", "Task3"]:
                dt = d[f"{sc}|{cname}|{tgt}"].values
                rho, p = spearmanr(d1, dt)
                pr, _ = pearsonr(d1, dt)
                pooled[(sc, cname, tgt)] = (d1, dt, rho, p)
                out(f"| {SCORES[sc]} | {cname} | ΔS1→ΔS{tgt[-1]} | {rho:+.3f} | {p:.2g} | {pr:+.3f} |")
    out("")

    # (2) per-expert correlations, Fisher-z aggregated
    out("## (2) Per-expert correlations (each expert ~30 cases), Fisher-z aggregated\n")
    out("Independent unit = expert. mean r = tanh(mean z); t-test on z across 10 experts.\n")
    out("| score | contrast | pair | mean r (10 experts) | t | p | #experts r>0 |")
    out("|---|---|---|---|---|---|---|")
    per_expert_rs = {}
    for sc in SCORES:
        for cname in CONTRASTS:
            for tgt in ["Task2", "Task3"]:
                rs = []
                for exp, g in d.groupby("expertId"):
                    a = g[f"{sc}|{cname}|Task1"].values
                    b = g[f"{sc}|{cname}|{tgt}"].values
                    if np.std(a) == 0 or np.std(b) == 0:
                        continue
                    rs.append(pearsonr(a, b)[0])
                per_expert_rs[(sc, cname, tgt)] = rs
                mr, t, p, k = fisher_aggregate(rs)
                npos = int(np.sum(np.array(rs) > 0))
                out(f"| {SCORES[sc]} | {cname} | ΔS1→ΔS{tgt[-1]} | {mr:+.3f} | {t:+.2f} | {p:.3g} | {npos}/{k} |")
    out("")

    out("## Read\n")
    out("- If within a single rater ΔStep1 tracks ΔStep2 (high r) but not ΔStep3, the perception→"
        "insight propagation (H1 for Step2) holds at the individual-judgment level, while Step3 stays "
        "perception-independent (H2).")
    out("- The per-expert aggregation (2) is the trustworthy inference (expert = independent unit); "
        "the pooled (1) is shown only for completeness and will look over-significant.")
    out("- ⚠ within-rater is the MOST exposed to a single-person halo (one rater's global impression "
        "of a case's outputs bleeding across steps). The Step2-vs-Step3 asymmetry is the safeguard: a "
        "pure halo would lift Step3 too.\n")

    # figure: per-expert r distributions for Step2 vs Step3 (gt-img)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, sc in zip(axes, SCORES):
        data, labels, colors = [], [], []
        for cname in CONTRASTS:
            for tgt in ["Task2", "Task3"]:
                data.append(per_expert_rs[(sc, cname, tgt)])
                labels.append(f"{cname}\nΔS1→ΔS{tgt[-1]}")
                colors.append("#2171b5" if tgt == "Task2" else "#bdbdbd")
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        for i, vals in enumerate(data, 1):
            ax.scatter(np.random.normal(i, 0.05, len(vals)), vals, s=18, color="k", alpha=0.6, zorder=3)
        ax.axhline(0, color="red", lw=1, ls="--")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("per-expert correlation r")
        ax.set_title(f"{SCORES[sc]}: within-rater ΔStep1 → ΔStep_k\n(each dot = 1 expert, ~30 cases)",
                     fontsize=10)
    fig.suptitle("Within-rater propagation: per-expert correlations (Step2 blue vs Step3 grey)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "step_propagation_withinrater.png", dpi=150)
    plt.close(fig)
    out("![step_propagation_withinrater](step_propagation_withinrater.png)")

    (OUT / "STEP_PROPAGATION_WITHINRATER_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[saved] {OUT}/step_propagation_withinrater.png + STEP_PROPAGATION_WITHINRATER_SUMMARY.md")


if __name__ == "__main__":
    main()
