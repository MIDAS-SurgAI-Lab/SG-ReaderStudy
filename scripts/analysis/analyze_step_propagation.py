#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does a perception (Step1) gain from the scene graph PROPAGATE to Step2/Step3?

Design: scene graph is the manipulation. For each case we measure the
manipulation-induced CHANGE in each step's reader score:
    ΔStep_k = score(graph condition) - score(img_only),   k in {1,2,3}
and ask whether ΔStep1 predicts ΔStep2 / ΔStep3 (paired, within case).

  strong ΔStep1 -> ΔStep2/3  =>  perception bottleneck (H1): perception gains flow downstream
  weak / none                =>  tacit-knowledge bottleneck (H2): Step1 rises but reasoning doesn't follow

Conditions are WITHIN-ROW (same expert rates img/pred/gt for the same case+task),
so each Δ is a clean within-rater difference; case-level = mean over the 3 experts.
Contrasts: gt-img (full perception oracle) and pred-img (predicted graph).
Scores: score1=정확성(accuracy), score2=타당성(validity).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import spearmanr, pearsonr, ttest_rel, wilcoxon

for _fp in ["sg-userstudy/Pretendard-Medium.otf", "sg-userstudy/Pretendard-ExtraBold.otf"]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("./analysis_out"); OUT.mkdir(exist_ok=True)
CSV = "관리자페이지 (씬그래프 리더스터디)의 사본 (최종_가영) - ALL.csv"

SCORES = {"score1": "정확성", "score2": "타당성"}
CONTRASTS = {"gt-img": ("gt", "img"), "pred-img": ("pred", "img")}


def load_case_level():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    # case-level mean over the 3 experts, per task
    rows = {}
    for sc in SCORES:
        for cond in ["img", "pred", "gt"]:
            col = f"{cond}_{sc}"
            piv = df.pivot_table(index="caseId", columns="taskKey", values=col, aggfunc="mean")
            for task in ["Task1", "Task2", "Task3"]:
                rows[(sc, cond, task)] = piv[task]
    return pd.DataFrame({f"{sc}|{cond}|{task}": s for (sc, cond, task), s in rows.items()})


def main():
    cl = load_case_level()
    N = len(cl)
    report = []

    def out(line=""):
        print(line); report.append(line)

    out(f"# Step1 → Step2/Step3 propagation analysis  (case-level, N={N})\n")
    out("ΔStep_k = score(graph) − score(img_only), per case (mean over 3 experts).")
    out("Question: does the perception gain (ΔStep1) predict the downstream gain (ΔStep2/3)?\n")

    deltas = {}  # (score, contrast, task) -> per-case array
    for sc in SCORES:
        for cname, (hi, lo) in CONTRASTS.items():
            for task in ["Task1", "Task2", "Task3"]:
                deltas[(sc, cname, task)] = (
                    cl[f"{sc}|{hi}|{task}"] - cl[f"{sc}|{lo}|{task}"]).values

    # (a) sanity: mean Δ per step (is the manipulation lifting each step?)
    out("## (a) Mean manipulation-induced change per step (does each step rise?)\n")
    out("| score | contrast | ΔStep1 | ΔStep2 | ΔStep3 |")
    out("|---|---|---|---|---|")
    for sc in SCORES:
        for cname in CONTRASTS:
            d1, d2, d3 = (deltas[(sc, cname, t)] for t in ["Task1", "Task2", "Task3"])
            out(f"| {SCORES[sc]} | {cname} | {d1.mean():+.3f} | {d2.mean():+.3f} | {d3.mean():+.3f} |")
    out("")

    # (b) propagation: ΔStep1 vs ΔStep2, ΔStep1 vs ΔStep3
    out("## (b) Propagation: does ΔStep1 predict ΔStep2 / ΔStep3?  (paired by case)\n")
    out("| score | contrast | pair | Spearman r | p | Pearson r | slope (OLS) | sign-concordance |")
    out("|---|---|---|---|---|---|---|---|")
    results = {}
    for sc in SCORES:
        for cname in CONTRASTS:
            d1 = deltas[(sc, cname, "Task1")]
            for tgt in ["Task2", "Task3"]:
                dt = deltas[(sc, cname, tgt)]
                rho, p = spearmanr(d1, dt)
                pr, _ = pearsonr(d1, dt)
                slope = np.polyfit(d1, dt, 1)[0]
                # concordance: among cases where Step1 moved, does target move same direction?
                moved = d1 != 0
                conc = np.mean(np.sign(d1[moved]) == np.sign(dt[moved])) * 100 if moved.any() else np.nan
                results[(sc, cname, tgt)] = (d1, dt, rho, p, slope)
                out(f"| {SCORES[sc]} | {cname} | ΔStep1→Δ{tgt[-1]} | {rho:+.3f} | {p:.3g} | "
                    f"{pr:+.3f} | {slope:+.3f} | {conc:.0f}% |")
    out("")

    # (c) conditional view: among cases where Step1 ROSE vs not, mean ΔStep2/3
    out("## (c) Conditional: when Step1 rose (Δ>0) vs not, what happened downstream?\n")
    out("| contrast | score | subset | n | mean ΔStep2 | mean ΔStep3 |")
    out("|---|---|---|---|---|---|")
    for cname in CONTRASTS:
        for sc in SCORES:
            d1 = deltas[(sc, cname, "Task1")]
            d2 = deltas[(sc, cname, "Task2")]
            d3 = deltas[(sc, cname, "Task3")]
            up = d1 > 0; flat = d1 <= 0
            out(f"| {cname} | {SCORES[sc]} | Step1↑ (Δ>0) | {up.sum()} | {d2[up].mean():+.3f} | {d3[up].mean():+.3f} |")
            out(f"| {cname} | {SCORES[sc]} | Step1 flat/↓ | {flat.sum()} | {d2[flat].mean():+.3f} | {d3[flat].mean():+.3f} |")
    out("")

    out("## Read\n")
    out("- (a) confirms the manipulation lifts all three steps on average.")
    out("- (b) **the key test.** High ΔStep1↔ΔStep2/3 correlation ⇒ perception gains propagate "
        "downstream (H1). Near-zero ⇒ Step1 can rise without Step2/3 following (H2).")
    out("- ⚠ confound: Step1/2/3 of a case are rated by the same 3 experts, so a per-case rating "
        "*halo* (a rater impressed by the graph rates every step higher) can inflate the correlation. "
        "Treat (b) as an upper bound on true propagation.\n")

    # ---- figures: scatter ΔStep1 vs ΔStep2/3 for each contrast, score1 & score2 ----
    fig_name = {"gt-img": "step_propagation.png", "pred-img": "step_propagation_pred.png"}
    for cname in CONTRASTS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for r, sc in enumerate(SCORES):
            for c, tgt in enumerate(["Task2", "Task3"]):
                ax = axes[r][c]
                d1, dt, rho, p, slope = results[(sc, cname, tgt)]
                ax.scatter(d1, dt, s=22, alpha=0.5, color="#2171b5")
                xs = np.linspace(d1.min(), d1.max(), 50)
                b, a = np.polyfit(d1, dt, 1)
                ax.plot(xs, b * xs + a, "r--", lw=1.5)
                ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
                ax.set_xlabel(f"ΔStep1 ({SCORES[sc]}, {cname})")
                ax.set_ylabel(f"ΔStep{tgt[-1]} ({SCORES[sc]}, {cname})")
                ax.set_title(f"{SCORES[sc]}: ΔStep1 → ΔStep{tgt[-1]}\n"
                             f"Spearman r={rho:+.3f}, p={p:.2g}, slope={slope:+.2f}", fontsize=10)
        fig.suptitle(f"Does the perception (Step1) gain propagate to Step2/Step3?  (per case, {cname})",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / fig_name[cname], dpi=150)
        plt.close(fig)
        out(f"![{fig_name[cname]}]({fig_name[cname]})")

    (OUT / "STEP_PROPAGATION_SUMMARY.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[saved] {OUT}/step_propagation.png + STEP_PROPAGATION_SUMMARY.md")


if __name__ == "__main__":
    main()
