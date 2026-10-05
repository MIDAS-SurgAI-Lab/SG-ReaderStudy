import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/tmp/krippendorff_pkg")
sys.path.insert(0, "/tmp/statsmodels_pkg")

import numpy as np
import pandas as pd
import krippendorff
from scipy.stats import sem, spearmanr, ttest_rel
from statsmodels.stats.anova import AnovaRM


CSV_PATH = Path("관리자페이지 (씬그래프 리더스터디)의 사본 (최종_가영) - ALL.csv")
SUMMARY_OUT = Path("official_case_mean_summary.csv")
TTEST_OUT = Path("official_paired_ttests.csv")
CORR_OUT = Path("official_task_correlations.csv")
ANOVA_OUT = Path("official_rm_anova.csv")
ALPHA_OUT = Path("official_krippendorff_alpha.csv")

SCORE_COLS = [
    "img_score1",
    "img_score2",
    "pred_score1",
    "pred_score2",
    "gt_score1",
    "gt_score2",
    "img_gc",
    "pred_gc",
    "gt_gc",
]
ALPHA_ITEMS = SCORE_COLS + ["bdi"]
TASK_PAIRS = [("Task1", "Task2"), ("Task1", "Task3"), ("Task2", "Task3")]
SCORE_SETS = {
    "score1": ("img_score1", "pred_score1", "gt_score1"),
    "score2": ("img_score2", "pred_score2", "gt_score2"),
    "gc": ("img_gc", "pred_gc", "gt_gc"),
}


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def holm_adjust(pvals):
    m = len(pvals)
    ordered = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [None] * m
    running = 0.0
    for rank, (idx, pval) in enumerate(ordered, start=1):
        value = (m - rank + 1) * pval
        value = max(value, running)
        running = value
        adjusted[idx] = min(value, 1.0)
    return adjusted


def build_case_means(rows):
    vals = defaultdict(list)
    for row in rows:
        task = row["taskKey"]
        case = row["caseId"]
        for col in SCORE_COLS:
            value = row.get(col, "")
            if value != "":
                vals[(task, case, col)].append(float(value))
    return {key: float(np.mean(values)) for key, values in vals.items()}


def compute_summary(case_means):
    records = []
    for (task, col), values in sorted(
        ((k[0], k[2]), []) for k in case_means
    ):
        pass
    grouped = defaultdict(list)
    for (task, case, col), value in case_means.items():
        grouped[(task, col)].append(value)
    for (task, col), values in sorted(grouped.items()):
        records.append(
            {
                "task": task,
                "item": col,
                "n_cases": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "se": float(sem(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return pd.DataFrame(records)


def compute_ttests_and_anova(case_means):
    ttest_records = []
    anova_records = []
    tasks = sorted({task for task, _, _ in case_means})
    for task in tasks:
        for score_type, cols in SCORE_SETS.items():
            records = []
            cases = sorted({case for t, case, _ in case_means if t == task})
            for case in cases:
                if all((task, case, col) in case_means for col in cols):
                    records.append(
                        {
                            "caseId": case,
                            "img": case_means[(task, case, cols[0])],
                            "pred": case_means[(task, case, cols[1])],
                            "gt": case_means[(task, case, cols[2])],
                        }
                    )
            if not records:
                continue

            long_df = pd.DataFrame(
                [
                    {"caseId": rec["caseId"], "condition": cond, "score": rec[cond]}
                    for rec in records
                    for cond in ["img", "pred", "gt"]
                ]
            )
            aov = AnovaRM(long_df, depvar="score", subject="caseId", within=["condition"]).fit()
            table = aov.anova_table.iloc[0]
            anova_records.append(
                {
                    "task": task,
                    "score_type": score_type,
                    "n_cases": len(records),
                    "F": float(table["F Value"]),
                    "df_num": float(table["Num DF"]),
                    "df_den": float(table["Den DF"]),
                    "p_value": float(table["Pr > F"]),
                }
            )

            pair_defs = [("img", "pred"), ("img", "gt"), ("pred", "gt")]
            raw_ps = []
            pending = []
            for left, right in pair_defs:
                x = [rec[left] for rec in records]
                y = [rec[right] for rec in records]
                res = ttest_rel(x, y)
                raw_ps.append(float(res.pvalue))
                pending.append(
                    {
                        "task": task,
                        "score_type": score_type,
                        "comparison": f"{left}-{right}",
                        "n_cases": len(records),
                        "mean_diff": float(np.mean(np.array(x) - np.array(y))),
                        "t_stat": float(res.statistic),
                        "p_value": float(res.pvalue),
                    }
                )
            for rec, p_holm in zip(pending, holm_adjust(raw_ps)):
                rec["p_holm"] = p_holm
                ttest_records.append(rec)

    return pd.DataFrame(ttest_records), pd.DataFrame(anova_records)


def compute_correlations(case_means):
    records = []
    for score_type, cols in SCORE_SETS.items():
        if score_type == "gc":
            continue
        for metric, col in zip(["img", "pred", "gt"], cols):
            for task_a, task_b in TASK_PAIRS:
                paired = []
                cases = sorted({case for task, case, item in case_means if item == col})
                for case in cases:
                    key_a = (task_a, case, col)
                    key_b = (task_b, case, col)
                    if key_a in case_means and key_b in case_means:
                        paired.append((case_means[key_a], case_means[key_b]))
                x = [p[0] for p in paired]
                y = [p[1] for p in paired]
                rho, pval = spearmanr(x, y)
                records.append(
                    {
                        "score_type": score_type,
                        "metric": metric,
                        "task_a": task_a,
                        "task_b": task_b,
                        "n_cases": len(paired),
                        "spearman_rho": float(rho),
                        "p_value": float(pval),
                    }
                )
    return pd.DataFrame(records)


def compute_alpha(rows):
    records = []
    tasks = sorted(set(row["taskKey"] for row in rows))
    for task in tasks:
        task_rows = [row for row in rows if row["taskKey"] == task]
        cases = sorted(set(row["caseId"] for row in task_rows))
        experts = sorted(set(row["expertId"] for row in task_rows))
        for item in ALPHA_ITEMS:
            matrix = []
            for expert in experts:
                expert_rows = {row["caseId"]: row for row in task_rows if row["expertId"] == expert}
                matrix.append(
                    [
                        np.nan if expert_rows.get(case, {}).get(item, "") == "" else float(expert_rows[case][item])
                        for case in cases
                    ]
                )
            arr = np.array(matrix, dtype=float)
            if np.isfinite(arr).sum() == 0:
                continue
            records.append(
                {
                    "task": task,
                    "item": item,
                    "n_cases": int(np.sum(np.sum(np.isfinite(arr), axis=0) >= 2)),
                    "alpha_ordinal": float(krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal")),
                    "alpha_interval": float(krippendorff.alpha(reliability_data=arr, level_of_measurement="interval")),
                }
            )
    return pd.DataFrame(records)


def main():
    rows = load_rows()
    case_means = build_case_means(rows)

    compute_summary(case_means).to_csv(SUMMARY_OUT, index=False)
    ttests, anova = compute_ttests_and_anova(case_means)
    ttests.to_csv(TTEST_OUT, index=False)
    anova.to_csv(ANOVA_OUT, index=False)
    compute_correlations(case_means).to_csv(CORR_OUT, index=False)
    compute_alpha(rows).to_csv(ALPHA_OUT, index=False)

    print(f"saved={SUMMARY_OUT}")
    print(f"saved={TTEST_OUT}")
    print(f"saved={ANOVA_OUT}")
    print(f"saved={CORR_OUT}")
    print(f"saved={ALPHA_OUT}")


if __name__ == "__main__":
    main()
