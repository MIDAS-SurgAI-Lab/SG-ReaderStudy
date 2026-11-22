"""Scale-free permutation test for the step dependence of the GT-minus-Image gain.
Methods spec: permute the step labels of the 3 per-frame gains within each
rater-and-frame block. Statistic: omnibus step dependence (variance of the three
step-mean gains); also the S1-S3 contrast. 20,000 permutations, two-sided."""
import argparse, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/reader_study/expert_ratings.csv",
                help="expert ratings CSV (repo layout default)")
args = ap.parse_args()

rows=[]
with open(args.raw) as f:
    r=csv.DictReader(f); km={c.strip():c for c in r.fieldnames}
    for row in r: rows.append(row)
def val(row,k):
    v=row[km[k]].strip(); return float(v) if v else None
STEP={'Task1':0,'Task2':1,'Task3':2}

def run(axis_col_img, axis_col_gt, label, seed=12345):
    blocks={}
    for row in rows:
        key=(row[km['expertId']].strip(), row[km['caseId']].strip())
        s=STEP[row[km['taskKey']].strip()]
        g=val(row,axis_col_gt); i=val(row,axis_col_img)
        if g is None or i is None: continue
        blocks.setdefault(key,[None]*3)[s]=g-i
    B=np.array([b for b in blocks.values() if None not in b])  # n_blocks x 3
    n=len(B)
    def stats(M):
        means=M.mean(axis=0)
        omni=((means-means.mean())**2).sum()      # omnibus step dependence
        c13=means[0]-means[2]                      # S1-S3 contrast
        return omni,c13
    omni_obs,c13_obs=stats(B)
    rng=np.random.RandomState(seed)
    NP=20000
    cnt_o=0; cnt_c=0
    for _ in range(NP):
        P=B.copy()
        idx=rng.rand(n,3).argsort(axis=1)
        P=np.take_along_axis(P,idx,axis=1)
        o,c=stats(P)
        if o>=omni_obs: cnt_o+=1
        if abs(c)>=abs(c13_obs): cnt_c+=1
    print(f"[{label}] blocks={n}  step-mean gains={np.round(B.mean(axis=0),3)}")
    print(f"  omnibus statistic={omni_obs:.4f}  p_perm={(cnt_o+1)/(NP+1):.5f}")
    print(f"  S1-S3 contrast={c13_obs:.3f}      p_perm(two-sided)={(cnt_c+1)/(NP+1):.5f}")
    return (cnt_o+1)/(NP+1),(cnt_c+1)/(NP+1)

print("=== Accuracy (score1), GT - Image ===")
run("img_score1","gt_score1","Accuracy GT-Image")
print("\n=== Detail (score2), GT - Image ===  (기대: 비유의, chi2 p=.61과 정합)")
run("img_score2","gt_score2","Detail GT-Image")
