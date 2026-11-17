import json, glob, numpy as np
from collections import defaultdict
print("=== CLASSIFICATION kNN acc (mean over reps) ===")
for f in sorted(glob.glob("res_diag_cls_*.jsonl")):
    R = [json.loads(l) for l in open(f)]
    for rate in sorted({r["rate"] for r in R}):
        S = [r["res"] for r in R if r["rate"] == rate]
        ms = ["EMK-cos","EMK-renorm","Taylor2","CMI","MeanImp","RatioMeans"]
        print(R[0]["data"].ljust(11), rate, " ".join(f"{m}={np.mean([s[m]['knn'] for s in S]):.3f}" for m in ms), f"compl={np.mean([s['_complete'] for s in S]):.3f}",
              f"nbnrm={np.mean([s['EMK-cos']['nb_nrm'] for s in S]):.3f}/{np.mean([s['_all_nrm'] for s in S]):.3f}")
print("\n=== REGRESSION R2 (KRR | 5NN-reg) ===")
agg = defaultdict(list)
for f in sorted(glob.glob("res_diag_reg_*.jsonl")):
    R = [json.loads(l) for l in open(f)]
    for rate in sorted({r["rate"] for r in R}):
        S = [r["res"] for r in R if r["rate"] == rate]
        ms = ["EMK-cos","EMK-nodiag","EMK-renorm","Taylor2","CMI","MeanImp","RatioMeans","ObsSubset"]
        print(R[0]["data"].ljust(11), rate, " ".join(f"{m}={np.mean([s[m]['r2'] for s in S]):.3f}|{np.mean([s[m]['r2knn'] for s in S]):.3f}" for m in ms), f"compl={np.mean([s['_complete']['r2'] for s in S]):.3f}")
        for m in ms: agg[m].append((np.mean([s[m]['r2'] for s in S]), np.mean([s[m]['r2knn'] for s in S])))
# pairwise run-level on KRR
print("\nrun-level KRR R2: EMK-cos vs other (wins/total, mean diff)")
runs = []
for f in glob.glob("res_diag_reg_*.jsonl"):
    runs += [json.loads(l)["res"] for l in open(f)]
for m in ["Taylor2","CMI","MeanImp","RatioMeans","EMK-nodiag","EMK-renorm"]:
    d = np.array([r["EMK-cos"]["r2"] - r[m]["r2"] for r in runs])
    print(m.ljust(11), f"{(d>1e-9).sum()}/{len(d)} wins, {(d<-1e-9).sum()} losses, mean diff {d.mean():+.4f}")
