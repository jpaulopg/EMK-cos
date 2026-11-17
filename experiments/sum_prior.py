"""Summary of exp_prior.py (Table tab:prior): RMSE against the true weights for four prior variances.
res_prior.jsonl: weights from four families; res_prior_iid.jsonl: weights drawn independently from N(0, 1.25 I)."""
import json
import numpy as np

METHODS = ("EMK-cos", "RatioMeans", "Taylor2", "Plug-in")
for fname in ("res_prior_iid.jsonl", "res_prior.jsonl"):
    R = [json.loads(l)["res"] for l in open(fname)]
    print(f"== {fname} ({len(R)} repetitions)")
    for nk in (15, 30):
        for tau in (0.3125, 1.25, 4.0, 20.0):
            k = f"nk{nk}_tau{tau}"
            v = {m: np.array([r[k][m] for r in R]) for m in METHODS}
            best = min(v, key=lambda m: v[m].mean())
            line = "  ".join(f"{m} {v[m].mean():.4f}" for m in METHODS)
            print(f"  rows={nk} prior={tau:<6} {line} | best {best} | EMK<plug-in {(v['EMK-cos'] < v['Plug-in']).sum()}/{len(R)}")
