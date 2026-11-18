"""Incomplete data stratified by posterior ambiguity (50% MCAR, full GMM, 3 repetitions).
For each record, H_i is the normalized entropy of its component responsibilities. Pairs are grouped by
max(H_i, H_j); per-pair squared errors against the complete-data cosine are stored by group, together with
the squared gap between EMK-cos and the ratio of means (the excess-risk term of eq. risk under the fitted law)."""
import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B
from em_struct import em_struct
from exp_real2 import masked, cosm, ridge_from
from exp_aux import CS, NS

EDGES = (0.0, 0.1, 0.5, 1.0001)

def run(name, rep, rate=0.5):
    X, Xm, mask, _ = masked(name, NS[name], 100 + rep, rate, 1000 * rep + int(rate * 100)); n, D = X.shape; C = CS[name]
    reg = ridge_from(Xm)
    pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct="full")
    cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    H = np.array([-(r[r > 0] * np.log(r[r > 0])).sum() / np.log(C) for r, _, _ in cond])
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(len(r))) for r, m, S in cond])
    Nr = np.array([B.ratio_comp(r, m, S) for r, m, S in cond])
    K = {"EMK-cos": Nu @ Nu.T, "RatioMeans": B.ratio_mean(Mean, Tr), "RatioComp": Nr @ Nr.T, "Plug-in": cosm(Mean)}
    Kt = cosm(X); iu = np.triu_indices(n, 1); amb = np.maximum(H[:, None], H[None, :])[iu]
    out = {}
    for g in range(len(EDGES) - 1):
        sel = (amb >= EDGES[g]) & (amb < EDGES[g + 1])
        out[f"g{g}"] = dict(pairs=int(sel.sum()),
                            sse={k: float(((v[iu] - Kt[iu])[sel] ** 2).sum()) for k, v in K.items()},
                            gap2=float(((K["EMK-cos"] - K["RatioMeans"])[iu][sel] ** 2).sum()),
                            gap2c=float(((K["EMK-cos"] - K["RatioComp"])[iu][sel] ** 2).sum()),
                            wins={k: int((((v[iu] - Kt[iu])[sel]) ** 2 > ((K["EMK-cos"][iu] - Kt[iu])[sel]) ** 2).sum()) for k, v in K.items()})
    out["_H"] = dict(mean=float(H.mean()), frac_amb=float((H >= 0.5).mean()))
    return out

if __name__ == "__main__":
    for name in sys.argv[1:]:
        with open(f"res_strat_{name}.jsonl", "w") as f:
            for rep in range(3):
                t = time.time(); r = run(name, rep); f.write(json.dumps(dict(data=name, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rep, round(time.time() - t, 1), flush=True)
