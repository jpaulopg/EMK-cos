"""Robustness of the incomplete-data comparison to the feature scaling (50% MCAR, full GMM, 3 repetitions).
For the five min-max representations, the same comparison is repeated with features standardized to zero mean and
unit variance, both estimated from the observed entries; cosines then range over [-1, 1] instead of concentrating
near 1. Reports RMSE against the complete-data cosine and the spread (standard deviation) of the targets."""
import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B
from em_struct import em_struct
from exp_real2 import load, cosm, scale_observed, ridge_from
from exp_aux import CS, NS


def standardize_observed(X, mask):
    Xo = np.where(mask, np.nan, X)
    with np.errstate(all="ignore"):
        mu, sd = np.nanmean(Xo, 0), np.nanstd(Xo, 0)
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    return (X - np.where(ok, mu, 0.0)) / np.where(ok, sd, 1.0)


def run(name, rep, rate=0.5):
    Xr, _ = load(name, NS[name], 100 + rep); n, D = Xr.shape; C = CS[name]
    mask = np.random.default_rng(1000 * rep + int(rate * 100)).random((n, D)) < rate
    out = {}
    for scaling, X in (("minmax", scale_observed(name, Xr, mask)), ("zscore", standardize_observed(Xr, mask))):
        Xm = np.where(mask, np.nan, X)
        pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=ridge_from(Xm), seed=rep, struct="full")
        cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
        Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
        Np = np.array([sum(r[c] * (B.nu_pn2025(m[c], S[c]) if m[c] @ m[c] > 1e-12 else np.zeros(D)) for c in range(len(r))) for r, m, S in cond])
        Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
        Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(len(r))) for r, m, S in cond])
        Nc = np.array([B.ratio_comp(r, m, S) for r, m, S in cond])
        O = (~mask).astype(float); Xo = np.where(mask, 0.0, X)
        with np.errstate(invalid="ignore", divide="ignore"):
            Kobs = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
        Kobs[~np.isfinite(Kobs)] = 0.0
        K = {"EMK-cos": Nu @ Nu.T, "RatioMeans": B.ratio_mean(Mean, Tr), "RatioComp": Nc @ Nc.T, "PN-Taylor": Np @ Np.T,
             "Plug-in": cosm(Mean), "MeanImp": cosm(np.where(mask, np.nanmean(Xm, 0), Xm)), "ObsSubset": Kobs}
        Kt = cosm(X); iu = np.triu_indices(n, 1)
        out[scaling] = dict(target_sd=float(Kt[iu].std()), target_mean=float(Kt[iu].mean()),
                            rmse={k: float(np.sqrt(np.mean((v[iu] - Kt[iu]) ** 2))) for k, v in K.items()})
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        with open(f"res_zscore_{name}.jsonl", "w") as f:
            for rep in range(3):
                t = time.time(); r = run(name, rep); f.write(json.dumps(dict(data=name, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rep, round(time.time() - t, 1), flush=True)
