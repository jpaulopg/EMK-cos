"""Missing-data mechanisms and protocol (50% missing, full GMM, 3 repetitions).
MCAR (transductive), MCAR (inductive: GMM fitted on 70% of records, kernels on the other 30%),
MAR (missingness driven by one always-observed column), MNAR (self-masking: large values more likely missing)."""
import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import load, cosm
from exp_aux import CS, NS

def sig(z): return 1.0 / (1.0 + np.exp(-z))

def calib(Z, target, beta=2.0):
    lo, hi = -30.0, 30.0
    for _ in range(80):
        a = 0.5 * (lo + hi)
        if sig(a + beta * Z).mean() < target: lo = a
        else: hi = a
    return sig(0.5 * (lo + hi) + beta * Z)

def fix_rows(mask, rng):
    full = np.where(mask.all(1))[0]
    for i in full: mask[i, rng.integers(mask.shape[1])] = False
    return mask

def make_mask(X, mech, rate, rng):
    n, D = X.shape; Zs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    if mech == "mcar":
        m = rng.random((n, D)) < rate
    elif mech == "mar":
        j0 = rng.integers(D); others = np.array([j for j in range(D) if j != j0])
        p = calib(Zs[:, j0], min(rate * D / (D - 1), 0.95))
        m = np.zeros((n, D), bool); m[:, others] = rng.random((n, len(others))) < p[:, None]
    elif mech == "mnar":
        p = calib(Zs, rate); m = rng.random((n, D)) < p
    return fix_rows(m, rng)

def kernels(Xm, cond, train_means):
    n, D = Xm.shape; mask = np.isnan(Xm)
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(len(r))) for r, m, S in cond])
    O = (~mask).astype(float); Xo = np.where(mask, 0.0, Xm)
    with np.errstate(invalid="ignore", divide="ignore"):
        Kobs = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
    Kobs[~np.isfinite(Kobs)] = 0.0
    return {"EMK-cos": Nu @ Nu.T, "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr)), "Plug-in": cosm(Mean),
            "MeanImp": cosm(np.where(mask, train_means, Xm)), "ObsSubset": Kobs}

def rmse(K, Kt):
    iu = np.triu_indices(len(Kt), 1); return {k: float(np.sqrt(((v[iu] - Kt[iu]) ** 2).mean())) for k, v in K.items()}

def run(name, rep, rate=0.5):
    X, _ = load(name, NS[name], 100 + rep); n, D = X.shape; C = CS[name]
    reg = 1e-3 * max(X.var(0).mean(), 1e-6) * 10; out = {}
    for mech in ("mcar", "mar", "mnar"):
        rng = np.random.default_rng(1000 * rep + int(rate * 100) + {"mcar": 0, "mar": 7, "mnar": 13}[mech])
        mask = make_mask(X, mech, rate, rng); Xm = np.where(mask, np.nan, X)
        pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct="full")
        cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
        out[mech] = dict(rate=float(mask.mean()), rmse=rmse(kernels(Xm, cond, np.nanmean(Xm, 0)), cosm(X)))
        if mech == "mcar":
            perm = np.random.default_rng(500 + rep).permutation(n); ntr = int(0.7 * n); tr, te = perm[:ntr], perm[ntr:]
            pi, mus, covs = em_struct(Xm[tr], C=C, iters=15, reg=reg, seed=rep, struct="full")
            cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in te]
            out["mcar_ind"] = dict(rate=float(mask[te].mean()), rmse=rmse(kernels(Xm[te], cond, np.nanmean(Xm[tr], 0)), cosm(X[te])))
    return out

if __name__ == "__main__":
    for name in sys.argv[1:]:
        with open(f"res_mech_{name}.jsonl", "w") as f:
            for rep in range(3):
                t = time.time(); r = run(name, rep); f.write(json.dumps(dict(data=name, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rep, round(time.time() - t, 1), flush=True)
