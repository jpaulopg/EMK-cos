import sys, json, time
import numpy as np
from sklearn import datasets as ds
import emk_cos as E
from exp_real2 import load, cosm, knn_acc, recall

def kernels(Xm, mask, X, C, rep):
    n, D = X.shape
    pi, mus, covs = E.em_gmm_missing(Xm, C=C, iters=20, reg=1e-3 * max(X.var(0).mean(), 1e-6) * 10, seed=rep)
    cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Nt = np.array([sum(r[c] * (E.nu_taylor2(m[c], S[c]) if m[c] @ m[c] > 1e-12 else np.zeros(D)) for c in range(C)) for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(C)) for r, m, S in cond])
    dvec = 1 - (Nu ** 2).sum(1)
    nrm = np.linalg.norm(Nu, axis=1, keepdims=True); Nn = Nu / np.maximum(nrm, 1e-12)
    Xmean = np.where(mask, np.nanmean(Xm, 0), Xm)
    O = (~mask).astype(float); Xo = np.where(mask, 0.0, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        Kobs = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
    Kobs[~np.isfinite(Kobs)] = 0.0
    K = {"EMK-cos": Nu @ Nu.T + np.diag(dvec), "EMK-nodiag": Nu @ Nu.T, "EMK-renorm": Nn @ Nn.T,
         "Taylor2": Nt @ Nt.T, "CMI": cosm(Mean), "MeanImp": cosm(Xmean),
         "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr)), "ObsSubset": Kobs}
    for k in K:
        if k not in ("EMK-cos", "EMK-nodiag"): np.fill_diagonal(K[k], 1.0)
    return K, nrm[:, 0]

def run_cls(name, rate, rep, n, C):
    X, y = load(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    Xm = np.where(mask, np.nan, X)
    K, nrm = kernels(Xm, mask, X, C, rep)
    Kt = cosm(X); iu = np.triu_indices(n, 1)
    out = {k: dict(knn=knn_acc(M, y), rmse=float(np.sqrt(np.mean((M[iu] - Kt[iu]) ** 2)))) for k, M in K.items()}
    # hubness: skewness of 5-occurrence counts, and mean ||nu|| of selected neighbours vs all
    for k in ["EMK-cos", "CMI"]:
        B = K[k].copy(); np.fill_diagonal(B, -np.inf); nb = np.argsort(-B, 1)[:, :5]
        out[k]["nb_nrm"] = float(nrm[nb].mean())
    out["_all_nrm"] = float(nrm.mean()); out["_complete"] = knn_acc(Kt, y)
    return out

def load_reg(name, n, seed):
    rng = np.random.default_rng(seed)
    if name == "diabetes":
        b = ds.load_diabetes(); X, y = b.data.astype(float), b.target.astype(float)
    elif name == "wine_reg":
        b = ds.load_wine(); X, y = b.data[:, 1:].astype(float), b.data[:, 0].astype(float)
    elif name == "bc_reg":
        b = ds.load_breast_cancer(); X, y = b.data[:, 1:].astype(float), b.data[:, 0].astype(float)
    elif name == "digits_reg":
        b = ds.load_digits(); X, y = b.data.astype(float), b.target.astype(float)
    idx = rng.choice(len(X), min(n, len(X)), replace=False); X, y = X[idx], y[idx]
    keep = X.std(0) > 1e-9; X = X[:, keep]
    lo, hi = X.min(0), X.max(0); X = (X - lo) / (hi - lo)
    y = (y - y.mean()) / y.std()
    return X, y

def krr_eval(K, y, tr, te, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    # inner 5-fold CV on train for lambda
    rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(tr), 5)
    best, bl = 1e18, lams[0]
    for lam in lams:
        err = 0
        for f in folds:
            t2 = np.setdiff1d(tr, f); ym = y[t2].mean()
            a = np.linalg.solve(K[np.ix_(t2, t2)] + lam * np.eye(len(t2)), y[t2] - ym)
            err += np.sum((K[np.ix_(f, t2)] @ a + ym - y[f]) ** 2)
        if err < best: best, bl = err, lam
    ym = y[tr].mean()
    a = np.linalg.solve(K[np.ix_(tr, tr)] + bl * np.eye(len(tr)), y[tr] - ym)
    p = K[np.ix_(te, tr)] @ a + ym
    r2 = 1 - np.sum((p - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
    # kernel-weighted 5-NN regression
    B = K[np.ix_(te, tr)]; nb = np.argsort(-B, 1)[:, :5]; pk = y[tr][nb].mean(1)
    r2k = 1 - np.sum((pk - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
    return float(r2), float(r2k), bl

def run_reg(name, rate, rep, n, C):
    X, y = load_reg(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    Xm = np.where(mask, np.nan, X)
    K, nrm = kernels(Xm, mask, X, C, rep)
    perm = np.random.default_rng(rep).permutation(n); tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    out = {}
    for k, M in K.items():
        r2, r2k, lam = krr_eval(0.5 * (M + M.T), y, tr, te)
        out[k] = dict(r2=r2, r2knn=r2k, lam=lam)
    Kt = cosm(X); r2, r2k, lam = krr_eval(Kt, y, tr, te); out["_complete"] = dict(r2=r2, r2knn=r2k)
    return out

if __name__ == "__main__":
    mode, name, n, C, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rates = [float(r) for r in sys.argv[6].split(",")]
    fn = run_cls if mode == "cls" else run_reg
    with open(f"res_diag_{mode}_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = fn(name, rate, rep, n, C)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
