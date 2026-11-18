"""Covariance structure of the GMM: observed-data BIC and RMSE of EMK-cos for full, diagonal and low-rank
covariance (paragraph "Choice of the density model"). Paired with exp_select2.py (held-out imputation score)
by sum_select2.py. Usage: exp_select.py NAME N C REPS RATES"""
import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import masked, cosm, knn_acc, ridge_from

def obs_loglik(X, pi, mus, covs):
    n, D = X.shape; M = ~np.isnan(X); C = len(pi); ll = 0.0
    for i in range(n):
        o = np.where(M[i])[0]
        if len(o) == 0: continue
        xo = X[i, o]; lp = np.zeros(C)
        for c in range(C):
            L = np.linalg.cholesky(covs[c][np.ix_(o, o)]); z = np.linalg.solve(L, xo - mus[c][o])
            lp[c] = np.log(pi[c]) - 0.5 * z @ z - np.log(np.diag(L)).sum() - 0.5 * len(o) * np.log(2 * np.pi)
        m = lp.max(); ll += m + np.log(np.exp(lp - m).sum())
    return ll

def npar(st, k, C, D):
    k = min(k, D - 1)
    cov = {"full": D * (D + 1) // 2, "diag": D}.get(st, D * k - k * (k - 1) // 2 + 1)
    return C * (D + cov) + C - 1

def run(name, rate, rep, n, C):
    X, Xm, mask, y = masked(name, n, 100 + rep, rate, 1000 * rep + int(rate * 100)); n, D = X.shape
    Kt = cosm(X); iu = np.triu_indices(n, 1)
    reg = ridge_from(Xm)
    out = {}
    for st, k in [("full", 0), ("diag", 0), ("lowrank3", 3), ("lowrank8", 8)]:
        s = "lowrank" if st.startswith("lowrank") else st
        pi, mus, covs = em_struct(Xm, C=C, iters=20, reg=reg, seed=rep, struct=s, k=k)
        ll = obs_loglik(Xm, pi, mus, covs); p = npar(s, k, C, D)
        cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
        Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
        nr = np.linalg.norm(Nu, axis=1, keepdims=True)
        K = Nu @ Nu.T; np.fill_diagonal(K, 1.0)
        Ng = Nu / np.maximum(nr, 1e-12); K1 = Ng @ Ng.T; np.fill_diagonal(K1, 1.0)
        out[st] = dict(ll=ll, bic=-2 * ll + p * np.log(n), rmse=float(np.sqrt(np.mean((K[iu] - Kt[iu]) ** 2))),
                       knn1=knn_acc(K1, y), rmse1=float(np.sqrt(np.mean((K1[iu] - Kt[iu]) ** 2))))
    return out

if __name__ == "__main__":
    name, n, C, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rates = [float(r) for r in sys.argv[5].split(",")]
    with open(f"res_select_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = run(name, rate, rep, n, C)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
