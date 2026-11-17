import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import load, cosm, knn_acc
from exp_knn import pair_d2, knn_impute, knn_posterior
from exp_select3 import gmm_post_mean

STRUCTS = [("full", 0), ("diag", 0), ("lowrank3", 3)]
CS = [2, 4, 8]
KS = [5, 10, 20]

def run(name, rate, rep, n, hold=0.15):
    X, y = load(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    Xm = np.where(mask, np.nan, X); Kt = cosm(X); iu = np.triu_indices(n, 1)
    r2 = np.random.default_rng(777 + rep); H = (~mask) & (r2.random((n, D)) < hold)
    for i in range(n):
        if (~mask[i] & ~H[i]).sum() == 0: H[i] = False
    Xt = np.where(mask | H, np.nan, X); rows = [i for i in range(n) if H[i].any()]
    reg = 1e-3 * max(X.var(0).mean(), 1e-6) * 10; out = {}
    def score(K, K1):
        return dict(rmse=float(np.sqrt(np.mean((K[iu] - Kt[iu]) ** 2))), knn1=knn_acc(K1, y))
    for C in CS:
        if C >= n // 10: continue
        for st, kk in STRUCTS:
            s = "lowrank" if st.startswith("lowrank") else st
            pi, mus, covs = em_struct(Xt, C=C, iters=15, reg=reg, seed=rep, struct=s, k=kk)
            se = 0.0; cnt = 0
            for i in rows:
                h = np.where(H[i])[0]; o = np.where(~mask[i] & ~H[i])[0]
                mu = gmm_post_mean(X[i], o, h, pi, mus, covs); se += ((mu - X[i, h]) ** 2).sum(); cnt += len(h)
            pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct=s, k=kk)
            cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
            Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
            nr = np.linalg.norm(Nu, axis=1, keepdims=True); K = Nu @ Nu.T; np.fill_diagonal(K, 1.0)
            N1 = Nu / np.maximum(nr, 1e-12); K1 = N1 @ N1.T; np.fill_diagonal(K1, 1.0)
            out[f"gmm_{st}_C{C}"] = dict(mse=se / cnt, **score(K, K1))
    d2t, obst = pair_d2(Xt)
    for k in KS:
        Xh = knn_impute(Xt, d2t, obst, k); se = 0.0; cnt = 0
        for i in rows:
            h = np.where(H[i])[0]; nb = np.argsort(d2t[i])[:k]; nb = nb[np.isfinite(d2t[i, nb])]
            mu = Xh[np.ix_(nb, h)].mean(0); se += ((mu - X[i, h]) ** 2).sum(); cnt += len(h)
        Nu, _ = knn_posterior(Xm, k, "mix", reg)
        nr = np.linalg.norm(Nu, axis=1, keepdims=True); K = Nu @ Nu.T; np.fill_diagonal(K, 1.0)
        N1 = Nu / np.maximum(nr, 1e-12); K1 = N1 @ N1.T; np.fill_diagonal(K1, 1.0)
        out[f"knn_k{k}"] = dict(mse=se / cnt, **score(K, K1))
    return out

if __name__ == "__main__":
    name, n, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rates = [float(r) for r in sys.argv[4].split(",")]
    with open(f"res_auto_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = run(name, rate, rep, n)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
