import sys, json, time
import numpy as np
from em_struct import em_struct
from exp_real2 import load
from exp_knn import pair_d2, knn_impute
from exp_select2 import rowlog

def gmm_post_mean(x, o, h, pi, mus, covs):
    C = len(pi); lp = np.zeros(C); m = np.zeros((C, len(h)))
    for c in range(C):
        L = np.linalg.cholesky(covs[c][np.ix_(o, o)]); d = x[o] - mus[c][o]; z = np.linalg.solve(L, d)
        lp[c] = np.log(pi[c]) - 0.5 * z @ z - np.log(np.diag(L)).sum()
        K = np.linalg.solve(covs[c][np.ix_(o, o)], covs[c][np.ix_(h, o)].T).T
        m[c] = mus[c][h] + K @ d
    r = np.exp(lp - lp.max()); r /= r.sum(); return r @ m

def run(name, rate, rep, n, C, k=10, hold=0.15):
    X, y = load(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    r2 = np.random.default_rng(777 + rep); H = (~mask) & (r2.random((n, D)) < hold)
    for i in range(n):
        if (~mask[i] & ~H[i]).sum() == 0: H[i] = False
    Xt = np.where(mask | H, np.nan, X)
    reg = 1e-3 * max(X.var(0).mean(), 1e-6) * 10
    out = {}
    rows = [i for i in range(n) if H[i].any()]
    for st, kk in [("full", 0), ("diag", 0), ("lowrank3", 3), ("lowrank8", 8)]:
        s = "lowrank" if st.startswith("lowrank") else st
        pi, mus, covs = em_struct(Xt, C=C, iters=20, reg=reg, seed=rep, struct=s, k=kk)
        se = 0.0; cnt = 0
        for i in rows:
            h = np.where(H[i])[0]; o = np.where(~mask[i] & ~H[i])[0]
            mu = gmm_post_mean(X[i], o, h, pi, mus, covs); se += ((mu - X[i, h]) ** 2).sum(); cnt += len(h)
        out[st] = dict(mse=se / cnt)
    # kNN model on Xt
    d2, obs = pair_d2(Xt); Xh = knn_impute(Xt, d2, obs, k); se = 0.0; ll = 0.0; cnt = 0
    for i in rows:
        h = np.where(H[i])[0]
        nb = np.argsort(d2[i])[:k]; nb = nb[np.isfinite(d2[i, nb])]
        B = Xh[np.ix_(nb, h)]; mu = B.mean(0); se += ((mu - X[i, h]) ** 2).sum(); cnt += len(h)
    out["knn"] = dict(mse=se / cnt)
    return out

if __name__ == "__main__":
    name, n, C, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rates = [float(r) for r in sys.argv[5].split(",")]
    with open(f"res_select3_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = run(name, rate, rep, n, C)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, mse=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
