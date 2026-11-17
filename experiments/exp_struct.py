import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import load, cosm, knn_acc

def run(name, rate, rep, n, C):
    X, y = load(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    Xm = np.where(mask, np.nan, X); Kt = cosm(X); iu = np.triu_indices(n, 1)
    out = {}
    reg = 1e-3 * max(X.var(0).mean(), 1e-6) * 10
    for st, k in [("full", 0), ("diag", 0), ("lowrank3", 3), ("lowrank8", 8)]:
        s = "lowrank" if st.startswith("lowrank") else st
        pi, mus, covs = em_struct(Xm, C=C, iters=20, reg=reg, seed=rep, struct=s, k=k)
        cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
        Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
        Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
        nr = np.linalg.norm(Nu, axis=1, keepdims=True); Nn = Nu / np.maximum(nr, 1e-12)
        res = {}
        for g in (0.0, 0.5, 1.0):
            Ng = Nu / np.maximum(nr, 1e-12) ** g
            Kg = Ng @ Ng.T; np.fill_diagonal(Kg, 1.0)
            res[f"EMK-g{g}"] = dict(knn=knn_acc(Kg, y), rmse=float(np.sqrt(np.mean((Kg[iu] - Kt[iu]) ** 2))))
        Kc = cosm(Mean); res["CMI"] = dict(knn=knn_acc(Kc, y), rmse=float(np.sqrt(np.mean((Kc[iu] - Kt[iu]) ** 2))))
        # model fit: mean log-lik not computed; store mean d
        res["dmean"] = float((1 - nr[:, 0] ** 2).mean())
        out[st] = res
    out["_complete"] = knn_acc(Kt, y)
    return out

if __name__ == "__main__":
    name, n, C, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rates = [float(r) for r in sys.argv[5].split(",")]
    with open(f"res_struct_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = run(name, rate, rep, n, C)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
