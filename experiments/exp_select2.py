import sys, json, time
import numpy as np
from em_struct import em_struct
from exp_real2 import masked, ridge_from

def rowlog(x, idx, pi, mus, covs):
    C = len(pi); lp = np.zeros(C)
    for c in range(C):
        L = np.linalg.cholesky(covs[c][np.ix_(idx, idx)]); z = np.linalg.solve(L, x[idx] - mus[c][idx])
        lp[c] = np.log(pi[c]) - 0.5 * z @ z - np.log(np.diag(L)).sum() - 0.5 * len(idx) * np.log(2 * np.pi)
    m = lp.max(); return m + np.log(np.exp(lp - m).sum())

def run(name, rate, rep, n, C, hold=0.15):
    X, Xm, mask, y = masked(name, n, 100 + rep, rate, 1000 * rep + int(rate * 100)); n, D = X.shape
    r2 = np.random.default_rng(777 + rep); H = (~mask) & (r2.random((n, D)) < hold)
    # keep at least 1 observed entry per row after holding out
    for i in range(n):
        if (~mask[i] & ~H[i]).sum() == 0: H[i] = False
    Xt = np.where(H, np.nan, Xm)
    reg = ridge_from(Xt)
    out = {}
    for st, k in [("full", 0), ("diag", 0), ("lowrank3", 3), ("lowrank8", 8)]:
        s = "lowrank" if st.startswith("lowrank") else st
        pi, mus, covs = em_struct(Xt, C=C, iters=20, reg=reg, seed=rep, struct=s, k=k)
        v = 0.0; cnt = 0
        for i in range(n):
            h = np.where(H[i])[0]
            if len(h) == 0: continue
            o = np.where(~mask[i] & ~H[i])[0]
            v += rowlog(X[i], np.concatenate([o, h]), pi, mus, covs) - rowlog(X[i], o, pi, mus, covs); cnt += len(h)
        out[st] = v / cnt
    return out

if __name__ == "__main__":
    name, n, C, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rates = [float(r) for r in sys.argv[5].split(",")]
    with open(f"res_select2_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = run(name, rate, rep, n, C)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, val=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
