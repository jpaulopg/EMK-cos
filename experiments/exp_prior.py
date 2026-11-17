"""Bayesian regression weights: effect of the prior variance on the error against the true weights.
Same data-generating process and random stream as exp_rv.blr_run; prior variance tau2 varies
(1.25 matches the marginal variance of the true weights; 4 is the value of Table 5)."""
import sys, json
import numpy as np
import emk_cos as E
from sklearn.datasets import load_diabetes

TAUS = (0.3125, 1.25, 4.0, 20.0)

def run(seed, n=150, noise=1.0, iid=False):
    rng = np.random.default_rng(seed)
    X, _ = load_diabetes(return_X_y=True); X = (X - X.mean(0)) / X.std(0); D = X.shape[1]
    fam = rng.normal(size=(4, D)); lab = rng.integers(0, 4, n)
    Wt = fam[lab] + 0.5 * rng.normal(size=(n, D))
    if iid: Wt = np.sqrt(1.25) * np.random.default_rng(10_000 + seed).normal(size=(n, D))
    data = {}
    for nk in (15, 30, 60):
        data[nk] = []
        for i in range(n):
            idx = rng.choice(len(X), nk, replace=False); Xk = X[idx]; yv = Xk @ Wt[i] + noise * rng.normal(size=nk)
            data[nk].append((Xk, yv))
    Wn = Wt / np.linalg.norm(Wt, axis=1, keepdims=True); Kt = Wn @ Wn.T; iu = np.triu_indices(n, 1); out = {}
    for nk in (15, 30):
        for tau2 in TAUS:
            M = np.zeros((n, D)); S = np.zeros((n, D, D))
            for i, (Xk, yv) in enumerate(data[nk]):
                S[i] = np.linalg.inv(Xk.T @ Xk / noise**2 + np.eye(D) / tau2); M[i] = S[i] @ Xk.T @ yv / noise**2
            Nu = np.array([E.nu_gauss(M[i], S[i]) for i in range(n)]); Nt = np.array([E.nu_taylor2(M[i], S[i]) for i in range(n)])
            tr = (M**2).sum(1) + np.trace(S, axis1=1, axis2=2); mn = M / np.linalg.norm(M, axis=1, keepdims=True)
            Ks = {"EMK-cos": Nu @ Nu.T, "RatioMeans": (M @ M.T) / np.sqrt(np.outer(tr, tr)), "Taylor2": Nt @ Nt.T, "Plug-in": mn @ mn.T}
            out[f"nk{nk}_tau{tau2}"] = {k: float(np.sqrt(((K[iu] - Kt[iu])**2).mean())) for k, K in Ks.items()}
    return out

if __name__ == "__main__":
    seeds = range(int(sys.argv[1])) if len(sys.argv) > 1 else range(3)
    iid = len(sys.argv) > 2 and sys.argv[2] == "iid"
    with open("res_prior_iid.jsonl" if iid else "res_prior.jsonl", "w") as f:
        for s in seeds:
            f.write(json.dumps(dict(seed=s, res=run(s, iid=iid))) + "\n"); f.flush(); print(s, flush=True)
