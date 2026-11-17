"""Random-vector kernels with FULL covariance: (A) GP posterior of curves, (B) Bayesian linear regression weights."""
import sys, json, time
import numpy as np
import emk_cos as E
from sklearn.datasets import load_diabetes
from sklearn.metrics import roc_auc_score

def mc_nu_full(m, S, N, rng):
    lam, Q = np.linalg.eigh(0.5 * (S + S.T)); A = Q * np.sqrt(np.clip(lam, 0, None))
    out = np.zeros_like(m); chunk = 20000
    for s in range(0, N, chunk):
        z = m + rng.normal(size=(chunk, len(m))) @ A.T
        out += (z / np.linalg.norm(z, axis=1, keepdims=True)).sum(0)
    return out / N

def evaluate(M, S, W, lab, rng, N=100000, th=0.5):
    n = len(M); iu = np.triu_indices(n, 1); same = (lab[:, None] == lab[None, :])[iu]
    Nu = np.array([E.nu_gauss(M[i], S[i]) for i in range(n)])
    Nt = np.array([E.nu_taylor2(M[i], S[i]) for i in range(n)])
    Nr = np.array([mc_nu_full(M[i], S[i], N, rng) for i in range(n)])
    Kref = Nr @ Nr.T
    tr = (M**2).sum(1) + np.array([np.trace(s) for s in S])
    mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True); Ktrue = Wn @ Wn.T
    Ks = {"EMK-cos": Nu @ Nu.T, "RatioMeans": (M @ M.T) / np.sqrt(np.outer(tr, tr)), "Taylor2": Nt @ Nt.T, "Plug-in": mn @ mn.T}
    th_far = np.quantile(Kref[iu][~same], 0.95)
    out = {"_snr": float(np.mean(np.linalg.norm(M, axis=1) / np.sqrt([np.trace(s) for s in S]))), "_th_far": float(th_far)}
    for k, K in Ks.items():
        e = K[iu] - Kref[iu]; et = K[iu] - Ktrue[iu]
        a = K[iu] > th_far
        out[k] = dict(rmse_ref=float(np.sqrt((e**2).mean())), maxerr=float(np.abs(e).max()), rmse_true=float(np.sqrt((et**2).mean())),
                      mineig=float(np.linalg.eigvalsh(0.5 * (K + K.T)).min()), auc=float(roc_auc_score(same, K[iu])),
                      far_at_exact_th=float(a[~same].mean()), tpr_at_exact_th=float(a[same].mean()),
                      dis_th=float(((K[iu] > th) != (Kref[iu] > th)).mean()), dis_true=float(((K[iu] > th) != (Ktrue[iu] > th)).mean()))
    out["MC ref"] = dict(rmse_true=float(np.sqrt(((Kref[iu] - Ktrue[iu])**2).mean())), dis_true=float(((Kref[iu] > th) != (Ktrue[iu] > th)).mean()),
                         auc=float(roc_auc_score(same, Kref[iu])))
    return out

def gp_run(seed, n=150, G=40, noise=0.3):
    rng = np.random.default_rng(seed); x = np.linspace(0, 1, G)
    def kern(a, b, s2=1.0, l=0.2): return s2 * np.exp(-0.5 * (a[:, None] - b[None, :])**2 / l**2)
    Kg = kern(x, x); Lc = np.linalg.cholesky(Kg + 1e-8 * np.eye(G))
    lab = rng.integers(0, 4, n); phase = rng.uniform(0, 2 * np.pi, n); amp = rng.uniform(0.7, 1.3, n)
    base = lambda c, ph, a: a * np.sin(2 * np.pi * (c + 1) * x * 0.75 + ph)
    F = np.array([base(lab[i], 0.3 * phase[i], amp[i]) + 0.5 * (Lc @ rng.normal(size=G)) for i in range(n)])
    out = {}
    for nobs in (3, 6, 12):
        M = np.zeros((n, G)); S = np.zeros((n, G, G))
        for i in range(n):
            idx = np.sort(rng.choice(G, nobs, replace=False)); yv = F[i, idx] + noise * rng.normal(size=nobs)
            Kxx = Kg[np.ix_(idx, idx)] + noise**2 * np.eye(nobs); Kxs = Kg[:, idx]
            A = np.linalg.solve(Kxx, Kxs.T)
            M[i] = A.T @ yv; S[i] = Kg - Kxs @ A
        out[f"nobs{nobs}"] = evaluate(M, S, F, lab, rng)
    return out

def blr_run(seed, n=150, noise=1.0, tau2=4.0):
    rng = np.random.default_rng(seed)
    X, _ = load_diabetes(return_X_y=True); X = (X - X.mean(0)) / X.std(0); D = X.shape[1]
    fam = rng.normal(size=(4, D)); lab = rng.integers(0, 4, n)
    Wt = fam[lab] + 0.5 * rng.normal(size=(n, D))
    out = {}
    for nk in (15, 30, 60):
        M = np.zeros((n, D)); S = np.zeros((n, D, D))
        for i in range(n):
            idx = rng.choice(len(X), nk, replace=False); Xk = X[idx]; yv = Xk @ Wt[i] + noise * rng.normal(size=nk)
            P = Xk.T @ Xk / noise**2 + np.eye(D) / tau2; S[i] = np.linalg.inv(P); M[i] = S[i] @ Xk.T @ yv / noise**2
        r = evaluate(M, S, Wt, lab, rng); r["_cond"] = float(np.mean([np.linalg.cond(s) for s in S])); out[f"nk{nk}"] = r
    return out

if __name__ == "__main__":
    kind, seed = sys.argv[1], int(sys.argv[2]); t = time.time()
    r = gp_run(seed) if kind == "gp" else blr_run(seed)
    with open(f"res_rv_{kind}_{seed}.jsonl", "w") as f: f.write(json.dumps(dict(seed=seed, res=r)) + "\n")
    print(kind, seed, round(time.time() - t, 1), flush=True)
