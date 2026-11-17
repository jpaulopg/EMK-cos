"""Uncertainty of the cosine: predicted mean and variance of the realized cosine.

Exact (EMK-cos): mean nu_i^T nu_j, variance tr(G_i G_j) - mean^2 (Theorems 1 and 2).
Ratio of moments (Baumann et al. 2024): mean E[g^T h]/sqrt(E||g||^2 E||h||^2), variance Var[g^T h]/(E||g||^2 E||h||^2).
PN-Taylor (Herrera-Esposito & Burge 2025): mean gamma_i^T gamma_j, variance tr(G~_i G~_j) - mean^2 with their
second-moment approximation. MC-100: per-input Monte Carlo moments with 100 draws.
Targets: 20 independent realized cosines per pair drawn from the laws; reference moments by Monte Carlo (2e4 draws).
Scores: Dawid-Sebastiani score log s^2 + (y-m)^2/s^2 (proper for mean-variance forecasts), the variance ratio
E[(y-m)^2]/E[s^2], and the RMSE of the predicted standard deviation against the reference."""
import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B
import exp_rv as R
from sklearn.datasets import load_digits
from vae_np import train_vae, encode

def sqrtm(S):
    lam, Q = np.linalg.eigh(0.5 * (S + S.T)); return Q * np.sqrt(np.clip(lam, 0, None))

def pn_second(m, S):
    d = np.trace(S) + m @ m; N = S + np.outer(m, m); Sm = S @ m
    vard = 2 * np.trace(S @ S) + 4 * m @ Sm
    C = 2 * (S @ S) + 2 * (np.outer(m, Sm) + np.outer(Sm, m))
    return N / d - C / d**2 + N * vard / d**3

def mc_moments(m, S, n, rng, A):
    z = m + rng.normal(size=(n, len(m))) @ A.T; u = z / np.linalg.norm(z, axis=1, keepdims=True)
    return u.mean(0), u.T @ u / n

def evaluate(M, S, rng, nreal=20, nref=20000):
    n, D = M.shape; A = [sqrtm(s) for s in S]; iu = np.triu_indices(n, 1)
    Nu = np.array([E.nu_gauss(M[i], S[i]) for i in range(n)]); G = np.array([E.m2_gauss(M[i], S[i]) for i in range(n)])
    Np = np.array([B.nu_pn2025(M[i], S[i]) for i in range(n)]); Gp = np.array([pn_second(M[i], S[i]) for i in range(n)])
    mc = [mc_moments(M[i], S[i], 100, rng, A[i]) for i in range(n)]; Nm = np.array([a for a, _ in mc]); Gm = np.array([b for _, b in mc])
    ref = [mc_moments(M[i], S[i], nref, rng, A[i]) for i in range(n)]; Nr = np.array([a for a, _ in ref]); Gr = np.array([b for _, b in ref])
    Tr = (M**2).sum(1) + np.trace(S, axis1=1, axis2=2)
    def mv(Nx, Gx):
        mu = Nx @ Nx.T; Gf = Gx.reshape(n, -1); return mu[iu], (Gf @ Gf.T - mu**2)[iu]
    preds = {"EMK-cos": mv(Nu, G), "PN-Taylor": mv(Np, Gp), "MC-100": mv(Nm, Gm),
             "RatioMoments": (B.ratio_mean(M, Tr)[iu], B.ratio_var(M, S, Tr)[iu])}
    mu_ref, var_ref = mv(Nr, Gr); sd_ref = np.sqrt(np.clip(var_ref, 0, None))
    Z = np.stack([np.stack([M[i] + rng.normal(size=(nreal, D)) @ A[i].T for i in range(n)], 1)])[0]   # nreal x n x D
    Z /= np.linalg.norm(Z, axis=2, keepdims=True)
    Y = np.einsum("rid,rjd->rij", Z, Z)[:, iu[0], iu[1]]                                           # nreal x pairs
    out = {}
    for k, (mu, var) in preds.items():
        v = np.clip(var, 1e-6, None)
        out[k] = dict(dss=float(np.mean(np.log(v) + (Y - mu) ** 2 / v)),
                      var_ratio=float(np.mean((Y - mu) ** 2) / np.mean(v)),
                      cover2=float(np.mean(np.abs(Y - mu) <= 2 * np.sqrt(v))),
                      rmse_mean_ref=float(np.sqrt(np.mean((mu - mu_ref) ** 2))),
                      rmse_sd_ref=float(np.sqrt(np.mean((np.sqrt(v) - sd_ref) ** 2))),
                      neg_var=float(np.mean(var < 0)))
    out["_mean_sd_ref"] = float(sd_ref.mean())
    return out

def vae_cases(seed, ntest=300):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    P = train_vae(X, zdim=10, epochs=400, seed=seed); mu, lv = encode(P, X)
    kl = 0.5 * (mu**2 + np.exp(lv) - 1 - lv).mean(0); act = np.where(kl > 0.1)[0]
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), ntest, replace=False)
    m = mu[idx][:, act]; v0 = np.exp(lv[idx][:, act])
    return {f"tau{t}": evaluate(m, np.array([np.diag(v) for v in v0 * t]), rng) for t in (0.25, 1.0, 4.0, 16.0)}

def capture(fn, seed):
    got = {}; old = R.evaluate
    def ev(M, S, W, lab, rng, **kw): got[len(got)] = evaluate(M, S, rng); return {}
    R.evaluate = ev
    try: fn(seed)
    finally: R.evaluate = old
    return got

if __name__ == "__main__":
    seed = int(sys.argv[1]); t = time.time()
    g = capture(R.gp_run, seed); b = capture(R.blr_run, seed)
    out = dict(vae=vae_cases(seed), gp={f"nobs{k}": g[i] for i, k in enumerate((3, 6, 12))},
               blr={f"nk{k}": b[i] for i, k in enumerate((15, 30, 60))})
    with open(f"res_unc_{seed}.jsonl", "w") as f: f.write(json.dumps(dict(seed=seed, res=out)) + "\n")
    print(seed, round(time.time() - t, 1), flush=True)
