"""PSD of expected-cosine estimators (unit diagonal) for the three random-vector applications.
Adds pairwise Monte Carlo (independent draws per pair, S samples) and per-vector MC (Gram of MC directions)."""
import sys, json, time
import numpy as np
import emk_cos as E
import exp_rv as R
from sklearn.datasets import load_digits
from vae_np import train_vae, encode

def sqrtm(S):
    lam, Q = np.linalg.eigh(0.5 * (S + S.T)); return Q * np.sqrt(np.clip(lam, 0, None))

def chol_ok(K, jit=1e-6):
    try: np.linalg.cholesky(K + jit * np.eye(len(K))); return True
    except np.linalg.LinAlgError: return False

def psd_eval(M, S, rng, Ss=(10, 100)):
    n, D = M.shape; A = [sqrtm(s) for s in S]
    draw = lambda i, k: M[i] + rng.normal(size=(k, D)) @ A[i].T
    Nu = np.array([E.nu_gauss(M[i], S[i]) for i in range(n)])
    Nt = np.array([E.nu_taylor2(M[i], S[i]) for i in range(n)])
    tr = (M**2).sum(1) + np.array([np.trace(s) for s in S]); mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    Ks = {"EMK-cos": Nu @ Nu.T, "RatioMeans": (M @ M.T) / np.sqrt(np.outer(tr, tr)), "Taylor2": Nt @ Nt.T, "Plug-in": mn @ mn.T}
    for s in Ss:
        Z = np.stack([draw(i, s) for i in range(n)]); Z /= np.linalg.norm(Z, axis=2, keepdims=True)
        Ks[f"MC-vec{s}"] = Z.mean(1) @ Z.mean(1).T
        Z2 = np.stack([draw(i, s) for i in range(n)]); Z2 /= np.linalg.norm(Z2, axis=2, keepdims=True)
        Kp = np.einsum("isd,jsd->ij", Z, Z2) / s; Ks[f"MC-pair{s}"] = 0.5 * (Kp + Kp.T)  # independent draws for the two arguments
    out = {}
    for k, K in Ks.items():
        K = K.copy(); np.fill_diagonal(K, 1.0); ev = np.linalg.eigvalsh(K)
        out[k] = dict(mineig=float(ev.min()), negmass=float(-ev[ev < 0].sum() / np.abs(ev).sum()), chol=chol_ok(K))
    return out

def vae_cases(seed, ntest=300):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    P = train_vae(X, zdim=10, epochs=400, seed=seed); mu, lv = encode(P, X)
    kl = 0.5 * (mu**2 + np.exp(lv) - 1 - lv).mean(0); act = np.where(kl > 0.1)[0]
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), ntest, replace=False)
    m = mu[idx][:, act]; v0 = np.exp(lv[idx][:, act]); res = {}
    for tau in (1.0, 4.0, 16.0):
        res[f"tau{tau}"] = psd_eval(m, np.array([np.diag(v) for v in v0 * tau]), rng)
    return res

def capture(fn, seed):
    got = {}; old = R.evaluate
    def ev(M, S, W, lab, rng, **kw): got[len(got)] = psd_eval(M, S, rng); return {}
    R.evaluate = ev
    try: fn(seed)
    finally: R.evaluate = old
    return got

if __name__ == "__main__":
    seed = int(sys.argv[1]); t = time.time()
    g = capture(R.gp_run, seed); b = capture(R.blr_run, seed)
    out = dict(vae=vae_cases(seed), gp={f"nobs{k}": g[i] for i, k in enumerate((3, 6, 12))}, blr={f"nk{k}": b[i] for i, k in enumerate((15, 30, 60))})
    with open(f"res_psd_{seed}.jsonl", "w") as f: f.write(json.dumps(dict(seed=seed, res=out)) + "\n")
    print(seed, round(time.time() - t, 1), flush=True)
