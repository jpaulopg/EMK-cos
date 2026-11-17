import sys, json, time
SMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
import numpy as np
import emk_cos as E
from sklearn.datasets import load_digits
from vae_np import train_vae, encode

def krr_cv(Ktr, Yt, lams=(1e-3, 1e-2, 1e-1, 1.0), folds=4, rng=None):
    n = len(Ktr); idx = rng.permutation(n); best = None
    for lam in lams:
        err = 0.0
        for f in range(folds):
            va = idx[f::folds]; tr = np.setdiff1d(idx, va)
            A = np.linalg.solve(Ktr[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Yt[tr]); err += ((Ktr[np.ix_(va, tr)] @ A - Yt[va]) ** 2).sum()
        if best is None or err < best[0]: best = (err, lam)
    return best[1]

def run(seed, ntr=800, nte=500):
    X, y = load_digits(return_X_y=True); X = X / 16.0; rng = np.random.default_rng(seed)
    n = len(X); sig = rng.uniform(0.0, SMAX, n); Xn = X + sig[:, None] * rng.normal(size=X.shape)
    P = train_vae(Xn, zdim=10, epochs=400, seed=seed)
    mu, lv = encode(P, Xn); kl = 0.5 * (mu**2 + np.exp(lv) - 1 - lv).mean(0); act = np.where(kl > 0.1)[0]
    m = mu[:, act]; V = np.exp(lv[:, act])
    idx = rng.permutation(n); tr = idx[:ntr]; te = idx[ntr:ntr + nte]
    Nu = np.array([E.nu_gauss(m[i], np.diag(V[i])) for i in range(n)])
    Nt = np.array([E.nu_taylor2(m[i], np.diag(V[i])) for i in range(n)])
    trV = (m**2).sum(1) + V.sum(1); mn = m / np.linalg.norm(m, axis=1, keepdims=True)
    def kern(name, A, B):
        if name == "EMK-cos": K = Nu[A] @ Nu[B].T
        elif name == "EMK-cos g1": a = Nu[A] / np.linalg.norm(Nu[A], axis=1, keepdims=True); b = Nu[B] / np.linalg.norm(Nu[B], axis=1, keepdims=True); K = a @ b.T
        elif name == "RatioMeans": K = (m[A] @ m[B].T) / np.sqrt(np.outer(trV[A], trV[B]))
        elif name == "Taylor2": K = Nt[A] @ Nt[B].T
        else: K = mn[A] @ mn[B].T
        return np.clip(K, -1, 1)
    Y = np.eye(10)[y]; t = X.mean(1); t = (t - t[tr].mean()) / t[tr].std()
    out = {"active": int(len(act)), "mean_sigma_post": float(np.sqrt(V).mean()), "snr": float(np.mean(np.linalg.norm(m, axis=1) / np.sqrt(V.sum(1))))}
    for name in ("EMK-cos", "EMK-cos g1", "RatioMeans", "Taylor2", "Plug-in"):
        Ktr = kern(name, tr, tr); np.fill_diagonal(Ktr, 1.0)
        if name == "EMK-cos": Ktr = Nu[tr] @ Nu[tr].T + np.diag(1 - (Nu[tr] ** 2).sum(1))
        Kte = kern(name, te, tr); r = {}
        for tag, T in (("acc", Y), ("r2", t[:, None])):
            lam = krr_cv(Ktr, T[tr], rng=np.random.default_rng(seed)); A = np.linalg.solve(Ktr + lam * np.eye(ntr), T[tr]); pred = Kte @ A
            r[tag] = float((pred.argmax(1) == y[te]).mean()) if tag == "acc" else float(1 - ((pred[:, 0] - t[te]) ** 2).sum() / ((t[te] - t[te].mean()) ** 2).sum())
        r["mineig"] = float(np.linalg.eigvalsh(0.5 * (Ktr + Ktr.T)).min()); out[name] = r
    return out

if __name__ == "__main__":
    with open("res_vae_task_" + str(SMAX) + ".jsonl", "w") as f:
        for seed in range(5):
            t = time.time(); r = run(seed); f.write(json.dumps(dict(seed=seed, res=r)) + "\n"); f.flush(); print(seed, round(time.time() - t, 1), flush=True)
