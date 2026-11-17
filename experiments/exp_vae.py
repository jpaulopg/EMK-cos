import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B
from sklearn.datasets import load_digits
from vae_np import train_vae, encode
from exp_real2 import knn_acc

def mc_nu(m, S, N, rng):
    out = np.zeros_like(m); chunk = 20000
    for s in range(0, N, chunk):
        z = m + np.sqrt(S) * rng.normal(size=(chunk, len(m)))
        out += (z / np.linalg.norm(z, axis=1, keepdims=True)).sum(0)
    return out / N

def run(seed, ntest=500, N=100000):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    P = train_vae(X, zdim=10, epochs=400, seed=seed)
    mu, lv = encode(P, X); kl = 0.5 * (mu**2 + np.exp(lv) - 1 - lv).mean(0); act = np.where(kl > 0.1)[0]
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), ntest, replace=False); m = mu[idx][:, act]; v0 = np.exp(lv[idx][:, act]); yy = y[idx]
    out = {"active": int(len(act))}
    iu = np.triu_indices(ntest, 1)
    for tau in (0.25, 1.0, 4.0, 16.0):
        V = v0 * tau; d = m.shape[1]
        Nu = np.array([E.nu_gauss(m[i], np.diag(V[i])) for i in range(ntest)])
        Nt = np.array([E.nu_taylor2(m[i], np.diag(V[i])) for i in range(ntest)])
        Np = np.array([B.nu_pn2025(m[i], np.diag(V[i])) for i in range(ntest)])
        Nr = np.array([mc_nu(m[i], V[i], N, rng) for i in range(ntest)])
        Kref = Nr @ Nr.T
        tr = (m**2).sum(1) + V.sum(1)
        Ks = {"EMK-cos": Nu @ Nu.T, "RatioMeans": (m @ m.T) / np.sqrt(np.outer(tr, tr)), "Taylor2": Nt @ Nt.T, "PN-Taylor": Np @ Np.T}
        mn = m / np.linalg.norm(m, axis=1, keepdims=True); Ks["Plug-in"] = mn @ mn.T
        z = m + np.sqrt(V) * rng.normal(size=m.shape); zn = z / np.linalg.norm(z, axis=1, keepdims=True); Kreal = zn @ zn.T
        r = {}
        for k, K in Ks.items():
            e = K[iu] - Kref[iu]; er = K[iu] - Kreal[iu]
            r[k] = dict(rmse_ref=float(np.sqrt((e**2).mean())), maxerr=float(np.abs(e).max()), rmse_real=float(np.sqrt((er**2).mean())),
                        knn=knn_acc(K + np.eye(ntest) * 1.0, yy))
        n1 = Nu / np.linalg.norm(Nu, axis=1, keepdims=True); K1 = n1 @ n1.T; r["EMK-cos g1"] = dict(knn=knn_acc(K1 + np.eye(ntest), yy))
        r["MC ref"] = dict(knn=knn_acc(Kref + np.eye(ntest), yy)); r["_normnu"] = float(np.linalg.norm(Nu, axis=1).mean())
        r["_snr"] = float(np.mean(np.linalg.norm(m, axis=1) / np.sqrt(V.sum(1))))
        out[f"tau{tau}"] = r
    return out

if __name__ == "__main__":
    with open("res_vae.jsonl", "w") as f:
        for seed in range(3):
            t = time.time(); r = run(seed); f.write(json.dumps(dict(seed=seed, res=r)) + "\n"); f.flush(); print(seed, round(time.time() - t, 1), flush=True)
