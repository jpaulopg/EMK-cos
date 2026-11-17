import sys, json, time
import numpy as np
import emk_cos as E
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from vae_np import train_vae, encode
from exp_vae import mc_nu

THS = (0.3, 0.5, 0.7, 0.9)

def run(seed, ntest=500, N=100000):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    P = train_vae(X, zdim=10, epochs=400, seed=seed)
    mu, lv = encode(P, X); kl = 0.5 * (mu**2 + np.exp(lv) - 1 - lv).mean(0); act = np.where(kl > 0.1)[0]
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), ntest, replace=False)
    m = mu[idx][:, act]; v0 = np.exp(lv[idx][:, act]); yy = y[idx]
    iu = np.triu_indices(ntest, 1); same = (yy[:, None] == yy[None, :])[iu]
    out = {}
    for tau in (0.25, 1.0, 4.0, 16.0):
        V = v0 * tau
        Nu = np.array([E.nu_gauss(m[i], np.diag(V[i])) for i in range(ntest)])
        Nt = np.array([E.nu_taylor2(m[i], np.diag(V[i])) for i in range(ntest)])
        Nr = np.array([mc_nu(m[i], V[i], N, rng) for i in range(ntest)])
        Kref = (Nr @ Nr.T)[iu]
        tr = (m**2).sum(1) + V.sum(1)
        mn = m / np.linalg.norm(m, axis=1, keepdims=True)
        Ks = {"EMK-cos": (Nu @ Nu.T)[iu], "RatioMeans": ((m @ m.T) / np.sqrt(np.outer(tr, tr)))[iu],
              "Taylor2": (Nt @ Nt.T)[iu], "Plug-in": (mn @ mn.T)[iu]}
        # threshold calibrated on the exact expected kernel: FAR = 5% on different-class pairs
        th_far = np.quantile(Kref[~same], 0.95)
        r = {"_th_far": float(th_far), "_auc_ref": float(roc_auc_score(same, Kref))}
        for k, K in Ks.items():
            d = {"auc": float(roc_auc_score(same, K))}
            # (1) fixed operational thresholds: disagreement with exact decision, accept-rate error
            for th in THS:
                a, a0 = K > th, Kref > th
                d[f"dis{th}"] = float((a != a0).mean()); d[f"acc{th}"] = float(a.mean()); d[f"acc0_{th}"] = float(a0.mean())
            # (2) threshold calibrated with exact kernel, applied to this estimator
            a = K > th_far
            d["far_at_exact_th"] = float(a[~same].mean()); d["tpr_at_exact_th"] = float(a[same].mean())
            d["dis_at_exact_th"] = float((a != (Kref > th_far)).mean())
            out_k = d; r[k] = out_k
        out[f"tau{tau}"] = r
    return out

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    for seed in seeds:
        t = time.time(); r = run(seed)
        with open(f"res_thr_{seed}.jsonl", "w") as f: f.write(json.dumps(dict(seed=seed, res=r)) + "\n")
        print(seed, round(time.time() - t, 1), flush=True)
