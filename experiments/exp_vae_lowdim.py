"""Variational autoencoders with a small latent space (Digits): posteriors of low effective dimension.

For latent dimensions 2 and 3, the VAE of exp_unc is trained (three seeds), and the diagonal posteriors of 300 test
images are evaluated with exp_unc.evaluate at posterior-variance scales tau in {1, 4, 16}. Besides the scores of exp_unc,
we report the excess risk of each predicted mean relative to the Bayes risk of the realized cosine,
    rmse(mu - mu_ref)^2 / mean Var_ref,
and summaries of the posteriors: SNR ||m|| / sqrt(tr S), effective dimension (tr S)^2 / tr S^2 and the coefficient of
variation of ||x||^2 (Proposition 'Concentration of the norm')."""
import sys, json, time
import numpy as np
from sklearn.datasets import load_digits
from vae_np import train_vae, encode
from exp_unc import evaluate


def cv_diag(m, v):
    return float(np.sqrt(2 * (v ** 2).sum() + 4 * (m * m * v).sum()) / (m @ m + v.sum()))


def run(zdim, seed, ntest=300, taus=(1.0, 4.0, 16.0)):
    X, _ = load_digits(return_X_y=True); X = X / 16.0
    P = train_vae(X, zdim=zdim, epochs=400, seed=seed); mu, lv = encode(P, X)
    kl = 0.5 * (mu ** 2 + np.exp(lv) - 1 - lv).mean(0); act = np.where(kl > 0.1)[0]
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), ntest, replace=False)
    m = mu[idx][:, act]; v0 = np.exp(lv[idx][:, act]); out = {"active": int(len(act))}
    for t in taus:
        v = v0 * t
        r = evaluate(m, np.array([np.diag(x) for x in v]), rng)
        for k, val in r.items():
            if isinstance(val, dict): val["rel_excess"] = val["rmse_mean_ref"] ** 2 / r["_mean_var_ref"]
        r["_snr"] = float(np.median(np.linalg.norm(m, axis=1) / np.sqrt(v.sum(1))))
        r["_deff"] = float(np.median(v.sum(1) ** 2 / (v ** 2).sum(1)))
        r["_cv"] = float(np.median([cv_diag(a, b) for a, b in zip(m, v)]))
        out[f"tau{t:g}"] = r
    return out


if __name__ == "__main__":
    with open("res_vae_lowdim.jsonl", "w") as f:
        for zdim in (2, 3):
            for seed in range(3):
                t = time.time(); r = run(zdim, seed)
                f.write(json.dumps(dict(zdim=zdim, seed=seed, res=r)) + "\n"); f.flush()
                print(zdim, seed, r["active"], round(time.time() - t, 1),
                      {k: (round(v["_snr"], 2), round(v["_cv"], 2), {m: round(100 * v[m]["rel_excess"], 2) for m in ("EMK-cos", "PN-Taylor", "RatioMoments")},
                           {m: round(v[m]["scale_ratio"], 2) for m in ("EMK-cos", "PN-Taylor", "RatioMoments")}) for k, v in r.items() if k.startswith("tau")}, flush=True)
