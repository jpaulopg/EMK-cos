"""Fidelity to the fitted law on incomplete data: mean and variance of the cosine under the conditional laws of a
full-covariance GMM fitted to the observed entries (50% MCAR, 3 repetitions), against a Monte Carlo reference with
2e4 draws from each conditional mixture. Pairs of distinct records; for the variance, pairs with at least one
incomplete record. Estimators as in exp_cmp.mixture_stats."""
import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import masked, ridge_from
from exp_cmp import mixture_stats
from exp_aux import CS, NS


def run(name, rep, rate=0.5, nmc=20000):
    X, Xm, mask, _ = masked(name, NS[name], 100 + rep, rate, 1000 * rep + int(rate * 100)); n, D = X.shape
    pi, mus, covs = em_struct(Xm, C=CS[name], iters=15, reg=ridge_from(Xm), seed=rep, struct="full")
    cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    F = mixture_stats(cond, D); rng = np.random.default_rng(55 + rep); Nr = []; Gr = []
    for r, m, S in cond:
        Z = E.sample_mixture(r, m, S, nmc, rng); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        Nr.append(Z.mean(0)); Gr.append(Z.T @ Z / nmc)
    Nr, Gr = np.array(Nr), np.array(Gr); iu = np.triu_indices(n, 1)
    inc = mask.any(1); P = inc[iu[0]] | inc[iu[1]]
    def mv(Nx, Gx):
        mu = (Nx @ Nx.T)[iu]; Gf = Gx.reshape(n, -1); return mu, (Gf @ Gf.T)[iu] - mu ** 2
    mr, vr = mv(Nr, Gr); out = {"mean_sd_ref": float(np.sqrt(np.clip(vr[P], 0, None)).mean())}
    for k, (Nx, Gx) in dict(EMK=(F["nu"], F["G"]), PN=(F["nu_pn"], F["G_pn"]), Ratio=(F["phi"], F["Gt"]),
                            RatioComp=(F["phi_c"], F["Gt_c"]), Plug=(F["mean"] / np.linalg.norm(F["mean"], axis=1, keepdims=True), None)).items():
        if Gx is None:
            mu = (Nx @ Nx.T)[iu]; out[k] = dict(rmse_mean=float(np.sqrt(np.mean((mu - mr) ** 2)))); continue
        mu, v = mv(Nx, Gx); sd = np.sqrt(np.clip(v[P], 0, None)); sdr = np.sqrt(np.clip(vr[P], 0, None))
        out[k] = dict(rmse_mean=float(np.sqrt(np.mean((mu - mr) ** 2))), rmse_sd=float(np.sqrt(np.mean((sd - sdr) ** 2))),
                      scale=float(np.clip(v[P], 0, None).sum() / np.clip(vr[P], 0, None).sum()))
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        with open(f"res_fitted_{name}.jsonl", "w") as f:
            for rep in range(3):
                t = time.time(); r = run(name, rep); f.write(json.dumps(dict(data=name, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rep, round(time.time() - t, 1), {k: v for k, v in r.items() if k != "mean_sd_ref"}, flush=True)
