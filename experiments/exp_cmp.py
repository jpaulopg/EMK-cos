import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B
from em_struct import em_struct
from exp_real2 import masked, cosm, knn_acc, ridge_from
from exp_knn import pair_d2, knn_impute

def knn_mixtures(Xm, k, ridge):
    n, D = Xm.shape; d2, obs = pair_d2(Xm); Xh = knn_impute(Xm, d2, obs, k)
    gm = Xh.mean(0); gc = np.cov(Xh.T) + ridge * np.eye(D); out = []
    for i in range(n):
        u = np.where(~obs[i])[0]; o = np.where(obs[i])[0]
        if len(u) == 0:
            out.append((np.ones(1), Xm[i][None].copy(), np.zeros((1, D, D)))); continue
        if len(o) == 0 or not np.isfinite(d2[i]).any():
            out.append((np.ones(1), gm[None].copy(), gc[None].copy())); continue
        nb = np.argsort(d2[i])[:k]; nb = nb[np.isfinite(d2[i, nb])]
        B = Xh[np.ix_(nb, u)]
        Sl = (np.cov(B.T) if len(nb) > 1 and len(u) > 1 else np.atleast_2d(B.var(0) if len(nb) > 1 else 0.0)).reshape(len(u), len(u)) + ridge * np.eye(len(u))
        h2 = len(nb) ** (-2.0 / (len(u) + 4)); Sc = np.zeros((D, D)); Sc[np.ix_(u, u)] = h2 * Sl
        ms = np.zeros((len(nb), D)); ms[:, o] = Xm[i, o]; ms[:, u] = B
        out.append((np.full(len(nb), 1.0 / len(nb)), ms, np.repeat(Sc[None], len(nb), 0)))
    return out

def mixture_stats(cond, D):
    """Per record: EMK-cos (nu, G), PN-Taylor (nu, G), moment-ratio features of the whole law and per component."""
    out = dict(nu=[], G=[], nu_pn=[], G_pn=[], phi=[], Gt=[], phi_c=[], Gt_c=[], mean=[], tr=[])
    for r, ms, Ss in cond:
        nu, G = E.moments_mixture(r, ms, Ss)
        nup = sum(r[c] * (B.nu_pn2025(ms[c], Ss[c]) if ms[c] @ ms[c] > 1e-12 else np.zeros(D)) for c in range(len(r)))
        Gp = sum(r[c] * B.pn_second(ms[c], Ss[c]) for c in range(len(r)) if r[c] > 0)
        mean = (r[:, None] * ms).sum(0)
        Mc = [Ss[c] + np.outer(ms[c], ms[c]) for c in range(len(r))]; e2c = np.array([np.trace(M) for M in Mc])
        e2 = float(r @ e2c); M = sum(r[c] * Mc[c] for c in range(len(r)))
        out["nu"].append(nu); out["G"].append(G); out["nu_pn"].append(nup); out["G_pn"].append(Gp)
        out["phi"].append(mean / np.sqrt(e2)); out["Gt"].append(M / e2)
        out["phi_c"].append(B.ratio_comp(r, ms, Ss)); out["Gt_c"].append(sum(r[c] * Mc[c] / e2c[c] for c in range(len(r)) if r[c] > 0 and e2c[c] > 0))
        out["mean"].append(mean); out["tr"].append(e2)
    return {k: np.array(v) for k, v in out.items()}


def variance_scores(F, mask, Kt, iu, qs=(0.5, 0.9), floor=1e-6):
    """External check of mean-variance forecasts of the complete-data cosine, on pairs with a missing entry:
    Dawid-Sebastiani score, ratio of mean squared error to mean predicted variance (1 = calibrated scale),
    coverage of the nominal 90% normal interval, and the Brier score of the predicted probability that the
    complete cosine exceeds its q-quantile theta_q (normal approximation); 'EMK_mean_only' predicts 1{mu > theta_q}."""
    from scipy.stats import norm
    n = len(Kt); inc = mask.any(1); P = (inc[iu[0]] | inc[iu[1]])
    y = Kt[iu][P]; th = {q: float(np.quantile(y, q)) for q in qs}
    out = dict(pairs=int(P.sum()), theta={str(q): th[q] for q in qs})
    for name, (Nx, Gx) in dict(EMK=(F["nu"], F["G"]), PN=(F["nu_pn"], F["G_pn"]),
                               Ratio=(F["phi"], F["Gt"]), RatioComp=(F["phi_c"], F["Gt_c"])).items():
        mu = (Nx @ Nx.T)[iu][P]; Gf = Gx.reshape(n, -1); v = ((Gf @ Gf.T)[iu][P] - mu ** 2)
        vf = np.maximum(v, floor); e2 = (y - mu) ** 2; sd = np.sqrt(vf)
        out[name] = dict(dss=float(np.mean(np.log(vf) + e2 / vf)), var_ratio=float(e2.mean() / vf.mean()),
                         cover90=float(np.mean(np.abs(y - mu) <= norm.ppf(0.95) * sd)), floor_frac=float(np.mean(v < floor)),
                         **{f"brier{q}": float(np.mean((1.0 - norm.cdf((th[q] - mu) / sd) - (y > th[q])) ** 2)) for q in qs})
    mu = (F["nu"] @ F["nu"].T)[iu][P]
    out["EMK_mean_only"] = {f"brier{q}": float(np.mean(((mu > th[q]).astype(float) - (y > th[q])) ** 2)) for q in qs}
    return out


def run(name, rate, rep, n, sel):
    X, Xm, mask, y = masked(name, n, 100 + rep, rate, 1000 * rep + int(rate * 100)); n, D = X.shape
    reg = ridge_from(Xm)
    if sel.startswith("knn"):
        cond = knn_mixtures(Xm, int(sel.split("_k")[1]), reg)
    else:
        _, st, c = sel.split("_"); C = int(c[1:]); s = "lowrank" if st.startswith("lowrank") else st; kk = 3 if s == "lowrank" else 0
        pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct=s, k=kk)
        cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    F = mixture_stats(cond, D)
    Nu, Np, Mean, Tr = F["nu"], F["nu_pn"], F["mean"], F["tr"]
    dvec = 1 - (Nu ** 2).sum(1); Xmean = np.where(mask, np.nanmean(Xm, 0), Xm)
    O = (~mask).astype(float); Xo = np.where(mask, 0.0, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        Kobs = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
    dfn = np.isfinite(Kobs); Kobs[~dfn] = 0.0
    K = {"EMK-cos": Nu @ Nu.T + np.diag(dvec), "PN-Taylor": Np @ Np.T, "Plug-in": cosm(Mean), "MeanImp": cosm(Xmean),
         "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr)), "RatioComp": F["phi_c"] @ F["phi_c"].T, "ObsSubset": Kobs}
    Kt = cosm(X); iu = np.triu_indices(n, 1); d = dfn[iu]; res = {"_sel": sel}
    for k_, M in K.items():
        Mx = M.copy(); np.fill_diagonal(Mx, 1.0)
        res[k_] = dict(rmse=float(np.sqrt(np.mean((Mx[iu] - Kt[iu]) ** 2))),
                       rmse_defined=float(np.sqrt(np.mean((Mx[iu][d] - Kt[iu][d]) ** 2))) if d.any() else float("nan"),
                       knn=knn_acc(Mx, y), mineig=float(np.linalg.eigvalsh(0.5 * (Mx + Mx.T)).min()))
    res["_defined"] = dict(frac_undefined=float(1 - d.mean()))
    res["_var"] = variance_scores(F, mask, Kt, iu)
    return res

if __name__ == "__main__":
    name, n = sys.argv[1], int(sys.argv[2])
    A = [json.loads(l) for l in open(f"res_auto_{name}.jsonl")]
    with open(f"res_cmp_{name}.jsonl", "w") as f:
        for a in A:
            sel = min(a["res"], key=lambda k: a["res"][k]["mse"]); t = time.time()
            r = run(name, a["rate"], a["rep"], n, sel)
            f.write(json.dumps(dict(data=name, rate=a["rate"], rep=a["rep"], res=r)) + "\n"); f.flush()
            print(name, a["rate"], a["rep"], sel, round(time.time() - t, 1), flush=True)
