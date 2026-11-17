import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import load, cosm, knn_acc, recall, spec_ari
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

def run(name, rate, rep, n, sel):
    X, y = load(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    Xm = np.where(mask, np.nan, X); reg = 1e-3 * max(X.var(0).mean(), 1e-6) * 10
    if sel.startswith("knn"):
        cond = knn_mixtures(Xm, int(sel.split("_k")[1]), reg)
    else:
        _, st, c = sel.split("_"); C = int(c[1:]); s = "lowrank" if st.startswith("lowrank") else st; kk = 3 if s == "lowrank" else 0
        pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct=s, k=kk)
        cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Nt = np.array([sum(r[c] * (E.nu_taylor2(m[c], S[c]) if m[c] @ m[c] > 1e-12 else np.zeros(D)) for c in range(len(r))) for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(len(r))) for r, m, S in cond])
    dvec = 1 - (Nu ** 2).sum(1); Xmean = np.where(mask, np.nanmean(Xm, 0), Xm)
    O = (~mask).astype(float); Xo = np.where(mask, 0.0, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        Kobs = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
    Kobs[~np.isfinite(Kobs)] = 0.0
    nr = np.linalg.norm(Nu, axis=1, keepdims=True); N1 = Nu / np.maximum(nr, 1e-12)
    K = {"EMK-cos": Nu @ Nu.T + np.diag(dvec), "EMK-cos g1": N1 @ N1.T, "Taylor2": Nt @ Nt.T, "Plug-in": cosm(Mean), "MeanImp": cosm(Xmean),
         "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr)), "ObsSubset": Kobs}
    Kt = cosm(X); iu = np.triu_indices(n, 1); res = {"_sel": sel}
    for k_, M in K.items():
        Mx = M.copy(); np.fill_diagonal(Mx, 1.0)
        res[k_] = dict(rmse=float(np.sqrt(np.mean((Mx[iu] - Kt[iu]) ** 2))), knn=knn_acc(Mx, y), mineig=float(np.linalg.eigvalsh(0.5 * (Mx + Mx.T)).min()))
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
