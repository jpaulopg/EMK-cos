import sys, json, time
import numpy as np
import emk_cos as E
from exp_real2 import load, cosm, knn_acc

def pair_d2(Xm):
    M = (~np.isnan(Xm)).astype(float); X0 = np.where(M > 0, Xm, 0.0)
    num = (M * X0**2) @ M.T + M @ (M * X0**2).T - 2 * (M * X0) @ (M * X0).T
    cnt = M @ M.T
    d2 = np.where(cnt > 0, np.maximum(num, 0) / np.maximum(cnt, 1), np.inf)
    np.fill_diagonal(d2, np.inf)
    return d2, M > 0

def knn_impute(Xm, d2, obs, k):
    n, D = Xm.shape; Xh = Xm.copy(); colm = np.nanmean(Xm, 0)
    for i in range(n):
        for c in np.where(~obs[i])[0]:
            cand = np.where(obs[:, c] & np.isfinite(d2[i]))[0]
            if len(cand) == 0: Xh[i, c] = colm[c]; continue
            nb = cand[np.argsort(d2[i, cand])[:k]]; Xh[i, c] = Xm[nb, c].mean()
    return Xh

def knn_posterior(Xm, k, mode, ridge):
    n, D = Xm.shape; d2, obs = pair_d2(Xm); Xh = knn_impute(Xm, d2, obs, k)
    gm = Xh.mean(0); gc = np.cov(Xh.T) + ridge * np.eye(D)
    nus, imp = [], []
    for i in range(n):
        u = np.where(~obs[i])[0]; o = np.where(obs[i])[0]
        if len(u) == 0:
            m = Xm[i].copy(); nus.append(E.nu_gauss(m, np.zeros((D, D)))); imp.append(m); continue
        if len(o) == 0 or not np.isfinite(d2[i]).any():
            m = gm.copy(); S = gc; nus.append(E.nu_gauss(m, S)); imp.append(m); continue
        nb = np.argsort(d2[i])[:k]; nb = nb[np.isfinite(d2[i, nb])]
        B = Xh[np.ix_(nb, u)]; mu = B.mean(0); imp.append(np.where(obs[i], Xm[i], 0.0) + np.bincount(u, weights=mu, minlength=D))
        Sl = (np.cov(B.T) if len(nb) > 1 and len(u) > 1 else np.atleast_2d(B.var(0) if len(nb) > 1 else 0.0)).reshape(len(u), len(u))
        Sl = Sl + ridge * np.eye(len(u))
        if mode == "gauss":
            m = np.zeros(D); m[o] = Xm[i, o]; m[u] = mu; S = np.zeros((D, D)); S[np.ix_(u, u)] = Sl
            nus.append(E.nu_gauss(m, S))
        else:
            h2 = len(nb) ** (-2.0 / (len(u) + 4)); Sc = np.zeros((D, D)); Sc[np.ix_(u, u)] = h2 * Sl
            acc = 0.0
            for j in range(len(nb)):
                m = np.zeros(D); m[o] = Xm[i, o]; m[u] = B[j]; acc = acc + E.nu_gauss(m, Sc)
            nus.append(acc / len(nb))
    return np.array(nus), np.array(imp)
