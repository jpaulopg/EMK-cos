import numpy as np
def proj(S, struct, k, reg):
    D=len(S)
    if struct=='full': return S+reg*np.eye(D)
    if struct=='diag': return np.diag(np.diag(S))+reg*np.eye(D)
    lam,Q=np.linalg.eigh(0.5*(S+S.T)); lam=np.clip(lam[::-1],0,None); Q=Q[:,::-1]
    k=min(k,D-1); s2=lam[k:].mean() if D>k else 0.0
    L=np.concatenate([lam[:k],np.full(D-k,s2)])
    return (Q*L)@Q.T+reg*np.eye(D)
def em_struct(X, C=3, iters=40, reg=1e-3, seed=0, struct='full', k=3):
    rng = np.random.default_rng(seed)
    n, D = X.shape
    M = ~np.isnan(X)
    mu_col = np.nanmean(X, 0)
    Xf = np.where(M, X, mu_col)
    pi = np.full(C, 1.0 / C)
    idx = rng.choice(n, C, replace=False)
    mus = Xf[idx].copy()
    cov0 = np.cov(Xf.T) + reg * np.eye(D)
    covs = np.array([proj(cov0.copy(), struct, k, reg) for _ in range(C)])
    for it in range(iters):
        Nk = np.zeros(C)
        Sm = np.zeros((C, D))
        Sc = np.zeros((C, D, D))
        for i in range(n):
            o = np.where(M[i])[0]
            u = np.where(~M[i])[0]
            xo = X[i, o]
            logp = np.zeros(C)
            xhat = np.zeros((C, D))
            Ci = np.zeros((C, D, D))
            for c in range(C):
                if len(o):
                    Soo = covs[c][np.ix_(o, o)]
                    Lc = np.linalg.cholesky(Soo)
                    d = xo - mus[c][o]
                    z = np.linalg.solve(Lc, d)
                    logp[c] = np.log(pi[c]) - 0.5 * z @ z - np.log(np.diag(Lc)).sum() - 0.5 * len(o) * np.log(2 * np.pi)
                    xhat[c, o] = xo
                    if len(u):
                        Suo = covs[c][np.ix_(u, o)]
                        K = np.linalg.solve(Soo, Suo.T).T
                        xhat[c, u] = mus[c][u] + K @ d
                        Ci[c][np.ix_(u, u)] = covs[c][np.ix_(u, u)] - K @ Suo.T
                else:
                    logp[c] = np.log(pi[c])
                    xhat[c] = mus[c]
                    Ci[c] = covs[c]
            r = np.exp(logp - logp.max())
            r /= r.sum()
            for c in range(C):
                Nk[c] += r[c]
                Sm[c] += r[c] * xhat[c]
                Sc[c] += r[c] * (np.outer(xhat[c], xhat[c]) + Ci[c])
        pi = Nk / n
        mus = Sm / Nk[:, None]
        for c in range(C):
            covs[c] = proj(Sc[c] / Nk[c] - np.outer(mus[c], mus[c]), struct, k, reg)
    return pi, mus, covs
