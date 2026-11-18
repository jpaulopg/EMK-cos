"""Expected cosine / normalized kernels for incomplete data (EMK-cos).

For x ~ N(m, S) (S may be singular):
    nu(m,S)  = E[x/||x||]      = (1/sqrt(pi)) * int_0^inf t^{-1/2} L(t) (I+2tS)^{-1} m dt
    M2(m,S)  = E[x x^T/||x||^2] = int_0^inf L(t) [ S(I+2tS)^{-1} + m_t m_t^T ] dt,  m_t=(I+2tS)^{-1}m
    L(t)     = |I+2tS|^{-1/2} exp(-t m^T (I+2tS)^{-1} m)
Integrals are evaluated by composite Gauss-Legendre in y = log t.
"""
import numpy as np

_GLX, _GLW = np.polynomial.legendre.leggauss(12)

# Quadrature in y = log(t * kappa), kappa = ||m||^2 + tr S.  The window starts at LO with an analytic
# correction for t < e^LO / kappa (there L(t) = 1 and m_t = m up to O(e^LO)).  The upper end is HI0, extended so
# that every direction with scale lambda_k + b_k^2 (eigenvalue plus squared mean coordinate) is integrated up to
# t * (lambda_k + b_k^2) = e^MARGIN; this matters only for directions much smaller than kappa.
LO, HI0, WIDTH, MARGIN, HI_CAP = -30.0, 40.0, 2.0, 40.0, 120.0
_GRIDS = {}


def _grid(lo=LO, hi=HI0, width=WIDTH):
    key = (lo, hi, width)
    if key not in _GRIDS:
        edges = np.arange(lo, hi + 1e-9, width)
        ys, ws = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            ys.append(0.5 * (b - a) * _GLX + 0.5 * (b + a))
            ws.append(0.5 * (b - a) * _GLW)
        _GRIDS[key] = (np.concatenate(ys), np.concatenate(ws))
    return _GRIDS[key]


def _window(lam, b, kappa, width=WIDTH):
    sc = lam + b * b
    pos = sc[sc > 1e-30 * kappa]
    hi = HI0 if len(pos) == 0 else max(HI0, np.log(kappa / pos.min()) + MARGIN)
    hi = min(hi, HI_CAP)
    npan = int(np.ceil((hi - LO) / width - 1e-9))
    return _grid(LO, LO + npan * width, width)


def _eig(S):
    lam, Q = np.linalg.eigh(0.5 * (S + S.T))
    return np.clip(lam, 0.0, None), Q


def _moments_eigbasis(lam, b, want_nu=True, want_G=True, width=WIDTH):
    """nu and G in the eigenbasis of S (lam: eigenvalues, b: mean coordinates)."""
    D = len(b)
    kappa = float(b @ b + lam.sum())
    if kappa <= 0:
        return np.zeros(D), np.eye(D) / D
    Y, W = _window(lam, b, kappa, width)
    T = np.exp(Y) / kappa
    den = 1.0 + 2.0 * T[:, None] * lam[None, :]
    logL = -0.5 * np.log(den).sum(1) - T * (b * b / den).sum(1)
    L = np.exp(logL)
    q = b[None, :] / den
    t0 = np.exp(LO) / kappa
    nu = G = None
    if want_nu:
        nu = ((W * np.exp(Y / 2.0) * L)[:, None] * q).sum(0) / np.sqrt(np.pi * kappa) + 2.0 * np.sqrt(t0 / np.pi) * b
    if want_G:
        w = W * T * L
        G = (q * w[:, None]).T @ q                      # BLAS: sum_k w_k q_k q_k^T
        G[np.diag_indices(D)] += (w[:, None] * lam[None, :] / den).sum(0)
        G += t0 * (np.diag(lam) + np.outer(b, b))       # left tail
    return nu, G


def moments_gauss(m, S, width=WIDTH):
    """(E[x/||x||], E[x x^T/||x||^2]) for x ~ N(m, S), S >= 0, sharing one eigendecomposition."""
    m = np.asarray(m, float)
    lam, Q = _eig(S)
    nu, G = _moments_eigbasis(lam, Q.T @ m, width=width)
    return Q @ nu, Q @ G @ Q.T


def nu_gauss(m, S, width=WIDTH):
    """E[x/||x||] for x ~ N(m, S)."""
    m = np.asarray(m, float)
    lam, Q = _eig(S)
    nu, _ = _moments_eigbasis(lam, Q.T @ m, want_G=False, width=width)
    return Q @ nu


def m2_gauss(m, S, width=WIDTH):
    """E[x x^T/||x||^2] for x ~ N(m, S)."""
    m = np.asarray(m, float)
    lam, Q = _eig(S)
    _, G = _moments_eigbasis(lam, Q.T @ m, want_nu=False, width=width)
    return Q @ G @ Q.T


def nu_diag(m, s, width=WIDTH):
    """E[x/||x||] for x ~ N(m, diag(s)): the eigenbasis is the coordinate basis."""
    nu, _ = _moments_eigbasis(np.clip(np.asarray(s, float), 0.0, None), np.asarray(m, float), want_G=False, width=width)
    return nu


def nu_mixture(pi, ms, Ss, H=None):
    """Mean direction of a Gaussian mixture; H optional linear map (e.g. centering)."""
    out = 0.0
    for p, m, S in zip(pi, ms, Ss):
        if H is not None:
            m, S = H @ m, H @ S @ H.T
        out = out + p * nu_gauss(m, S)
    return out


def moments_mixture(pi, ms, Ss, H=None):
    """(nu, G) of a Gaussian mixture, one eigendecomposition per component."""
    nu = G = 0.0
    for p, m, S in zip(pi, ms, Ss):
        if p <= 0:
            continue
        if H is not None:
            m, S = H @ m, H @ S @ H.T
        a, b = moments_gauss(m, S)
        nu = nu + p * a; G = G + p * b
    return nu, G


def m2_mixture(pi, ms, Ss, H=None):
    out = 0.0
    for p, m, S in zip(pi, ms, Ss):
        if H is not None:
            m, S = H @ m, H @ S @ H.T
        out = out + p * m2_gauss(m, S)
    return out


# ---------------- baselines -----------------
def nu_taylor2(m, S):
    """2nd-order Taylor approximation of E[x/||x||] (as in projected-normal literature)."""
    r2 = float(m @ m)
    f = m / np.sqrt(r2)
    return f + (1.0 / (2 * r2)) * (-2 * (S @ f) - np.trace(S) * f + 3 * f * (f @ S @ f))


# ---------------- GMM with missing data (EM) -----------------
def em_gmm_missing(X, C=3, iters=40, reg=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n, D = X.shape
    M = ~np.isnan(X)
    mu_col = np.nanmean(X, 0)
    Xf = np.where(M, X, mu_col)
    pi = np.full(C, 1.0 / C)
    idx = rng.choice(n, C, replace=False)
    mus = Xf[idx].copy()
    cov0 = np.cov(Xf.T) + reg * np.eye(D)
    covs = np.array([cov0.copy() for _ in range(C)])
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
            covs[c] = Sc[c] / Nk[c] - np.outer(mus[c], mus[c]) + reg * np.eye(D)
    return pi, mus, covs


def conditional_mixture(x, pi, mus, covs):
    """Return (resp, means, covs) of the missing-data posterior as a full-D mixture.
    Observed coords: mean = x_o, cov = 0."""
    D = len(x)
    o = np.where(~np.isnan(x))[0]
    u = np.where(np.isnan(x))[0]
    C = len(pi)
    logp = np.zeros(C)
    ms = np.zeros((C, D))
    Ss = np.zeros((C, D, D))
    for c in range(C):
        if len(o):
            Soo = covs[c][np.ix_(o, o)]
            Lc = np.linalg.cholesky(Soo)
            d = x[o] - mus[c][o]
            z = np.linalg.solve(Lc, d)
            logp[c] = np.log(pi[c]) - 0.5 * z @ z - np.log(np.diag(Lc)).sum()
            ms[c, o] = x[o]
            if len(u):
                Suo = covs[c][np.ix_(u, o)]
                K = np.linalg.solve(Soo, Suo.T).T
                ms[c, u] = mus[c][u] + K @ d
                Ss[c][np.ix_(u, u)] = covs[c][np.ix_(u, u)] - K @ Suo.T
        else:
            logp[c] = np.log(pi[c])
            ms[c] = mus[c]
            Ss[c] = covs[c]
    r = np.exp(logp - logp.max())
    r /= r.sum()
    return r, ms, Ss


def sample_mixture(r, ms, Ss, n, rng):
    D = ms.shape[1]
    comp = rng.choice(len(r), n, p=r)
    out = np.empty((n, D))
    for c in range(len(r)):
        k = (comp == c).sum()
        if k:
            lam, Q = np.linalg.eigh(Ss[c])
            A = Q * np.sqrt(np.clip(lam, 0, None))
            out[comp == c] = ms[c] + rng.standard_normal((k, D)) @ A.T
    return out
