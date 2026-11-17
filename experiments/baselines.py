"""Published approximations used as baselines.

* nu_pn2025: first moment of the projected normal by the second-order Taylor expansion of
  Herrera-Esposito & Burge (2025), around (mu_i, E z_i) with z_i = x^T x - x_i^2 and exact
  Gaussian moments of the quadratic form z_i.
* ratio_mean / ratio_var: mean and variance of the cosine of two independent Gaussian vectors in
  the form of Baumann et al. (2024), E[g^T h]/sqrt(E||g||^2 E||h||^2) and Var[g^T h]/(E||g||^2 E||h||^2),
  written for full covariance (it reduces to their diagonal formulas).
* nearest_correlation: matrix correction of a pairwise similarity estimate to the nearest unit-diagonal
  PSD matrix in Frobenius norm (alternating projections with Dykstra's correction, Higham 2002), the
  type of correction used by Yu et al. (2023).
"""
import numpy as np


def nu_pn2025(m, S):
    m = np.asarray(m, float); S = np.asarray(S, float); D = len(m)
    d = np.diag(S); tr = d.sum(); mm = m @ m
    zbar = (tr - d) + (mm - m**2)                                   # E z_i
    SS = S @ S; Sm = S @ m
    # var(z_i) = 2 tr(S_{-i}^2) + 4 m_{-i}^T S_{-i} m_{-i}
    tr_SS = np.trace(SS); row2 = (S**2).sum(1)
    tr_Sm_i = tr_SS - 2 * row2 + d**2
    mSm = m @ Sm; q_i = mSm - 2 * m * Sm + m**2 * d
    varz = 2 * tr_Sm_i + 4 * q_i
    covxz = 2 * (Sm - m * d)                                        # 2 (mu^T col_i(S) - mu_i S_ii)
    u = m**2 + zbar
    g = m / np.sqrt(u)
    g += 0.5 * d * (-3 * m * zbar) / u**2.5
    g += 0.5 * varz * (3 * m) / (4 * u**2.5)
    g += covxz * (m**2 - 0.5 * zbar) / u**2.5
    return g


def ratio_mean(M, Tr):
    """M: n x D means, Tr: n vector of E||x||^2 = ||m||^2 + tr S."""
    return (M @ M.T) / np.sqrt(np.outer(Tr, Tr))


def ratio_var(M, S, Tr):
    """Var[g^T h] = tr(S_g S_h) + m_h^T S_g m_h + m_g^T S_h m_g, divided by E||g||^2 E||h||^2."""
    n = len(M)
    Sf = S.reshape(n, -1)
    trSS = Sf @ Sf.T                                                # tr(S_i S_j) (symmetric S)
    SM = np.einsum("iab,jb->ija", S, M)                             # S_i m_j
    q = np.einsum("ija,ja->ij", SM, M)                              # m_j^T S_i m_j
    V = trSS + q + q.T
    return V / np.outer(Tr, Tr)


def nearest_correlation(A, iters=500, tol=1e-9):
    Y = 0.5 * (A + A.T); dS = np.zeros_like(Y)
    for _ in range(iters):
        R = Y - dS
        lam, Q = np.linalg.eigh(R); X = (Q * np.clip(lam, 0, None)) @ Q.T
        dS = X - R
        Yn = X.copy(); np.fill_diagonal(Yn, 1.0)
        if np.linalg.norm(Yn - Y) <= tol * max(1.0, np.linalg.norm(Y)):
            Y = Yn; break
        Y = Yn
    return Y


def ratio_comp(r, ms, Ss):
    """Per-component ratio of means: sum_c r_c m_c / sqrt(||m_c||^2 + tr S_c).
    Normalizes each Gaussian component before mixing; feature norm <= 1, so the Gram construction is PSD."""
    out = 0.0
    for rc, m, S in zip(r, ms, Ss):
        if rc <= 0:
            continue
        tr = np.trace(S) if np.ndim(S) == 2 else np.sum(S)
        e2 = float(m @ m + tr)
        if e2 > 0:
            out = out + rc * m / np.sqrt(e2)
    return out
