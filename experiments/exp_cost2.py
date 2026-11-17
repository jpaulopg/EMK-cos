"""Cost of the mean AND variance of the cosine, per input and per pair.

Inputs: full covariance with eigenvalues log-spaced from 1 to 1e-2 in a random basis, ||m||^2 = tr S (as in the
cost table of nu). Per input we time the computation of (nu, G) for EMK-cos (528 nodes), PN-Taylor (first moment
and second-moment approximation), Monte Carlo per input with S draws, and the moment ratio (which needs only m, S).
Per pair we time the assembly of the variance tr(G_i G_j) - (nu_i^T nu_j)^2 (or the moment-ratio variance).
Error: RMSE over pairs of the predicted standard deviation of the cosine against a reference computed with a finer
and wider quadrature (width 1, y in [-60, 60]), and of the predicted mean.
"""
import json, time
import numpy as np
import emk_cos as E
import baselines as B
from exp_unc import pn_second, sqrtm

YR, WR = E._grid(-60.0, 60.0, 1.0)


def moments_ref(m, S):
    lam, Q = E._eig(S); mt = Q.T @ m; sc = float(m @ m + lam.sum()); T = np.exp(YR) / sc
    den = 1.0 + 2.0 * T[:, None] * lam[None, :]
    logL = -0.5 * np.log(den).sum(1) - T * (mt**2 / den).sum(1); L = np.exp(logL)
    nu = Q @ (((WR * np.exp(YR / 2) * L)[:, None] * (mt[None] / den)).sum(0) / np.sqrt(np.pi * sc))
    w = WR * T * L; a = mt[None] / den
    M = np.einsum("k,ki,kj->ij", w, a, a); M[np.diag_indices(len(m))] += (w[:, None] * lam[None] / den).sum(0)
    return nu, Q @ M @ Q.T


def mc(m, A, n, rng):
    D = len(m); s1 = np.zeros(D); s2 = np.zeros((D, D)); ch = 20000
    for s0 in range(0, n, ch):
        k = min(ch, n - s0); z = m + rng.normal(size=(k, D)) @ A.T; u = z / np.linalg.norm(z, axis=1, keepdims=True)
        s1 += u.sum(0); s2 += u.T @ u
    return s1 / n, s2 / n


def timed(f, reps):
    t0 = time.perf_counter()
    for _ in range(reps): out = f()
    return out, 1e3 * (time.perf_counter() - t0) / reps


def pair_var(Nx, Gx):
    n = len(Nx); mu = Nx @ Nx.T; Gf = Gx.reshape(n, -1); return mu, Gf @ Gf.T - mu**2


if __name__ == "__main__":
    rng = np.random.default_rng(0); res = []
    for D, n in ((10, 40), (100, 30), (500, 12)):
        Ms, Ss = [], []
        for _ in range(n):
            Q = np.linalg.qr(rng.normal(size=(D, D)))[0]; lam = np.logspace(0, -2, D); S = (Q * lam) @ Q.T
            v = rng.normal(size=D); Ms.append(v / np.linalg.norm(v) * np.sqrt(np.trace(S))); Ss.append(S)
        M = np.array(Ms); S = np.array(Ss); A = [sqrtm(s) for s in S]
        ref = [moments_ref(M[i], S[i]) for i in range(n)]
        mu_r, var_r = pair_var(np.array([a for a, _ in ref]), np.array([b for _, b in ref]))
        iu = np.triu_indices(n, 1); sd_r = np.sqrt(np.clip(var_r[iu], 0, None))
        reps = 5 if D < 500 else 1
        meth = {}
        def run(name, per_input):
            outs, ts = [], []
            for i in range(n):
                o, t = timed(lambda: per_input(i), reps); outs.append(o); ts.append(t)
            Nx = np.array([a for a, _ in outs]); Gx = np.array([b for _, b in outs])
            (mu, var), tp = timed(lambda: pair_var(Nx, Gx), 3)
            meth[name] = dict(ms_input=float(np.mean(ts)), us_pair=1e3 * tp / len(iu[0]),
                              rmse_mean=float(np.sqrt(np.mean((mu[iu] - mu_r[iu]) ** 2))),
                              rmse_sd=float(np.sqrt(np.mean((np.sqrt(np.clip(var[iu], 0, None)) - sd_r) ** 2))))
            print(D, name, meth[name], flush=True)
        run("EMK-cos", lambda i: (E.nu_gauss(M[i], S[i]), E.m2_gauss(M[i], S[i])))
        run("PN-Taylor", lambda i: (B.nu_pn2025(M[i], S[i]), pn_second(M[i], S[i])))
        for ns in (10**3, 10**4, 10**5):
            run(f"MC-{ns}", lambda i: mc(M[i], A[i], ns, rng))
        Tr = (M**2).sum(1) + np.trace(S, axis1=1, axis2=2)
        (Vr,), tp = timed(lambda: (B.ratio_var(M, S, Tr),), 3); mr = B.ratio_mean(M, Tr)
        meth["RatioMoments"] = dict(ms_input=0.0, us_pair=1e3 * tp / len(iu[0]),
                                    rmse_mean=float(np.sqrt(np.mean((mr[iu] - mu_r[iu]) ** 2))),
                                    rmse_sd=float(np.sqrt(np.mean((np.sqrt(Vr[iu]) - sd_r) ** 2))))
        print(D, "RatioMoments", meth["RatioMoments"], flush=True)
        res.append(dict(D=D, n=n, mean_sd_ref=float(sd_r.mean()), methods=meth))
    json.dump(res, open("res_cost2.json", "w"), indent=1)
