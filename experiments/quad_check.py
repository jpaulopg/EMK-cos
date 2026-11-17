"""Accuracy of the composite quadrature against an adaptive reference, and cost against Monte Carlo per input.
Reference: scipy quad_vec on the same integrand in y = log(t s), tolerance 1e-14, plus the analytic left tail."""
import json, time
import numpy as np
from scipy.integrate import quad_vec

GLX, GLW = np.polynomial.legendre.leggauss(12)

def grid(width, lo=-48.0, hi=40.0):
    e = np.arange(lo, hi + 1e-9, width); a, b = e[:-1], e[1:]
    y = (0.5 * (b - a)[:, None] * GLX + 0.5 * (b + a)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * GLW).ravel(); return y, w

def integrand(y, lam, bb, s):
    t = np.exp(y) / s; den = 1.0 + 2.0 * t * lam
    logL = -0.5 * np.log(den).sum(-1) - t * (bb**2 / den).sum(-1)
    return (np.exp(y / 2.0 + logL))[..., None] * (bb / den)

def nu_grid(m, S, width):
    lam, U = np.linalg.eigh(0.5 * (S + S.T)); lam = np.clip(lam, 0, None); bb = U.T @ m; s = m @ m + lam.sum()
    y, w = grid(width); t = np.exp(y)[:, None] / s; den = 1.0 + 2.0 * t * lam[None]
    logL = -0.5 * np.log(den).sum(1) - t[:, 0] * (bb**2 / den).sum(1)
    I = ((w * np.exp(y / 2.0 + logL))[:, None] * (bb[None] / den)).sum(0)
    return U @ I / np.sqrt(np.pi * s)

def nu_ref(m, S):
    lam, U = np.linalg.eigh(0.5 * (S + S.T)); lam = np.clip(lam, 0, None); bb = U.T @ m; s = m @ m + lam.sum()
    I, _ = quad_vec(lambda y: integrand(np.array([y]), lam, bb, s)[0], -70.0, 90.0, epsabs=1e-15, epsrel=1e-13, limit=20000)
    I = I + 2.0 * np.exp(-35.0) * bb   # left tail, L ~ 1 and (I+2tS)^{-1} m ~ m for t < e^{-70}/s
    return U @ I / np.sqrt(np.pi * s)

def cases(rng):
    def rnd_m(D, snr, S): v = rng.normal(size=D); return v / np.linalg.norm(v) * snr * np.sqrt(np.trace(S))
    out = []
    S = np.eye(10); out.append(("isotropic, D=10", rnd_m(10, 1.0, S), S))
    Q = np.linalg.qr(rng.normal(size=(100, 100)))[0]
    lam = 10.0 ** (-6 * np.arange(100) / 99); S = (Q * lam) @ Q.T; out.append(("exponential spectrum 1..1e-6, D=100", rnd_m(100, 1.0, S), S))
    lam = 1.0 / np.arange(1, 101) ** 2; S = (Q * lam) @ Q.T; out.append(("polynomial spectrum k^-2, D=100", rnd_m(100, 0.5, S), S))
    B = rng.normal(size=(50, 2)); S = B @ B.T; m = rng.normal(size=50); out.append(("rank 2, m outside range, D=50", m, S))
    S = np.eye(10); out.append(("near-zero mean direction, D=10", 1e-3 * rng.normal(size=10) / np.sqrt(10), S))
    A = rng.normal(size=(500, 500)) / np.sqrt(500); lam = np.logspace(0, -4, 500); Q5 = np.linalg.qr(A)[0]; S = (Q5 * lam) @ Q5.T
    out.append(("full, condition 1e4, D=500", rnd_m(500, 0.5, S), S))
    S = np.eye(10) * 1e12; out.append(("isotropic, scale 1e6", rnd_m(10, 1.0, S), S))
    S = np.eye(10) * 1e-12; out.append(("isotropic, scale 1e-6", rnd_m(10, 1.0, S), S))
    return out

def mc_vec(m, S, n, rng, A=None):
    if A is None:
        lam, U = np.linalg.eigh(S); A = U * np.sqrt(np.clip(lam, 0, None))
    acc = np.zeros_like(m); ch = 10000
    for s0 in range(0, n, ch):
        k = min(ch, n - s0); z = m + rng.normal(size=(k, len(m))) @ A.T; acc += (z / np.linalg.norm(z, axis=1, keepdims=True)).sum(0)
    return acc / n

if __name__ == "__main__":
    rng = np.random.default_rng(0); res = {"accuracy": [], "cost": []}
    for name, m, S in cases(rng):
        ref = nu_ref(m, S); row = {"case": name, "norm_nu": float(np.linalg.norm(ref))}
        for width, R in ((8, 132), (4, 264), (2, 528), (1, 1056)):
            e = nu_grid(m, S, width) - ref; row[f"abs{R}"] = float(np.abs(e).max()); row[f"rel{R}"] = float(np.linalg.norm(e) / np.linalg.norm(ref))
        res["accuracy"].append(row); print(row, flush=True)
    for D in (10, 100, 500):
        Q = np.linalg.qr(rng.normal(size=(D, D)))[0]; lam = np.logspace(0, -2, D); S = (Q * lam) @ Q.T
        v = rng.normal(size=D); m = v / np.linalg.norm(v) * np.sqrt(np.trace(S)); ref = nu_ref(m, S)
        reps = 20 if D < 500 else 3
        t0 = time.perf_counter()
        for _ in range(reps): nu = nu_grid(m, S, 2)
        row = {"D": D, "emk_time_ms": 1e3 * (time.perf_counter() - t0) / reps, "emk_err": float(np.abs(nu - ref).max())}
        for n in (10**3, 10**4, 10**5):
            r2 = 3 if n < 10**5 else 1; errs = []; t0 = time.perf_counter()
            for _ in range(r2): errs.append(np.abs(mc_vec(m, S, n, rng) - ref).max())
            row[f"mc{n}_time_ms"] = 1e3 * (time.perf_counter() - t0) / r2; row[f"mc{n}_err"] = float(np.mean(errs))
        res["cost"].append(row); print(row, flush=True)
    json.dump(res, open("res_quad_check.json", "w"), indent=1)
