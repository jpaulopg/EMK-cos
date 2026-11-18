"""Accuracy of the composite quadrature (emk_cos) against an adaptive reference, and cost of the expected direction
against Monte Carlo per input.
Reference: scipy quad_vec on the integrand in y = log(t kappa) over a wide window, plus the analytic left tail.
The quadrature under test uses panels of width 4, 2 and 1 (12 Gauss-Legendre nodes each) on the adaptive window."""
import json, time
import numpy as np
from scipy.integrate import quad_vec
import emk_cos as E


def _basis(m, S):
    lam, U = np.linalg.eigh(0.5 * (S + S.T)); lam = np.clip(lam, 0, None); b = U.T @ m
    return lam, U, b, float(b @ b + lam.sum())


def _hi(lam, b, kappa):
    sc = lam + b * b; pos = sc[sc > 1e-30 * kappa]
    return max(40.0, np.log(kappa / pos.min()) + 40.0) + 50.0


def nu_ref(m, S):
    lam, U, b, k = _basis(m, S)
    def f(y):
        t = np.exp(y) / k; den = 1.0 + 2.0 * t * lam
        return np.exp(y / 2.0 - 0.5 * np.log(den).sum() - t * (b * b / den).sum()) * b / den
    I, _ = quad_vec(f, -70.0, _hi(lam, b, k), epsabs=1e-17, epsrel=1e-13, limit=20000)
    return U @ (I / np.sqrt(np.pi * k) + 2.0 * np.sqrt(np.exp(-70.0) / k / np.pi) * b)


def G_ref(m, S):
    lam, U, b, k = _basis(m, S)
    def f(y):
        t = np.exp(y) / k; den = 1.0 + 2.0 * t * lam; L = np.exp(-0.5 * np.log(den).sum() - t * (b * b / den).sum())
        q = b / den; return (t * L * (np.outer(q, q) + np.diag(lam / den))).ravel()
    I, _ = quad_vec(f, -70.0, _hi(lam, b, k), epsabs=1e-19, epsrel=1e-13, limit=20000)
    G = I.reshape(len(b), len(b)) + np.exp(-70.0) / k * (np.diag(lam) + np.outer(b, b))
    return U @ G @ U.T


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
    out.append(("tiny deterministic component, x=(Z, 1e-14)", np.array([0.0, 1e-14]), np.diag([1.0, 0.0])))
    v = rng.normal(size=6); S = np.outer(v, v); out.append(("rank 1, m in range, D=6", 0.7 * v, S))
    return out


def mc_vec(m, S, n, rng):
    """Monte Carlo per input, including the Cholesky factorization of S (S positive definite here)."""
    A = np.linalg.cholesky(S); acc = np.zeros_like(m); ch = 10000
    for s0 in range(0, n, ch):
        k = min(ch, n - s0); z = m + rng.normal(size=(k, len(m))) @ A.T; acc += (z / np.linalg.norm(z, axis=1, keepdims=True)).sum(0)
    return acc / n


if __name__ == "__main__":
    rng = np.random.default_rng(0); res = {"accuracy": [], "cost": []}
    for name, m, S in cases(rng):
        rn = nu_ref(m, S); row = {"case": name, "norm_nu": float(np.linalg.norm(rn))}
        rG = G_ref(m, S) if len(m) <= 100 else None
        for width in (4.0, 2.0, 1.0):
            lam, U, b, k = _basis(m, S); R = len(E._window(lam, b, k, width)[0])
            nu, G = E.moments_gauss(m, S, width=width)
            row[f"nodes_w{width:g}"] = R
            row[f"nu_rel_w{width:g}"] = float(np.linalg.norm(nu - rn) / np.linalg.norm(rn))
            if rG is not None:
                row[f"G_rel_w{width:g}"] = float(np.linalg.norm(G - rG) / np.linalg.norm(rG))
                row[f"G22_rel_w{width:g}"] = float(abs(G[-1, -1] - rG[-1, -1]) / abs(rG[-1, -1]))
        res["accuracy"].append(row); print(row, flush=True)
    for D in (10, 100, 500):
        Q = np.linalg.qr(rng.normal(size=(D, D)))[0]; lam = np.logspace(0, -2, D); S = (Q * lam) @ Q.T
        v = rng.normal(size=D); m = v / np.linalg.norm(v) * np.sqrt(np.trace(S)); ref = E.nu_gauss(m, S, width=0.5)
        reps = 20 if D < 500 else 3
        t0 = time.perf_counter()
        for _ in range(reps): nu = E.nu_gauss(m, S)
        row = {"D": D, "emk_time_ms": 1e3 * (time.perf_counter() - t0) / reps, "emk_err": float(np.abs(nu - ref).max())}
        for n in (10**3, 10**4, 10**5):
            r2 = 3 if n < 10**5 else 1; errs = []; t0 = time.perf_counter()
            for _ in range(r2): errs.append(np.abs(mc_vec(m, S, n, rng) - ref).max())
            row[f"mc{n}_time_ms"] = 1e3 * (time.perf_counter() - t0) / r2; row[f"mc{n}_err"] = float(np.mean(errs))
        res["cost"].append(row); print(row, flush=True)
    json.dump(res, open("res_quad_check.json", "w"), indent=1)
