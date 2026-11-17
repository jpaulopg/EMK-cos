"""Regime map: when does the exact expectation matter for predicting the realized cosine?

For pairs of independent random inputs we compute the exact conditional mean mu = nu_i^T nu_j (quadrature)
and, by Monte Carlo over realized pairs, the conditional variance Var(Y). For an approximation f, the
excess squared-error risk is E[(mu - f)^2] (eq. risk), reported relative to the Bayes risk E[Var(Y)].
Laws: (a) diagonal Gaussians, variances log-uniform over a factor 100 (as in probabilistic embeddings),
at signal-to-noise ratios ||m||/sqrt(tr S) in {0.5, 1, 2}; (b) two-component mixtures with equal weights,
tight isotropic components (sd 0.1 x norm / sqrt(D)) with norms 1 and rho in random directions."""
import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B

def nu_diag(m, s):
    """Expected direction for N(m, diag(s)): eigenbasis = coordinate basis."""
    sc = float(m @ m + s.sum()); T = E.EY / sc
    den = 1.0 + 2.0 * T[:, None] * s[None, :]
    logL = -0.5 * np.log(den).sum(1) - T * (m**2 / den).sum(1)
    return ((E.WY * E.EY2 * np.exp(logL))[:, None] * (m[None, :] / den)).sum(0) / np.sqrt(np.pi * sc)

def gauss_input(D, snr, rng):
    s = 10 ** rng.uniform(-1, 1, D); v = rng.normal(size=D)
    return [(1.0, v / np.linalg.norm(v) * snr * np.sqrt(s.sum()), s)]

def mixture_input(D, par, rng):
    rho, spread = par if isinstance(par, tuple) else (par, 0.1)
    comps = []
    for norm in (1.0, rho):
        v = rng.normal(size=D); m = v / np.linalg.norm(v) * norm
        comps.append((0.5, m, np.full(D, (spread * norm) ** 2 / D)))
    return comps

def embed_input(D, snr, rng, eps=1e-4):
    """A two-dimensional Gaussian law (variances log-uniform, given SNR) embedded in D dimensions:
    the other D-2 coordinates have zero mean and variance eps times the mean 2-D variance."""
    s2 = 10 ** rng.uniform(-1, 1, 2); v = rng.normal(size=2)
    m = np.zeros(D); m[:2] = v / np.linalg.norm(v) * snr * np.sqrt(s2.sum())
    s = np.full(D, eps * s2.mean()); s[:2] = s2
    return [(1.0, m, s)]

def stats(comps):
    nu = sum(w * nu_diag(m, s) for w, m, s in comps)
    mean = sum(w * m for w, m, s in comps); e2 = sum(w * (m @ m + s.sum()) for w, m, s in comps)
    pn = sum(w * B.nu_pn2025(m, np.diag(s)) for w, m, s in comps) if len(comps[0][1]) <= 300 else None
    rc = B.ratio_comp([w for w, _, _ in comps], [m for _, m, _ in comps], [s for _, _, s in comps])
    return nu, mean, e2, pn, rc

def draws(comps, n, rng):
    w = np.array([c[0] for c in comps]); k = rng.choice(len(comps), n, p=w); D = len(comps[0][1])
    z = np.empty((n, D))
    for c, (_, m, s) in enumerate(comps):
        idx = k == c; z[idx] = m + rng.normal(size=(idx.sum(), D)) * np.sqrt(s)
    return z / np.linalg.norm(z, axis=1, keepdims=True)

def cell(make, D, par, rng, pairs=40, nmc=4000):
    ex = {"RatioMeans": [], "RatioComp": [], "PN-Taylor": [], "Plug-in": []}; bayes = []
    for _ in range(pairs):
        a, b = make(D, par, rng), make(D, par, rng)
        (nu1, m1, e1, p1, r1), (nu2, m2, e2, p2, r2) = stats(a), stats(b)
        mu = nu1 @ nu2
        y = (draws(a, nmc, rng) * draws(b, nmc, rng)).sum(1); bayes.append(y.var())
        ex["RatioMeans"].append((mu - m1 @ m2 / np.sqrt(e1 * e2)) ** 2)
        ex["RatioComp"].append((mu - r1 @ r2) ** 2)
        ex["Plug-in"].append((mu - m1 @ m2 / np.linalg.norm(m1) / np.linalg.norm(m2)) ** 2)
        if p1 is not None: ex["PN-Taylor"].append((mu - p1 @ p2) ** 2)
    br = float(np.mean(bayes))
    return {"bayes_risk": br, **{k: dict(rmse=float(np.sqrt(np.mean(v))), rel_excess=float(np.mean(v) / br)) for k, v in ex.items() if v}}

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0); out = []
    for D in (2, 5, 10, 30, 100, 300, 1000):
        for snr in (0.5, 1.0, 2.0):
            t = time.time(); r = cell(gauss_input, D, snr, rng); out.append(dict(law="gauss", D=D, par=snr, res=r))
            print("gauss", D, snr, round(time.time() - t, 1), {k: v for k, v in r.items()}, flush=True)
        for spread in (0.1, 0.5, 1.0):
            for rho in ((1.0, 3.0, 10.0) if spread == 0.1 else (3.0, 10.0)):
                t = time.time(); r = cell(mixture_input, D, (rho, spread), rng)
                out.append(dict(law="mixture", D=D, par=rho, spread=spread, res=r))
                print("mixture", D, rho, spread, round(time.time() - t, 1), {k: v.get("rel_excess") for k, v in r.items() if k != "bayes_risk"}, flush=True)
    for D in (2, 10, 100, 300):
        for snr in (0.5, 1.0):
            t = time.time(); r = cell(embed_input, D, snr, rng); out.append(dict(law="embed2", D=D, par=snr, res=r))
            print("embed2", D, snr, round(time.time() - t, 1), {k: v.get("rel_excess") for k, v in r.items() if k != "bayes_risk"}, flush=True)
    json.dump(out, open("res_regime.json", "w"), indent=1)
