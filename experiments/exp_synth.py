"""Multimodal conditional laws (Appendix, Table with norms 6 and 6/rho): exp_synth.py s6"""
import sys, json, time
import numpy as np
import emk_cos as E
import baselines as B


def gen_params(D, C, rng, norms, k=3):
    """Gaussian mixture with centers of given norms (list of length C) and factor covariances."""
    mus = []
    for c in range(C):
        u = rng.standard_normal(D); u /= np.linalg.norm(u)
        mus.append(norms[c] * u)
    mus = np.array(mus)
    covs = []
    for c in range(C):
        L = rng.standard_normal((D, k))
        S = L @ L.T
        S *= 0.7 / np.mean(np.diag(S))
        covs.append(S + 0.3 * np.eye(D))
    pi = np.full(C, 1.0 / C)
    return pi, mus, np.array(covs)


def sample(pi, mus, covs, n, rng):
    C, D = mus.shape
    comp = rng.choice(C, n, p=pi)
    X = np.empty((n, D))
    for c in range(C):
        idx = np.where(comp == c)[0]
        if len(idx):
            L = np.linalg.cholesky(covs[c])
            Z = rng.standard_normal((len(idx), D))
            X[idx] = mus[c] + Z @ L.T
    return X


def cosm(V):
    n = np.linalg.norm(V, axis=1, keepdims=True); n[n == 0] = 1
    U = V / n
    return U @ U.T


def estimators(Xm, pi, mus, covs):
    n, D = Xm.shape; C = len(pi)
    cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Nt = np.array([sum(r[c] * (E.nu_taylor2(m[c], S[c]) if m[c] @ m[c] > 1e-12 else np.zeros(D)) for c in range(C))
                   for r, m, S in cond])
    Np = np.array([sum(r[c] * (B.nu_pn2025(m[c], S[c]) if m[c] @ m[c] > 1e-12 else np.zeros(D)) for c in range(C))
                   for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(C)) for r, m, S in cond])
    Nr = np.array([B.ratio_comp(r, m, S) for r, m, S in cond])
    K = {"EMK-cos": Nu @ Nu.T, "Taylor2": Nt @ Nt.T, "PN-Taylor": Np @ Np.T, "CMI": cosm(Mean), "RatioComp": Nr @ Nr.T,
         "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr))}
    return K


def score(K, Xtrue, Kref=None):
    Kt = cosm(Xtrue); n = len(Kt); iu = np.triu_indices(n, 1)
    out = {k: float(np.sqrt(np.mean((M[iu] - Kt[iu]) ** 2))) for k, M in K.items()}
    if Kref is not None:   # approximation/model error against exact expected cosine under the true model
        out.update({"ref_" + k: float(np.sqrt(np.mean((M[iu] - Kref[iu]) ** 2))) for k, M in K.items()})
    return out


def mcar(n, D, rate, rng):
    return rng.random((n, D)) < rate


def emit(f, **kw):
    f.write(json.dumps(kw) + "\n"); f.flush()
    print(kw.get("exp"), {k: v for k, v in kw.items() if k not in ("res", "exp")}, flush=True)


def s6(f, reps=10, D=10, C=2, ntest=200):
    """Ambiguous posterior: tight components with different norms, high missing rate."""
    for rho in [1, 3, 10]:
        for rate in [0.7, 0.85]:
            for rep in range(reps):
                rng = np.random.default_rng(600 + rep)
                pi, mus, covs = gen_params(D, C, rng, [6.0, 6.0 / rho])
                covs = covs * 0.1
                Xt = sample(pi, mus, covs, ntest, rng); Mt = mcar(ntest, D, rate, rng)
                Xm = np.where(Mt, np.nan, Xt)
                Ko = estimators(Xm, pi, mus, covs)
                # Monte Carlo reference for the expected cosine (independent of the quadrature)
                mrng = np.random.default_rng(9000 + rep)
                Nmc = []
                for i in range(ntest):
                    r, ms, Ss = E.conditional_mixture(Xm[i], pi, mus, covs)
                    Z = E.sample_mixture(r, ms, Ss, 20000, mrng); Nmc.append((Z / np.linalg.norm(Z, axis=1, keepdims=True)).mean(0))
                Nmc = np.array(Nmc)
                emit(f, exp="S6", rho=rho, rate=rate, rep=rep, res=score(Ko, Xt, Nmc @ Nmc.T))


if __name__ == "__main__":
    which = sys.argv[1].split(",")
    with open(f"res_synth_{'_'.join(which)}.jsonl", "w") as f:
        for w in which:
            globals()[w](f)
