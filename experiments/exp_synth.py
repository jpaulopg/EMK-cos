import sys, json, time
import numpy as np
import emk_cos as E


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


def sample(pi, mus, covs, n, rng, df=None):
    C, D = mus.shape
    comp = rng.choice(C, n, p=pi)
    X = np.empty((n, D))
    for c in range(C):
        idx = np.where(comp == c)[0]
        if len(idx):
            L = np.linalg.cholesky(covs[c])
            Z = rng.standard_normal((len(idx), D))
            if df is not None:   # multivariate t with same scale matrix
                g = rng.chisquare(df, len(idx)) / df
                Z = Z / np.sqrt(g)[:, None]
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
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(C)) for r, m, S in cond])
    K = {"EMK-cos": Nu @ Nu.T, "Taylor2": Nt @ Nt.T, "CMI": cosm(Mean),
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


def mnar(X, rate, rng):
    # missing probability grows with |x|; calibrated to the target mean rate
    A = np.abs(X); A = (A - A.mean()) / A.std()
    lo, hi = -10, 10
    for _ in range(40):
        b = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(1.5 * A + b)))
        if p.mean() > rate: hi = b
        else: lo = b
    return rng.random(X.shape) < p


def emit(f, **kw):
    f.write(json.dumps(kw) + "\n"); f.flush()
    print(kw.get("exp"), {k: v for k, v in kw.items() if k not in ("res", "exp")}, flush=True)


def s1(f, reps=10, D=8, C=3, rate=0.5, ntest=200):
    for rep in range(reps):
        rng = np.random.default_rng(10 + rep)
        par = gen_params(D, C, rng, [3.0] * C)
        Xt = sample(*par, ntest, rng); Mt = mcar(ntest, D, rate, rng)
        Xm = np.where(Mt, np.nan, Xt)
        Ko = estimators(Xm, *par); Kref = Ko["EMK-cos"]
        emit(f, exp="S1", model="oracle", ntrain=0, rep=rep, res=score(Ko, Xt, Kref))
        for ntr in [100, 300, 1000]:
            Xtr = sample(*par, ntr, rng); Mtr = mcar(ntr, D, rate, rng)
            fit = E.em_gmm_missing(np.where(Mtr, np.nan, Xtr), C=C, iters=25, reg=1e-3, seed=rep)
            emit(f, exp="S1", model="fitted", ntrain=ntr, rep=rep, res=score(estimators(Xm, *fit), Xt, Kref))


def s2(f, reps=10, D=10, C=3, ntest=200):
    for a in [0.5, 1.0, 2.0, 4.0]:
        for rate in [0.3, 0.6]:
            for rep in range(reps):
                rng = np.random.default_rng(100 + rep)
                par = gen_params(D, C, rng, [a] * C)
                Xt = sample(*par, ntest, rng); Mt = mcar(ntest, D, rate, rng)
                Ko = estimators(np.where(Mt, np.nan, Xt), *par)
                emit(f, exp="S2", a=a, rate=rate, rep=rep, res=score(Ko, Xt, Ko["EMK-cos"]))


def s3(f, reps=10, D=10, C=2, rate=0.5, ntest=200):
    for rho in [1, 3, 10]:
        for rep in range(reps):
            rng = np.random.default_rng(200 + rep)
            par = gen_params(D, C, rng, [6.0, 6.0 / rho])
            Xt = sample(*par, ntest, rng); Mt = mcar(ntest, D, rate, rng)
            Xm = np.where(Mt, np.nan, Xt)
            Ko = estimators(Xm, *par); Kref = Ko["EMK-cos"]
            emit(f, exp="S3", rho=rho, model="oracle", rep=rep, res=score(Ko, Xt, Kref))
            Xtr = sample(*par, 500, rng); Mtr = mcar(500, D, rate, rng)
            fit = E.em_gmm_missing(np.where(Mtr, np.nan, Xtr), C=C, iters=25, reg=1e-3, seed=rep)
            emit(f, exp="S3", rho=rho, model="fitted", rep=rep, res=score(estimators(Xm, *fit), Xt, Kref))


def s4(f, reps=3, C=3, rate=0.5, ntest=150):
    for D in [10, 30, 60]:
        for rep in range(reps):
            rng = np.random.default_rng(300 + rep)
            par = gen_params(D, C, rng, [2.0 * np.sqrt(D / 10)] * C)
            Xt = sample(*par, ntest, rng); Mt = mcar(ntest, D, rate, rng)
            t = time.time()
            r = score(estimators(np.where(Mt, np.nan, Xt), *par), Xt)
            emit(f, exp="S4", D=D, rep=rep, sec=round(time.time() - t, 1), res=r)


def s5(f, reps=4, D=8, C=3, rate=0.5, ntest=200, ntr=600):
    for kind in ["gauss", "t5", "t3", "mnar"]:
        for rep in range(reps):
            rng = np.random.default_rng(400 + rep)
            par = gen_params(D, C, rng, [2.0] * C)
            df = {"gauss": None, "t5": 5, "t3": 3, "mnar": None}[kind]
            Xt = sample(*par, ntest, rng, df); Xtr = sample(*par, ntr, rng, df)
            if kind == "mnar":
                Mt = mnar(Xt, rate, rng); Mtr = mnar(Xtr, rate, rng)
            else:
                Mt = mcar(ntest, D, rate, rng); Mtr = mcar(ntr, D, rate, rng)
            fit = E.em_gmm_missing(np.where(Mtr, np.nan, Xtr), C=C, iters=25, reg=1e-3, seed=rep)
            emit(f, exp="S5", kind=kind, rep=rep,
                 res=score(estimators(np.where(Mt, np.nan, Xt), *fit), Xt))


def s6(f, reps=10, D=10, C=2, ntest=200):
    """Ambiguous posterior: tight components with different norms, high missing rate."""
    for rho in [1, 3, 10]:
        for rate in [0.7, 0.85]:
            for rep in range(reps):
                rng = np.random.default_rng(600 + rep)
                pi, mus, covs = gen_params(D, C, rng, [6.0, 6.0 / rho])
                covs = covs * 0.1
                Xt = sample(pi, mus, covs, ntest, rng); Mt = mcar(ntest, D, rate, rng)
                Ko = estimators(np.where(Mt, np.nan, Xt), pi, mus, covs)
                emit(f, exp="S6", rho=rho, rate=rate, rep=rep, res=score(Ko, Xt, Ko["EMK-cos"]))


if __name__ == "__main__":
    which = sys.argv[1].split(",")
    with open(f"res_synth_{'_'.join(which)}.jsonl", "w") as f:
        for w in which:
            globals()[w](f)
