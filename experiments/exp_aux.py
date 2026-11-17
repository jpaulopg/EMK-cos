"""Where does the real-data error come from?  Error decomposition at 50% MCAR.
 A  real data, GMM estimated from the incomplete data          (what the paper reports)
 B  semi-synthetic data drawn from a GMM fitted to the complete real data, posterior computed with the TRUE GMM
 C  same semi-synthetic data, GMM re-estimated from the incomplete data
Method error = EMK-cos vs Monte Carlo under the same posterior (A and B)."""
import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import load, cosm

CS = dict(iris=3, wine=3, vehicle=4, bc=3, oliv_pca40=5, news50=5, digits8=5, oliv_pool8=5)
NS = dict(iris=150, wine=178, vehicle=400, bc=300, oliv_pca40=400, news50=600, digits8=400, oliv_pool8=400)

def kernels(Xm, cond, mc_rng=None, nmc=20000):
    n, D = Xm.shape; iu = np.triu_indices(n, 1)
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(len(r))) for r, m, S in cond])
    K = {"EMK-cos": Nu @ Nu.T, "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr)), "Plug-in": cosm(Mean)}
    meth = None
    if mc_rng is not None:
        Nr = []
        for r, m, S in cond:
            Z = E.sample_mixture(r, m, S, nmc, mc_rng); Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); Nr.append(Z.mean(0))
        Nr = np.array(Nr); Kmc = Nr @ Nr.T; meth = {k: float(np.sqrt(((v[iu] - Kmc[iu])**2).mean())) for k, v in K.items()}
    return K, meth

def rmse(K, Kt):
    iu = np.triu_indices(len(Kt), 1); return {k: float(np.sqrt(((v[iu] - Kt[iu])**2).mean())) for k, v in K.items()}

def run(name, rep, rate=0.5):
    X, _ = load(name, NS[name], 100 + rep); n, D = X.shape; C = CS[name]
    reg = 1e-3 * max(X.var(0).mean(), 1e-6) * 10; rng = np.random.default_rng(1000 * rep + int(rate * 100))
    mask = rng.random((n, D)) < rate; mcr = np.random.default_rng(55 + rep)
    out = {}
    # A: real data, estimated model
    Xm = np.where(mask, np.nan, X); pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct="full")
    K, meth = kernels(Xm, [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)], mcr)
    out["A"] = dict(rmse=rmse(K, cosm(X)), method_err=meth)
    # true model fitted on the complete real data; sample semi-synthetic data of the same size
    pt, mt, ct = em_struct(X, C=C, iters=30, reg=reg, seed=rep, struct="full")
    srng = np.random.default_rng(7000 + rep); z = srng.choice(C, n, p=pt)
    Xs = np.array([srng.multivariate_normal(mt[c], ct[c]) for c in z]); Xsm = np.where(mask, np.nan, Xs); Kt = cosm(Xs)
    K, meth = kernels(Xsm, [E.conditional_mixture(Xsm[i], pt, mt, ct) for i in range(n)], mcr)
    out["B"] = dict(rmse=rmse(K, Kt), method_err=meth)
    pe, me, ce = em_struct(Xsm, C=C, iters=15, reg=reg, seed=rep, struct="full")
    K, _ = kernels(Xsm, [E.conditional_mixture(Xsm[i], pe, me, ce) for i in range(n)])
    out["C"] = dict(rmse=rmse(K, Kt))
    return out

if __name__ == "__main__":
    name = sys.argv[1]
    with open(f"res_aux_{name}.jsonl", "w") as f:
        for rep in range(3):
            t = time.time(); r = run(name, rep); f.write(json.dumps(dict(data=name, rep=rep, res=r)) + "\n"); f.flush()
            print(name, rep, round(time.time() - t, 1), flush=True)
