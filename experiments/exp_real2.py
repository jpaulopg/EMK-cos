import sys, json, time, os
import numpy as np
from sklearn import datasets as ds
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import emk_cos as E


DATA_DIR = os.environ.get("EMKCOS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def load(name, n, seed):
    rng = np.random.default_rng(seed)
    if name in ("vehicle", "glass", "concrete"):
        d = np.load(os.path.join(DATA_DIR, name + ".npz")); X, y = d["X"].astype(float), d["y"]
        if name == "concrete":
            y = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
        else:
            y = np.unique(y, return_inverse=True)[1]
        lo, hi = X.min(0), X.max(0); keep = hi > lo
        X = (X[:, keep] - lo[keep]) / (hi[keep] - lo[keep])
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx], y[idx]
    if name == "news50":
        d = np.load(os.path.join(DATA_DIR, "news50.npz")); X, y = d["X"], d["y"]
    elif name == "oliv_pca40":
        d = np.load(os.path.join(DATA_DIR, "olivetti.npz")); X = PCA(40, random_state=0).fit_transform(d["X"].astype(float)); y = d["y"]
    elif name == "oliv_pool8":
        d = np.load(os.path.join(DATA_DIR, "olivetti.npz")); X = d["X"].astype(float).reshape(-1, 8, 8, 8, 8).mean((2, 4)).reshape(len(d["y"]), 64); y = d["y"]
    else:
        L = dict(iris=ds.load_iris, wine=ds.load_wine, bc=ds.load_breast_cancer)
        if name == "digits8":
            b = ds.load_digits()
        else:
            b = L[name]()
        X, y = b.data.astype(float), b.target
        lo, hi = X.min(0), X.max(0); keep = hi > lo
        X = (X[:, keep] - lo[keep]) / (hi[keep] - lo[keep])
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    X, y = X[idx], y[idx]
    keep = X.std(0) > 1e-9
    return X[:, keep], y


def cosm(V):
    n = np.linalg.norm(V, axis=1, keepdims=True); n[n == 0] = 1
    U = V / n
    return U @ U.T


def recall(Kt, Ke, k=10):
    A = Kt.copy(); B = Ke.copy(); np.fill_diagonal(A, -np.inf); np.fill_diagonal(B, -np.inf)
    ta = np.argsort(-A, 1)[:, :k]; tb = np.argsort(-B, 1)[:, :k]
    return float(np.mean([len(set(ta[i]) & set(tb[i])) / k for i in range(len(A))]))


def knn_acc(K, y, k=5):
    B = K.copy(); np.fill_diagonal(B, -np.inf)
    nb = np.argsort(-B, 1)[:, :k]
    pred = [np.bincount(y[nb[i]]).argmax() for i in range(len(y))]
    return float(np.mean(np.array(pred) == y))


def spec_ari(K, y):
    A = np.clip(0.5 * (K + K.T), 0, None); np.fill_diagonal(A, 1.0)
    lab = SpectralClustering(n_clusters=len(np.unique(y)), affinity="precomputed", random_state=0, n_init=5).fit_predict(A)
    return float(adjusted_rand_score(y, lab))


def run(name, rate, rep, n, C):
    X, y = load(name, n, 100 + rep)
    n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100))
    mask = rng.random((n, D)) < rate
    Xm = np.where(mask, np.nan, X)
    pi, mus, covs = E.em_gmm_missing(Xm, C=C, iters=20, reg=1e-3 * max(X.var(0).mean(), 1e-6) * 10, seed=rep)
    cond = [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(n)]
    Nu = np.array([E.nu_mixture(r, m, S) for r, m, S in cond])
    Nt = np.array([sum(r[c] * (E.nu_taylor2(m[c], S[c]) if m[c] @ m[c] > 1e-12 else np.zeros(D)) for c in range(C)) for r, m, S in cond])
    Mean = np.array([(r[:, None] * m).sum(0) for r, m, S in cond])
    Tr = np.array([sum(r[c] * (m[c] @ m[c] + np.trace(S[c])) for c in range(C)) for r, m, S in cond])
    dvec = 1 - (Nu ** 2).sum(1)
    Xmean = np.where(mask, np.nanmean(Xm, 0), Xm)
    O = (~mask).astype(float); Xo = np.where(mask, 0.0, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        Kobs = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
    Kobs[~np.isfinite(Kobs)] = 0.0
    K = {"EMK-cos": Nu @ Nu.T + np.diag(dvec), "Taylor2": Nt @ Nt.T, "CMI": cosm(Mean), "MeanImp": cosm(Xmean),
         "RatioMeans": (Mean @ Mean.T) / np.sqrt(np.outer(Tr, Tr)), "ObsSubset": Kobs}
    Kt = cosm(X); iu = np.triu_indices(n, 1)
    res = {}
    for k, M in K.items():
        Mx = M.copy(); np.fill_diagonal(Mx, 1.0)
        res[k] = dict(rmse=float(np.sqrt(np.mean((Mx[iu] - Kt[iu]) ** 2))), recall10=recall(Kt, Mx),
                      knn=knn_acc(Mx, y), ari=spec_ari(Mx, y), mineig=float(np.linalg.eigvalsh(0.5 * (Mx + Mx.T)).min()))
    res["_complete"] = dict(knn=knn_acc(Kt, y), ari=spec_ari(Kt, y))
    res["_pd"] = dict(dmin=float(dvec.min()), dmean=float(dvec.mean()), mineig=res["EMK-cos"]["mineig"])
    res["_dim"] = dict(n=int(n), D=int(D))
    return res


if __name__ == "__main__":
    name = sys.argv[1]; n = int(sys.argv[2]); C = int(sys.argv[3]); reps = int(sys.argv[4])
    rates = [float(r) for r in sys.argv[5].split(",")]
    with open(f"res_real2_{name}.jsonl", "w") as f:
        for rate in rates:
            for rep in range(reps):
                t = time.time(); r = run(name, rate, rep, n, C)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
