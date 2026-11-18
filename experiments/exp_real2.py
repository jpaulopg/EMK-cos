import sys, json, time, os
import numpy as np
from sklearn import datasets as ds
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
import emk_cos as E


DATA_DIR = os.environ.get("EMKCOS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


MINMAX = ("iris", "wine", "bc", "digits8", "vehicle")   # representations that are min-max scaled


def load(name, n, seed):
    """Random subsample of n records of a representation, unscaled; columns that are constant are removed.
    Min-max scaling, where it applies, is estimated from the observed entries only (scale_observed)."""
    rng = np.random.default_rng(seed)
    if name in ("vehicle", "glass", "concrete"):
        d = np.load(os.path.join(DATA_DIR, name + ".npz")); X, y = d["X"].astype(float), d["y"]
        if name == "concrete":
            y = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
        else:
            y = np.unique(y, return_inverse=True)[1]
    elif name == "news50":
        d = np.load(os.path.join(DATA_DIR, "news50.npz")); X, y = d["X"], d["y"]
    elif name == "oliv_pca40":
        d = np.load(os.path.join(DATA_DIR, "olivetti.npz")); X = PCA(40, random_state=0).fit_transform(d["X"].astype(float)); y = d["y"]
    elif name == "oliv_pool8":
        d = np.load(os.path.join(DATA_DIR, "olivetti.npz")); X = d["X"].astype(float).reshape(-1, 8, 8, 8, 8).mean((2, 4)).reshape(len(d["y"]), 64); y = d["y"]
    else:
        L = dict(iris=ds.load_iris, wine=ds.load_wine, bc=ds.load_breast_cancer)
        b = ds.load_digits() if name == "digits8" else L[name]()
        X, y = b.data.astype(float), b.target
    X = X[:, X.max(0) > X.min(0)]
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    X, y = X[idx], y[idx]
    keep = X.std(0) > 1e-9
    return X[:, keep], y


def scale_observed(name, X, mask, rows=None):
    """Min-max scaling of the complete matrix X with column ranges estimated from the observed entries
    (of the given rows only, e.g. the training records); identity for the other representations."""
    if name not in MINMAX:
        return X
    R = slice(None) if rows is None else rows
    Xo = np.where(mask[R], np.nan, X[R])
    with np.errstate(all="ignore"):
        lo, hi = np.nanmin(Xo, 0), np.nanmax(Xo, 0)
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
    lo = np.where(ok, lo, 0.0); rg = np.where(ok, hi - lo, 1.0)
    return (X - lo) / rg


def ridge_from(Xm):
    """EM ridge from the observed entries only: 1e-2 times the mean observed variance (floor 1e-8)."""
    with np.errstate(all="ignore"):
        v = np.nanvar(Xm, 0)
    v = v[np.isfinite(v)]
    return 1e-3 * max(float(v.mean()) if len(v) else 0.0, 1e-6) * 10


def masked(name, n, seed, rate, mask_seed):
    """Load, draw an MCAR mask and scale from the observed entries. Returns X (complete, scaled), Xm, mask, y."""
    X, y = load(name, n, seed); n_, D = X.shape
    mask = np.random.default_rng(mask_seed).random((n_, D)) < rate
    X = scale_observed(name, X, mask)
    return X, np.where(mask, np.nan, X), mask, y


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
