"""Observed-subset cosine with and without the PSD matrix correction (nearest correlation matrix),
on the same data subsamples and masks as exp_cmp.py."""
import sys, json, time
import numpy as np
from exp_real2 import load, cosm
from baselines import nearest_correlation

def run(name, rate, rep, n):
    X, _ = load(name, n, 100 + rep); n, D = X.shape
    rng = np.random.default_rng(1000 * rep + int(rate * 100)); mask = rng.random((n, D)) < rate
    O = (~mask).astype(float); Xo = np.where(mask, 0.0, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        K = (Xo @ Xo.T) / np.sqrt(((Xo ** 2) @ O.T) * ((Xo ** 2) @ O.T).T)
    K[~np.isfinite(K)] = 0.0; np.fill_diagonal(K, 1.0)
    Kc = nearest_correlation(K, iters=300, tol=1e-7)
    Kt = cosm(X); iu = np.triu_indices(n, 1)
    r = lambda M: float(np.sqrt(np.mean((M[iu] - Kt[iu]) ** 2)))
    return {"ObsSubset": r(K), "ObsSubset+PSD": r(Kc), "mineig_corrected": float(np.linalg.eigvalsh(Kc).min())}

if __name__ == "__main__":
    name, n = sys.argv[1], int(sys.argv[2])
    with open(f"res_yu_{name}.jsonl", "w") as f:
        for rate in (0.3, 0.5, 0.7):
            for rep in range(3):
                t = time.time(); r = run(name, rate, rep, n)
                f.write(json.dumps(dict(data=name, rate=rate, rep=rep, res=r)) + "\n"); f.flush()
                print(name, rate, rep, round(time.time() - t, 1), flush=True)
