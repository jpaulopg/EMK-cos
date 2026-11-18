import sys, json, time
import numpy as np
from em_struct import em_struct
from exp_real2 import load
from exp_knn import pair_d2, knn_impute
from exp_select2 import rowlog

def gmm_post_mean(x, o, h, pi, mus, covs):
    C = len(pi); lp = np.zeros(C); m = np.zeros((C, len(h)))
    for c in range(C):
        L = np.linalg.cholesky(covs[c][np.ix_(o, o)]); d = x[o] - mus[c][o]; z = np.linalg.solve(L, d)
        lp[c] = np.log(pi[c]) - 0.5 * z @ z - np.log(np.diag(L)).sum()
        K = np.linalg.solve(covs[c][np.ix_(o, o)], covs[c][np.ix_(h, o)].T).T
        m[c] = mus[c][h] + K @ d
    r = np.exp(lp - lp.max()); r /= r.sum(); return r @ m
