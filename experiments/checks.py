import numpy as np
from scipy.special import erf
import emk_cos as E

rng = np.random.default_rng(1)
out = {}

# T1: 1-D closed form  E[sign(x)] = erf(m/(s sqrt2))
errs = []
for m, s in [(0.3, 1.0), (2.0, 0.5), (-1.0, 3.0), (0.0, 1.0), (10.0, 1e-3), (1e-3, 5.0)]:
    v = E.nu_gauss(np.array([m]), np.array([[s * s]]))[0]
    errs.append(abs(v - erf(m / (s * np.sqrt(2)))))
out["T1_1d_maxerr"] = max(errs)

# T2: isotropic, compare with 1F1 closed form (Presnell-Rumcheva type): mean = c * m/||m||
from scipy.special import hyp1f1, gamma
D = 6
m = np.array([1.0, -0.5, 0.7, 0, 0.2, 0.1])
s2 = 0.8
nu = E.nu_gauss(m, s2 * np.eye(D))
# E[x/||x||] direction coefficient via MC with many samples
X = m + np.sqrt(s2) * rng.standard_normal((4_000_000, D))
mc = (X / np.linalg.norm(X, axis=1, keepdims=True)).mean(0)
out["T2_iso_maxerr_vs_MC"] = np.abs(nu - mc).max()
out["T2_parallel_to_m"] = float(np.abs(nu / np.linalg.norm(nu) - m / np.linalg.norm(m)).max())

# T3: full covariance, singular, vs MC
D = 5
A = rng.standard_normal((D, 2))
S = A @ A.T * 0.6  # rank 2
m = np.array([0.4, -0.2, 0.9, 0.3, 0.0])
nu = E.nu_gauss(m, S)
L = np.linalg.cholesky(S + 1e-12 * np.eye(D))
X = m + rng.standard_normal((4_000_000, 2)) @ A.T * np.sqrt(0.6)
mc = (X / np.linalg.norm(X, axis=1, keepdims=True)).mean(0)
out["T3_singular_maxerr_vs_MC"] = np.abs(nu - mc).max()
out["T3_taylor2_maxerr_vs_MC"] = np.abs(E.nu_taylor2(m, S) - mc).max()
out["T3_plugin_maxerr_vs_MC"] = np.abs(m / np.linalg.norm(m) - mc).max()

# T4: second moment
M2 = E.m2_gauss(m, S)
mc2 = (X[:, :, None] * X[:, None, :] / (np.linalg.norm(X, axis=1) ** 2)[:, None, None]).mean(0)
out["T4_M2_maxerr_vs_MC"] = np.abs(M2 - mc2).max()
out["T4_trace_M2"] = float(np.trace(M2))
out["T4_1d_M2"] = float(E.m2_gauss(np.array([0.7]), np.array([[2.0]]))[0, 0])

# T5: mixture of two incomplete vectors; E[cos] vs MC
D = 6
Sig = np.cov(rng.standard_normal((200, D)) @ rng.standard_normal((D, D)), rowvar=False)
pi = np.array([0.5, 0.3, 0.2])
mus = rng.standard_normal((3, D)) * 1.5 + 1.0
covs = np.array([Sig * k for k in (0.5, 1.0, 0.3)])
xi = np.array([1.0, np.nan, 0.5, np.nan, np.nan, 1.2])
xj = np.array([np.nan, 0.8, np.nan, 1.0, 0.7, np.nan])
ri, mi, Si = E.conditional_mixture(xi, pi, mus, covs)
rj, mj, Sj = E.conditional_mixture(xj, pi, mus, covs)
nui = E.nu_mixture(ri, mi, Si)
nuj = E.nu_mixture(rj, mj, Sj)
emk = float(nui @ nuj)
Xi = E.sample_mixture(ri, mi, Si, 4_000_000, rng)
Xj = E.sample_mixture(rj, mj, Sj, 4_000_000, rng)
cs = (Xi * Xj).sum(1) / (np.linalg.norm(Xi, axis=1) * np.linalg.norm(Xj, axis=1))
mc, se = cs.mean(), cs.std() / np.sqrt(len(cs))
mean_i = (ri[:, None] * mi).sum(0)
mean_j = (rj[:, None] * mj).sum(0)
cmi = float(mean_i @ mean_j / (np.linalg.norm(mean_i) * np.linalg.norm(mean_j)))
tr_i = sum(r * (m @ m + np.trace(S)) for r, m, S in zip(ri, mi, Si))
tr_j = sum(r * (m @ m + np.trace(S)) for r, m, S in zip(rj, mj, Sj))
ratio = float(mean_i @ mean_j / np.sqrt(tr_i * tr_j))
out["T5_EMKcos"] = emk
out["T5_MC"] = float(mc)
out["T5_MC_se"] = float(se)
out["T5_CMI"] = cmi
out["T5_ratio_of_means"] = ratio

# T6: Pearson (centered) version
H = np.eye(D) - np.ones((D, D)) / D
nui_c = E.nu_mixture(ri, mi, Si, H)
nuj_c = E.nu_mixture(rj, mj, Sj, H)
Xic, Xjc = Xi @ H, Xj @ H
pc = (Xic * Xjc).sum(1) / (np.linalg.norm(Xic, axis=1) * np.linalg.norm(Xjc, axis=1))
out["T6_pearson_EMK"] = float(nui_c @ nuj_c)
out["T6_pearson_MC"] = float(pc.mean())
out["T6_pearson_se"] = float(pc.std() / np.sqrt(len(pc)))

# T7: squared cosine
M2i = E.m2_mixture(ri, mi, Si)
M2j = E.m2_mixture(rj, mj, Sj)
sq = (cs ** 2)
out["T7_cos2_EMK"] = float((M2i * M2j).sum())
out["T7_cos2_MC"] = float(sq.mean())
out["T7_cos2_se"] = float(sq.std() / np.sqrt(len(sq)))

for k, v in out.items():
    print(f"{k:28s} {v:.6g}")
