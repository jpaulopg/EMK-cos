"""Synthetic stress test: error of cosine estimators vs. signal-to-noise ratio ||m||/sqrt(tr S)."""
import json
import numpy as np
import emk_cos as E
import baselines as B

rng = np.random.default_rng(7)
D = 10
NS = 300_000
levels = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]  # SNR = ||m||/sqrt(trS)
res = {}
for snr in levels:
    errs = {"EMK-cos": [], "Plug-in (CMI)": [], "Taylor2": [], "PN-Taylor": [], "RatioMeans": []}
    for rep in range(30):
        ests = []
        data = []
        for _ in range(2):
            A = rng.standard_normal((D, D))
            S = A @ A.T
            m = rng.standard_normal(D) + 1.0
            S *= (m @ m) / (snr ** 2 * np.trace(S))
            data.append((m, S))
        (m1, S1), (m2, S2) = data
        X1 = E.sample_mixture(np.array([1.0]), m1[None], S1[None], NS, rng)
        X2 = E.sample_mixture(np.array([1.0]), m2[None], S2[None], NS, rng)
        cs = (X1 * X2).sum(1) / (np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1))
        ref = cs.mean()
        e = {}
        e["EMK-cos"] = E.nu_gauss(m1, S1) @ E.nu_gauss(m2, S2)
        e["Plug-in (CMI)"] = m1 @ m2 / (np.linalg.norm(m1) * np.linalg.norm(m2))
        e["Taylor2"] = E.nu_taylor2(m1, S1) @ E.nu_taylor2(m2, S2)
        e["PN-Taylor"] = B.nu_pn2025(m1, S1) @ B.nu_pn2025(m2, S2)
        e["RatioMeans"] = m1 @ m2 / np.sqrt((m1 @ m1 + np.trace(S1)) * (m2 @ m2 + np.trace(S2)))
        for k in errs:
            errs[k].append(abs(e[k] - ref))
    res[snr] = {k: float(np.mean(v)) for k, v in errs.items()}
    print(snr, {k: round(v, 4) for k, v in res[snr].items()}, flush=True)
json.dump(res, open("res_snr.json", "w"))
