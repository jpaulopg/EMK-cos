"""Summary of the PSD analysis (Table tab:psd).
Posterior applications: res_psd_{seed}.jsonl from exp_psd.py.
Incomplete data: smallest eigenvalues stored in res_cmp_*.jsonl by exp_cmp.py."""
import glob, json
import numpy as np

R = [json.loads(open(f"res_psd_{s}.jsonl").read())["res"] for s in range(3)]
methods = ["EMK-cos", "RatioMeans", "Plug-in", "MC-vec100", "Taylor2", "MC-pair10", "MC-pair100"]
for app in ("vae", "gp", "blr"):
    counts = {m: 0 for m in methods}; chol = {m: 0 for m in methods}; total = 0
    for r in R:
        for case in r[app].values():
            total += 1
            for m in methods:
                counts[m] += case[m]["mineig"] < -1e-8
                chol[m] += not case[m]["chol"]
    print(app.ljust(4), " ".join(f"{m}:{counts[m]}/{total}(chol {chol[m]})" for m in methods))

runs = [json.loads(l)["res"] for f in sorted(glob.glob("res_cmp_*.jsonl")) for l in open(f)]
for m in ("EMK-cos", "RatioMeans", "Plug-in", "Taylor2", "ObsSubset"):
    e = np.array([r[m]["mineig"] for r in runs])
    print(f"incomplete {m}: {(e < -1e-8).sum()}/{len(e)} non-PSD, min eigenvalue {e.min():.3g}")
