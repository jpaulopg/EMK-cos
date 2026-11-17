"""Summary of exp_mech.py: missingness mechanisms and inductive protocol (Table tab:mech)."""
import json
import numpy as np

NAMES = ["iris", "wine", "bc", "digits8", "vehicle", "oliv_pool8", "oliv_pca40", "news50"]
METHODS = ["EMK-cos", "RatioMeans", "Plug-in", "MeanImp", "ObsSubset"]
CONDS = ["mcar", "mcar_ind", "mar", "mnar"]

tab = {c: {m: [] for m in METHODS} for c in CONDS}
for name in NAMES:
    runs = [json.loads(l)["res"] for l in open(f"res_mech_{name}.jsonl")]
    for c in CONDS:
        for m in METHODS:
            tab[c][m].append(np.mean([r[c]["rmse"][m] for r in runs]))

for c in CONDS:
    emk = np.array(tab[c]["EMK-cos"])
    means = " ".join(f"{m} {np.mean(tab[c][m]):.4f}" for m in METHODS)
    better = int((emk < np.array(tab[c]["Plug-in"])).sum())
    red = 100 * (1 - emk / np.array(tab[c]["Plug-in"])).mean()
    print(f"{c:9s} | {means} | EMK<plug-in on {better}/8 sets, mean reduction {red:.1f}%")
