"""Print the LaTeX rows of the paper's tables from the stored results (run from results/)."""
import json, glob, collections
import numpy as np

def b(vals, fmt, lower=True):
    """Bold the best value(s) of a row (ties at the printed precision)."""
    s = [fmt.format(v) for v in vals]; key = [float(x) for x in s]
    best = min(key) if lower else max(key)
    return [f"\\textbf{{{x}}}" if k == best else x for x, k in zip(s, key)]

out = {}
# SNR figure
S = json.load(open("res_snr.json")); lv = ["4.0", "2.0", "1.0", "0.5", "0.25", "0.1"]
for m, k in (("EMK", "EMK-cos"), ("PLUG", "Plug-in (CMI)"), ("PN", "PN-Taylor"), ("RATIO", "RatioMeans")):
    out["snr_" + m] = "".join(f"({i+1},{np.mean(S[l][k]):.3g})" for i, l in enumerate(lv))

# regime map: worst case (max over SNR, rho and spread) of the excess risk relative to the Bayes risk, in %
R = json.load(open("res_regime.json")); rows = []
pc = lambda x, k: 100 * x[k]["rel_excess"] if k in x else None
fmt = lambda v, d=2: "--" if v is None else f"{v:.{d}f}"
for D in (2, 5, 10, 30, 100, 300, 1000):
    rr = [r for r in R if r["D"] == D]
    gs = [r["res"] for r in rr if r["law"] == "gauss"]
    es = [r["res"] for r in rr if r["law"] == "embed2"]
    ms = [r["res"] for r in rr if r["law"] == "mixture" and r["par"] > 1]
    mx = lambda L, k: (max(v for v in (pc(x, k) for x in L) if v is not None) if L and any(k in x for x in L) else None)
    cells = [fmt(mx(gs, "RatioMeans")), fmt(mx(gs, "PN-Taylor")), fmt(mx(es, "RatioMeans")), fmt(mx(es, "PN-Taylor")),
             fmt(mx(ms, "RatioMeans"), 1), fmt(mx(ms, "RatioComp")), fmt(mx(ms, "PN-Taylor"))]
    rows.append(f"{D} & " + " & ".join(cells) + r" \\")
out["regime"] = "\n".join(rows)
# mixture detail by spread (max over D and rho)
out["regime_spread"] = {sp: {k: round(max(100 * r["res"][k]["rel_excess"] for r in R if r["law"] == "mixture" and r.get("spread") == sp and r["par"] > 1 and k in r["res"]), 3)
                             for k in ("RatioMeans", "RatioComp", "PN-Taylor", "Plug-in")} for sp in (0.1, 0.5, 1.0)}
out["regime_gauss_min_max"] = {k: [round(min(100*r["res"][k]["rel_excess"] for r in R if r["law"]=="gauss" and k in r["res"]),3), round(max(100*r["res"][k]["rel_excess"] for r in R if r["law"]=="gauss" and k in r["res"]),3)] for k in ("Plug-in",)}

# multimodal (S6) with MC reference
L = [json.loads(l) for l in open("res_synth_s6.jsonl")]
G = collections.defaultdict(list)
for l in L: G[(l["exp"], l.get("rho"), l.get("rate"), l.get("kind"))].append(l["res"])
rows = []
for rho in (1, 3, 10):
    for rate in (0.7, 0.85):
        v = G[("S6", rho, rate, None)]; a = lambda k: np.mean([x[k] for x in v])
        ref = b([a("ref_RatioMeans"), a("ref_RatioComp"), a("ref_PN-Taylor"), a("ref_CMI")], "{:.3f}")
        real = b([a("EMK-cos"), a("RatioMeans"), a("RatioComp"), a("PN-Taylor")], "{:.3f}")
        rows.append(f"{rho} & {int(rate*100)}\\% & {a('ref_EMK-cos'):.4f} & " + " & ".join(ref) + " & " + " & ".join(real) + r" \\")
out["s6"] = "\n".join(rows)
# s5 kept from the earlier run
G5 = collections.defaultdict(list)
for l in (json.loads(l) for l in open("res_synth_s5_s6.jsonl")): G5[(l["exp"], l.get("kind"))].append(l["res"])
out["s5"] = {k: {m: round(np.mean([x[m] for x in v]), 4) for m in ("EMK-cos", "RatioMeans", "PN-Taylor", "CMI")} for (e, k), v in G5.items() if e == "S5"}

# VAE
V = [json.loads(l)["res"] for l in open("res_vae.jsonl")]; rows = []
M4 = ("EMK-cos", "RatioMeans", "PN-Taylor", "Plug-in")
for t, lab in (("tau0.25", "0.25"), ("tau1.0", "1"), ("tau4.0", "4"), ("tau16.0", "16")):
    snr = np.mean([v[t]["_snr"] for v in V])
    ref = b([np.mean([v[t][m]["rmse_ref"] for v in V]) for m in M4], "{:.4f}")
    real = [f"{np.mean([v[t][m]['rmse_real'] for v in V]):.4f}" for m in M4]
    rows.append(f"{lab} ({snr:.2f}) & " + " & ".join(ref) + " & " + " & ".join(real) + r" \\")
out["vae"] = "\n".join(rows)

# thresholds
T = [json.loads(open(f"res_thr_{s}.jsonl").read())["res"] for s in range(3)]; rows = []
for t, lab in (("tau0.25", "0.25"), ("tau1.0", "1"), ("tau4.0", "4"), ("tau16.0", "16")):
    far = [np.mean([x[t][m]["far_at_exact_th"] for x in T]) * 100 for m in M4]
    dis = [np.mean([x[t][m]["dis_at_exact_th"] for x in T]) * 100 for m in M4]
    auc = [np.mean([x[t][m]["auc"] for x in T]) for m in M4]
    rows.append(f"{lab} & " + " & ".join(f"{f:.2f}" for f in far) + " & " + " & ".join(f"{d:.2f}" for d in dis) + " & " + " & ".join(f"{a:.3f}" for a in auc) + r" \\")
out["thr"] = "\n".join(rows)

# Bayesian
rows = []
for kind, keys, labs in (("gp", ["nobs3", "nobs6", "nobs12"], ["GP, 3 obs.", "GP, 6 obs.", "GP, 12 obs."]),
                         ("blr", ["nk15", "nk30", "nk60"], ["Regression, 15", "Regression, 30", "Regression, 60"])):
    Rr = [json.loads(open(f"res_rv_{kind}_{s}.jsonl").read())["res"] for s in range(3)]
    for k, lab in zip(keys, labs):
        g = lambda m, f: np.mean([r[k][m][f] for r in Rr])
        ref = b([g(m, "rmse_ref") for m in M4], "{:.4f}"); tru = b([g(m, "rmse_true") for m in M4], "{:.4f}")
        rows.append(f"{lab} & " + " & ".join(ref) + " & " + " & ".join(tru) + r" \\")
out["bayes"] = "\n".join(rows)

# prior
rows = []
for fn, lab in (("res_prior_iid.jsonl", "Independent, $\\N(0,1.25\\,I)$"), ("res_prior.jsonl", "Four families")):
    Rp = [json.loads(l)["res"] for l in open(fn)]
    for i, (tau, tl) in enumerate(((0.3125, "0.31"), (1.25, "1.25"), (4.0, "4"), (20.0, "20"))):
        k = f"nk15_tau{tau}"; vals = b([np.mean([r[k][m] for r in Rp]) for m in M4], "{:.4f}")
        rows.append(f"{lab if i == 0 else ''} & {tl} & " + " & ".join(vals) + r" \\")
    rows.append(r"\midrule")
out["prior"] = "\n".join(rows[:-1])

# uncertainty
U = [json.loads(open(f"res_unc_{s}.jsonl").read())["res"] for s in range(3)]; rows = []
MU = ("EMK-cos", "PN-Taylor", "RatioMoments")
for app, cases in (("vae", [("tau0.25", "VAE, $\\tau=0.25$"), ("tau1.0", "VAE, $\\tau=1$"), ("tau4.0", "VAE, $\\tau=4$"), ("tau16.0", "VAE, $\\tau=16$")]),
                   ("gp", [("nobs3", "GP, 3 obs."), ("nobs6", "GP, 6 obs."), ("nobs12", "GP, 12 obs.")]),
                   ("blr", [("nk15", "Regression, 15"), ("nk30", "Regression, 30"), ("nk60", "Regression, 60")])):
    for k, lab in cases:
        g = lambda m, f: np.mean([u[app][k][m][f] for u in U])
        dss = b([g(m, "dss") for m in MU], "{:.3f}")
        vr = [f"{g(m, 'var_ratio'):.2f}" for m in MU]
        sd = b([g(m, "rmse_sd_ref") for m in MU], "{:.4f}")
        vr = [f"{g(m, 'scale_ratio'):.2f}" for m in MU]
        rows.append(f"{lab} & {np.mean([u[app][k]['_mean_sd_ref'] for u in U]):.2f} & " + " & ".join(dss) + " & " + " & ".join(vr) + " & " + " & ".join(sd) + r" \\")
out["unc"] = "\n".join(rows)

# incomplete data
names = [("iris", "Iris"), ("wine", "Wine"), ("bc", "Breast Cancer"), ("digits8", "Digits $8\\times8$"), ("vehicle", "Vehicle"),
         ("oliv_pool8", "Olivetti, $8\\times8$ pooled"), ("oliv_pca40", "Olivetti, PCA-40"), ("news50", "20 Newsgroups, SVD-50")]
MR = ("EMK-cos", "RatioMeans", "PN-Taylor", "Plug-in", "MeanImp", "ObsSubset")
rows = []; cols = collections.defaultdict(list)
for n, lab in names:
    C = [json.loads(l)["res"] for l in open(f"res_cmp_{n}.jsonl")]; Y = [json.loads(l)["res"] for l in open(f"res_yu_{n}.jsonl")]
    vals = [np.mean([c[m]["rmse"] for c in C]) for m in MR] + [np.mean([y["ObsSubset+PSD"] for y in Y])]
    for m, v in zip(MR + ("ObsPSD",), vals): cols[m].append(v)
    rows.append(f"{lab} & " + " & ".join(b(vals, "{:.4f}")) + r" \\")
rows.append(r"\midrule")
rows.append("Mean & " + " & ".join(b([np.mean(cols[m]) for m in MR + ("ObsPSD",)], "{:.4f}")) + r" \\")
out["real"] = "\n".join(rows)
E = np.array(cols["EMK-cos"])
out["real_red"] = {m: round(100 * (1 - E / np.array(cols[m])).mean(), 1) for m in MR[1:] + ("ObsPSD",)}

# PSD counts
P = [json.loads(open(f"res_psd_{s}.jsonl").read())["res"] for s in range(3)]
MP = ("EMK-cos", "RatioMeans", "PN-Taylor", "Plug-in", "MC-vec100", "Taylor2", "MC-pair10", "MC-pair100")
rows = []
for app, lab in (("vae", "VAE ($n=300$)"), ("gp", "GP curves ($n=150$)"), ("blr", "Regression weights ($n=150$)")):
    c = [sum(case[m]["mineig"] < -1e-8 for p in P for case in p[app].values()) for m in MP]
    rows.append(f"{lab} & " + " & ".join(f"{x}/9" for x in c) + r" & -- & -- \\")
runs = [json.loads(l)["res"] for n, _ in names for l in open(f"res_cmp_{n}.jsonl")]
inc = [sum(r[m]["mineig"] < -1e-8 for r in runs) if m in runs[0] else None for m in MP]
rows.append("Incomplete data & " + " & ".join("--" if x is None else f"{x}/72" for x in inc) + r" & 72/72 & 0/72 \\")
out["psd"] = "\n".join(rows)

json.dump(out, open("tables.json", "w"), indent=1)
for k, v in out.items(): print(f"=== {k}\n{v}")
