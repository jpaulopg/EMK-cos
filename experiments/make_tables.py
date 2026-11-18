"""Print the LaTeX rows of the paper's tables from the stored results (run from results/)."""
import json, glob, collections
import numpy as np

def b(vals, fmt, lower=True):
    """Bold the best value(s) of a row (ties at the printed precision)."""
    s = [fmt.format(v) for v in vals]; key = [float(x) for x in s]
    best = min(key) if lower else max(key)
    return [f"\\textbf{{{x}}}" if k == best else x for x, k in zip(s, key)]

out = {}
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

# concentration diagnostic: per-pair |mu - f| against the bound sqrt(2)(CV_i + CV_j), and SE of the cell excess
pts = dict(global_=[], comp=[]); se = []
for r in R:
    P_ = r["res"].get("_pairs")
    if not P_: continue
    br = np.mean(P_["bayes"])
    for key, bk, lab in (("RatioMeans", "bound_RatioMeans", "global_"), ("RatioComp", "bound_RatioComp", "comp")):
        for e, bd in zip(P_[key], P_[bk]): pts[lab].append((float(np.sqrt(e)), float(bd), r["law"]))
        se.append(100 * np.std(P_[key], ddof=1) / np.sqrt(len(P_[key])) / br)
out["cv_bound_holds"] = {k: bool(all(a <= b_ + 1e-12 for a, b_, _ in v)) for k, v in pts.items()}
out["cv_logcorr"] = {k: float(np.corrcoef(np.log([max(a, 1e-12) for a, _, _ in v]), np.log([b_ for _, b_, _ in v]))[0, 1]) for k, v in pts.items()}
out["cv_ratio_median"] = {k: float(np.median([b_ / max(a, 1e-12) for a, b_, _ in v])) for k, v in pts.items()}
out["regime_se_max"] = float(max(se)) if se else None
json.dump(pts, open("cv_points.json", "w"))
# per-cell coordinates for the figure: mean bound against RMS error, by family and estimator
fig = collections.defaultdict(list)
for r in R:
    P_ = r["res"].get("_pairs")
    if not P_: continue
    for key, bk in (("RatioMeans", "bound_RatioMeans"), ("RatioComp", "bound_RatioComp")):
        if r["law"] != "mixture" and key == "RatioComp": continue
        if r["law"] == "mixture" and r["par"] <= 1: continue
        fig[(r["law"], key)].append(f"({np.mean(P_[bk]):.4g},{max(np.sqrt(np.mean(P_[key])), 1e-6):.4g})")
out["cv_fig"] = {f"{a}|{k}": " ".join(v) for (a, k), v in fig.items()}

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
# VAE
V = [json.loads(l)["res"] for l in open("res_vae.jsonl")]; rows = []
M4 = ("EMK-cos", "RatioMeans", "PN-Taylor", "Plug-in")
for t, lab in (("tau0.25", "0.25"), ("tau1.0", "1"), ("tau4.0", "4"), ("tau16.0", "16")):
    snr = np.mean([v[t]["_snr"] for v in V])
    ref = b([np.mean([v[t][m]["rmse_ref"] for v in V]) for m in M4], "{:.4f}")
    real = [f"{np.mean([v[t][m]['rmse_real'] for v in V]):.4f}" for m in M4]
    rows.append(f"{lab} ({snr:.2f}) & " + " & ".join(ref) + " & " + " & ".join(real) + r" \\")
out["vae"] = "\n".join(rows)

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
    for i, (tau, tl) in enumerate(((1.25, "1.25"), (4.0, "4"))):
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

# incomplete data: RMSE against the complete-data cosine (mean over rates and repetitions), with the standard error
# of the mean over the 9 runs of each set for EMK-cos and the paired differences to the per-component ratio
names = [("iris", "Iris"), ("wine", "Wine"), ("bc", "Breast Cancer"), ("digits8", "Digits $8\\times8$"), ("vehicle", "Vehicle"),
         ("oliv_pool8", "Olivetti, $8\\times8$ pooled"), ("oliv_pca40", "Olivetti, PCA-40"), ("news50", "20 Newsgroups, SVD-50")]
MR = ("EMK-cos", "RatioMeans", "RatioComp", "PN-Taylor", "Plug-in", "MeanImp", "ObsSubset")
rows = []; cols = collections.defaultdict(list); runs_all = collections.defaultdict(list); drows = []; vrows = []
VM = ("EMK", "PN", "Ratio", "RatioComp"); vagg = collections.defaultdict(list)
for n, lab in names:
    C = [json.loads(l)["res"] for l in open(f"res_cmp_{n}.jsonl")]; Y = [json.loads(l)["res"] for l in open(f"res_yu_{n}.jsonl")]
    vals = [np.mean([c[m]["rmse"] for c in C]) for m in MR] + [np.mean([y["ObsSubset+PSD"] for y in Y])]
    for m in MR: runs_all[m] += [c[m]["rmse"] for c in C]
    runs_all["ObsPSD"] += [y["ObsSubset+PSD"] for y in Y]
    for m, v in zip(MR + ("ObsPSD",), vals): cols[m].append(v)
    rows.append(f"{lab} & " + " & ".join(b(vals, "{:.4f}")) + r" \\")
    # defined pairs: both records share at least one observed coordinate with nonzero norms
    fu = np.mean([y["defined"]["frac_undefined"] for y in Y])
    dv = [np.nanmean([c["EMK-cos"]["rmse_defined"] for c in C]), np.nanmean([c["Plug-in"]["rmse_defined"] for c in C]),
          np.nanmean([y["defined"]["ObsSubset"] for y in Y]), np.nanmean([y["defined"]["ObsPSD"] for y in Y])]
    for m, v in zip(("EMK", "Plug", "Obs", "ObsPSD"), dv): cols["def_" + m].append(v)
    cols["frac_undef"].append(fu)
    drows.append(f"{lab} & {100*fu:.1f} & " + " & ".join(b(dv, "{:.4f}")) + r" \\")
    # external variance: pairs with at least one incomplete record
    for m in VM:
        for f in ("var_ratio", "cover90", "dss", "brier0.9", "floor_frac"): vagg[(n, m, f)] = [c["_var"][m][f] for c in C]
    vagg[(n, "MeanOnly", "brier0.9")] = [c["_var"]["EMK_mean_only"]["brier0.9"] for c in C]
    g = lambda m, f: np.mean(vagg[(n, m, f)])
    vrows.append(f"{lab} & " + " & ".join(f"{g(m,'var_ratio'):.2f}" for m in VM) + " & " + " & ".join(f"{g(m,'cover90'):.2f}" for m in VM)
                 + " & " + " & ".join(b([g(m, "brier0.9") for m in VM] + [g("MeanOnly", "brier0.9")], "{:.3f}")) + r" \\")
rows.append(r"\midrule")
rows.append("Mean & " + " & ".join(b([np.mean(cols[m]) for m in MR + ("ObsPSD",)], "{:.4f}")) + r" \\")
out["real"] = "\n".join(rows)
E = np.array(cols["EMK-cos"])
out["real_red"] = {m: round(100 * (1 - E / np.array(cols[m])).mean(), 1) for m in MR[1:] + ("ObsPSD",)}
ea = np.array(runs_all["EMK-cos"])
out["real_wins"] = {m: int((ea < np.array(runs_all[m])).sum()) for m in MR[1:]}
out["real_nruns"] = len(ea)
out["real_paired_diff"] = {m: dict(mean=float(np.mean(np.array(runs_all[m]) - ea)), se=float(np.std(np.array(runs_all[m]) - ea, ddof=1) / np.sqrt(len(ea)))) for m in MR[1:]}
out["real_se_emk_per_set"] = [round(float(np.std([json.loads(l)["res"]["EMK-cos"]["rmse"] for l in open(f"res_cmp_{n}.jsonl")], ddof=1) / 3), 4) for n, _ in names]
drows.append(r"\midrule")
drows.append(f"Mean & {100*np.mean(cols['frac_undef']):.1f} & " + " & ".join(b([np.mean(cols['def_' + m]) for m in ('EMK', 'Plug', 'Obs', 'ObsPSD')], "{:.4f}")) + r" \\")
out["defined"] = "\n".join(drows)
de = np.array(cols["def_EMK"])
out["defined_red"] = {m: round(100 * (1 - de / np.array(cols["def_" + m])).mean(), 1) for m in ("Plug", "Obs", "ObsPSD")}
G = lambda m, f: np.mean([np.mean(vagg[(n, m, f)]) for n, _ in names])
vrows.append(r"\midrule")
vrows.append("Mean & " + " & ".join(f"{G(m,'var_ratio'):.2f}" for m in VM) + " & " + " & ".join(f"{G(m,'cover90'):.2f}" for m in VM)
             + " & " + " & ".join(b([G(m, "brier0.9") for m in VM] + [G("MeanOnly", "brier0.9")], "{:.3f}")) + r" \\")
out["extvar"] = "\n".join(vrows)
out["extvar_dss"] = {m: round(G(m, "dss"), 3) for m in VM}
out["extvar_floor_max"] = max(max(vagg[(n, m, "floor_frac")]) for n, _ in names for m in VM)
out["extvar_brier_wins"] = {m: int(sum(np.mean(vagg[(n, "EMK", "brier0.9")]) < np.mean(vagg[(n, m, "brier0.9")]) for n, _ in names)) for m in VM[1:] + ("MeanOnly",)}

# feature-scaling robustness (five min-max sets, 50% missing)
try:
    zr = []
    for n, lab in names[:5]:
        Z = [json.loads(l)["res"] for l in open(f"res_zscore_{n}.jsonl")]
        z = lambda sc, m: np.mean([r[sc]["rmse"][m] for r in Z])
        zr.append(f"{lab} & {np.mean([r['minmax']['target_sd'] for r in Z]):.2f} & {np.mean([r['zscore']['target_sd'] for r in Z]):.2f} & "
                  + " & ".join(b([z("zscore", m) for m in ("EMK-cos", "RatioMeans", "RatioComp", "PN-Taylor", "Plug-in", "MeanImp", "ObsSubset")], "{:.4f}")) + r" \\")
    out["zscore"] = "\n".join(zr)
except FileNotFoundError:
    pass

# cost of mean and variance (tab:cost2) and of the expected direction (tab:cost)
def sci(x):
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
C2 = json.load(open("res_cost2.json")); rows = []
for r in C2:
    M = r["methods"]; ms = lambda k: f"{M[k]['ms_input']:.2g}" if M[k]['ms_input'] < 10 else f"{M[k]['ms_input']:.0f}"
    rows.append(f"{r['D']} & " + " & ".join(f"{ms(k)} & {sci(M[k]['rmse_sd'])}" for k in ("EMK-cos", "PN-Taylor", "MC-1000", "MC-10000"))
                + f" & {M['RatioMoments']['us_pair']:.2g} & {sci(M['RatioMoments']['rmse_sd'])} & {M['EMK-cos']['us_pair']:.2g}" + r" \\")
out["cost2"] = "\n".join(rows)
Q = json.load(open("res_quad_check.json")); rows = []
for r in Q["cost"]:
    rows.append(f"{r['D']} & {r['emk_time_ms']:.2g} & {sci(max(r['emk_err'],1e-16))} & " + " & ".join(f"{r[f'mc{n}_time_ms']:.3g} & {sci(r[f'mc{n}_err'])}" for n in (1000, 10000, 100000)) + r" \\")
out["cost"] = "\n".join(rows)
acc = Q["accuracy"]
out["quad_acc"] = {f"w{w}": dict(nu=max(a[f"nu_rel_w{w}"] for a in acc), G=max(a.get(f"G_rel_w{w}", 0) for a in acc),
                                 nodes=[min(a[f"nodes_w{w}"] for a in acc), max(a[f"nodes_w{w}"] for a in acc)]) for w in ("4", "2", "1")}
out["quad_cases"] = acc

# mechanisms (tab:mech)
MM = ["EMK-cos", "RatioMeans", "Plug-in", "MeanImp", "ObsSubset"]; rows = []
for c, lab in (("mcar", "MCAR"), ("mcar_ind", "MCAR, inductive"), ("mar", "MAR"), ("mnar", "MNAR (self-masking)")):
    per = {m: [np.mean([json.loads(l)["res"][c]["rmse"][m] for l in open(f"res_mech_{n}.jsonl")]) for n, _ in names] for m in MM}
    better = int((np.array(per["EMK-cos"]) < np.array(per["Plug-in"])).sum())
    rows.append(f"{lab} & " + " & ".join(b([np.mean(per[m]) for m in MM], "{:.4f}")) + f" & {better}/8" + r" \\")
out["mech"] = "\n".join(rows)

# PSD counts
P = [json.loads(open(f"res_psd_{s}.jsonl").read())["res"] for s in range(3)]
MP = ("EMK-cos", "RatioMeans", "PN-Taylor", "Plug-in", "MC-vec100", "MC-pair10", "MC-pair100")
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
