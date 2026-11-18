"""Recalibration of the predicted variance of the cosine on incomplete data, using only observed entries.

Two factors are estimated after hiding a further 15% of the observed entries and refitting the selected model:
(a) entry level: r = sum (x - E[x|obs])^2 / sum Var[x|obs] over the hidden entries; the conditional covariances of
    the final model are multiplied by r;
(b) cosine level: for random pairs (i, j), the cosine restricted to the coordinates O_ij observed in both records is
    known; its mean and variance are predicted from the marginal of the refitted conditional laws on O_ij (pairs with
    a hidden entry in O_ij). c = sum (y - mu)^2 / sum v, separately for each estimator; the final variances are
    multiplied by c.

For each run, the density model selected in res_auto_NAME.jsonl is refitted after hiding a further 15% of the observed
entries. On the hidden entries, the factor
    r = sum (x - E[x | obs])^2 / sum Var[x | obs]
measures how much the fitted conditional laws understate the imputation error (r = 1 when calibrated). The model
fitted on all observed entries is then used with every conditional covariance multiplied by r, which leaves the
conditional means and responsibilities unchanged and widens each conditional law. Variance forecasts of the
complete-data cosine are scored as in exp_cmp.variance_scores, with and without the factor. The complete data are
used only for scoring."""
import sys, json, time
import numpy as np
import emk_cos as E
from em_struct import em_struct
from exp_real2 import masked, cosm, ridge_from
from exp_cmp import mixture_stats, variance_scores, knn_mixtures


def fit_cond(Xm, sel, reg, rep):
    if sel.startswith("knn"):
        return knn_mixtures(Xm, int(sel.split("_k")[1]), reg)
    _, st, c = sel.split("_"); C = int(c[1:]); s = "lowrank" if st.startswith("lowrank") else st; kk = 3 if s == "lowrank" else 0
    pi, mus, covs = em_struct(Xm, C=C, iters=15, reg=reg, seed=rep, struct=s, k=kk)
    return [E.conditional_mixture(Xm[i], pi, mus, covs) for i in range(len(Xm))]


def calibration_factor(Xm, sel, rep, frac=0.15):
    rng = np.random.default_rng(7000 + rep); obs = ~np.isnan(Xm)
    hide = obs & (rng.random(Xm.shape) < frac)
    hide[(obs & ~hide).sum(1) == 0] = False         # no record loses all of its observed entries
    Xv = np.where(hide, np.nan, Xm)
    cond = fit_cond(Xv, sel, ridge_from(Xv), rep)
    err2 = var = 0.0; nh = 0
    for i, (r, ms, Ss) in enumerate(cond):
        h = np.where(hide[i])[0]
        if not len(h): continue
        mean = (r[:, None] * ms[:, h]).sum(0)
        second = sum(r[c] * (np.diag(Ss[c])[h] + ms[c][h] ** 2) for c in range(len(r)))
        err2 += float(((Xm[i, h] - mean) ** 2).sum()); var += float((second - mean ** 2).sum()); nh += len(h)
    return err2 / var, nh


def cosine_factor(Xm, sel, rep, frac=0.15, npairs=3000):
    rng = np.random.default_rng(7000 + rep); obs = ~np.isnan(Xm)
    hide = obs & (rng.random(Xm.shape) < frac)
    hide[(obs & ~hide).sum(1) == 0] = False
    Xv = np.where(hide, np.nan, Xm)
    cond = fit_cond(Xv, sel, ridge_from(Xv), rep)
    n = len(Xm); rng2 = np.random.default_rng(8000 + rep); acc = {}; used = 0
    for _ in range(20 * npairs):
        if used >= npairs: break
        i, j = rng2.choice(n, 2, replace=False); O = np.where(obs[i] & obs[j])[0]
        if len(O) < 2 or not (hide[i, O].any() or hide[j, O].any()): continue
        a, b = Xm[i, O], Xm[j, O]
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0: continue
        sub = [(r, ms[:, O], Ss[:, O][:, :, O]) for r, ms, Ss in (cond[i], cond[j])]
        F = mixture_stats(sub, len(O)); y = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
        for name, (Nx, Gx) in dict(EMK=(F["nu"], F["G"]), PN=(F["nu_pn"], F["G_pn"]),
                                   Ratio=(F["phi"], F["Gt"]), RatioComp=(F["phi_c"], F["Gt_c"])).items():
            mu = Nx[0] @ Nx[1]; v = max(float(np.sum(Gx[0] * Gx[1]) - mu ** 2), 0.0)
            e, w = acc.get(name, (0.0, 0.0)); acc[name] = (e + (y - mu) ** 2, w + v)
        used += 1
    return {k: e / w for k, (e, w) in acc.items()}, used


def run(name, rate, rep, n, sel):
    X, Xm, mask, _ = masked(name, n, 100 + rep, rate, 1000 * rep + int(rate * 100)); n, D = X.shape
    r_hat, nh = calibration_factor(Xm, sel, rep)
    cond = fit_cond(Xm, sel, ridge_from(Xm), rep)
    Kt = cosm(X); iu = np.triu_indices(n, 1)
    out = dict(r_hat=r_hat, n_hidden=nh)
    out["base"] = variance_scores(mixture_stats(cond, D), mask, Kt, iu)
    cal = [(r, ms, np.array([r_hat * S for S in Ss])) for r, ms, Ss in cond]
    Fc = mixture_stats(cal, D)
    out["cal"] = variance_scores(Fc, mask, Kt, iu)
    mu = (Fc["nu"] @ Fc["nu"].T)[iu]
    out["cal_rmse"] = float(np.sqrt(np.mean((mu - Kt[iu]) ** 2)))   # the factor also changes the expected cosine
    c, used = cosine_factor(Xm, sel, rep)
    out["c_hat"] = c; out["c_pairs"] = used
    out["cos_cal"] = variance_scores(mixture_stats(cond, D), mask, Kt, iu, scale=c)
    return out


if __name__ == "__main__":
    name, n = sys.argv[1], int(sys.argv[2])
    A = [json.loads(l) for l in open(f"res_auto_{name}.jsonl")]
    with open(f"res_recal_{name}.jsonl", "w") as f:
        for a in A:
            sel = min(a["res"], key=lambda k: a["res"][k]["mse"]); t = time.time()
            r = run(name, a["rate"], a["rep"], n, sel)
            f.write(json.dumps(dict(data=name, rate=a["rate"], rep=a["rep"], sel=sel, res=r)) + "\n"); f.flush()
            print(name, a["rate"], a["rep"], sel, round(r["r_hat"], 3), round(time.time() - t, 1), flush=True)
