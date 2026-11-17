# EMK-cos: exact expected cosine kernels for random vectors

Code, data and stored results for the paper *Exact expected cosine kernels for random vectors*.

For a Gaussian vector `x ~ N(m, S)` with any covariance `S ⪰ 0` (including singular covariance), the expected direction

    nu(m, S) = E[x/||x||] = pi^{-1/2} ∫_0^∞ t^{-1/2} L(t) (I + 2tS)^{-1} m dt,
    L(t) = det(I + 2tS)^{-1/2} exp(-t m^T (I + 2tS)^{-1} m),

is a one-dimensional integral. For independent random inputs the expected cosine is `nu_i^T nu_j`, and the matrix
`K = V V^T + diag(1 - ||nu_i||^2)` is the expected Gram matrix (positive semidefinite, unit diagonal).

## Repository layout

| Folder | Contents |
| --- | --- |
| `experiments/` | Library (`emk_cos.py`) and one script per experiment (`exp_*.py`) with its summary (`sum_*.py`) |
| `data/` | Data files not available in scikit-learn (see `data/README.md`) |
| `results/` | Result files used for the tables and figures of the paper |

The core library is `experiments/emk_cos.py`:

| Function | Purpose |
| --- | --- |
| `nu_gauss(m, S)` | Expected direction `E[x/||x||]` of `N(m, S)` (Theorem 1, first integral), composite Gauss–Legendre quadrature in `log t`, 528 nodes |
| `m2_gauss(m, S)` | Second angular moment `E[xx^T/||x||^2]` (Theorem 1, second integral) |
| `nu_mixture`, `m2_mixture` | Same quantities for Gaussian mixtures, optionally after a linear map |
| `nu_taylor2(m, S)` | Second-order Taylor (delta-method) baseline |
| `em_gmm_missing`, `conditional_mixture` | GMM fitted by EM with missing values and the conditional law of a record |

Baselines from the literature are in `baselines.py`: the PN-Taylor approximation of Herrera-Esposito & Burge (2025) (`nu_pn2025`), the ratio of means and moment-ratio variance of Baumann et al. (2024) (`ratio_mean`, `ratio_var`), the per-component ratio of means used as a control (`ratio_comp`) and the nearest correlation matrix used for the PSD correction (`nearest_correlation`). `make_tables.py` prints the LaTeX rows of every table from the stored results and writes `results/tables.json`.

Other helpers: `em_struct.py` (EM with full, diagonal or low-rank covariance), `exp_knn.py` (k-NN imputation and k-NN mixtures),
`exp_real2.py` (data loaders and evaluation helpers), `vae_np.py` (NumPy variational autoencoder).

## Requirements

Python 3.10 or later with NumPy, SciPy and scikit-learn (`pip install -r requirements.txt`). The results were produced with
Python 3.13, NumPy 2.5, SciPy 1.18 and scikit-learn 1.9. Everything runs on a CPU.

## Running

Run every script **from the `results/` folder**, so that result files are written and read there:

```bash
cd results
python ../experiments/sum_vae.py          # rebuild a table from the stored results
python ../experiments/exp_vae.py          # or rerun the experiment (overwrites res_vae.jsonl)
```

`run_all.sh` lists the commands for every experiment. Scripts that use the UCI or projected data sets read them from
`data/` (or from the folder given in the environment variable `EMKCOS_DATA`); the other data sets are downloaded by scikit-learn.

## Paper elements and scripts

| Paper element | Experiment | Summary | Results |
| --- | --- | --- | --- |
| Table "Exactness against Monte Carlo" (Appendix B) | `checks.py` (prints) | — | — |
| Quadrature accuracy and Table "Time per input" (Appendix B) | `quad_check.py` | (prints) | `res_quad_check.json` |
| Figure "Error against Monte Carlo as uncertainty grows" (Appendix B) | `exp_snr.py` | (prints) | `res_snr.json` |
| Synthetic GMM, multimodal laws (Table with norms 6 and 6/ρ, Appendix B; s6). s1–s5 (estimated laws, t mixtures, MNAR) are auxiliary and not in the paper | `exp_synth.py s1,s3,s6` and `exp_synth.py s2,s4,s5` | `sum_synth.py` | `res_synth_*.jsonl` |
| VAE posteriors (Table "Fidelity to the expected cosine under the posterior laws", Appendix B) | `exp_vae.py` | `sum_vae.py` | `res_vae.jsonl` |
| Decision thresholds (FAR, AUC) — auxiliary, not in the paper | `exp_thr.py 0 1 2` | `sum_thr.py` | `res_thr_*.jsonl` |
| Downstream tasks on VAE posteriors — auxiliary, not in the paper | `exp_vae_task.py 0.6` and `exp_vae_task.py 1.2` | (prints) | `res_vae_task_*.jsonl` |
| Bayesian posteriors: GP curves and regression weights (fidelity table; true-vector comparison in Appendix B) | `exp_rv.py gp SEED`, `exp_rv.py blr SEED` (seeds 0–2) | `sum_rv.py` | `res_rv_*.jsonl` |
| Prior variance and dependence between inputs (Table "Regression weights, 15 rows", Appendix B) | `exp_prior.py 10` and `exp_prior.py 10 iid` | `sum_prior.py` | `res_prior*.jsonl` |
| Incomplete data, density-model selection | `exp_auto.py NAME N 3 0.3,0.5,0.7` | `sum_auto.py` | `res_auto_*.jsonl` |
| Incomplete data, comparison of estimators (Table "Incomplete data") | `exp_cmp.py NAME N` (needs `res_auto_NAME.jsonl`) | `sum_cmp.py` | `res_cmp_*.jsonl` |
| Covariance structure, BIC vs validation (paragraph "Choice of the density model") | `exp_struct.py`, `exp_select2.py` (`NAME N C 4 0.3,0.5,0.7`) | `sum_select2.py` | `res_struct_*`, `res_select2_*` |
| Missingness mechanisms and inductive protocol (Table "Missingness mechanisms", Appendix B) | `exp_mech.py NAME ...` | `sum_mech.py` | `res_mech_*.jsonl` |
| Error decomposition — auxiliary; only the fitted-law fidelity (first block) is quoted in the paper | `exp_aux.py NAME` | `sum_aux.py` | `res_aux_*.jsonl` |
| Positive semidefiniteness (Table "Kernel matrices ...", Appendix B; the delta-expansion column is auxiliary) | `exp_psd.py SEED` (seeds 0–2) | `sum_psd.py` (also reads `res_cmp_*`) | `res_psd_*.jsonl` |
| Regime map: excess risk for diagonal Gaussians, 2-D laws embedded in D dimensions and mixtures with three spreads, including the per-component ratio of means (Table "Excess squared-error risk") | `exp_regime.py 0` | `make_tables.py` | `res_regime.json` |
| Uncertainty of the cosine: DSS, scale ratio of variances, SD error (Table "Mean and variance forecasts") | `exp_unc.py SEED` (seeds 0–2) | `make_tables.py` | `res_unc_*.jsonl` |
| Cost of mean and variance per input and per pair (Table "Mean and variance of the cosine") | `exp_cost2.py` | (prints) | `res_cost2.json` |
| Observed-subset cosine with nearest-correlation PSD correction (column "Observed+PSD") | `exp_yu.py NAME N` | `make_tables.py` | `res_yu_*.jsonl` |
| Stratification by responsibility entropy (paragraph "Why the moment-based estimators tie here") | `exp_strat.py NAMES` | (prints) | `res_strat_*.jsonl` |
| Kernel ridge regression with the observed-subset cosine (R² = −144) — auxiliary, not in the paper | `exp_diag.py reg NAME N C 5 0.3,0.5,0.7` | `sum_diag.py` | `res_diag_reg_*.jsonl` |

Data-set names, sizes `N` and numbers of mixture components `C` used for the incomplete-data experiments:

| NAME | iris | wine | bc | digits8 | vehicle | oliv_pool8 | oliv_pca40 | news50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 150 | 178 | 300 | 400 | 400 | 400 | 400 | 600 |
| C | 3 | 3 | 3 | 5 | 4 | 5 | 5 | 5 |

## Notes on reproducibility

* Seeds are fixed inside the scripts (data subsample `100 + rep`, missingness mask `1000·rep + 100·rate`), so reruns on the
  same library versions reproduce the stored results up to floating-point differences across platforms.
* The sizes `N` and components `C` of the kernel-ridge-regression runs (`exp_diag.py reg`) were not recorded; this auxiliary
  run is reported in the paper only for the failure of an indefinite matrix.
* Monte Carlo references use 10^5 draws per input (noise ≈ 4×10^-4 on the cosine).
