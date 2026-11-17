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
python ../experiments/make_tables.py      # rebuild the LaTeX rows of the tables from the stored results
python ../experiments/exp_unc.py 0        # or rerun an experiment (overwrites res_unc_0.jsonl)
```

`run_all.sh` lists the commands for every experiment. Scripts that use the UCI or projected data sets read them from
`data/` (or from the folder given in the environment variable `EMKCOS_DATA`); the other data sets are downloaded by scikit-learn.

## Paper elements and scripts

| Paper element | Experiment | Summary | Results |
| --- | --- | --- | --- |
| **Main text** | | | |
| Regime map (Table "Excess squared-error risk") | `exp_regime.py 0` | `make_tables.py` | `res_regime.json` |
| Uncertainty of the cosine (Table "Mean and variance forecasts") | `exp_unc.py SEED` (seeds 0–2) | `make_tables.py` | `res_unc_*.jsonl` |
| Cost of mean and variance (Table "Mean and variance of the cosine") | `exp_cost2.py` | (prints) | `res_cost2.json` |
| Incomplete data, density-model selection | `exp_auto.py NAME N 3 0.3,0.5,0.7` | `sum_auto.py` | `res_auto_*.jsonl` |
| Incomplete data, comparison of estimators (Table "Incomplete data") | `exp_cmp.py NAME N` (needs `res_auto_NAME.jsonl`) | `sum_cmp.py`, `make_tables.py` | `res_cmp_*.jsonl` |
| Observed-subset cosine with PSD projection (column "Observed+PSD") | `exp_yu.py NAME N` | `make_tables.py` | `res_yu_*.jsonl` |
| Choice of the density model: BIC vs held-out imputation (paragraph "Choice of the density model") | `exp_select.py` and `exp_select2.py` (`NAME N C 4 0.3,0.5,0.7`) | `sum_select2.py` | `res_select_*`, `res_select2_*` |
| Stratification by responsibility entropy, and fitted-law fidelity (paragraph "Why the moment-based estimators tie here") | `exp_strat.py NAMES`, `exp_aux.py NAME` | (prints), `sum_aux.py` | `res_strat_*.jsonl`, `res_aux_*.jsonl` |
| **Appendix B** | | | |
| Table "Exactness against Monte Carlo" | `checks.py` (prints) | — | — |
| Quadrature accuracy and Table "Time per input" | `quad_check.py` | (prints) | `res_quad_check.json` |
| Fidelity of the mean under posterior laws (VAE, GP curves, regression weights) and comparison with the true vectors | `exp_vae.py`; `exp_rv.py gp SEED`, `exp_rv.py blr SEED` (seeds 0–2) | `sum_vae.py`, `sum_rv.py`, `make_tables.py` | `res_vae.jsonl`, `res_rv_*.jsonl` |
| Prior variance and dependence between inputs (Table "Regression weights, 15 rows") | `exp_prior.py 10` and `exp_prior.py 10 iid` | `sum_prior.py`, `make_tables.py` | `res_prior*.jsonl` |
| Multimodal conditional laws (Table with norms 6 and 6/ρ) | `exp_synth.py s6` | `make_tables.py` | `res_synth_s6.jsonl` |
| Missingness mechanisms and inductive protocol (Table "Missingness mechanisms") | `exp_mech.py NAME ...` | `sum_mech.py` | `res_mech_*.jsonl` |
| Positive semidefiniteness (Table "Kernel matrices ...") | `exp_psd.py SEED` (seeds 0–2) | `sum_psd.py`, `make_tables.py` (also read `res_cmp_*`) | `res_psd_*.jsonl` |

`exp_knn.py` (k-NN imputation and k-NN mixtures) and `exp_select3.py` (GMM posterior means) are used as libraries by
`exp_auto.py` and `exp_cmp.py`; their own command-line runs are not part of the paper.

Data-set names, sizes `N` and numbers of mixture components `C` used for the incomplete-data experiments:

| NAME | iris | wine | bc | digits8 | vehicle | oliv_pool8 | oliv_pca40 | news50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 150 | 178 | 300 | 400 | 400 | 400 | 400 | 600 |
| C | 3 | 3 | 3 | 5 | 4 | 5 | 5 | 5 |

## Notes on reproducibility

* Seeds are fixed inside the scripts (data subsample `100 + rep`, missingness mask `1000·rep + 100·rate`), so reruns on the
  same library versions reproduce the stored results up to floating-point differences across platforms.
* Monte Carlo references use 10^5 draws per input (noise ≈ 4×10^-4 on the cosine).
