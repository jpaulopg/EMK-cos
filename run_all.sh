#!/usr/bin/env bash
# Commands for every experiment of the paper. Run from the repository root; outputs go to results/.
# Several experiments take minutes to an hour on one CPU core; run the lines you need.
set -e
cd "$(dirname "$0")/results"
E=../experiments

# Numerics (Appendix B)
python $E/checks.py
python $E/quad_check.py

# Synthetic data: regime map (main text) and multimodal conditional laws (Appendix B)
python $E/exp_regime.py 0
python $E/exp_synth.py s6

# Posterior laws: fidelity (Appendix B), uncertainty and cost (main text)
python $E/exp_vae.py
for s in 0 1 2; do python $E/exp_rv.py gp $s; python $E/exp_rv.py blr $s; done
python $E/exp_prior.py 10
python $E/exp_prior.py 10 iid
for s in 0 1 2; do python $E/exp_unc.py $s; done
python $E/exp_cost2.py

# Incomplete data (N and C per data set, see README)
declare -A N=([iris]=150 [wine]=178 [bc]=300 [digits8]=400 [vehicle]=400 [oliv_pool8]=400 [oliv_pca40]=400 [news50]=600)
declare -A C=([iris]=3 [wine]=3 [bc]=3 [digits8]=5 [vehicle]=4 [oliv_pool8]=5 [oliv_pca40]=5 [news50]=5)
ALL="iris wine bc digits8 vehicle oliv_pool8 oliv_pca40 news50"
for d in $ALL; do
  python $E/exp_auto.py $d ${N[$d]} 3 0.3,0.5,0.7
  python $E/exp_cmp.py $d ${N[$d]}
  python $E/exp_yu.py $d ${N[$d]}
  python $E/exp_select.py $d ${N[$d]} ${C[$d]} 4 0.3,0.5,0.7
  python $E/exp_select2.py $d ${N[$d]} ${C[$d]} 4 0.3,0.5,0.7
  python $E/exp_aux.py $d
done
python $E/exp_mech.py $ALL
python $E/exp_strat.py $ALL

# Positive semidefiniteness (Appendix B)
for s in 0 1 2; do python $E/exp_psd.py $s; done

# Summaries and LaTeX rows
for f in sum_vae sum_rv sum_prior sum_auto sum_cmp sum_select2 sum_mech sum_aux sum_psd; do
  echo "== $f"; python $E/$f.py
done
python $E/make_tables.py
