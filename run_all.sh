#!/usr/bin/env bash
# Commands for every experiment of the paper. Run from the repository root; outputs go to results/.
# Several experiments take minutes to an hour on one CPU core; run the lines you need.
set -e
cd "$(dirname "$0")/results"
E=../experiments

# Synthetic data and numerics
python $E/checks.py
python $E/quad_check.py
python $E/exp_snr.py
python $E/exp_synth.py s1,s3,s6
python $E/exp_synth.py s2,s4,s5

# Application 1: VAE posteriors
python $E/exp_vae.py
python $E/exp_thr.py 0 1 2
python $E/exp_vae_task.py 0.6
python $E/exp_vae_task.py 1.2

# Application 2: Bayesian posteriors
for s in 0 1 2; do python $E/exp_rv.py gp $s; python $E/exp_rv.py blr $s; done
python $E/exp_prior.py 10
python $E/exp_prior.py 10 iid

# Application 3: incomplete data (N and C per data set, see README)
declare -A N=([iris]=150 [wine]=178 [bc]=300 [digits8]=400 [vehicle]=400 [oliv_pool8]=400 [oliv_pca40]=400 [news50]=600)
declare -A C=([iris]=3 [wine]=3 [bc]=3 [digits8]=5 [vehicle]=4 [oliv_pool8]=5 [oliv_pca40]=5 [news50]=5)
for d in iris wine bc digits8 vehicle oliv_pool8 oliv_pca40 news50; do
  python $E/exp_auto.py $d ${N[$d]} 3 0.3,0.5,0.7
  python $E/exp_cmp.py $d ${N[$d]}
  python $E/exp_struct.py $d ${N[$d]} ${C[$d]} 4 0.3,0.5,0.7
  python $E/exp_select2.py $d ${N[$d]} ${C[$d]} 4 0.3,0.5,0.7
done
python $E/exp_mech.py iris wine bc vehicle digits8 oliv_pool8 oliv_pca40 news50
python $E/exp_aux.py iris; python $E/exp_aux.py wine; python $E/exp_aux.py bc; python $E/exp_aux.py digits8
python $E/exp_aux.py vehicle; python $E/exp_aux.py oliv_pool8; python $E/exp_aux.py oliv_pca40; python $E/exp_aux.py news50

# Positive semidefiniteness
for s in 0 1 2; do python $E/exp_psd.py $s; done

# Summaries
for f in sum_synth sum_vae sum_thr sum_rv sum_prior sum_auto sum_cmp sum_select2 sum_mech sum_aux sum_psd; do
  echo "== $f"; python $E/$f.py
done
