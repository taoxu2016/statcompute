#!/usr/bin/env bash
# Runs every probe and writes one log per probe under out/. Needs Python 3 with NumPy and a JDK 17+.
cd "$(dirname "$0")" && mkdir -p out
{ python3 --version; python3 -c "import numpy; print('numpy', numpy.__version__)"; java -version 2>&1 | head -1; grep -m1 "model name" /proc/cpuinfo; echo "cores: $(nproc)"; } > out/00_environment.log 2>&1
python3 check_cox_gram.py       > out/A1_cox_gram_identity.log 2>&1
python3 check_tilt.py           > out/A2_tilting.log 2>&1
python3 check_twosum.py         > out/A3_twosum.log 2>&1
java -Xmx1500m GramBench.java 20000 0.30 > out/A4_gram_kernels_density_0.30.log 2>&1
java -Xmx1500m GramBench.java 20000 0.05 > out/A4_gram_kernels_density_0.05.log 2>&1
python3 check_cox_centering.py  > out/A7_cox_centering.log 2>&1
python3 iter_logit.py           > out/A6_logistic_profile.log 2>&1
python3 iter_cox.py             > out/A5a_cox_relative_loglik_rule.log 2>&1
python3 iter_cox2.py            > out/A5b_cox_curvature_variants.log 2>&1
python3 iter_cox3.py            > out/A5c_cox_bfgs_and_warm_start.log 2>&1
echo done > out/zz_done.log
