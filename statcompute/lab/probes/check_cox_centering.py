"""
Probe A-7 (sandbox, toy scale): cancellation in the Cox information when a covariate's stratum means are
spread much more widely than its values within a stratum, and the effect of centring covariates per stratum.
The partial likelihood, score and information are exactly invariant to adding a stratum-specific constant
to a covariate, so every variant below estimates the same matrix; they differ only in rounding.
  (a) classical running-sum form, covariates centred at their overall means (today's convention)
  (b) weighted-Gram form, covariates centred at their overall means
  (c) weighted-Gram form, covariates centred per stratum
Reference: the classical form on per-stratum-centred covariates.
"""
import numpy as np
from check_cox_gram import simulate, classical, gram_form

beta = np.array([0.3, 0.02, 0.15, 0.1])
print(f"{'spread of stratum means / within-stratum SD':>44s} {'(a) sweep, overall':>20s} {'(b) Gram, overall':>19s} {'(c) Gram, per stratum':>22s}")
for tau in (0.0, 1.0, 10.0, 100.0, 1000.0):
    rng = np.random.default_rng(20261113)
    X, s, entry, exit_, event, w, off = simulate(n=3000, p=4, strata=30)
    X = X.copy()
    X[:, 1] += tau * rng.normal(size=30)[s]                     # stratum-level shifts of one covariate
    means = np.vstack([X[s == k].mean(0) for k in range(30)])
    Xs = X - means[s]                                           # centred per stratum
    Xo = X - X.mean(0)                                          # centred overall
    for efron in (False,):
        ref = classical(beta, Xs, s, entry, exit_, event, w, off, efron)[2]
        scale = np.abs(ref).max()
        ea = np.abs(classical(beta, Xo, s, entry, exit_, event, w, off, efron)[2] - ref).max() / scale
        eb = np.abs(gram_form(beta, Xo, s, entry, exit_, event, w, off, efron)[1] - ref).max() / scale
        ec = np.abs(gram_form(beta, Xs, s, entry, exit_, event, w, off, efron)[1] - ref).max() / scale
        print(f"{tau:44.0f} {ea:20.1e} {eb:19.1e} {ec:22.1e}")
