"""
Sanity check (sandbox): exact Poisson-binomial tails for a shifted provider effect g,
from ONE recursion at a reference g0 plus exponential tilting, against re-running the recursion at g.
Model: logit pi_i(g) = g + a_i, S = sum of Bernoulli(pi_i(g)).
Identity: P_g(S = s) = P_g0(S = s) * exp((g - g0) s) / prod_i (1 - pi_i(g0) + pi_i(g0) exp(g - g0)).
"""
import numpy as np, time
rng = np.random.default_rng(7)

def pmf_truncated(p, cap):
    pmf = np.zeros(cap + 1); pmf[0] = 1.0; reached = 0
    for pi in p:
        q = 1.0 - pi
        reached = min(reached + 1, cap)
        pmf[1:reached + 1] = pmf[1:reached + 1] * q + pmf[0:reached] * pi
        pmf[0] *= q
    return pmf

def lower_tail_direct(a, g, o):
    p = 1.0 / (1.0 + np.exp(-(g + a)))
    return pmf_truncated(p, o).sum()            # P_g(S <= o)

def make_tilted(a, g0, o):
    p0 = 1.0 / (1.0 + np.exp(-(g0 + a)))
    with np.errstate(divide="ignore"):
        logpmf0 = np.log(pmf_truncated(p0, o))  # one O(n * o) recursion
    s = np.arange(o + 1)
    def lower_tail(g):
        dlt = g - g0
        logM = np.log1p(p0 * np.expm1(dlt)).sum()            # O(n)
        z = logpmf0 + dlt * s - logM                         # O(o)
        zmax = z.max()
        return np.exp(zmax) * np.exp(z - zmax).sum()
    return lower_tail

n = 4000
a = rng.normal(scale=0.8, size=n) - 1.5
g_hat = 0.2
y = rng.random(n) < 1.0 / (1.0 + np.exp(-(g_hat + a)))
o = int(y.sum())
tilt = make_tilted(a, g_hat, o)
worst = 0.0
for g in np.linspace(g_hat - 0.6, g_hat + 0.6, 25):
    d = lower_tail_direct(a, g, o); t = tilt(g)
    worst = max(worst, abs(t - d) / d)
    if abs(g - g_hat) > 0.59 or abs(g - g_hat) < 1e-12:
        print(f"g = {g:+.3f}: direct {d:.15e}  tilted {t:.15e}")
print(f"n = {n}, observed o = {o}; worst relative difference over 25 shifts: {worst:.2e}")

t0 = time.perf_counter(); [lower_tail_direct(a, g, o) for g in np.linspace(-0.4, 0.8, 40)]; t1 = time.perf_counter()
tilt = make_tilted(a, g_hat, o); t2 = time.perf_counter(); [tilt(g) for g in np.linspace(-0.4, 0.8, 40)]; t3 = time.perf_counter()
print(f"40 evaluations: re-running the recursion {t1 - t0:.3f} s; one recursion + 40 tilts {t3 - t1:.3f} s (NumPy, one core)")
