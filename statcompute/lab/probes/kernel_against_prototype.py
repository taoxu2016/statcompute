"""Probe A-12: one partition of the cluster's shape through the engine's Cox kernel and through the
prototype's (OPEN_ITEMS OI-35). 50,000 rows, 80 strata, sorted by stratum and time descending as both
working sets are. Run: python kernel_against_prototype.py [FEATURES]"""
import sys, time
import numpy as np
from statcompute.engine import kernels as engine
from statcompute.lab import kernels as lab

p = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rng = np.random.default_rng(7)
n, strata = 50_000, 80
s = np.sort(rng.integers(0, strata, n))
t = np.ceil(rng.exponential(300.0, n)).clip(1, 1825)
order = np.lexsort((-t, s))
s, t = s[order], t[order]
X = rng.integers(0, 4, size=(n, p)).astype(np.float64) / 3.0
d = (rng.random(n) < 0.55).astype(np.float64)
beta = rng.normal(0, 0.05, p)

def best(f, repeat=7):
    out = []
    for _ in range(repeat):
        a = time.perf_counter(); r = f(); out.append(time.perf_counter() - a)
    return min(out) * 1000, r

lab_ms, (ll0, u0, i0) = best(lambda: lab.cox_partial(X, t, d, s, beta, presorted=True))
eng_ms, out = best(lambda: engine.cox_partial(X, t, d, s, beta, ties="breslow", presorted=True))
ll1, u1, i1 = out["loglik"], out["score"], out["information"]
print(f"p={p}: prototype kernel {lab_ms:6.1f} ms | engine kernel {eng_ms:6.1f} ms | ratio {eng_ms / lab_ms:.2f}")
se = 1 / np.sqrt(np.diag(i0))
print(f"   same sums: log-likelihood {abs(ll1 - ll0) / abs(ll0):.1e} relative, score {np.max(np.abs(u1 - u0)):.1e}, information {np.max(np.abs(i1 - i0)) / np.max(np.abs(i0)):.1e} relative")
