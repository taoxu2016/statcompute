"""
Sanity check (sandbox): the rounding-error term of Neumaier's branchy update equals, bit for bit,
the branch-free TwoSum (Knuth) error term, so a branch-free rewrite of the compensated accumulators
would leave every result bit unchanged (finite sums).
"""
import numpy as np
rng = np.random.default_rng(11)

def neumaier_err(s, x):
    t = s + x
    return np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s), t

def twosum_err(s, x):
    t = s + x
    z = t - s
    return (s - (t - z)) + (x - z), t

N = 4_000_000
total = 0; mism = 0
for name, gen in {
    "wide exponents": lambda: (rng.standard_normal(N) * 10.0 ** rng.integers(-150, 150, N),
                               rng.standard_normal(N) * 10.0 ** rng.integers(-150, 150, N)),
    "similar magnitudes": lambda: (rng.standard_normal(N), rng.standard_normal(N)),
    "near cancellation": lambda: (lambda u: (u, -u * (1 + rng.integers(-4, 5, N) * 2.0 ** -52)))(rng.standard_normal(N) * 1e8),
    "subnormals and zeros": lambda: (rng.integers(-3, 4, N) * 5e-324 * rng.integers(0, 2**20, N),
                                     rng.integers(-3, 4, N) * 5e-324 * rng.integers(0, 2**20, N)),
}.items():
    s, x = gen()
    e1, t1 = neumaier_err(s, x); e2, t2 = twosum_err(s, x)
    bad = int((e1.view(np.uint64) != e2.view(np.uint64)).sum())
    total += N; mism += bad
    print(f"{name:22s}: {N:,} pairs, bitwise mismatches in the error term: {bad}")

# running accumulation: whole (sum, compensation) state over a long stream
x = rng.standard_normal(2_000_000) * 10.0 ** rng.integers(-8, 9, 2_000_000)
def run(update):
    s = 0.0; c = 0.0
    for v in x[:300_000]:
        e, t = update(s, v); c += e; s = t
    return s, c
import struct
a = run(lambda s, v: ((s - (s + v)) + v if abs(s) >= abs(v) else (v - (s + v)) + s, s + v))
def ts(s, v):
    t = s + v; z = t - s
    return (s - (t - z)) + (v - z), t
b = run(ts)
print("running state identical:", struct.pack('dd', *a) == struct.pack('dd', *b), "| total pairs", f"{total:,}", "| mismatches", mism)
