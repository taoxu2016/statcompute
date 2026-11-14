"""
Toy experiment (sandbox, NumPy): logistic regression with one fixed effect per provider.
(1) joint Newton by block elimination (the SerBIN step, full steps, no bound) -- 2 data passes per step in pprof_spark;
(2) Newton on the profile likelihood: each pass solves every provider's gamma_j in place (1-D Newton inside the block),
    then emits the profile score and the Schur complement -- 1 data pass per step;
(3) profile likelihood with the Schur complement evaluated once and BFGS updates afterwards.
All three target the same MLE; stopping rule: Newton decrement of beta below 1e-10.
"""
import numpy as np
rng = np.random.default_rng(17)

def simulate(m=2000, p=20, mean_size=100):
    sizes = np.maximum(20, rng.lognormal(np.log(mean_size) - 0.5, 1.0, m).astype(int))
    prov = np.repeat(np.arange(m), sizes); n = len(prov)
    X = np.column_stack([(rng.random(n) < 0.3) if j % 2 == 0 else rng.normal(size=n) for j in range(p)]).astype(float)
    X += 0.3 * rng.normal(size=(m, p))[prov]                 # case mix differs between providers
    X -= X.mean(0)
    beta = np.array([(-1) ** k * 0.5 / np.sqrt(p) * (1 + (k % 3)) for k in range(p)])
    gamma = -1.0 + 0.5 * rng.normal(size=m)
    y = (rng.random(n) < 1 / (1 + np.exp(-(gamma[prov] + X @ beta)))).astype(float)
    o = np.bincount(prov, weights=y, minlength=m)
    ok = (o > 0) & (o < sizes)                               # screen degenerate providers, as the package does
    keep = ok[prov]; relabel = np.cumsum(ok) - 1
    return X[keep], y[keep], relabel[prov[keep]], int(ok.sum())

def sums(X, y, prov, m, gamma, beta, need_S=True):
    eta = gamma[prov] + X @ beta
    pi = 1 / (1 + np.exp(-eta)); q = pi * (1 - pi); res = y - pi
    ll = (y * eta - np.logaddexp(0, eta)).sum()
    g = np.bincount(prov, weights=res, minlength=m); h = np.bincount(prov, weights=q, minlength=m)
    B = np.zeros((m, X.shape[1])); np.add.at(B, prov, X * q[:, None])
    xbar = B / h[:, None]
    r = res @ X - xbar.T @ g
    S = None
    if need_S:
        S = (X * q[:, None]).T @ X - (xbar * h[:, None]).T @ xbar
    return ll, g, h, xbar, r, S

def joint_newton(X, y, prov, m, tol=1e-10, max_iter=100):
    p = X.shape[1]; beta = np.zeros(p); gamma = np.full(m, np.log(y.mean() / (1 - y.mean())))
    for it in range(1, max_iter + 1):
        ll, g, h, xbar, r, S = sums(X, y, prov, m, gamma, beta)
        db = np.linalg.solve(S, r)
        if (g * g / h).sum() + r @ db <= tol: return beta, gamma, it - 1, it, 2 * (it - 1) + 1
        gamma = gamma + g / h - xbar @ db; beta = beta + db
    return beta, gamma, it, it, 2 * it

def profile_gamma(X, y, prov, m, beta, gamma, inner_tol=1e-13):
    xb = X @ beta; o = np.bincount(prov, weights=y, minlength=m)
    for _ in range(50):                                       # 1-D Newton per provider, in place, no extra data pass
        pi = 1 / (1 + np.exp(-(gamma[prov] + xb)))
        g = o - np.bincount(prov, weights=pi, minlength=m); h = np.bincount(prov, weights=pi * (1 - pi), minlength=m)
        gamma = gamma + g / h
        if np.abs(g / h).max() < inner_tol: break
    return gamma

def profile(X, y, prov, m, exact_steps=None, tol=1e-10, max_iter=100):
    p = X.shape[1]; beta = np.zeros(p); gamma = np.full(m, np.log(y.mean() / (1 - y.mean())))
    gamma = profile_gamma(X, y, prov, m, beta, gamma)
    Sinv = None; evalsS = 0; passes = 0
    ll, g, h, xbar, r, S = sums(X, y, prov, m, gamma, beta, need_S=True); passes += 1; evalsS += 1
    Sinv = np.linalg.inv(S)
    for it in range(1, max_iter + 1):
        db = Sinv @ r
        if r @ db <= tol: return beta, gamma, it - 1, evalsS, passes
        v = 1.0
        while True:
            gamma_new = profile_gamma(X, y, prov, m, beta + v * db, gamma)
            need = exact_steps is None or it < exact_steps
            ll_new, g2, h2, xbar2, r_new, S_new = sums(X, y, prov, m, gamma_new, beta + v * db, need_S=need); passes += 1
            if ll_new >= ll - 1e-12 * abs(ll) or v < 1e-6: break
            v *= 0.5
        s = v * db; yv = r - r_new
        if need: Sinv = np.linalg.inv(S_new); evalsS += 1
        elif s @ yv > 0:
            rho = 1 / (yv @ s); V = np.eye(p) - rho * np.outer(s, yv); Sinv = V @ Sinv @ V.T + rho * np.outer(s, s)
        beta = beta + s; gamma = gamma_new; ll, r = ll_new, r_new
    return beta, gamma, it, evalsS, passes

X, y, prov, m = simulate()
n, p = X.shape
print(f"n = {n:,}, providers = {m:,}, p = {p}, event rate = {y.mean():.3f}")
bj, gj, steps, evS, passes = joint_newton(X, y, prov, m)
print(f"{'variant':46s} {'steps':>5s} {'O(np^2) evals':>14s} {'data passes':>12s} {'max|beta - joint|':>18s} {'max|gamma - joint|':>19s}")
print(f"{'joint Newton, block elimination (SerBIN step)':46s} {steps:5d} {evS:14d} {passes:12d} {'-':>18s} {'-':>19s}")
for label, k in [("profile Newton", None), ("profile, Schur complement x2 then BFGS", 2), ("profile, Schur complement x1 then BFGS", 1)]:
    b, gm, st, ev, ps = profile(X, y, prov, m, exact_steps=k)
    print(f"{label:46s} {st:5d} {ev:14d} {ps:12d} {np.abs(b - bj).max():18.2e} {np.abs(gm - gj).max():19.2e}")
