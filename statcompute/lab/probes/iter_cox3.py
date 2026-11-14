"""
Toy experiment, continued: (a) chord + BFGS secant updates on top of an exact curvature evaluated early;
(b) warm starts from a fit on overlapping data ("last period").  Same path-independent stopping rule.
"""
import numpy as np
exec(open("iter_cox2.py").read().split("rng = np.random.default_rng(5)")[0])
rng = np.random.default_rng(5)
exec("def simulate_hard" + open("iter_cox2.py").read().split("def simulate_hard")[1].split("for n, p, strength")[0])

def fit3(c, p, exact_steps, tol=1e-10, max_iter=200, beta0=None, always_exact=False):
    beta = np.zeros(p) if beta0 is None else beta0.copy(); grams = 0; passes = 0
    ll, U, rw, H, M = c.cheap(beta); passes += 1
    Binv = None
    for it in range(1, max_iter + 1):
        if always_exact or it <= exact_steps:
            Binv = np.linalg.inv(c.gram(rw, H) - M); grams += 1
        step = Binv @ U
        if U @ step <= tol: it -= 1; break
        v = 1.0
        while True:
            ll_new, U_new, rw_new, H_new, M_new = c.cheap(beta + v * step); passes += 1
            if ll_new >= ll - 1e-12 * abs(ll) or v < 1e-6: break
            v *= 0.5
        s = v * step; y = U - U_new                      # gradient of -loglik is -U
        if not always_exact and it >= exact_steps and s @ y > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            rho = 1.0 / (y @ s); V = np.eye(p) - rho * np.outer(s, y)
            Binv = V @ Binv @ V.T + rho * np.outer(s, s)  # BFGS update of the inverse curvature
        beta = beta + s; ll, U, rw, H, M = ll_new, U_new, rw_new, H_new, M_new
    return beta, it, grams, passes, np.abs(U).max()

for n, p, strength in [(200_000, 60, 0.5), (200_000, 60, 2.0)]:
    X, time, event = simulate_hard(n, p, strength)
    c = Cox(X, time, event)
    ref = fit3(c, p, 0, tol=1e-22, always_exact=True)[0]
    print(f"\nn = {n:,}, p = {p}, effect scale {strength}")
    print(f"{'variant':44s} {'steps':>5s} {'O(np^2) evals':>14s} {'O(np) passes':>13s} {'max|beta - ref|':>16s}")
    for label, k, ae in [("Newton", 0, True), ("exact curvature x3, then BFGS updates", 3, False),
                         ("exact curvature x2, then BFGS updates", 2, False), ("exact curvature x1, then BFGS updates", 1, False)]:
        b, it, g, ps, sc = fit3(c, p, k, always_exact=ae)
        print(f"{label:44s} {it:5d} {g:14d} {ps:13d} {np.abs(b - ref).max():16.2e}")
    # warm start: "last period" = 80% of the same rows plus 20% different rows
    keep = rng.random(n) < 0.8
    X2, t2, e2 = simulate_hard(int((~keep).sum()), p, strength)
    X2 = X2 + 0.0
    c_prev = Cox(np.vstack([X[keep], X2]), np.r_[time[keep], t2], np.r_[event[keep], e2])
    prev = fit3(c_prev, p, 0, always_exact=True)[0]
    for label, k, ae in [("warm start from last period: Newton", 0, True),
                         ("warm start: exact curvature x1, then BFGS", 1, False)]:
        b, it, g, ps, sc = fit3(c, p, k, always_exact=ae, beta0=prev)
        print(f"{label:44s} {it:5d} {g:14d} {ps:13d} {np.abs(b - ref).max():16.2e}")
    print(f"   distance of the warm start from the solution: max|beta_prev - ref| = {np.abs(prev - ref).max():.3f}")
