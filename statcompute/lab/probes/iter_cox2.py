"""
Same toy experiment with a path-independent stopping rule: stop when the Newton decrement
U' I^{-1} U (with the curvature in hand) is below 1e-10, i.e. certify the root of the exact score equation.
"""
import numpy as np
exec(open("iter_cox.py").read().split("for n, p in")[0])     # reuse simulate / Cox

def fit2(c, mode, p, freeze_after=None, tol=1e-10, max_iter=200, scale=1.0):
    beta = np.zeros(p); grams = 0; passes = 0
    ll, U, rw, H, M = c.cheap(beta); passes += 1
    H_prev = H; info = None
    for it in range(1, max_iter + 1):
        if mode == "newton" or (mode == "chord" and it <= freeze_after):
            info = c.gram(rw, H) - M; grams += 1
        elif mode == "lagged":
            info = c.gram(rw, H_prev) - M; grams += 1
        step = np.linalg.solve(info, U)
        if U @ step <= tol: it -= 1; break
        v = 1.0
        while True:
            ll_new, U_new, rw_new, H_new, M_new = c.cheap(beta + v * step); passes += 1
            if ll_new >= ll - 1e-12 * abs(ll) or v < 1e-6: break
            v *= 0.5
        beta = beta + v * step
        H_prev = H; ll, U, rw, H, M = ll_new, U_new, rw_new, H_new, M_new
    return beta, it, grams, passes, np.abs(U).max()

rng = np.random.default_rng(5)
def simulate_hard(n, p, strength):
    X = np.column_stack([(rng.random(n) < 0.3) if j % 2 == 0 else rng.normal(size=n) for j in range(p)]).astype(float)
    X -= X.mean(0)
    beta = np.array([(-1) ** k * strength / np.sqrt(p) * (1 + (k % 3)) for k in range(p)])
    T = (-np.log(rng.random(n)) / (0.1 * np.exp(X @ beta))) ** (1 / 1.2)
    C = rng.random(n) * 5
    time = np.ceil(365 * np.minimum(np.minimum(T, C), 5)).astype(int)
    event = (T <= C) & (T <= 5)
    return X, time, event

for n, p, strength in [(200_000, 20, 0.5), (200_000, 60, 0.5), (200_000, 60, 2.0)]:
    X, time, event = simulate_hard(n, p, strength)
    c = Cox(X, time, event)
    ref = fit2(c, "newton", p, tol=1e-22)[0]
    print(f"\nn = {n:,}, p = {p}, effect scale {strength}, events = {int(event.sum()):,}, max|beta| = {np.abs(ref).max():.2f}")
    print(f"{'variant':36s} {'steps':>5s} {'O(np^2) evals':>14s} {'O(np) passes':>13s} {'max|score|':>11s} {'max|beta - ref|':>16s}")
    for label, mode, fa in [("Newton, exact curvature each step", "newton", None),
                            ("lagged-hazard Gram (one pass/step)", "lagged", None),
                            ("chord, curvature frozen after 3", "chord", 3),
                            ("chord, curvature frozen after 2", "chord", 2),
                            ("chord, curvature frozen after 1", "chord", 1)]:
        b, it, g, ps, sc = fit2(c, mode, p, fa)
        print(f"{label:36s} {it:5d} {g:14d} {ps:13d} {sc:11.2e} {np.abs(b - ref).max():16.2e}")
