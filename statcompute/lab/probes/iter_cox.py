"""
Toy experiment (sandbox, NumPy): how many O(n p^2) curvature evaluations does a Cox fit need
if the score stays exact but the curvature is reused or lagged?  Breslow ties, right-censored, unstratified.
All variants converge to the same root of the exact score equation; only the path differs.
"""
import numpy as np
rng = np.random.default_rng(3)

def simulate(n, p):
    X = np.column_stack([(rng.random(n) < 0.3) if j % 2 == 0 else rng.normal(size=n) for j in range(p)]).astype(float)
    X -= X.mean(0)
    beta = np.array([(-1) ** k * 0.5 / np.sqrt(p) * (1 + (k % 3)) for k in range(p)])
    T = (-np.log(rng.random(n)) / (0.1 * np.exp(X @ beta))) ** (1 / 1.2)
    C = rng.random(n) * 5
    time = np.ceil(365 * np.minimum(np.minimum(T, C), 5)).astype(int)
    event = (T <= C) & (T <= 5)
    return X, time, event

class Cox:
    def __init__(self, X, time, event):
        o = np.argsort(-time, kind="stable")
        self.X, self.time, self.event = X[o], time[o], event[o]
        t = self.time
        self.last = np.r_[np.nonzero(np.diff(t))[0], len(t) - 1]        # last row of each tie group (descending time)
        self.grp = np.searchsorted(-t[self.last], -t)                   # group index per row
        self.d = np.bincount(self.grp, weights=self.event, minlength=len(self.last))
        self.ev_x = self.event @ self.X
    def cheap(self, beta):
        """O(n p): log PL, exact score, hazard increments, risk-set means."""
        eta = self.X @ beta; rw = np.exp(eta)
        S0 = np.cumsum(rw)[self.last]
        S1 = np.cumsum(rw[:, None] * self.X, axis=0)[self.last]
        xbar = S1 / S0[:, None]
        ll = eta[self.event].sum() - (self.d * np.log(S0)).sum()
        U = self.ev_x - self.d @ xbar
        h = self.d / S0
        H = np.cumsum(h[::-1])[::-1]                                    # cumulative hazard at each group's time
        M = (xbar * self.d[:, None]).T @ xbar
        return ll, U, rw, H, M
    def gram(self, rw, H):
        """O(n p^2): order-free weighted Gram with weights E_i = rw_i * Lambda0(T_i)."""
        E = rw * H[self.grp]
        return (self.X * E[:, None]).T @ self.X

def fit(c, mode, p, freeze_after=None, eps=1e-9, max_iter=60):
    beta = np.zeros(p); grams = 0; passes = 0
    ll, U, rw, H, M = c.cheap(beta); passes += 1
    H_prev = H                                 # at beta = 0 the lagged hazard is the Nelson-Aalen estimate
    info = None
    for it in range(1, max_iter + 1):
        if mode == "newton":
            info = c.gram(rw, H) - M; grams += 1
        elif mode == "lagged":                 # Gram weights use the previous iterate's hazard: one pass per iteration
            info = c.gram(rw, H_prev) - M; grams += 1
        elif mode == "chord":
            if it <= freeze_after:
                info = c.gram(rw, H) - M; grams += 1
        step = np.linalg.solve(info, U)
        v = 1.0
        while True:
            ll_new, U_new, rw_new, H_new, M_new = c.cheap(beta + v * step); passes += 1
            if ll_new >= ll or v < 1e-4: break
            v *= 0.5
        done = abs(ll_new - ll) <= eps * abs(ll_new)
        beta = beta + v * step
        H_prev = H; ll, U, rw, H, M = ll_new, U_new, rw_new, H_new, M_new
        if done: break
    return beta, it, grams, passes, np.abs(U).max(), ll

for n, p in [(200_000, 20), (200_000, 60)]:
    X, time, event = simulate(n, p)
    c = Cox(X, time, event)
    ref = fit(c, "newton", p, eps=1e-15)[0]
    print(f"\nn = {n:,}, p = {p}, events = {int(event.sum()):,}, distinct event times = {int((c.d > 0).sum())}")
    print(f"{'variant':34s} {'iters':>5s} {'O(np^2) evals':>14s} {'max|score|':>11s} {'max|beta - ref|':>16s}")
    for label, mode, fa in [("Newton (exact curvature each step)", "newton", None),
                            ("lagged-hazard Gram", "lagged", None),
                            ("chord, curvature frozen after 2", "chord", 2),
                            ("chord, curvature frozen after 1", "chord", 1)]:
        b, it, g, ps, sc, ll = fit(c, mode, p, fa)
        print(f"{label:34s} {it:5d} {g:14d} {sc:11.2e} {np.abs(b - ref).max():16.2e}")
