"""
Sanity check (sandbox, toy scale): the Cox information matrix as
    I = X' diag(E) X  -  sum over event-time steps of  m * xbar xbar'
where E_i is row i's expected event count (Efron-correct), against
(a) the classical running-S2 formula and (b) central differences of the score.
Covers strata, left truncation (entry, exit], case weights, offsets, Breslow and Efron ties.
"""
import numpy as np

rng = np.random.default_rng(20261113)

def simulate(n=600, p=4, strata=5):
    X = np.column_stack([rng.integers(0, 2, n) if j % 2 == 0 else np.round(rng.normal(size=n) * 64) / 64
                         for j in range(p)]).astype(float)
    s = rng.integers(0, strata, n)
    exit_ = rng.integers(1, 25, n).astype(float)          # heavy ties on purpose
    entry = np.where(rng.random(n) < 0.4, np.floor(rng.random(n) * exit_), -np.inf)
    event = rng.random(n) < 0.6
    w = rng.integers(1, 4, n).astype(float)
    off = np.round(rng.normal(size=n) * 4) / 16
    return X, s, entry, exit_, event, w, off

def classical(beta, X, s, entry, exit_, event, w, off, efron):
    """log PL, score, information by explicit risk sets (O(n * K), brute force)."""
    n, p = X.shape
    rw = w * np.exp(X @ beta + off)
    ll = 0.0; U = np.zeros(p); I = np.zeros((p, p))
    for st in np.unique(s):
        idx = np.where(s == st)[0]
        for t in np.unique(exit_[idx][event[idx]]):
            R = idx[(entry[idx] < t) & (exit_[idx] >= t)]
            D = idx[(exit_[idx] == t) & event[idx]]
            d = len(D); dw = w[D].sum()
            S0 = rw[R].sum(); S1 = rw[R] @ X[R]; S2 = (X[R] * rw[R, None]).T @ X[R]
            ll += (w[D] * (X[D] @ beta + off[D])).sum(); U += w[D] @ X[D]
            if not efron or d == 1:
                ll -= dw * np.log(S0); xb = S1 / S0
                U -= dw * xb; I += dw * (S2 / S0 - np.outer(xb, xb))
            else:
                T0 = rw[D].sum(); T1 = rw[D] @ X[D]; T2 = (X[D] * rw[D, None]).T @ X[D]
                m = dw / d
                for k in range(1, d + 1):
                    f = k / d; a = (S0 - T0) + f * T0
                    xb = ((S1 - T1) + f * T1) / a
                    ll -= m * np.log(a); U -= m * xb
                    I += m * (((S2 - T2) + f * T2) / a - np.outer(xb, xb))
    return ll, U, I

def gram_form(beta, X, s, entry, exit_, event, w, off, efron):
    """Same quantities; the information as a weighted Gram minus rank-one mean terms.
    Needs only S0 (scalar) and S1 (p-vector) per event time, never S2."""
    n, p = X.shape
    rw = w * np.exp(X @ beta + off)
    E = np.zeros(n)                       # expected events per row
    M = np.zeros((p, p)); U = np.zeros(p)
    for st in np.unique(s):
        idx = np.where(s == st)[0]
        for t in np.unique(exit_[idx][event[idx]]):
            atrisk = (entry[idx] < t) & (exit_[idx] >= t)
            R = idx[atrisk]
            D = idx[(exit_[idx] == t) & event[idx]]
            d = len(D); dw = w[D].sum()
            S0 = rw[R].sum(); S1 = rw[R] @ X[R]
            U += w[D] @ X[D]
            if not efron or d == 1:
                h = dw / S0; xb = S1 / S0
                E[R] += rw[R] * h
                U -= dw * xb; M += dw * np.outer(xb, xb)
            else:
                T0 = rw[D].sum(); T1 = rw[D] @ X[D]; m = dw / d
                h = 0.0; hD = 0.0
                for k in range(1, d + 1):
                    f = k / d; a = (S0 - T0) + f * T0
                    xb = ((S1 - T1) + f * T1) / a
                    h += m / a; hD += m * (1.0 - f) / a
                    U -= m * xb; M += m * np.outer(xb, xb)
                E[R] += rw[R] * h
                E[D] -= rw[D] * hD
    G = (X * E[:, None]).T @ X            # order-free weighted Gram
    return U, G - M, E


if __name__ == "__main__":
    data = simulate()
    X = data[0]; p = X.shape[1]
    beta = np.array([0.3, -0.2, 0.15, 0.1])
    for efron in (False, True):
        ll, U, I = classical(beta, *data, efron)
        U2, I2, E = gram_form(beta, *data, efron)
        # central differences of the classical score
        H = np.zeros((p, p)); eps = 1e-5
        for j in range(p):
            e = np.zeros(p); e[j] = eps
            H[:, j] = -(classical(beta + e, *data, efron)[1] - classical(beta - e, *data, efron)[1]) / (2 * eps)
        scale = np.abs(I).max()
        w = data[5]; event = data[4]
        print(f"{'Efron' if efron else 'Breslow'}: max|I_gram - I_classical|/max|I| = {np.abs(I2 - I).max()/scale:.2e}; "
              f"max|U_gram - U|/max|U| = {np.abs(U2 - U).max()/np.abs(U).max():.2e}; "
              f"max|I_gram - FD|/max|I| = {np.abs(I2 - H).max()/scale:.2e}; "
              f"sum(E) = {E.sum():.10f} vs weighted events = {(w*event).sum():.10f}; min E = {E.min():.3e}")
