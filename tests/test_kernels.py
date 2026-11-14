"""The lab kernels against brute-force definitions."""
import numpy as np
import pytest

from statcompute.lab import kernels
from statcompute.lab.newton import maximize


def brute_cox(X, time, event, strata, beta):
    p = X.shape[1]
    rw = np.exp(X @ beta)
    ll, U, I = 0.0, np.zeros(p), np.zeros((p, p))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        for t in np.unique(time[idx][event[idx] == 1]):
            risk = idx[time[idx] >= t]
            dead = idx[(time[idx] == t) & (event[idx] == 1)]
            s0, s1 = rw[risk].sum(), rw[risk] @ X[risk]
            s2 = (X[risk] * rw[risk, None]).T @ X[risk]
            xbar = s1 / s0
            ll += (X[dead] @ beta).sum() - len(dead) * np.log(s0)
            U += X[dead].sum(0) - len(dead) * xbar
            I += len(dead) * (s2 / s0 - np.outer(xbar, xbar))
    return ll, U, I


def cox_data(n=500, p=3, strata=7, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, p)), rng.integers(1, 16, n).astype(float), rng.integers(0, 2, n),
            rng.integers(0, strata, n), rng.normal(size=p) * 0.3)


def test_cox_partial_matches_explicit_risk_sets():
    X, time, event, strata, beta = cox_data()
    ll, U, I = kernels.cox_partial(X, time, event, strata, beta)
    ll0, U0, I0 = brute_cox(X, time, event, strata, beta)
    assert abs(ll - ll0) <= 1e-10 * abs(ll0)
    assert np.max(np.abs(U - U0)) <= 1e-10 * np.max(np.abs(U0))
    assert np.max(np.abs(I - I0)) <= 1e-10 * np.max(np.abs(I0))


def test_cox_partial_adds_over_blocks_of_whole_strata():
    X, time, event, strata, beta = cox_data(seed=2)
    order = np.argsort(strata, kind="stable")
    X, time, event, strata = X[order], time[order], event[order], strata[order]
    whole = kernels.cox_partial(X, time, event, strata, beta)
    blocks = kernels.whole_group_blocks(strata, 120)
    assert len(blocks) > 1 and blocks[0][0] == 0 and blocks[-1][1] == len(strata)
    parts = [kernels.cox_partial(X[a:b], time[a:b], event[a:b], strata[a:b], beta) for a, b in blocks]
    assert abs(sum(part[0] for part in parts) - whole[0]) <= 1e-10 * abs(whole[0])
    assert np.allclose(sum(part[2] for part in parts), whole[2], rtol=1e-10, atol=0)


def logistic_data(n=4000, m=15, p=3, seed=3):
    rng = np.random.default_rng(seed)
    provider = np.sort(rng.integers(0, m, n))
    X = rng.normal(size=(n, p)) + 0.5 * rng.normal(size=(m, p))[provider]
    eta = -0.5 + 0.4 * rng.normal(size=m)[provider] + X @ np.array([0.5, -0.3, 0.2])
    return X, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float), provider


def test_logistic_profile_reaches_the_joint_maximum():
    X, y, provider = logistic_data()
    n, p = X.shape
    fit = maximize(lambda b: kernels.logistic_profile_partial(X, y, provider, b, presorted=True)[:3], p, tol=1e-11)
    Z = np.hstack([np.eye(provider.max() + 1)[provider], X])           # one dummy per provider
    theta = np.zeros(Z.shape[1])
    for _ in range(50):
        pi = 1 / (1 + np.exp(-(Z @ theta)))
        step = np.linalg.solve((Z * (pi * (1 - pi))[:, None]).T @ Z, Z.T @ (y - pi))
        theta += step
        if np.max(np.abs(step)) < 1e-13:
            break
    joint_se = np.sqrt(np.diag(np.linalg.inv((Z * (pi * (1 - pi))[:, None]).T @ Z)))[-p:]
    assert fit["converged"] and np.max(np.abs(fit["beta"] - theta[-p:])) < 1e-10
    assert np.max(np.abs(fit["se"] / joint_se - 1)) < 1e-8


def test_logistic_profile_gradient_is_the_derivative_of_the_profile_likelihood():
    X, y, provider = logistic_data(seed=4)
    beta = np.array([0.2, -0.1, 0.05])
    value = lambda b: kernels.logistic_profile_partial(X, y, provider, b, presorted=True)[0]  # noqa: E731
    gradient = kernels.logistic_profile_partial(X, y, provider, beta, presorted=True)[1]
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1e-5
        assert abs((value(beta + e) - value(beta - e)) / 2e-5 - gradient[k]) < 1e-5 * max(1.0, abs(gradient[k]))


def test_a_provider_without_events_is_refused():
    X, y, provider = logistic_data(seed=5)
    y[provider == 0] = 0.0
    with pytest.raises(kernels.DegenerateProviders):
        kernels.logistic_profile_partial(X, y, provider, np.zeros(3), presorted=True)
