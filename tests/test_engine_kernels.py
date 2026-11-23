"""The engine's kernels and single-machine fits: against explicit risk sets, and against pprof_py
for every option the engine implements."""
import numpy as np
import pytest

from statcompute.engine import local
from statcompute.engine.kernels import cox_partial

OPTIONS = [(w, o, e) for w in (0, 1) for o in (0, 1) for e in (0, 1)]


def explicit(X, time, event, strata, beta, w, o, entry, ties):
    """Log partial likelihood, score and information from explicit risk sets, one event time at a time."""
    p = X.shape[1]
    eta = X @ beta + o
    r = w * np.exp(eta)
    ll, U, I = 0.0, np.zeros(p), np.zeros((p, p))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        for t in np.unique(time[idx][event[idx] == 1]):
            risk = idx[(time[idx] >= t) & ~(entry[idx] >= t)]
            dead = idx[(time[idx] == t) & (event[idx] == 1)]
            k, total = len(dead), w[dead].sum()
            S0, S1, S2 = r[risk].sum(), r[risk] @ X[risk], (X[risk] * r[risk, None]).T @ X[risk]
            ll += (w[dead] * eta[dead]).sum()
            U += w[dead] @ X[dead]
            if ties == "breslow":
                xbar = S1 / S0
                ll -= total * np.log(S0)
                U -= total * xbar
                I += total * (S2 / S0 - np.outer(xbar, xbar))
            else:
                D0, D1, D2 = r[dead].sum(), r[dead] @ X[dead], (X[dead] * r[dead, None]).T @ X[dead]
                for j in range(k):
                    f = j / k
                    A = S0 - f * D0
                    xbar = (S1 - f * D1) / A
                    ll -= total / k * np.log(A)
                    U -= total / k * xbar
                    I += total / k * ((S2 - f * D2) / A - np.outer(xbar, xbar))
    return ll, U, I


def survival_data(n=700, p=4, strata=9, seed=11):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, strata, n)
    X = rng.normal(size=(n, p)) + 3.0 * rng.normal(size=(strata, p))[s]      # stratum means far apart
    time = rng.integers(2, 20, n).astype(float)
    event = (rng.random(n) < 0.6).astype(int)
    entry = np.where(rng.random(n) < 0.4, np.floor(time * rng.random(n)), np.nan)
    return X, time, event, s, entry, rng.integers(1, 4, n).astype(float), 0.2 * rng.normal(size=n), 0.3 * rng.normal(size=p)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("use_weights,use_offset,use_entry", OPTIONS)
def test_cox_kernel_matches_explicit_risk_sets(ties, use_weights, use_offset, use_entry):
    X, time, event, strata, entry, w, o, beta = survival_data()
    n = len(time)
    out = cox_partial(X, time, event, strata, beta, ties=ties, weights=w if use_weights else None,
                      offset=o if use_offset else None, entry=entry if use_entry else None)
    ll, U, I = explicit(X, time, event, strata, beta, w if use_weights else np.ones(n), o if use_offset else np.zeros(n),
                        entry if use_entry else np.full(n, np.nan), ties)
    assert abs(out["loglik"] - ll) <= 1e-12 * abs(ll)
    assert np.max(np.abs(out["score"] - U)) <= 1e-11 * np.max(np.abs(U))
    assert np.max(np.abs(out["information"] - I)) <= 1e-10 * np.max(np.abs(I))
    assert out["deaths"] == int(event.sum())


def test_cox_kernel_adds_over_blocks_and_ignores_row_order():
    X, time, event, strata, entry, w, o, beta = survival_data(seed=12)
    whole = cox_partial(X, time, event, strata, beta, ties="efron", weights=w, offset=o, entry=entry)
    shuffled = np.random.default_rng(0).permutation(len(time))
    again = cox_partial(X[shuffled], time[shuffled], event[shuffled], strata[shuffled], beta, ties="efron",
                        weights=w[shuffled], offset=o[shuffled], entry=entry[shuffled])
    assert np.allclose(again["information"], whole["information"], rtol=1e-11, atol=0)
    low = strata < 4
    parts = [cox_partial(X[m], time[m], event[m], strata[m], beta, ties="efron", weights=w[m], offset=o[m], entry=entry[m])
             for m in (low, ~low)]
    assert abs(parts[0]["loglik"] + parts[1]["loglik"] - whole["loglik"]) <= 1e-12 * abs(whole["loglik"])
    assert np.allclose(parts[0]["information"] + parts[1]["information"], whole["information"], rtol=1e-11, atol=0)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_rows_given_in_the_kernels_order_need_no_sort_and_give_the_same_bits(ties):
    """Round 12: the working set is stored sorted, so no pass sorts or copies it again."""
    X, time, event, strata, entry, w, o, beta = survival_data(seed=13)
    key = np.random.default_rng(1).permutation(len(time)).astype(float)
    more = dict(baseline=True, rows=True, robust=ties == "breslow")
    sorted_here = cox_partial(X, time, event, strata, beta, ties=ties, weights=w, offset=o, entry=entry, order=key, **more)
    perm = np.lexsort((key, -time, strata))
    given = cox_partial(X[perm], time[perm], event[perm], strata[perm], beta, ties=ties, weights=w[perm], offset=o[perm],
                        entry=entry[perm], presorted=True, **more)
    for name in ("loglik", "score", "information", "events", "deaths") + (("meat",) if ties == "breslow" else ()):
        assert np.array_equal(sorted_here[name], given[name]), name
    assert np.array_equal(sorted_here["expected"][perm], given["expected"])
    assert all(np.array_equal(sorted_here["baseline"][k], given["baseline"][k]) for k in ("stratum", "time", "increment"))
    # the strata may lie in any arrangement and carry any labels: the same sums, to rounding
    other = np.lexsort((key, -time, -strata))
    labels = np.array([f"unit {s}" for s in strata], dtype=object)
    relaid = cox_partial(X[other], time[other], event[other], labels[other], beta, ties=ties, weights=w[other],
                         offset=o[other], entry=entry[other], presorted=True)
    assert abs(relaid["loglik"] - sorted_here["loglik"]) <= 1e-12 * abs(sorted_here["loglik"])
    assert np.max(np.abs(relaid["information"] - sorted_here["information"])) <= 1e-11 * np.max(np.abs(sorted_here["information"]))


def test_the_information_survives_strata_whose_means_are_far_apart():
    """Roadmap F-4: without centring within strata the weighted-Gram form loses digits."""
    rng = np.random.default_rng(3)
    n, p = 3000, 3
    strata = rng.integers(0, 30, n)
    time, event = rng.integers(1, 40, n).astype(float), (rng.random(n) < 0.7).astype(int)
    Z = rng.normal(size=(n, p))
    beta = np.array([0.3, -0.2, 0.1])
    reference = cox_partial(Z, time, event, strata, beta)["information"]
    X = Z.copy()
    X[:, 0] += 1000.0 * rng.normal(size=30)[strata]                # the likelihood does not depend on this shift
    shifted = cox_partial(X, time, event, strata, beta)["information"]
    assert np.max(np.abs(shifted - reference)) <= 1e-9 * np.max(np.abs(reference))


# ---------------------------------------------------------------- against pprof_py, the validated reference
@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("use_weights,use_offset,use_entry", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
def test_cox_fit_agrees_with_pprof_py(ties, use_weights, use_offset, use_entry):
    pprof_py = pytest.importorskip("pprof_py")
    X, time, event, strata, entry, w, o, _ = survival_data(n=4000, strata=25, seed=21)
    ours = local.cox_fit(X, time, event, strata, ties=ties, weights=w if use_weights else None,
                         offset=o if use_offset else None, entry=entry if use_entry else None, tol=1e-11)
    theirs = pprof_py.CoxPH(ties=ties, eps=1e-13, max_iter=50)
    arguments = {"event": event, "strata": strata}
    if use_entry:
        arguments.update(start=np.where(np.isnan(entry), 0.0, entry), stop=time)
    else:
        arguments["duration"] = time
    if use_weights:
        arguments["sample_weight"] = w
    if use_offset:
        arguments["offset"] = o
    theirs.fit(X, **arguments)
    se = np.asarray(theirs.standard_errors_)
    assert ours["converged"] and ours["newton_step_in_se"] <= 1e-11
    assert np.max(np.abs(ours["se"] / se - 1)) < 1e-7
    assert abs(ours["loglik"] - theirs.log_likelihood_) <= 1e-9 * abs(theirs.log_likelihood_)
    # pprof_py stops on the change in the log-likelihood, which can leave it a little short of the
    # root. So the claim tested is the sharp one: the two estimates differ by exactly the Newton step
    # that remains at pprof_py's estimate (the roadmap's T_CONV measure), and that step is small.
    at_theirs = cox_partial(X, time, event, strata, np.asarray(theirs.coef_, dtype=float), ties=ties,
                            weights=w if use_weights else None, offset=o if use_offset else None,
                            entry=entry if use_entry else None)
    remaining = np.linalg.solve(at_theirs["information"], at_theirs["score"])
    assert np.max(np.abs(remaining) / se) < 1e-5
    assert np.max(np.abs(ours["beta"] - (theirs.coef_ + remaining)) / se) < 1e-9


def logistic_data(seed=5, n=30000, p=4, m=30):
    rng = np.random.default_rng(seed)
    provider = np.sort(rng.integers(0, m, n))
    X = rng.normal(size=(n, p))
    eta = -1 + 0.5 * rng.normal(size=m)[provider] + X @ np.array([0.4, -0.3, 0.2, 0.1])
    return X, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float), provider


@pytest.mark.parametrize("degenerate", [False, True])
def test_logistic_fit_agrees_with_pprof_py_also_with_providers_at_the_bound(degenerate):
    pprof_py = pytest.importorskip("pprof_py")
    X, y, provider = logistic_data()
    if degenerate:
        y[np.isin(provider, [3, 7])] = 0.0
        y[provider == 11] = 1.0
    ours = local.logistic_fit(X, y, provider, tol=1e-11)
    theirs = pprof_py.LogisticFixedEffectModel()
    theirs.fit(X, y, provider, use_dataprep=False, tol=1e-12, bound=10.0, backtrack=True)
    beta, gamma = np.ravel(theirs.coefficients_["beta"]), np.ravel(theirs.coefficients_["gamma"])
    se = np.sqrt(np.diag(np.asarray(theirs.variances_["beta"])))
    assert np.max(np.abs(ours["beta"] - beta) / se) < 1e-9 and np.max(np.abs(ours["se"] / se - 1)) < 1e-9
    assert np.max(np.abs(ours["gamma"] - gamma)) < 1e-9
    assert int(ours["bounded"].sum()) == (3 if degenerate else 0)
    if degenerate:
        assert np.allclose(ours["gamma"][[3, 7]], ours["median_gamma"] - 10.0, rtol=0, atol=1e-12)
        assert abs(ours["gamma"][11] - (ours["median_gamma"] + 10.0)) < 1e-12
    loglik = (2 * (X.shape[1] + len(gamma)) - theirs.aic_) / 2
    assert abs(ours["loglik"] - loglik) <= 1e-9 * abs(loglik)


# ---------------------------------------------------------------- round 27: the same sums in less memory
def partition(n, strata, p, seed, every_time_an_event=False, far_apart=False):
    """Rows as a partition of the working set holds them: whole strata, sorted by stratum and time descending."""
    rng = np.random.default_rng(seed)
    s = np.sort(rng.integers(0, strata, n))
    t = np.ceil(rng.exponential(60.0, n)).clip(1, 200)
    order = np.lexsort((-t, s))
    s, t = s[order], t[order]
    X = rng.normal(size=(n, p)) + (5.0 * rng.normal(size=strata)[s][:, None] if far_apart else 0.0)
    d = np.ones(n) if every_time_an_event else (rng.random(n) < 0.55).astype(np.float64)
    entry = np.where(rng.random(n) < 0.3, np.maximum(t - rng.integers(1, 30, n), 0.0), -np.inf)
    return X, t, d, s, rng.normal(0, 0.2, p), rng.uniform(0.5, 2.0, n), rng.normal(0, 0.1, n), entry


def same_bits(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if isinstance(a[name], dict):
            same_bits(a[name], b[name])
        elif isinstance(a[name], np.ndarray):
            assert a[name].dtype == b[name].dtype and a[name].tobytes() == b[name].tobytes(), name
        else:
            assert a[name] == b[name], name


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("shape", [(3000, 40, 5, False, False), (3000, 1500, 3, False, True), (2500, 9, 4, True, False), (7, 1, 2, False, False)])
def test_the_kernel_adds_up_the_bits_it_added_up_before_it_reused_its_arrays(ties, shape):
    """Round 27 (OPEN_ITEMS OI-47): up to version 0.24.0 the kernel made eleven arrays the size of the rows in every call;
    it now makes three and reuses them. The arithmetic is the same operation for operation, so every output is the same bit
    for bit as the kernel kept in lab/cox_kernel_0_24.py: with few large strata (centred stratum by stratum) and with many
    small ones (centred at once), with an event at every time, and with every option."""
    from statcompute.lab.cox_kernel_0_24 import cox_partial_0_24

    n, strata, p, every, far = shape
    X, t, d, s, beta, w, o, entry = partition(n, strata, p, seed=n + p, every_time_an_event=every, far_apart=far)
    for weights, offset, late, extras in ((None, None, None, 0), (w, o, None, 1), (None, None, entry, 2), (w, o, entry, 3)):
        more = dict(ties=ties, weights=weights, offset=offset, entry=late, baseline=extras in (1, 3), rows=extras in (2, 3),
                    robust=extras == 3 and ties == "breslow")
        before = cox_partial_0_24(X, t, d, s, beta, presorted=True, **more)
        kept = X.copy()
        same_bits(cox_partial(X, t, d, s, beta, presorted=True, **more), before)
        assert X.tobytes() == kept.tobytes()                       # rows the kernel was not given stay as they were
        mine = X.copy()
        same_bits(cox_partial(mine, t, d, s, beta, presorted=True, overwrite=True, **more), before)
        centred = X - (np.add.reduceat(X, np.flatnonzero(np.r_[True, s[1:] != s[:-1]]), axis=0)
                       / np.bincount(s - s.min())[np.unique(s) - s.min()][:, None])[np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1]
        assert mine.tobytes() == centred.tobytes()                 # given the rows, it centres them where they lie
        locked = X.copy()
        locked.flags.writeable = False                             # as a single Arrow batch arrives: not to be written to
        same_bits(cox_partial(locked, t, d, s, beta, presorted=True, overwrite=True, **more), before)
        single = X.astype(np.float32)
        same_bits(cox_partial(single, t, d, s, beta, presorted=True, overwrite=True, **more),
                  cox_partial_0_24(single, t, d, s, beta, presorted=True, **more))
        assert single.dtype == np.float32 and single.tobytes() == X.astype(np.float32).tobytes()
        mixed = np.random.default_rng(1).permutation(n)
        there = dict(more, weights=None if weights is None else weights[mixed], offset=None if offset is None else offset[mixed],
                     entry=None if late is None else late[mixed], order=np.arange(n, dtype=float)[mixed])
        same_bits(cox_partial(X[mixed], t[mixed], d[mixed], s[mixed], beta, **there),
                  cox_partial_0_24(X[mixed], t[mixed], d[mixed], s[mixed], beta, **there))


def test_subtracting_group_by_group_gives_the_bits_of_subtracting_at_once():
    from statcompute.engine.kernels import _less

    rng = np.random.default_rng(5)
    for n, groups in ((4000, 7), (4000, 3000), (1, 1)):
        codes = np.sort(rng.integers(0, groups, n))
        first = np.r_[True, codes[1:] != codes[:-1]]
        start, sid = np.flatnonzero(first), np.cumsum(first) - 1
        stop = np.r_[start[1:], n]
        values, parts = rng.normal(size=(n, 3)), rng.normal(size=(len(start), 3))
        plain = values - parts[sid]
        assert _less(values, parts, sid, start, stop, np.empty((n, 3))).tobytes() == plain.tobytes()
        assert _less(values.copy(), parts, sid, start, stop).tobytes() == plain.tobytes()          # in the array itself
