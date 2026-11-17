"""Cox regression without strata on the time grid: two passes, rows in any order, first moments
only. Against explicit risk sets, against the stratum kernel run as one stratum, against pprof_py,
and on Spark."""
import numpy as np
import pytest

from statcompute.engine import CoxPH, local, timegrid
from statcompute.testkit import DatasetSpec, materialize
from test_engine_kernels import OPTIONS, explicit

FEATURES = [f"x{k}" for k in range(1, 6)]


def one_table(n=900, p=4, seed=31):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) + np.array([60.0, 0.0, -5.0, 2.0])[:p]          # a covariate with a large mean, like age
    time = rng.integers(2, 30, n).astype(float)
    event = (rng.random(n) < 0.6).astype(int)
    entry = np.where(rng.random(n) < 0.4, np.floor(time * rng.random(n)), np.nan)
    return X, time, event, entry, rng.integers(1, 4, n).astype(float), 0.2 * rng.normal(size=n), 0.3 * rng.normal(size=p)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("use_weights,use_offset,use_entry", OPTIONS)
def test_the_time_grid_matches_explicit_risk_sets(ties, use_weights, use_offset, use_entry):
    X, time, event, entry, w, o, beta = one_table()
    n = len(time)
    out = timegrid.evaluate_arrays(X, time, event, beta, ties=ties, weights=w if use_weights else None,
                                   offset=o if use_offset else None, entry=entry if use_entry else None, chunk_rows=250)
    ll, U, I = explicit(X, time, event, np.zeros(n, dtype=int), beta, w if use_weights else np.ones(n),
                        o if use_offset else np.zeros(n), entry if use_entry else np.full(n, np.nan), ties)
    assert abs(out["loglik"] - ll) <= 1e-12 * abs(ll)
    assert np.max(np.abs(out["score"] - U)) <= 1e-11 * np.max(np.abs(U))
    assert np.max(np.abs(out["information"] - I)) <= 1e-10 * np.max(np.abs(I))
    assert out["deaths"] == int(event.sum()) and out["rows"] == n


def test_the_result_does_not_depend_on_how_the_rows_are_cut_or_ordered():
    X, time, event, entry, w, o, beta = one_table(seed=32)
    whole = timegrid.evaluate_arrays(X, time, event, beta, ties="efron", weights=w, offset=o, entry=entry, chunk_rows=10 ** 6)
    order = np.random.default_rng(1).permutation(len(time))
    cut = timegrid.evaluate_arrays(X[order], time[order], event[order], beta, ties="efron", weights=w[order],
                                   offset=o[order], entry=entry[order], chunk_rows=37)
    assert abs(cut["loglik"] - whole["loglik"]) <= 1e-12 * abs(whole["loglik"])
    assert np.allclose(cut["information"], whole["information"], rtol=1e-10, atol=0)


def test_efron_steps_are_expanded_in_chunks_without_changing_the_result():
    X, time, event, entry, w, o, beta = one_table(seed=33)
    grid, mu = np.unique(time), X.mean(axis=0)
    tables = timegrid.new_tables(len(grid), X.shape[1], True, False)
    timegrid.accumulate(tables, grid, X, time, event, beta, mu, 0.0, efron=True, weights=w)
    all_at_once = timegrid.reduce(tables, len(grid), X.shape[1], efron=True, entry=False)
    in_pieces = timegrid.reduce(tables, len(grid), X.shape[1], efron=True, entry=False, max_steps=7)
    assert all_at_once["loglik"] == pytest.approx(in_pieces["loglik"], rel=1e-14)
    assert np.allclose(all_at_once["M"], in_pieces["M"], rtol=1e-12, atol=0)
    assert np.allclose(all_at_once["tied"], in_pieces["tied"], rtol=1e-12, atol=0)


def follow_up(n=20000, p=6, seed=41):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) + np.array([60, 0, -5, 2, 0, 1.0])
    rate = 0.02 * np.exp((X - X.mean(0)) @ np.array([0.02, 0.3, -0.2, 0.1, 0.0, 0.25]))
    t, c = np.ceil(rng.exponential(1 / rate)).clip(1, 400), rng.integers(1, 400, n)
    time, event = np.minimum(t, c).astype(float), (t <= c).astype(int)
    entry = np.where(rng.random(n) < 0.3, np.floor(time * rng.random(n)), np.nan)
    return X, time, event, entry, rng.integers(1, 4, n).astype(float), 0.1 * rng.normal(size=n)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("options", [False, True])
def test_the_fit_equals_the_stratum_kernel_run_as_one_stratum(ties, options):
    X, time, event, entry, w, o = follow_up()
    extra = dict(weights=w, offset=o, entry=entry) if options else {}
    grid = local.cox_fit(X, time, event, None, ties=ties, tol=1e-11, block_rows=3000, **extra)
    kernel = local.cox_fit(X, time, event, None, ties=ties, tol=1e-11, layout="stratum", **extra)
    assert grid["converged"] and grid["distinct_times"] > 300
    assert np.max(np.abs(grid["beta"] - kernel["beta"]) / kernel["se"]) < 1e-10
    assert np.max(np.abs(grid["se"] / kernel["se"] - 1)) < 1e-10
    assert abs(grid["loglik"] - kernel["loglik"]) <= 1e-12 * abs(kernel["loglik"])


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("options", [False, True])
def test_the_fit_agrees_with_pprof_py(ties, options):
    pprof_py = pytest.importorskip("pprof_py")
    X, time, event, entry, w, o = follow_up(n=8000, seed=42)
    ours = local.cox_fit(X, time, event, None, ties=ties, tol=1e-11, **(dict(weights=w, offset=o, entry=entry) if options else {}))
    theirs = pprof_py.CoxPH(ties=ties, eps=1e-13, max_iter=50)
    if options:
        theirs.fit(X, start=np.where(np.isnan(entry), 0.0, entry), stop=time, event=event, sample_weight=w, offset=o)
    else:
        theirs.fit(X, duration=time, event=event)
    se = np.asarray(theirs.standard_errors_)
    at_theirs = timegrid.evaluate_arrays(X, time, event, np.asarray(theirs.coef_, dtype=float), ties=ties,
                                         weights=w if options else None, offset=o if options else None,
                                         entry=entry if options else None)
    remaining = np.linalg.solve(at_theirs["information"], at_theirs["score"])     # what pprof_py's stopping rule left
    assert np.max(np.abs(remaining) / se) < 1e-5
    assert np.max(np.abs(ours["beta"] - (theirs.coef_ + remaining)) / se) < 1e-9
    assert np.max(np.abs(ours["se"] / se - 1)) < 1e-7
    assert abs(ours["loglik"] - theirs.log_likelihood_) <= 1e-9 * abs(theirs.log_likelihood_)


# ---------------------------------------------------------------- on Spark
@pytest.fixture(scope="module")
def survival(spark, data_root):
    frame = materialize(spark, DatasetSpec("cox", 6000, 20, 5), data_root).frame(spark).selectExpr(
        "*", "1.0D + pmod(id, 3) AS w", "(pmod(id, 7) - 3) / 20.0D AS o",
        "CASE WHEN pmod(id, 5) < 2 THEN floor(0.3D * time) END AS entry")
    return frame, frame.orderBy("id").toPandas()


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_unstratified_cox_on_spark_equals_the_single_machine_fit(spark, survival, ties):
    frame, pdf = survival
    estimator = CoxPH("time", "event", FEATURES, None, ties=ties, entry="entry", weights="w", offset="o", tol=1e-11)
    model = estimator.fit(frame)
    alone = local.cox_fit(pdf[FEATURES].to_numpy(), pdf["time"].to_numpy(), pdf["event"].to_numpy(), None, ties=ties,
                          weights=pdf["w"].to_numpy(), offset=pdf["o"].to_numpy(), entry=pdf["entry"].to_numpy(dtype=float),
                          tol=1e-11)
    assert model.converged and model.rows == 6000 and model.settings["layout"] == "time grid"
    assert model.settings["distinct_times"] == alone["distinct_times"]
    assert np.max(np.abs(model.coefficients - alone["beta"]) / alone["se"]) < 1e-9
    assert np.max(np.abs(model.standard_errors / alone["se"] - 1)) < 1e-10
    assert abs(model.log_likelihood - alone["loglik"]) <= 1e-12 * abs(alone["loglik"])
    # other partitions and other batch sizes: the same fit
    key = "spark.sql.execution.arrow.maxRecordsPerBatch"
    before = spark.conf.get(key)
    spark.conf.set(key, "700")
    try:
        recut = CoxPH("time", "event", FEATURES, None, ties=ties, entry="entry", weights="w", offset="o", tol=1e-11,
                      partitions=7).fit(frame)
    finally:
        spark.conf.set(key, before)
    assert recut.partitions == 7
    assert np.max(np.abs(recut.coefficients - model.coefficients) / model.standard_errors) < 1e-9


def test_unstratified_and_stratified_fits_differ_as_they_should(survival):
    """A guard against fitting the wrong model: the data have provider effects, so the two estimates differ."""
    frame, _ = survival
    without = CoxPH("time", "event", FEATURES, None).fit(frame)
    within = CoxPH("time", "event", FEATURES, "provider", row_id="id").fit(frame)
    assert np.max(np.abs(without.coefficients - within.coefficients) / within.standard_errors) > 1e-3
    assert without.settings["passes_per_evaluation"] == 2 and within.settings["passes_per_evaluation"] == 1


def test_the_harness_runs_the_engine_without_strata(spark, tmp_path):
    from statcompute.bench import run_plan
    from statcompute.validation import compare_records

    plan = {"name": "u", "datasets": [{"family": "cox", "rows": 4000, "providers": 10, "features": 4}],
            "models": {"cox": [{"family": "cox", "ties": "breslow", "strata": None, "convergence": "tight"}]},
            "adapters": ["lab_numpy", "engine"], "repeats": 1}
    records = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))
    by = {r["adapter"]["name"]: r for r in records}
    assert by["engine"]["status"] == "ok" and by["engine"]["fit"]["extras"]["layout"] == "time grid"
    comparison = compare_records(records, reference="lab_numpy")
    assert len(comparison) == 1 and comparison[0]["coefficients_over_T_COEF"] <= 1 and comparison[0]["like_for_like"]
