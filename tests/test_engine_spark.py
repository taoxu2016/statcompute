"""The engine on Spark: the same numbers as its single-machine fits and as pprof_py, with every
option, with providers at the bound, and whatever the layout."""
import numpy as np
import pytest

from statcompute.engine import CoxPH, LogisticFE, local
from statcompute.testkit import DatasetSpec, materialize

FEATURES = [f"x{k}" for k in range(1, 6)]


@pytest.fixture(scope="module")
def survival(spark, data_root):
    """A stored Cox dataset with three more columns: case weights, an offset and entry times."""
    frame = materialize(spark, DatasetSpec("cox", 6000, 20, 5), data_root).frame(spark).selectExpr(
        "*", "1.0D + pmod(id, 3) AS w", "(pmod(id, 7) - 3) / 20.0D AS o",
        "CASE WHEN pmod(id, 5) < 2 THEN floor(0.3D * time) END AS entry", "concat('unit-', provider) AS unit")
    return frame, frame.orderBy("id").toPandas()


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_on_spark_equals_the_single_machine_fit_with_every_option(survival, ties):
    frame, pdf = survival
    model = CoxPH("time", "event", FEATURES, "provider", ties=ties, entry="entry", weights="w", offset="o",
                  row_id="id", rows_per_partition=2500, tol=1e-11).fit(frame)
    alone = local.cox_fit(pdf[FEATURES].to_numpy(), pdf["time"].to_numpy(), pdf["event"].to_numpy(),
                          pdf["provider"].to_numpy(), ties=ties, weights=pdf["w"].to_numpy(),
                          offset=pdf["o"].to_numpy(), entry=pdf["entry"].to_numpy(dtype=float), tol=1e-11)
    assert model.converged and model.partitions == 3 and model.rows == 6000
    assert model.events == pytest.approx(float((pdf["w"] * pdf["event"]).sum()))
    assert np.max(np.abs(model.coefficients - alone["beta"]) / alone["se"]) < 1e-9
    assert np.max(np.abs(model.standard_errors / alone["se"] - 1)) < 1e-10
    assert abs(model.log_likelihood - alone["loglik"]) <= 1e-12 * abs(alone["loglik"])
    table = model.summary()
    assert list(table["feature"]) == FEATURES and np.all((table["p_value"] >= 0) & (table["p_value"] <= 1))


def test_cox_on_spark_repeats_bit_for_bit_and_takes_strata_of_any_type(survival):
    frame, _ = survival
    fit = lambda strata, rows: CoxPH("time", "event", FEATURES, strata, row_id="id", rows_per_partition=rows).fit(frame)  # noqa: E731
    first, second = fit("provider", 2500), fit("provider", 2500)
    assert np.array_equal(first.coefficients, second.coefficients) and first.log_likelihood == second.log_likelihood
    named, relaid = fit("unit", 2500), fit("provider", 700)          # string strata; another layout
    for other in (named, relaid):
        assert np.max(np.abs(other.coefficients - first.coefficients) / first.standard_errors) < 1e-9
    assert relaid.partitions == 9
    # the working set does not depend on how the input happens to be partitioned: the same bits again
    moved = CoxPH("time", "event", FEATURES, "provider", row_id="id", rows_per_partition=2500).fit(frame.repartition(5))
    assert np.array_equal(first.coefficients, moved.coefficients) and first.log_likelihood == moved.log_likelihood
    assert len(first.evaluation_seconds) == first.evaluations
    assert first.execution["layout"] == "whole strata" and first.execution["partitions"] == 3 and first.execution["row_id"] == "id"
    assert first.execution["statcompute"] and first.execution["spark"] == frame.sparkSession.version


def test_by_default_a_working_set_has_one_partition_per_core_and_more_only_for_size(spark, data_root, survival, monkeypatch):
    """Decision D-21: a pass costs for its tasks, so by default it has a task for every core and no more;
    a partition that would hold too many bytes of features is cut. Explicit settings fix the layout as before."""
    from statcompute.engine import spark as engine

    frame, _ = survival
    cores = spark.sparkContext.defaultParallelism
    assert engine.working_partitions(spark, 5_000_000, 100) == (max(cores, 20), "one per core" if cores >= 20 else "by size", cores)
    assert engine.working_partitions(spark, 100_000_000, 180)[:2] == (720, "by size")            # 144 GB of features in 200 MB
    assert engine.working_partitions(spark, 5_000_000, 100, rows_per_partition=50_000)[:2] == (100, "rows per partition")
    assert engine.working_partitions(spark, 5_000_000, 100, partitions=7, rows_per_partition=50_000)[:2] == (7, "given")

    default = CoxPH("time", "event", FEATURES, "provider", row_id="id").fit(frame)
    assert default.partitions == cores and default.settings["partitions_from"] == "one per core" and default.settings["rows_per_partition"] is None
    assert default.execution["cores"] == cores and default.execution["partitions"] == cores
    fixed = CoxPH("time", "event", FEATURES, "provider", row_id="id", rows_per_partition=2500).fit(frame)
    assert fixed.partitions == 3 and fixed.settings["partitions_from"] == "rows per partition"
    assert np.max(np.abs(default.coefficients - fixed.coefficients) / fixed.standard_errors) < 1e-9

    logistic_frame = materialize(spark, DatasetSpec("logistic_fe", 6000, 20, 5), data_root).frame(spark)
    by_core = LogisticFE("y", FEATURES, "provider", row_id="id").fit(logistic_frame)
    assert by_core.partitions == cores and by_core.settings["partitions_from"] == "one per core" and by_core.execution["cores"] == cores
    # with a small limit on a partition's bytes the size decides: 6,000 rows by 5 features are 240,000 bytes
    monkeypatch.setattr(engine, "PARTITION_BYTES", 40_000)
    by_size = LogisticFE("y", FEATURES, "provider", row_id="id").fit(logistic_frame)
    assert by_size.partitions == 6 and by_size.settings["partitions_from"] == "by size"
    assert np.max(np.abs(by_size.coefficients - by_core.coefficients) / by_core.standard_errors) < 1e-9
    assert set(by_size.provider_effects["provider"]) == set(by_core.provider_effects["provider"])


def test_cox_on_spark_agrees_with_pprof_py(survival):
    pprof_py = pytest.importorskip("pprof_py")
    frame, pdf = survival
    model = CoxPH("time", "event", FEATURES, "provider", ties="efron", weights="w", row_id="id", tol=1e-11).fit(frame)
    theirs = pprof_py.CoxPH(ties="efron", eps=1e-13, max_iter=50)
    theirs.fit(pdf[FEATURES].to_numpy(), duration=pdf["time"].to_numpy(), event=pdf["event"].to_numpy(),
               strata=pdf["provider"].to_numpy(), sample_weight=pdf["w"].to_numpy())
    se = np.asarray(theirs.standard_errors_)
    assert np.max(np.abs(model.coefficients - theirs.coef_) / se) < 1e-5      # pprof_py's stopping rule; see the kernel tests
    assert np.max(np.abs(model.standard_errors / se - 1)) < 1e-7
    assert abs(model.log_likelihood - theirs.log_likelihood_) <= 1e-9 * abs(theirs.log_likelihood_)


def test_logistic_on_spark_with_providers_at_the_bound(spark, data_root):
    pprof_py = pytest.importorskip("pprof_py")
    frame = materialize(spark, DatasetSpec("logistic_fe", 8000, 20, 5), data_root).frame(spark).selectExpr(
        "id", "provider", *FEATURES, "CASE WHEN provider = 3 THEN 0 WHEN provider = 7 THEN 1 ELSE y END AS y")
    pdf = frame.orderBy("id").toPandas()
    model = LogisticFE("y", FEATURES, "provider", row_id="id", rows_per_partition=3000, tol=1e-11).fit(frame)
    X, y, provider = pdf[FEATURES].to_numpy(), pdf["y"].to_numpy(dtype=float), pdf["provider"].to_numpy()
    alone = local.logistic_fit(X, y, provider, tol=1e-11)
    theirs = pprof_py.LogisticFixedEffectModel()
    theirs.fit(X, y, provider, use_dataprep=False, tol=1e-12, bound=10.0, backtrack=True)
    beta, gamma = np.ravel(theirs.coefficients_["beta"]), np.ravel(theirs.coefficients_["gamma"])
    se = np.sqrt(np.diag(np.asarray(theirs.variances_["beta"])))
    assert model.converged and (model.providers, model.bounded_low, model.bounded_high) == (20, 1, 1)
    assert np.max(np.abs(model.coefficients - alone["beta"]) / alone["se"]) < 1e-9
    assert np.max(np.abs(model.coefficients - beta) / se) < 1e-9 and np.max(np.abs(model.standard_errors / se - 1)) < 1e-9
    effects = model.provider_effects
    assert list(effects["provider"]) == list(range(20)) and list(effects["bounded"]) == [i in (3, 7) for i in range(20)]
    assert np.max(np.abs(effects["gamma"].to_numpy() - gamma)) < 1e-9
    assert effects["gamma"][3] == pytest.approx(model.median_gamma - 10.0) and effects["gamma"][7] == pytest.approx(model.median_gamma + 10.0)
    assert int(effects["rows"].sum()) == 8000 and float(effects["events"].sum()) == float(y.sum())
    variance = np.ravel(np.asarray(theirs.variances_["gamma"], dtype=float))
    solved = ~effects["bounded"].to_numpy()
    assert np.max(np.abs(effects["standard_error"].to_numpy()[solved] ** 2 / variance[solved] - 1)) < 1e-8


def test_data_that_cannot_be_fitted_are_refused_with_the_reason(survival):
    frame, _ = survival
    with pytest.raises(ValueError, match="missing value"):
        CoxPH("time", "event", FEATURES, "provider").fit(frame.selectExpr(
            "provider", "time", "event", "CASE WHEN id = 5 THEN NULL ELSE x1 END AS x1", *FEATURES[1:]))
    with pytest.raises(ValueError, match="event not 0 or 1"):
        CoxPH("time", "event", FEATURES, "provider").fit(frame.selectExpr(
            "provider", "time", *FEATURES, "CASE WHEN id = 5 THEN 2 ELSE event END AS event"))
    with pytest.raises(ValueError, match="entry not before exit"):
        CoxPH("time", "event", FEATURES, "provider", entry="late").fit(frame.selectExpr("*", "time AS late"))
    with pytest.raises(NotImplementedError, match="distinct values"):
        CoxPH("time", "event", FEATURES, None, max_times=5).fit(frame)


def test_missing_features_are_refused_by_the_pass_that_builds_the_working_set(spark, data_root, survival):
    """Round 12: the light job reads no features; the first pass over them counts the rows it cannot use."""
    frame, _ = survival
    holed = "CASE WHEN id = 5 THEN double('NaN') WHEN id = 77 THEN NULL ELSE x1 END AS x1"
    with pytest.raises(ValueError, match="missing value': 2"):
        CoxPH("time", "event", FEATURES, "provider", row_id="id", rows_per_partition=2500).fit(
            frame.selectExpr("id", "provider", "time", "event", holed, *FEATURES[1:]))
    logit = materialize(spark, DatasetSpec("logistic_fe", 8000, 20, 5), data_root).frame(spark)
    with pytest.raises(ValueError, match="missing value': 2"):
        LogisticFE("y", FEATURES, "provider", row_id="id").fit(logit.selectExpr("id", "provider", "y", holed, *FEATURES[1:]))
    with pytest.raises(ValueError, match="outcome not 0 or 1': 1"):
        LogisticFE("y", FEATURES, "provider").fit(logit.selectExpr("provider", *FEATURES, "CASE WHEN id = 9 THEN 3 ELSE y END AS y"))


def test_the_engine_runs_in_the_harness(spark, tmp_path):
    from statcompute.bench import run_plan
    from statcompute.validation import compare_records

    plan = {"name": "e", "datasets": [{"family": f, "rows": 4000, "providers": 10, "features": 4} for f in ("cox", "logistic_fe")],
            "models": {"cox": [{"family": "cox", "ties": "efron", "convergence": "tight"}],
                       "logistic_fe": [{"family": "logistic_fe", "convergence": "tight"}]},
            "adapters": ["lab_numpy", "engine"], "repeats": 1}
    records = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))
    engine = [r for r in records if r["adapter"]["name"] == "engine"]
    assert [r["status"] for r in engine] == ["ok", "ok"] and engine[0]["adapter"]["role"] == "engine"
    assert set(engine[1]["timing"]["phases"]) == {"prepare", "fit", "provider_effects"}
    assert [r["status"] for r in records if r["adapter"]["name"] == "lab_numpy"] == ["unsupported", "ok"]   # the prototype has no Efron
    comparison = compare_records(records, reference="lab_numpy")
    assert len(comparison) == 1 and comparison[0]["coefficients_over_T_COEF"] <= 1
    # the logistic record says which providers were flagged, as counts and a digest, and the report compares runs by it
    from statcompute.bench.report import render

    flags = engine[1]["fit"]["extras"]["flags"]
    assert set(flags) == {"score", "wald", "exact"} and len(flags["score"]["digest"]) == 16 and "flags" not in engine[0]["fit"]["extras"]
    assert flags["score"]["above"] + flags["score"]["below"] <= 10 and flags["exact"]["smallest_distance_from_cutoff"] > 0
    assert "## Provider flags across runs" in render(records, reference="lab_numpy")
