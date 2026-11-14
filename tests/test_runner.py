"""The harness records every outcome, and never one run's values under another run."""
import sys
import types

from statcompute import adapters
from statcompute.adapters import Adapter, Support
from statcompute.bench import run_plan
from statcompute.bench.report import render
from statcompute.bench.results import load_records, summary
from statcompute.testkit import DatasetSpec, ModelSpec
from statcompute.validation import compare_records


class AlwaysFails(Adapter):
    name, kind, role = "always_fails", "driver", "prototype"

    def support(self, model, data):
        return Support()

    def fit(self, spark, dataset, model):
        raise RuntimeError("boom")


PLAN = {"name": "t", "datasets": [{"family": "logistic_fe", "rows": 4000, "providers": 10, "features": 4}],
        "models": {"logistic_fe": [{"family": "logistic_fe", "convergence": "tight"}]},
        "adapters": ["lab_numpy", "always_fails", "lab_spark", "pprof_spark"], "repeats": 1}


def test_a_plan_records_every_outcome(spark, tmp_path):
    adapters.register("always_fails", AlwaysFails)
    records = run_plan(spark, PLAN, str(tmp_path / "data"), str(tmp_path / "results"))
    by = {r["adapter"]["name"]: r for r in records}
    assert [by[n]["status"] for n in PLAN["adapters"]] == ["ok", "failed", "ok", "unavailable"]
    failed = by["always_fails"]
    assert failed["fit"] is None and failed["truth"] is None and "boom" in failed["error"]["message"]
    assert by["lab_spark"]["fit"]["coefficients"] and by["lab_spark"]["run_id"] != by["lab_numpy"]["run_id"]
    assert by["lab_numpy"]["truth"]["recovered"] is True
    assert by["lab_numpy"]["process"]["peak_rss_bytes"] > 0
    metrics = by["lab_spark"]["spark"]
    if metrics is not None:                      # the UI's REST API was reachable
        assert metrics["jobs"] > 0 and metrics["tasks"] > 0 and metrics["executor_run_s"] > 0
    loaded = load_records(str(tmp_path / "results"))
    assert len(loaded) == 4 and len(summary(loaded)) == 4
    comparisons = compare_records(loaded, reference="lab_numpy")
    assert [c["adapter"] for c in comparisons] == ["lab_spark"] and comparisons[0]["like_for_like"]
    assert comparisons[0]["coefficients_over_T_COEF"] <= 1
    text = render(loaded, reference="lab_numpy")
    assert "lab_spark" in text and "boom" in text


def test_the_memory_guard_skips_a_fit_that_would_not_fit_the_driver(spark, tmp_path):
    plan = dict(PLAN, adapters=["lab_numpy"], options={"driver_memory_bytes": 1000})
    record = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))[0]
    assert record["status"] == "skipped_memory" and record["fit"] is None


def test_the_pprof_spark_adapter_maps_options_and_results(monkeypatch):
    calls = {}

    class Coefficient:
        def __init__(self, feature, estimate, standard_error):
            self.feature, self.estimate, self.standard_error = feature, estimate, standard_error

    class Model:
        coefficients = [Coefficient("x2", 2.0, 0.2), Coefficient("x1", 1.0, 0.1)]
        log_likelihood, iterations, converged, halvings, warnings, summary = -12.5, 4, True, 0, [], {}

    class CoxPH:
        def __init__(self, time, event, features, **options):
            calls["cox"] = (time, event, list(features), options)

        def fit(self, frame):
            return Model()

    fake = types.ModuleType("pprof_spark")
    fake.CoxPH, fake.api, fake.__version__ = CoxPH, (lambda spark: object()), "0.0-test"
    monkeypatch.setitem(sys.modules, "pprof_spark", fake)

    class Stored:
        spec = DatasetSpec("cox", 100, 5, 2)

        def frame(self, spark):
            return "frame"

    adapter = adapters.get("pprof_spark")
    assert adapter.available().runnable
    result = adapter.fit(None, Stored(), ModelSpec("cox", convergence="tight"))
    assert result.coefficients == [1.0, 2.0] and result.standard_errors == [0.1, 0.2]
    assert result.iterations == 4 and result.converged is True and result.log_likelihood == -12.5
    assert calls["cox"] == ("time", "event", ["x1", "x2"],
                            {"ties": "breslow", "row_id": "id", "strata": "provider", "eps": 1e-13, "max_iterations": 50})


def test_task_metrics_come_from_the_status_store_when_the_rest_api_is_closed(spark):
    import time

    from statcompute.bench.metrics import SparkMetrics

    sc = spark.sparkContext
    sc.setJobGroup("statcompute-store-test", "status store fallback")
    assert spark.range(0, 4000, 1, 4).selectExpr("sum(id) AS s").first()["s"] == 7998000
    metrics = SparkMetrics(spark)
    metrics.base = None                                   # as on a cluster whose UI cannot be reached
    out = None
    for _ in range(20):                                   # the listener bus is asynchronous
        out = metrics.collect("statcompute-store-test")
        if out and out["tasks"] >= 4:
            break
        time.sleep(0.25)
    assert out is not None and out["source"] == "status_store"
    assert out["jobs"] >= 1 and out["tasks"] >= 4 and out["executor_run_s"] >= 0
    assert metrics.executors()["cores"] >= 1
    assert metrics.collect("no-such-group")["jobs"] == 0


def test_records_are_scrubbed_before_they_are_written():
    import json

    from statcompute.bench.runner import scrub

    record = {"plan": "p", "environment": {"static": {"databricks": {"cluster_id": "1114-123456-abcdefgh"}}},
              "error": {"message": "/Workspace/Users/jane.doe@example.edu/a.py failed for jane.doe@example.edu",
                        "trace": "at /home/alice/x and /Users/bob/y"}}
    text = scrub(record)
    assert "jane" not in text and "alice" not in text and "bob" not in text and "1114-123456" not in text
    clean = json.loads(text)
    assert clean["scrubbed"] and clean["environment"]["static"]["databricks"]["cluster_id"].startswith("sha256:")
    assert record["environment"]["static"]["databricks"]["cluster_id"] == "1114-123456-abcdefgh"   # the original is untouched


def test_a_fit_that_kills_the_process_would_leave_a_running_record(spark, tmp_path):
    from statcompute.adapters import FitResult

    seen = []

    class Peeks(Adapter):
        name, kind, role = "peeks", "driver", "prototype"

        def support(self, model, data):
            return Support()

        def fit(self, spark, dataset, model):
            seen.extend(r["status"] for r in load_records(str(tmp_path / "results")) if r["adapter"]["name"] == "peeks")
            return FitResult(dataset.spec.feature_names(), [0.0] * dataset.spec.features, caveats=["a test caveat"])

    adapters.register("peeks", Peeks)
    record = run_plan(spark, dict(PLAN, adapters=["peeks"]), str(tmp_path / "data"), str(tmp_path / "results"))[0]
    assert seen == ["running"]                            # what would remain had the process died in the fit
    final = load_records(str(tmp_path / "results"))
    assert [r["status"] for r in final] == ["ok"] and final[0]["scrubbed"] and final[0]["caveats"] == ["a test caveat"]
    assert record["memory_guard"] is None and summary(final)[0]["caveats"] == "a test caveat"
