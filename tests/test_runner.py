"""The harness records every outcome, and never one run's values under another run."""
import sys
import types

import pytest

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
    # every Spark implementation runs before any single-machine one, whatever the plan's order
    assert [r["adapter"]["name"] for r in records] == ["lab_spark", "pprof_spark", "lab_numpy", "always_fails"]
    assert by["lab_spark"]["dataset"]["input_partitions"] >= 1 and "input_partitions" not in by["lab_numpy"]["dataset"]
    rows = {row["adapter"]: row for row in summary(records)}
    assert rows["lab_spark"]["compare_s"] == rows["lab_spark"]["wall_s"]        # the whole call, preparation included
    assert rows["lab_numpy"]["compare_s"] == rows["lab_numpy"]["fit_s"] < rows["lab_numpy"]["wall_s"]
    failed = by["always_fails"]
    assert failed["fit"] is None and failed["truth"] is None and "boom" in failed["error"]["message"]
    assert by["lab_spark"]["fit"]["coefficients"] and by["lab_spark"]["run_id"] != by["lab_numpy"]["run_id"]
    assert by["lab_numpy"]["truth"]["recovered"] is True
    assert by["lab_numpy"]["process"]["peak_rss_bytes"] > 0
    metrics = by["lab_spark"]["spark"]
    if metrics is not None:                      # the UI's REST API was reachable
        assert metrics["jobs"] > 0 and metrics["tasks"] > 0 and metrics["executor_run_s"] > 0
        heaviest = metrics["heaviest_stages"][0]
        assert heaviest["name"] and heaviest["tasks"] >= 1 and heaviest["run_s"] >= 0
        assert "slowest_task" in metrics and (metrics["slowest_task"] is None or metrics["slowest_task"]["executor"] == "driver")
    loaded = load_records(str(tmp_path / "results"))
    assert len(loaded) == 4 and len(summary(loaded)) == 4
    comparisons = compare_records(loaded, reference="lab_numpy")
    assert [c["adapter"] for c in comparisons] == ["lab_spark"] and comparisons[0]["like_for_like"]
    assert comparisons[0]["coefficients_over_T_COEF"] <= 1
    text = render(loaded, reference="lab_numpy")
    assert "lab_spark" in text and "boom" in text


def a_record(adapter, version, wall, phases, coefficients, role="engine", kind="spark", caveats=(), seconds=None, stages=None,
             changes=None):
    return {"statcompute": version, "plan": "p", "repeat": 0, "status": "ok", "caveats": list(caveats), "error": None,
            "adapter": {"name": adapter, "role": role, "kind": kind, "version": "0.7.0"},
            "dataset": {"id": "d", "spec": {"rows": 100, "providers": 4, "features": 2, "design": "simple"}},
            "model": {"label": "cox/stratified/breslow/default"}, "support": {"status": "exact", "notes": []},
            "timing": {"wall_s": wall, "phases": phases}, "spark": {"tasks": 10, "heaviest_stages": stages or []},
            "environment": {"static": {}, "after": {"executors": 2}, "cluster_changed": False,
                            "during": {"executors_min": 2, "executors_max": 2, "changes": changes or [[0.0, 2, 8]]}},
            "process": {}, "truth": None,
            "fit": {"feature_names": ["x1", "x2"], "coefficients": coefficients, "standard_errors": [0.1, 0.1],
                    "log_likelihood": -10.0, "iterations": 3, "extras": {"evaluation_seconds": seconds} if seconds else {}}}


def test_a_report_reads_a_folder_of_several_rounds_round_by_round():
    """Round 14: a results folder collects runs of several versions; every run is compared, and the time is attributed."""
    from statcompute.bench.results import medians, where_time_went

    stalled = [{"stage": 3, "name": "collect at /Workspace/x/statcompute/engine/spark.py:10", "tasks": 17, "run_s": 60.0,
                "median_task_s": 0.5, "max_task_s": 45.0},
               {"stage": 4, "name": "mapInArrow at spark.py:20", "tasks": 100, "run_s": 40.0, "median_task_s": 0.4, "max_task_s": 2.0},
               {"stage": 5, "name": "no task summary", "tasks": 1, "run_s": 1.0}]
    records = [
        a_record("pprof_py", "0.9.0", 30.0, {"fit": 20.0}, [1.0, 2.0], role="reference", kind="driver"),
        a_record("engine", "0.9.0", 20.0, {"prepare": 8.0, "fit": 12.0}, [1.0, 2.0]),
        a_record("engine", "0.10.0", 60.0, {"prepare": 50.0, "fit": 9.0}, [1.0, 2.0], seconds=[48.0, 3.1, 2.9, 3.0], stages=stalled,
                 changes=[[0.0, 9, 36], [20.0, 10, 40]], caveats=["executors ranged from 9 to 10 during the fit"]),
        a_record("engine", "0.10.0", 16.0, {"prepare": 7.0, "fit": 9.0}, [1.0, 2.001], seconds=[5.0, 3.0, 3.0, 3.0]),
    ]
    records[2]["spark"]["slowest_task"] = {"stage": 3, "executor": "7", "executor_attached_s_before": 3.21, "gc_s": 1.5,
                                           "fetch_wait_s": 0.0, "locality": "ANY", "input_bytes": 20_000_000}
    records[3]["warm_input"] = {"mode": "all", "seconds": 7.2, "rows": 100, "python_seconds": 3.4}
    records[0]["fit"]["extras"] = {"likelihood_calls": 11, "halved_steps": 6}       # the reference halved its last step
    by = {(c["adapter"], c["version"]): c for c in compare_records(records, reference="pprof_py")}
    assert set(by) == {("engine", "0.9.0"), ("engine", "0.10.0")}
    assert all(row["reference_halved_steps"] == 6 for row in by.values())
    assert by[("engine", "0.9.0")]["runs"] == 1 and by[("engine", "0.9.0")]["max_abs_coefficient_difference"] == 0
    newest = by[("engine", "0.10.0")]                    # both runs are compared, and the one furthest from the reference is shown
    assert newest["runs"] == 2 and newest["max_coefficient_difference_in_se"] == pytest.approx(0.01)
    middle = {(m["adapter"], m["version"]): m for m in medians(records)}
    assert middle[("engine", "0.10.0")]["runs"] == 2 and middle[("engine", "0.10.0")]["median_compare_s"] == 38.0
    assert middle[("engine", "0.10.0")]["runs_without_caveat"] == 1 and middle[("engine", "0.10.0")]["median_compare_s_without_caveat"] == 16.0
    assert middle[("pprof_py", "0.9.0")]["median_compare_s"] == 20.0                    # a single-machine fit: its fit phase
    went = where_time_went(records)
    assert [w["version"] for w in went] == ["0.9.0", "0.10.0", "0.10.0"]               # Spark implementations only
    slow = went[1]
    assert slow["prepare_s"] == 50.0 and slow["fit_s"] == 9.0 and slow["other_s"] == 1.0
    assert slow["first_evaluation_s"] == 48.0 and slow["later_evaluations_s"] == "3.1 2.9 3.0"
    assert slow["input_warmed"] is None and went[2]["input_warmed"] == "all in 7 s" and slow["task_input_gb"] == 0.02
    assert slow["python_started"] is None and went[2]["python_started"] == "in 3 s"
    assert slow["slowest_task_s"] == 45.0 and slow["its_stage"] == "3: collect at spark.py:10" and slow["stage_tasks"] == 17
    assert slow["stage_median_task_s"] == 0.5 and slow["executors"] == 2 and slow["executors_seen"] == "9>10"
    assert slow["on_executor"] == "7" and slow["executor_attached_s_before"] == 3.2 and slow["task_gc_s"] == 1.5
    assert went[0]["slowest_task_s"] is None and went[0]["first_evaluation_s"] is None and went[0]["on_executor"] is None
    text = render(records, reference="pprof_py")
    assert "## Medians by version" in text and "## Where the time went" in text and "| 0.10.0 | 2 |" in text


def test_a_plan_can_have_the_stored_table_read_once_before_every_timed_spark_fit(spark, tmp_path):
    """Round 15: an executor's first read of a stored file stalled the fits on the cluster (OI-31)."""
    for mode in ("all", "footers"):
        plan = dict(PLAN, adapters=["lab_spark", "lab_numpy"], options={"warm_input": mode})
        records = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / mode))
        by = {r["adapter"]["name"]: r for r in records}
        assert by["lab_spark"]["status"] == "ok" and by["lab_spark"]["warm_input"]["mode"] == mode
        assert by["lab_spark"]["warm_input"]["rows"] == by["lab_spark"]["dataset"]["rows"] and by["lab_spark"]["warm_input"]["seconds"] > 0
        # round 18 (decision D-20): the same step starts the Python workers, and they report their libraries (D-16)
        started = by["lab_spark"]["warm_input"]
        assert started["python_seconds"] > 0 and sum(found["tasks"] for found in started["executor_libraries"]) >= 8
        assert all(found["numpy"] and found["python"] for found in started["executor_libraries"])
        assert by["lab_numpy"]["warm_input"] is None                      # a single-machine fit reads nothing through Spark
    assert run_plan(spark, dict(PLAN, adapters=["lab_spark"]), str(tmp_path / "data"), str(tmp_path / "plain"))[0]["warm_input"] is None


def test_every_plan_in_the_repository_loads_under_its_own_name():
    import glob
    import os

    from statcompute.bench.runner import PLANS, load_plan

    files = sorted(glob.glob(os.path.join(PLANS, "*.json")))
    assert len(files) >= 10
    for path in files:
        plan = load_plan(path)
        assert plan["name"] == os.path.basename(path)[:-5] and plan["datasets"] and plan["adapters"]
        assert all(family in plan["models"] for family in {d["family"] for d in plan["datasets"]})
    assert load_plan("compare_5m_warm")["options"]["warm_input"] == "all"
    assert load_plan("unstratified_5m_warm_input")["options"]["engine_partitions"] == "input"


def test_the_executors_report_their_libraries(spark):
    """Round 18: the executors' side of the execution record, by the job that also starts their Python workers."""
    import numpy

    from statcompute.engine import executor_libraries

    found = executor_libraries(spark, tasks=4, hold_seconds=0.0)
    assert sum(entry["tasks"] for entry in found) == 4 and found[0]["numpy"] == numpy.__version__
    assert set(found[0]) >= {"python", "numpy", "pandas", "pyarrow", "blas", "tasks"}


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

    class Frame:
        events = []

        def persist(self):
            self.events.append("persist")
            return self

        def count(self):
            self.events.append("count")
            return 100

        def unpersist(self):
            self.events.append("unpersist")

    class Stored:
        spec = DatasetSpec("cox", 100, 5, 2)

        def frame(self, spark):
            return Frame()

    adapter = adapters.get("pprof_spark")
    assert adapter.available().runnable
    result = adapter.fit(None, Stored(), ModelSpec("cox", convergence="tight"))
    assert result.coefficients == [1.0, 2.0] and result.standard_errors == [0.1, 0.2]
    assert result.iterations == 4 and result.converged is True and result.log_likelihood == -12.5
    assert calls["cox"] == ("time", "event", ["x1", "x2"],
                            {"ties": "breslow", "row_id": "id", "strata": "provider", "eps": 1e-13, "max_iterations": 50})
    assert Frame.events == [] and result.settings["cache_input"] is False and "cache_input" not in result.phases
    # the second adapter caches the input first, times that separately, and releases it
    cached = adapters.get("pprof_spark_cached")
    assert cached.pin == "pprof_spark" and cached.kind == "spark"
    again = cached.fit(None, Stored(), ModelSpec("cox"))
    assert Frame.events == ["persist", "count", "unpersist"] and again.settings["cache_input"] is True
    assert again.phases["cache_input"] >= 0 and again.coefficients == [1.0, 2.0]


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
    assert all(stage["name"] for stage in out["heaviest_stages"]) and out["heaviest_stages"]
    slowest = out["slowest_task"]                         # by the same route: where the longest task ran and what it was made of
    if slowest is not None:                               # None only when no stage had tasks long enough to have a median
        assert slowest["executor"] == "driver" and slowest["duration_s"] >= 0 and slowest["executor_attached_s_before"] >= 0
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


def test_the_executor_sampler_reports_what_was_attached():
    import time

    from statcompute.bench.metrics import ExecutorSampler

    readings = iter([(4, 16), (4, 16), (10, 40), (10, 40), (10, 40), (10, 40), (10, 40), (10, 40)])
    with ExecutorSampler(lambda: next(readings), interval=0.05) as watch:
        time.sleep(0.22)
    out = watch.result()
    assert out["executors_min"] == 4 and out["executors_max"] == 10 and out["cores_min"] == 16 and out["cores_max"] == 40
    assert 16 < out["cores_time_average"] < 40 and out["samples"] >= 3
    assert [c[1:] for c in out["changes"]] == [[4, 16], [10, 40]]


def test_holding_a_cluster_at_size_runs_tasks_until_the_target_or_the_timeout(spark):
    from statcompute.bench import environment

    assert environment.hold_at(spark, 10)["skipped"] == "local session"
    assert environment.attached(spark) == (0, spark.sparkContext.defaultParallelism)
    said = []
    out = environment.hold_at(spark, 1, timeout_s=1.0, task_seconds=0.05, allow_local=True, log=said.append)   # a local session never gets there
    assert out["target"] == 1 and out["reached"] == 0 and out["rounds"] >= 1
    assert len(out["seen"]) == out["rounds"] + 1 and said and said[0].startswith("warm-up: 0 of 1 executors")


def test_a_run_says_how_many_executors_it_had_and_flags_a_short_cluster(spark, tmp_path, monkeypatch):
    from statcompute.bench import environment

    monkeypatch.setattr(environment, "hold_at", lambda spark, executors, timeout: {"target": executors, "reached": 4, "seconds": 1.0, "rounds": 2})
    plan = dict(PLAN, adapters=["lab_spark"], options={"hold_executors": 10})
    record = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))[0]
    assert record["status"] == "ok" and record["warmup"]["reached"] == 4
    assert record["caveats"] == ["the cluster had 4 of 10 executors when the fit started"]
    during = record["environment"]["during"]
    assert during["executors_min"] == during["executors_max"] == 0 and during["cores_time_average"] > 0
    assert record["environment"]["cluster_changed"] is False
