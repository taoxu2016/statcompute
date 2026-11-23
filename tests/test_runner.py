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
    assert load_plan("engine_partitions_5m_warm")["adapters"] == ["engine", "engine_50k", "engine_125k", "engine_250k"]
    assert load_plan("compare_5m_warm")["options"]["lab_rows_per_partition"] == 125000


def test_the_executors_report_their_libraries(spark):
    """Round 18: the executors' side of the execution record, by the job that also starts their Python workers."""
    import numpy

    from statcompute.engine import executor_libraries

    found = executor_libraries(spark, tasks=4, hold_seconds=0.0)
    assert sum(entry["tasks"] for entry in found) == 4 and found[0]["numpy"] == numpy.__version__
    assert set(found[0]) >= {"python", "numpy", "pandas", "pyarrow", "blas", "tasks"}


def test_the_report_judges_the_pin_when_it_is_made_and_lists_the_builds_that_ran():
    """Round 19: runs of a build that is pinned afterwards lose the caveat about the pin, with no new fit."""
    from statcompute.bench.results import NOT_PINNED, caveats, implementations, medians, summary

    def run(name, commit, seconds, more=()):
        record = a_record(name, "0.16.0", seconds, {"fit": seconds}, [1.0, 2.0], role="reference", caveats=list(more) + [NOT_PINNED])
        record["adapter"].update(version="0.3.1", commit=commit)
        return record

    records = [run("pprof_spark", "b92919c0", 50.0), run("pprof_spark_cached", "b92919c0", 60.0, ["executors ranged from 6 to 10 during the fit"]),
               run("pprof_spark", "7a146242", 70.0)]
    old = {"references": {"pprof_spark": {"commit": "5db844e23669d194b00996abf13e46804124f834"}}}
    new = {"references": {"pprof_spark": {"commit": "b92919c0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}, "pprof_py": {"commit": "94b5476"}}}
    assert all(NOT_PINNED in caveats(r) and NOT_PINNED in caveats(r, old) for r in records)        # as recorded; and under the old lock
    assert [NOT_PINNED in caveats(r, new) for r in records] == [False, False, True]                # the variant follows its package's pin
    assert caveats(records[1], new) == ["executors ranged from 6 to 10 during the fit"]            # other caveats stay
    listed = {(row["adapter"], row["commit"]): row for row in implementations(records, new)}
    assert listed[("pprof_spark", "b92919c0")]["is_the_pinned_build"] == "yes" and listed[("pprof_spark", "7a146242")]["is_the_pinned_build"] == "no"
    assert listed[("pprof_spark_cached", "b92919c0")]["pinned_commit"].startswith("b92919c0") and listed[("pprof_spark", "b92919c0")]["records"] == 1
    assert listed[("pprof_spark", "b92919c0")]["code_sent"] is None
    records[0]["fit"]["extras"] = {"execution": {"code_sent": "by value"}}
    assert {row["code_sent"] for row in implementations(records, new) if row["commit"] == "b92919c0"} == {"by value", None}
    assert [row["caveats"] for row in summary(records, new)] == ["", "executors ranged from 6 to 10 during the fit", NOT_PINNED]
    by = {(m["adapter"], m["runs"]): m for m in medians(records, new)}
    assert by[("pprof_spark", 2)]["runs_without_caveat"] == 1 and by[("pprof_spark", 2)]["median_compare_s_without_caveat"] == 50.0
    assert by[("pprof_spark_cached", 1)]["runs_without_caveat"] == 0


def test_the_engine_has_adapters_with_larger_partitions(spark, tmp_path):
    """Round 19: the same engine at other rows per partition, side by side in one plan (OI-25)."""
    assert {"engine", "engine_50k", "engine_125k", "engine_250k"} <= set(adapters.names())
    assert adapters.get("engine_50k").rows_per_partition == 50_000
    assert adapters.get("engine_125k").rows_per_partition == 125_000 and adapters.get("engine").rows_per_partition is None
    plan = dict(PLAN, adapters=["engine", "engine_125k"])
    records = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))
    by = {r["adapter"]["name"]: r for r in records if r["model"]["family"] == "logistic_fe"}
    assert by["engine"]["status"] == by["engine_125k"]["status"] == "ok"
    assert by["engine_125k"]["fit"]["settings"]["rows_per_partition"] == 125_000 and by["engine_125k"]["adapter"]["rows_per_partition"] == 125_000
    assert by["engine"]["fit"]["coefficients"] == pytest.approx(by["engine_125k"]["fit"]["coefficients"], rel=1e-9, abs=1e-12)


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
    # round 23: the executors are counted again after the reads, just before the fit, and that count is the one flagged
    assert record["warmup"]["before_fit"] == 0 and record["warmup"]["asked"] == 1
    assert record["caveats"] == ["the cluster had 0 of 10 executors when the fit started"]
    during = record["environment"]["during"]
    assert during["executors_min"] == during["executors_max"] == 0 and during["cores_time_average"] > 0
    assert record["environment"]["cluster_changed"] is False


def test_a_fit_on_fewer_executors_than_the_plan_holds_is_flagged(spark, tmp_path, monkeypatch):
    """Round 23: the warm-up can report the full number and the fit still run on fewer from its first
    moment to its last. The count did not move during the fit, so nothing flagged it: on the cluster
    7 of the 64 runs of 2026-11-22 ran on six of ten executors without a caveat."""
    from statcompute.bench import environment
    from statcompute.bench.results import caveats, medians

    short = "the fit ran on 6 of the 10 executors the plan holds"
    held = {"target": 10, "reached": 10, "seconds": 0.0, "rounds": 0}
    assert environment.short_of(held, {"executors_min": 6, "executors_max": 6}) == short
    assert environment.short_of(held, {"executors_min": 10, "executors_max": 10}) is None
    assert environment.short_of(held, {"executors_min": 6, "executors_max": 10}) is None                 # it moved: another caveat says so
    assert environment.short_of(dict(held, reached=7), {"executors_min": 7, "executors_max": 7}) is None  # the warm-up said so already
    assert environment.short_of({"target": 10, "reached": None, "skipped": "local session"}, {"executors_min": 0, "executors_max": 0}) is None
    assert environment.short_of(None, {"executors_min": 6, "executors_max": 6}) is None                  # the plan holds nothing
    # a record written before the caveat existed gets it when it is read, and leaves the runs without a caveat
    old = a_record("engine", "0.20.0", 18.6, {"prepare": 8.0, "fit": 9.0}, [0.1, 0.2])
    old["warmup"] = held
    old["environment"]["during"].update(executors_min=6, executors_max=6)
    fair = a_record("engine", "0.20.0", 13.3, {"prepare": 5.4, "fit": 6.2}, [0.1, 0.2])
    assert caveats(old) == [short] and caveats(dict(old, caveats=[short])) == [short] and caveats(fair) == []
    row = medians([old, fair])[0]
    assert row["runs"] == 2 and row["runs_without_caveat"] == 1 and row["median_compare_s_without_caveat"] == 13.3
    # A run made now counts the executors again after the reads, just before the fit, and says what it saw.
    calls = []

    def hold(spark, executors, timeout):
        calls.append(executors)
        return dict(held, target=executors, reached=executors, seconds=2.0, rounds=1)

    def count(scheduler, active=None):
        active = scheduler if active is None else active
        return {"scheduler": scheduler, "active": active, "cores": 4 * active, "by_cores": active, "executors": min(scheduler, active)}

    monkeypatch.setattr(environment, "hold_at", hold)
    monkeypatch.setattr(environment, "counted", lambda spark: count(6))
    monkeypatch.setattr(environment, "attached", lambda spark: (6, 24))          # what the sampler follows during the fit
    plan = dict(PLAN, adapters=["lab_spark"], options={"hold_executors": 10})
    record = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "a"))[0]
    assert record["status"] == "ok" and calls == [10]
    assert record["warmup"]["before_fit"] == 6 and record["warmup"]["asked"] == 1 and record["warmup"]["counts"]["cores"] == 24
    assert record["caveats"] == ["the cluster had 6 of 10 executors when the fit started"] == caveats(record)
    from statcompute.bench.results import where_time_went
    assert where_time_went([record])[0]["executors_held"] == "10 of 10 in 2 s, 6 before the fit"
    # With `hold_again` it asks again, that many times at most, and adds up what the asking cost (decision D-22).
    del calls[:]
    counts = iter([count(6), count(10)])
    monkeypatch.setattr(environment, "counted", lambda spark: next(counts))
    monkeypatch.setattr(environment, "attached", lambda spark: (10, 40))
    again = run_plan(spark, dict(plan, options={"hold_executors": 10, "hold_again": 1}), str(tmp_path / "data"), str(tmp_path / "b"))[0]
    assert calls == [10, 10] and again["warmup"]["asked"] == 2 and again["warmup"]["before_fit"] == 10
    assert again["warmup"]["seconds"] == 4.0 and again["warmup"]["rounds"] == 2 and again["caveats"] == []
    assert where_time_went([again])[0]["executors_held"] == "10 of 10 in 4 s, asked 2 times"
    del calls[:]
    monkeypatch.setattr(environment, "counted", lambda spark: count(6))
    monkeypatch.setattr(environment, "attached", lambda spark: (6, 24))
    never = run_plan(spark, dict(plan, options={"hold_executors": 10, "hold_again": 2}), str(tmp_path / "data"), str(tmp_path / "c"))[0]
    assert calls == [10, 10, 10] and never["warmup"]["asked"] == 3
    assert never["caveats"] == ["the cluster had 6 of 10 executors when the fit started"]


def test_the_count_before_a_fit_is_the_lowest_of_three_and_waits_first(spark, tmp_path, monkeypatch):
    """Round 24: on the cluster the scheduler's count said ten just before a fit that ran on six and
    counted 24 cores. The count before the fit is now the lowest of the scheduler's block managers,
    the executors listed as active and the scheduler's cores over the cores of one executor, taken
    after a pause (`hold_settle_s`), because the loss comes about a second after the reads end."""
    import time

    from statcompute.bench import environment
    from statcompute.bench.metrics import SparkMetrics
    from statcompute.bench.results import where_time_went

    monkeypatch.setattr(environment, "attached", lambda spark: (10, 24))
    monkeypatch.setattr(SparkMetrics, "executors", lambda self: {"executors": 6, "cores": 24})
    assert environment.counted(spark) == {"scheduler": 10, "active": 6, "cores": 24, "by_cores": 6, "executors": 6}
    monkeypatch.setattr(SparkMetrics, "executors", lambda self: {"executors": 10, "cores": 40})   # the list is late, the cores are not
    assert environment.counted(spark)["by_cores"] == 6 and environment.counted(spark)["executors"] == 6
    monkeypatch.setattr(SparkMetrics, "executors", lambda self: None)                             # no list: the scheduler alone
    assert environment.counted(spark) == {"scheduler": 10, "active": None, "cores": 24, "by_cores": None, "executors": 10}
    monkeypatch.setattr(SparkMetrics, "executors", lambda self: {"executors": 6, "cores": 24})
    monkeypatch.setattr(environment, "hold_at", lambda spark, executors, timeout: {"target": executors, "reached": executors, "seconds": 0.0, "rounds": 0})
    waited, sleep = [], time.sleep
    monkeypatch.setattr(time, "sleep", lambda seconds: (waited.append(seconds), sleep(min(seconds, 0.05)))[1])
    plan = dict(PLAN, adapters=["lab_spark"], options={"hold_executors": 10, "hold_settle_s": 3})
    record = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))[0]
    assert 3.0 in waited and record["status"] == "ok"
    assert record["warmup"]["before_fit"] == 6 and record["warmup"]["counts"]["scheduler"] == 10
    assert "the cluster had 6 of 10 executors when the fit started" in record["caveats"]
    assert where_time_went([record])[0]["executors_held"] == "10 of 10 in 0 s, 6 before the fit (scheduler 10, active 6, by cores 6)"


def test_every_plan_that_holds_the_cluster_asks_again(spark):
    """Decision D-22, approved 2026-11-22."""
    import glob
    import json
    import os

    from statcompute.bench.runner import PLANS

    holding = []
    for path in sorted(glob.glob(os.path.join(PLANS, "*.json"))):
        with open(path) as f:
            options = json.load(f).get("options", {})
        if options.get("hold_executors"):
            holding.append(os.path.basename(path))
            assert options.get("hold_again") == 1 and options.get("hold_settle_s") == 3, path
    assert len(holding) == 9 and "compare_5m_warm.json" in holding and "cox_kernel_5m_warm.json" in holding


def test_the_adapters_of_a_plan_can_take_turns(spark, tmp_path):
    """Round 23: with `interleave` the repeats of the adapters alternate, so that what the cluster does
    over a session does not fall on the adapter that goes first."""
    plan = dict(PLAN, adapters=["lab_numpy", "always_fails", "pprof_py"], repeats=2)
    adapters.register("always_fails", AlwaysFails)
    one_after_another = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "a"))
    assert [(r["adapter"]["name"], r["repeat"]) for r in one_after_another] == [
        ("lab_numpy", 0), ("lab_numpy", 1), ("always_fails", 0), ("pprof_py", 0), ("pprof_py", 1)]
    in_turns = run_plan(spark, dict(plan, options={"interleave": True}), str(tmp_path / "data"), str(tmp_path / "b"))
    assert [(r["adapter"]["name"], r["repeat"]) for r in in_turns] == [
        ("lab_numpy", 0), ("always_fails", 0), ("pprof_py", 0), ("lab_numpy", 1), ("pprof_py", 1)]   # a failed fit is not repeated


def test_without_strata_the_rows_can_be_spread_over_one_partition_per_core(spark, tmp_path):
    """Round 23 (OI-39): two adapters that give a fit without strata as many partitions as a fit with
    strata has by default, one placing the rows by row identifier and sorting them, one by a hash of
    the row without the sort. Same estimates; with strata they stand aside."""
    from statcompute.bench.results import where_time_went

    assert adapters.get("engine_cores").spread == "row_id" and adapters.get("engine_cores_hash").spread == "hash"
    assert adapters.get("engine").spread is None
    plan = {"name": "t", "datasets": [{"family": "cox", "rows": 3000, "providers": 12, "features": 3}],
            "models": {"cox": [{"family": "cox", "ties": "breslow", "strata": None, "convergence": "default"},
                               {"family": "cox", "ties": "breslow", "strata": "provider", "convergence": "default"}]},
            "adapters": ["engine", "engine_input", "engine_cores", "engine_cores_hash", "pprof_py"], "repeats": 1,
            "options": {"interleave": True}}
    records = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))
    plain = {r["adapter"]["name"]: r for r in records if r["model"]["strata"] is None}
    cores = spark.sparkContext.defaultParallelism
    names = ("engine", "engine_input", "engine_cores", "engine_cores_hash")
    placed = {name: plain[name]["fit"]["extras"]["execution"]["rows_placed"] for name in names}
    assert placed == {"engine": "by a hash of the row", "engine_input": "as in the input", "engine_cores": "by row_id",
                      "engine_cores_hash": "by a hash of the row"}                  # the default is the hash layout (decision D-23)
    assert plain["engine"]["fit"]["extras"]["partitions"] == cores and "without_strata" not in plain["engine"]["adapter"]
    assert plain["engine_cores"]["fit"]["extras"]["partitions"] == plain["engine_cores_hash"]["fit"]["extras"]["partitions"] == cores
    assert plain["engine_cores_hash"]["adapter"]["without_strata"].endswith("not sorted")
    compared = compare_records(records, "pprof_py")
    assert sorted(row["adapter"] for row in compared) == ["engine", "engine", "engine_cores", "engine_cores_hash", "engine_input"]
    for row in compared:
        assert row["coefficients_over_T_COEF"] <= 1 and row["like_for_like"], row
    with_strata = {r["adapter"]["name"]: r["status"] for r in records if r["model"]["strata"]}
    assert with_strata == {"engine": "ok", "engine_input": "unsupported", "engine_cores": "unsupported", "engine_cores_hash": "unsupported",
                           "pprof_py": "ok"}
    # the report says how each working set was laid out
    shown = {(row["adapter"], row["model"].split("/")[1]): row for row in where_time_went(records)}
    assert shown[("engine_cores", "unstratified")]["partitions"] == cores and shown[("engine_cores", "unstratified")]["partitions_from"] == "by row_id"
    assert shown[("engine", "stratified")]["partitions_from"] == "one per core" and shown[("engine", "stratified")]["cores_at_start"] == cores
    assert shown[("engine", "unstratified")]["partitions_from"] == "one per core" and shown[("engine", "unstratified")]["cores_at_start"] == cores
    assert shown[("engine_input", "unstratified")]["partitions_from"] == "as in the input"


def test_a_report_can_be_narrowed_to_the_sessions_of_some_versions():
    """Round 23: a results folder keeps every session of a plan; the report of one is a third of the size."""
    from statcompute.bench.results import of_versions, where_time_went

    records = [a_record("engine", version, 10.0, {"prepare": 4.0, "fit": 5.0}, [0.1, 0.2]) for version in ("0.16.0", "0.20.0", "0.20.0")]
    assert len(of_versions(records, None)) == len(of_versions(records, [""])) == 3
    assert [r["statcompute"] for r in of_versions(records, ["0.20.0"])] == ["0.20.0", "0.20.0"]
    assert len(of_versions(records, [" 0.16.0", "0.20.0 "])) == 3 and of_versions(records, ["0.21.0"]) == []
    # what the warm-up reported before a fit is in the table of where the time went
    records[1]["warmup"] = {"target": 10, "reached": 10, "seconds": 0.4, "rounds": 0}
    records[1]["fit"]["extras"] = {"partitions": 24, "execution": {"partitions": 24, "partitions_from": "one per core", "cores": 24}}
    rows = where_time_went(records)
    assert rows[1]["executors_held"] == "10 of 10 in 0 s" and rows[0]["executors_held"] is None
    assert (rows[1]["partitions"], rows[1]["partitions_from"], rows[1]["cores_at_start"]) == (24, "one per core", 24)
    text = render(of_versions(records, ["0.20.0"]), title="t")
    assert "executors_held" in text and "| 24 | one per core | 24 |" in text and "0.16.0" not in text
