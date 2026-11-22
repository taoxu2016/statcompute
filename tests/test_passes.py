"""The probe of what a pass costs (statcompute.lab.passes, OPEN_ITEMS OI-40 and OI-35). It measures the
engine's working set with lines of its own, so the first thing tested is that the two are the same."""
import pytest

from statcompute.bench import run_plan
from statcompute.bench.report import render
from statcompute.bench.results import medians, pass_costs
from statcompute.engine import CoxPH
from statcompute.lab import passes
from statcompute.testkit import DatasetSpec, materialize
from statcompute.validation import compare_records


def test_the_probe_evaluates_on_the_working_set_the_engine_builds(spark, tmp_path):
    spec = DatasetSpec.from_dict({"family": "cox", "rows": 6000, "providers": 30, "features": 5})
    dataset = materialize(spark, spec, str(tmp_path / "data"))
    frame, names = dataset.frame(spark), spec.feature_names()
    model = CoxPH("time", "event", names, "provider", row_id="id").fit(frame)
    out = passes.measure(spark, frame, "cox", names, dataset.rows, model.coefficients, repeats=2)
    # the same rows in the same partitions in the same order give the same bits as the engine's last evaluation
    assert out["value"] == model.log_likelihood
    assert out["partitions"] == model.partitions and out["partitions_from"] == "one per core" and out["repeats"] == 2
    assert out["kinds"] == list(passes.KINDS) and out["build_s"] > 0
    for kind in passes.KINDS:
        found = out["passes"][kind]
        assert len(found["seconds"]) == 2 and all(s > 0 for s in found["seconds"]) and len(found["cores"]) == 2, kind
    P = out["partitions"]
    assert out["passes"]["job"]["tasks"] == [] and out["passes"]["evaluation"]["tasks"] == []      # these two have no tasks that time themselves
    assert out["passes"]["idle"]["tasks"] == out["passes"]["rows"]["tasks"] == out["passes"]["kernel"]["tasks"] == [P, P]
    assert all(0 < tasks <= (P + 1) // 2 for tasks in out["passes"]["half"]["tasks"])
    assert out["passes"]["rows"]["kernel_median_s"] == [0.0, 0.0] and all(s > 0 for s in out["passes"]["kernel"]["kernel_median_s"])
    assert all(a >= b for a, b in zip(out["passes"]["kernel"]["task_max_s"], out["passes"]["kernel"]["kernel_max_s"]))
    # the engine's kernel gives the evaluation's log-likelihood with or without the BLAS limit, the prototype's to rounding
    values = out["kernel_values"]
    assert values["kernel"] == pytest.approx(out["value"], rel=1e-12) and values["kernel_free"] == pytest.approx(out["value"], rel=1e-12)
    assert values["kernel_plain"] == pytest.approx(out["value"], rel=1e-9)
    # only some kinds on request, and no prototype kernel for the logistic model
    few = passes.measure(spark, frame, "cox", names, dataset.rows, model.coefficients, repeats=1, kinds=("job", "evaluation"))
    assert few["kinds"] == ["job", "evaluation"] and few["value"] == model.log_likelihood


def test_a_plan_runs_the_probe_and_the_report_shows_what_a_pass_costs(spark, tmp_path):
    plan = {"name": "p",
            "datasets": [{"family": "cox", "rows": 4000, "providers": 20, "features": 4},
                         {"family": "logistic_fe", "rows": 4000, "providers": 20, "features": 4}],
            "models": {"cox": [{"family": "cox", "ties": "breslow", "strata": "provider", "convergence": "default"},
                               {"family": "cox", "ties": "breslow", "strata": None, "convergence": "default"}],
                       "logistic_fe": [{"family": "logistic_fe", "convergence": "default"}]},
            "adapters": ["engine_passes", "engine", "lab_numpy"], "repeats": 1, "options": {"pass_repeats": 2}}
    records = run_plan(spark, plan, str(tmp_path / "data"), str(tmp_path / "results"))
    probes = {r["model"]["label"]: r for r in records if r["adapter"]["name"] == "engine_passes"}
    assert {label: r["status"] for label, r in probes.items()} == {
        "cox/stratified/breslow/default": "ok", "cox/unstratified/breslow/default": "unsupported", "logistic_fe/default": "ok"}
    cox, logistic = probes["cox/stratified/breslow/default"], probes["logistic_fe/default"]
    assert cox["adapter"]["role"] == "probe" and cox["fit"]["coefficients"] == [] and cox["truth"] is None
    assert cox["fit"]["extras"]["pass_kinds"] == list(passes.KINDS) and cox["fit"]["extras"]["pass_repeats"] == 2
    assert logistic["fit"]["extras"]["pass_kinds"] == [kind for kind in passes.KINDS if kind != "kernel_plain"]
    assert cox["fit"]["extras"]["execution"]["partitions_from"] == "one per core" and cox["timing"]["phases"]["prepare"] > 0
    rows = pass_costs(records)
    assert [row["kind"] for row in rows] == list(passes.KINDS) + [kind for kind in passes.KINDS if kind != "kernel_plain"]
    kernel = next(row for row in rows if row["kind"] == "kernel" and row["model"].startswith("cox"))
    assert kernel["passes"] == 2 and kernel["tasks"] == cox["fit"]["extras"]["partitions"] and kernel["kernel_s"] is not None
    assert kernel["around_tasks_s"] is not None and kernel["median_s"] >= kernel["slowest_task_s"]
    job = next(row for row in rows if row["kind"] == "job")
    assert job["tasks"] is None and job["around_tasks_s"] is None and job["median_s"] > 0
    text = render(records, "lab_numpy")
    assert "## What a pass costs" in text and "| kernel_plain |" in text and "| half |" in text
    # a probe has no coefficients: nothing is compared with it, and its seconds are not among the fits' medians
    compared = compare_records(records, "lab_numpy")
    assert {row["adapter"] for row in compared} == {"engine"} and len(compared) == 3
    assert all(row["adapter"] != "engine_passes" for row in medians(records))
    assert "## What a pass costs" not in render([r for r in records if r["adapter"]["name"] != "engine_passes"], "lab_numpy")
