"""Provider-level flags between two computations of the same fit: the rule proposed as D-15."""
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from statcompute.engine import LogisticFE
from statcompute.testkit import DatasetSpec, materialize
from statcompute.validation import compare_flags, flag_summary, flags_across_records

CRITICAL = NormalDist().inv_cdf(0.975)
FEATURES = [f"x{k}" for k in range(1, 6)]


def provider_tests(z, p):
    z, p = np.asarray(z, dtype=float), np.asarray(p, dtype=float)
    flag = np.where(z > CRITICAL, 1, np.where(z < -CRITICAL, -1, 0))
    return pd.DataFrame({"provider": np.arange(len(z)) * 3 + 1, "score_z": z, "score_flag": flag, "wald_z": z, "wald_flag": flag,
                         "exact_p": p, "exact_flag": np.where(p < 0.05, np.sign(z), 0).astype(int)})


def scattered(n=500, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 2.0, n), rng.random(n)


def test_a_flag_may_differ_only_for_a_provider_on_the_cut_off():
    z, p = scattered()
    base = provider_tests(z, p)
    same = compare_flags(provider_tests(z + 1e-11, p), base)
    assert set(same) == {"score", "wald", "exact"} and same["score"]["providers"] == 500
    assert all(v["flags_that_differ"] == 0 and v["stable"] for v in same.values())
    assert same["score"]["largest_difference_in_statistic"] < 2e-11
    # moved across the cut-off by less than the tolerance: the flag differs, which is accepted and counted
    below, above = z.copy(), z.copy()
    below[10], above[10] = CRITICAL - 2e-8, CRITICAL + 2e-8
    near = compare_flags(provider_tests(above, p), provider_tests(below, p))["score"]
    assert near["flags_that_differ"] == 1 and near["of_them_borderline"] == 1 and near["of_them_not"] == 0
    assert near["stable"] and near["within_tolerance_of_cutoff"] == 1 and near["smallest_distance_from_cutoff"] < 3e-8
    # moved by more: a disagreement between the two computations
    above[10] = CRITICAL + 1e-3
    far = compare_flags(provider_tests(above, p), provider_tests(below, p))["score"]
    assert far["flags_that_differ"] == 1 and far["of_them_not"] == 1 and not far["stable"]
    # the exact test, on the scale of its p-value
    under, over = p.copy(), p.copy()
    under[20], over[20] = 0.05 - 4e-10, 0.05 + 4e-10
    exact = compare_flags(provider_tests(z, under), provider_tests(z, over))["exact"]
    assert exact["flags_that_differ"] == 1 and exact["of_them_borderline"] == 1 and exact["stable"]
    under[20] = 0.05 - 1e-6
    assert not compare_flags(provider_tests(z, under), provider_tests(z, over))["exact"]["stable"]
    # a looser tolerance can be asked for; other providers are an error; a missing test is left out
    assert compare_flags(provider_tests(z, under), provider_tests(z, over), tolerance={"p": 1e-5})["exact"]["stable"]
    with pytest.raises(ValueError, match="same providers"):
        compare_flags(base.iloc[1:], base)
    assert set(compare_flags(base.drop(columns=["exact_p"]), base)) == {"score", "wald"}


def test_a_record_keeps_enough_to_tell_whether_runs_flagged_the_same_providers():
    z, p = scattered(seed=8)
    first, again = flag_summary(provider_tests(z, p)), flag_summary(provider_tests(z + 1e-11, p).sample(frac=1.0, random_state=1))
    assert first["score"]["digest"] == again["score"]["digest"]                 # the order of the rows does not matter
    assert first["score"]["above"] == int(np.sum(z > CRITICAL)) and first["score"]["below"] == int(np.sum(z < -CRITICAL))
    assert first["exact"]["smallest_distance_from_cutoff"] == pytest.approx(np.min(np.abs(p - 0.05)))
    moved = z.copy()
    moved[3] = 5.0 if z[3] < CRITICAL else 0.0
    other = flag_summary(provider_tests(moved, p))
    assert other["score"]["digest"] != first["score"]["digest"] and other["exact"]["digest"] == first["exact"]["digest"]
    record = lambda flags, status="ok": {"status": status, "dataset": {"id": "d"}, "model": {"label": "m"},  # noqa: E731
                                         "adapter": {"name": "engine"}, "fit": {"extras": {"flags": flags}}}
    rows = flags_across_records([record(first), record(again), record(other), record(other, "failed"),
                                 {"status": "ok", "dataset": {"id": "d"}, "model": {"label": "m"}, "adapter": {"name": "x"}, "fit": {"extras": {}}}])
    assert len(rows) == 1 and rows[0]["runs"] == 3 and rows[0]["adapter"] == "engine"
    assert rows[0]["score_same_in_all_runs"] is False and rows[0]["exact_same_in_all_runs"] is True
    assert rows[0]["score_flagged"] == f"{first['score']['above']} above, {first['score']['below']} below"


def test_the_engines_flags_do_not_depend_on_the_layout_and_equal_pprof_pys(spark, data_root):
    pprof_py = pytest.importorskip("pprof_py")
    frame = materialize(spark, DatasetSpec("logistic_fe", 8000, 40, 5), data_root).frame(spark)
    fit = lambda rows: LogisticFE("y", FEATURES, "provider", row_id="id", tol=1e-11, rows_per_partition=rows).fit(frame).provider_effects  # noqa: E731
    ours, relaid = fit(3000), fit(1100)
    between = compare_flags(relaid, ours)
    assert set(between) == {"score", "wald", "exact"}
    for test, result in between.items():
        assert result["flags_that_differ"] == 0 and result["stable"] and result["providers"] == 40, test
        assert result["largest_difference_in_statistic"] < 1e-9, test
    pdf = frame.orderBy("id").toPandas()
    theirs = pprof_py.LogisticFixedEffectModel()
    theirs.fit(pdf[FEATURES].to_numpy(), pdf["y"].to_numpy(dtype=float), pdf["provider"].to_numpy(), use_dataprep=False, tol=1e-12)
    score, wald, exact = (theirs.test(test_method=m) for m in ("score", "wald", "poibin_exact"))
    reference = pd.DataFrame({"provider": ours["provider"].to_numpy(),
                              "score_z": score["z_raw"].to_numpy(), "score_flag": score["flag"].to_numpy(dtype=int),
                              "wald_z": wald["z_raw"].to_numpy(), "wald_flag": wald["flag"].to_numpy(dtype=int),
                              "exact_p": exact["p_value"].to_numpy(), "exact_flag": exact["flag"].to_numpy(dtype=int)})
    against = compare_flags(ours, reference)
    for test, result in against.items():
        assert result["flags_that_differ"] == 0 and result["stable"], test
    assert against["score"]["largest_difference_in_statistic"] < 1e-9 and against["wald"]["largest_difference_in_statistic"] < 1e-7
    assert against["exact"]["largest_difference_in_statistic"] < 1e-12
