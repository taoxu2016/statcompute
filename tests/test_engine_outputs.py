"""The engine's outputs beyond the fit, each against pprof_py: standardized ratios and provider tests
for the logistic model, the baseline hazard for Cox, and the linear predictor."""
import numpy as np
import pandas as pd
import pytest

from statcompute.engine import CoxPH, LogisticFE
from test_engine_kernels import logistic_data, survival_data

pprof_py = pytest.importorskip("pprof_py")
NAMES = [f"x{k}" for k in range(1, 5)]


def gap(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def test_standardized_ratios_and_provider_tests_match_pprof_py(spark):
    X, y, provider = logistic_data(n=6000, m=12)
    y[provider == 3] = 0.0                                      # one provider at the bound
    theirs = pprof_py.LogisticFixedEffectModel()
    theirs.fit(X, y, provider, use_dataprep=False, tol=1e-12)
    measures = theirs.calculate_standardized_measures(stdz="indirect")["indirect"]
    score, wald = theirs.test(test_method="score"), theirs.test(test_method="wald")
    frame = spark.createDataFrame(pd.DataFrame(X, columns=NAMES).assign(y=y, provider=provider, id=np.arange(len(y))))
    ours = LogisticFE("y", NAMES, "provider", row_id="id", tol=1e-11, rows_per_partition=2000).fit(frame).provider_effects
    assert list(ours["provider"]) == list(measures["provider_id"]) and bool(ours["bounded"][3])
    assert gap(ours["events"], measures["observed"]) == 0 and gap(ours["expected"], measures["expected"]) < 1e-9
    assert gap(ours["indirect_ratio"], measures["indirect_ratio"]) < 1e-10
    assert gap(ours["indirect_rate"], measures["indirect_rate"]) < 1e-9
    assert gap(ours["score_z"], score["z_raw"]) < 1e-9 and gap(ours["score_p"], score["p_value"]) < 1e-10
    assert list(ours["score_flag"]) == list(score["flag"])
    assert gap(ours["standard_error_case_mix"] ** 2, np.ravel(theirs.variances_["gamma_case_mix"])) < 1e-9 * 1e2
    assert gap(ours["wald_z"], wald["z_raw"]) < 1e-7 and list(ours["wald_flag"]) == list(wald["flag"])


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("stratified", [True, False])
def test_the_baseline_hazard_matches_pprof_py(spark, ties, stratified):
    X, time, event, strata, _, w, _, _ = survival_data(n=3000, strata=6, seed=21)
    frame = spark.createDataFrame(pd.DataFrame(X, columns=NAMES).assign(time=time, event=event, s=strata, w=w,
                                                                       id=np.arange(len(time))))
    ours = CoxPH("time", "event", NAMES, "s" if stratified else None, ties=ties, weights="w", row_id="id", tol=1e-11,
                 baseline=True, rows_per_partition=1000).fit(frame)
    theirs = pprof_py.CoxPH(ties=ties, eps=1e-13, max_iter=50)
    theirs.fit(X, duration=time, event=event, strata=strata if stratified else None, sample_weight=w)
    reference = theirs.baseline_hazard_.sort_values(["stratum", "time"] if stratified else ["time"])
    table = ours.baseline_hazard
    assert list(table.columns) == (["s"] if stratified else []) + ["time", "hazard", "survival"]
    assert np.array_equal(table["time"].to_numpy(), reference["time"].to_numpy())
    assert np.max(np.abs(table["hazard"].to_numpy() / reference["hazard"].to_numpy() - 1)) < 1e-8
    assert gap(table["survival"], reference["survival"]) < 1e-8
    assert CoxPH("time", "event", NAMES, "s").fit(frame).baseline_hazard is None        # only on request


def test_the_linear_predictor_column(spark):
    X, time, event, strata, _, _, _, _ = survival_data(n=1000, strata=4, seed=22)
    frame = spark.createDataFrame(pd.DataFrame(X, columns=NAMES).assign(time=time, event=event, s=strata))
    model = CoxPH("time", "event", NAMES, "s").fit(frame)
    column = model.linear_predictor(frame).select("linear_predictor").toPandas()["linear_predictor"].to_numpy()
    assert gap(np.sort(column), np.sort(X @ model.coefficients)) < 1e-12


def test_the_exact_test_and_direct_standardization_match_pprof_py(spark):
    X, y, provider = logistic_data(n=6000, m=12)
    y[provider == 3] = 0.0
    theirs = pprof_py.LogisticFixedEffectModel()
    theirs.fit(X, y, provider, use_dataprep=False, tol=1e-12)
    exact = theirs.test(test_method="poibin_exact")
    direct = theirs.calculate_standardized_measures(stdz="direct")["direct"]
    frame = spark.createDataFrame(pd.DataFrame(X, columns=NAMES).assign(y=y, provider=provider, id=np.arange(len(y))))
    ours = LogisticFE("y", NAMES, "provider", row_id="id", tol=1e-11, direct=True, rows_per_partition=2000).fit(frame).provider_effects
    assert gap(ours["exact_p"], exact["p_value"]) < 1e-12              # absolute: the smallest values are of order 1e-60
    assert list(ours["exact_flag"]) == [int(f) for f in exact["flag"]]
    assert gap(ours["direct_expected"], direct["expected"]) < 1e-9
    assert gap(ours["direct_ratio"], direct["direct_ratio"]) < 1e-12 and gap(ours["direct_rate"], direct["direct_rate"]) < 1e-10
    plain = LogisticFE("y", NAMES, "provider", row_id="id").fit(frame).provider_effects
    assert "direct_ratio" not in plain.columns and "exact_p" in plain.columns          # direct only on request


def test_cox_robust_variance_residuals_and_standardized_ratios_match_pprof_py(spark):
    X, time, event, strata, _, _, _, _ = survival_data(n=3000, strata=6, seed=21)
    frame = spark.createDataFrame(pd.DataFrame(X, columns=NAMES).assign(time=time, event=event, s=strata, id=np.arange(len(time))))
    ours = CoxPH("time", "event", NAMES, "s", row_id="id", tol=1e-11, robust=True, residuals=True, standardized=True,
                 rows_per_partition=1000).fit(frame)
    theirs = pprof_py.CoxPH(ties="breslow", eps=1e-13, max_iter=50, robust=True)
    theirs.fit(X, duration=time, event=event, strata=strata)
    assert np.max(np.abs(ours.robust_covariance / theirs.robust_covariance_ - 1)) < 1e-6
    assert np.max(np.abs(ours.robust_standard_errors / theirs.standard_errors_ - 1)) < 1e-6
    assert list(ours.residuals["row_id"]) == list(range(len(time)))
    assert gap(ours.residuals["martingale"], theirs.martingale_residuals_) < 1e-7
    measures = theirs.calculate_standardized_measures(X, duration=time, event=event, provider_id=strata, stdz="indirect")["indirect"]
    table = ours.standardized
    assert list(table.columns) == ["s", "observed", "expected", "indirect_ratio", "person_time"]
    assert gap(table["observed"], measures["observed"]) == 0 and gap(table["person_time"], measures["person_time"]) < 1e-9
    assert gap(table["expected"], measures["expected"]) < 1e-6 and gap(table["indirect_ratio"], measures["indirect_ratio"]) < 1e-7
    assert abs(table["expected"].sum() - table["observed"].sum()) < 1e-8 * table["observed"].sum()   # expected events add up to the observed
    with pytest.raises(NotImplementedError):
        CoxPH("time", "event", NAMES, "s", ties="efron", robust=True)
    with pytest.raises(ValueError, match="row_id"):
        CoxPH("time", "event", NAMES, "s", residuals=True)
