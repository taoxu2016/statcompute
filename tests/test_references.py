"""Cross-implementation agreement on small data: pprof_py (the validated reference) against the
lab prototypes, each at its tightest tolerance. Skipped where pprof_py is not installed."""
from dataclasses import asdict

import pytest

from statcompute import adapters
from statcompute.testkit import DatasetSpec, ModelSpec, materialize
from statcompute.validation import compare_fits

pytest.importorskip("pprof_py")


def agreement(spark, data_root, spec, model, others):
    dataset = materialize(spark, spec, data_root)
    reference = asdict(adapters.get("pprof_py").fit(spark, dataset, model))
    assert reference["log_likelihood"] is not None and len(reference["coefficients"]) == spec.features
    for name in others:
        fit = asdict(adapters.get(name).fit(spark, dataset, model))
        c = compare_fits(fit, reference)
        assert c["coefficients_over_T_COEF"] <= 1, (name, c)
        assert c["standard_errors_over_T_VAR"] <= 1, (name, c)
        assert c["log_likelihood_over_T_LOGLIK"] <= 1, (name, c)


@pytest.mark.parametrize("design", ["simple", "correlated"])
def test_cox_stratified_breslow(spark, data_root, design):
    agreement(spark, data_root, DatasetSpec("cox", 6000, 20, 5, design=design),
              ModelSpec("cox", convergence="tight"), ["lab_numpy", "lab_spark"])


def test_cox_unstratified(spark, data_root):
    agreement(spark, data_root, DatasetSpec("cox", 6000, 20, 5), ModelSpec("cox", strata=None, convergence="tight"),
              ["lab_numpy"])


@pytest.mark.parametrize("design", ["simple", "correlated"])
def test_logistic_fixed_effects(spark, data_root, design):
    agreement(spark, data_root, DatasetSpec("logistic_fe", 8000, 20, 5, design=design),
              ModelSpec("logistic_fe", convergence="tight"), ["lab_numpy", "lab_spark"])


# ---- the R package, where Rscript and pprof are installed -------------------------------------
needs_r = pytest.mark.skipif(not adapters.get("pprof_r").available().runnable, reason="Rscript with pprof is not available")


@needs_r
def test_r_logistic_fixed_effects_agree_with_pprof_py(spark, data_root):
    dataset = materialize(spark, DatasetSpec("logistic_fe", 8000, 20, 5), data_root)
    model = ModelSpec("logistic_fe", convergence="tight")
    reference = asdict(adapters.get("pprof_py").fit(spark, dataset, model))
    c = compare_fits(asdict(adapters.get("pprof_r").fit(spark, dataset, model)), reference)
    assert c["coefficients_over_T_COEF"] <= 1 and c["standard_errors_over_T_VAR"] <= 1, c
    assert c["log_likelihood_over_T_LOGLIK"] <= 1, c


@needs_r
@pytest.mark.parametrize("strata", ["provider", None])
def test_r_cox_agrees_with_pprof_py_as_far_as_its_fixed_stopping_rule_allows(spark, data_root, strata):
    dataset = materialize(spark, DatasetSpec("cox", 6000, 20, 5), data_root)
    model = ModelSpec("cox", strata=strata, convergence="tight")
    reference = asdict(adapters.get("pprof_py").fit(spark, dataset, model))
    fit = adapters.get("pprof_r").fit(spark, dataset, model)
    c = compare_fits(asdict(fit), reference)
    assert fit.converged and fit.extras["pprof_version"]
    assert c["max_coefficient_difference_in_se"] < 1e-6 and c["max_rel_se_difference"] < 1e-6, c
    assert c["log_likelihood_over_T_LOGLIK"] <= 1, c
