"""The new implementation: Cox regression and logistic regression with provider fixed effects on
Spark, with NumPy kernels inside partitions that hold whole strata or providers.

    from statcompute.engine import CoxPH, LogisticFE
    model = CoxPH("time", "event", features, strata="provider").fit(frame)
    model.summary()

`statcompute.engine.local` fits the same models from arrays on one machine, with the same kernels.
See docs/spec/engine.md for the estimators, the layout and what is not implemented yet.
"""
from .record import executor_libraries
from .spark import CoxModel, CoxPH, LogisticFE, LogisticModel

__all__ = ["CoxPH", "CoxModel", "LogisticFE", "LogisticModel", "executor_libraries"]
