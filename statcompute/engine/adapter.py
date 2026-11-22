"""The engine in the harness, under the name "engine"."""
from __future__ import annotations

from .. import __version__
from ..adapters import EXACT, UNSUPPORTED, Adapter, FitResult, Support, register
from ..testkit.specs import COX


SPREAD = {"row_id": "one partition per core, the rows placed by row identifier and sorted by it",
          "hash": "one partition per core, the rows placed by a hash of the row and not sorted",
          "input": "the rows left in the input's own partitions, the default until decision D-23"}


class Engine(Adapter):
    name, kind, role = "engine", "spark", "engine"
    rows_per_partition = None                              # None: the engine's own default, one partition per core (D-21)
    spread = None                                          # without strata. None: the engine's default, one per core by a hash (D-23)

    def describe(self, spark=None):
        out = {"name": self.name, "kind": self.kind, "role": self.role, "version": __version__,
               "rows_per_partition": self.rows_per_partition,
               "algorithm": "weighted-Gram Cox; profile-likelihood logistic with providers at the bound; "
                            "NumPy kernels per partition, one pass per evaluation"}
        if self.spread:
            out["without_strata"] = SPREAD[self.spread]
        return out

    def support(self, model, data) -> Support:
        notes = ["stops when the next Newton step is below the tolerance in standard errors (T_CONV)"]
        if model.family == COX and model.strata is None:
            notes.append("without strata: the time grid, two passes per evaluation")
            if self.spread:
                notes.append(SPREAD[self.spread])
        elif self.spread:
            return Support(UNSUPPORTED, ["a layout for fits without strata: with strata or providers this adapter "
                                         "would be the engine itself"])
        return Support(EXACT, notes)

    def fit(self, spark, dataset, model) -> FitResult:
        from . import CoxPH, LogisticFE

        names = dataset.spec.feature_names()
        frame = dataset.frame(spark)
        options = {"tol": 1e-10 if model.convergence == "tight" else 1e-8, "row_id": "id",
                   "max_iterations": model.max_iterations or 50}
        if self.options.get("engine_rows_per_partition") or self.rows_per_partition:
            options["rows_per_partition"] = int(self.options.get("engine_rows_per_partition") or self.rows_per_partition)
        if model.family == COX:
            if not model.strata and self.spread == "input":
                options["partitions"] = "input"
            elif not model.strata and self.spread:
                # As many partitions as a fit with strata gets by default (decision D-21): the cores at this moment,
                # more by size. Given a number, the engine places the rows by the row identifier and sorts them by
                # it; without a row identifier it places them by a hash of the row and does not sort.
                from .spark import working_partitions

                options["partitions"] = working_partitions(spark, dataset.rows, len(names))[0]
                if self.spread == "hash":
                    options["row_id"] = None
            elif not model.strata and self.options.get("engine_partitions"):   # "input" (the default), a number, or "rows"
                options["partitions"] = self.options["engine_partitions"]
            fitted = CoxPH("time", "event", names, "provider" if model.strata else None, ties=model.ties,
                           **options).fit(frame)
            extras = {"layout": fitted.settings["layout"], "distinct_times": fitted.settings.get("distinct_times")}
        else:
            fitted = LogisticFE("y", names, "provider", **options).fit(frame)
            extras = {"providers": fitted.providers, "bounded_low": fitted.bounded_low,
                      "bounded_high": fitted.bounded_high, "median_gamma": fitted.median_gamma}
            if fitted.provider_effects is not None:       # which providers are flagged, as counts and a digest (D-15)
                from ..validation.flags import flag_summary

                extras["flags"] = flag_summary(fitted.provider_effects, key="provider", level=0.95)
        extras.update(evaluations=fitted.evaluations, newton_step_in_se=fitted.newton_step_in_se,
                      partitions=fitted.partitions, events=fitted.events,
                      evaluation_seconds=[round(s, 3) for s in fitted.evaluation_seconds],
                      execution=fitted.execution)
        return FitResult(names, [float(v) for v in fitted.coefficients], [float(v) for v in fitted.standard_errors],
                         fitted.log_likelihood, int(fitted.iterations), bool(fitted.converged),
                         {k: v for k, v in fitted.settings.items() if v is not None}, dict(fitted.seconds), extras)


class Engine50k(Engine):
    """The engine with 50,000 rows per partition, its default until version 0.19.0: at 5 million
    rows, 100 tasks per pass. Kept to set beside the default of decision D-21."""
    name, rows_per_partition = "engine_50k", 50_000


class Engine125k(Engine):
    """The engine with 125,000 rows per partition: at 5 million rows, 40 tasks per pass. Beside
    the others in one plan it shows what a task costs on a cluster apart from its rows (OPEN_ITEMS
    OI-25)."""
    name, rows_per_partition = "engine_125k", 125_000


class Engine250k(Engine):
    """As `engine_125k`, with 250,000 rows per partition: 20 tasks per pass at 5 million rows."""
    name, rows_per_partition = "engine_250k", 250_000


class EngineCores(Engine):
    """Without strata: the rows spread over one partition per core instead of left in the input's
    partitions, placed by row identifier and sorted by it (OPEN_ITEMS OI-39). The number is the one
    a fit with strata gets by default: the cores when the fit starts, more when a partition would
    exceed 200 MB of features. Only for fits without strata."""
    name, spread = "engine_cores", "row_id"


class EngineCoresHash(Engine):
    """As `engine_cores`, with the rows placed by a hash of the row and not sorted, which is what
    the engine does when it is given no row identifier: the shuffle without the sort. Since
    decision D-23 this is the layout of the engine's default; the adapter stays for the records
    made under its name."""
    name, spread = "engine_cores_hash", "hash"


class EngineInput(Engine):
    """Without strata: the rows left in the input's own partitions, which was the default until
    decision D-23. Only for fits without strata."""
    name, spread = "engine_input", "input"


register("engine", Engine)
register("engine_input", EngineInput)
register("engine_cores", EngineCores)
register("engine_cores_hash", EngineCoresHash)
register("engine_50k", Engine50k)
register("engine_125k", Engine125k)
register("engine_250k", Engine250k)
