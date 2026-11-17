# statcompute

Estimator-preserving computation for classical statistical models. This repository is the
independent workspace of the project: new algorithms are tried here, the existing implementations
are benchmarked here on equal terms, and the next Spark implementation of provider profiling will
be built here.

**Status, version 0.5.0 (2026-11-16).** The engine's first version exists: `CoxPH` and `LogisticFE`,
verified against explicit risk sets and against pprof_py in a local Spark 4.0 session, and not yet
run on a cluster. Around it: the benchmark framework, adapters for the three reference packages,
and the analysis of the first cluster run in `docs/reports/2026-11-16-databricks-compare-5m.md`.
`STATUS.md` has the details.

## Three rules

1. **The existing packages are read-only references.** `pprof` (R), `pprof_py` and `pprof_spark`
   are never copied into, changed from, or pushed to from this repository. Adapters
   call them as installed, at the commits recorded in `statcompute/references/REFERENCES.lock`.
2. **Evidence over reasoning.** A number is quoted with the environment that produced it. A
   comparison is called like-for-like only when both implementations fit exactly the same model.
3. **Synthetic data only.** No protected data, here or in any result that leaves a cluster.

## Layout

| Area | Where | What it holds |
|---|---|---|
| New Spark implementation | `statcompute/engine/` | Cox regression and logistic regression with provider fixed effects on Spark, with NumPy kernels inside partitions. Version 0.1; `docs/spec/engine.md` |
| Experimental algorithms and prototypes | `statcompute/lab/` | NumPy kernels, a one-machine prototype and a Spark prototype of the recommended algorithms; the probes behind the research roadmap |
| Reference implementations and adapters | `statcompute/references/` | One adapter per reference package, and the lock file with their commits |
| Shared test utilities and synthetic data | `statcompute/testkit/` | Dataset and model specifications, the deterministic generator, tolerance classes |
| Correctness and numerical validation | `statcompute/validation/`, `tests/` | Fit-to-fit comparison; tests of the kernels against brute force and of the prototypes against pprof_py |
| Scale and performance benchmarking | `statcompute/bench/`, `notebooks/databricks/` | Plans, the runner, metrics, records, reports; the notebooks that run them on a cluster |
| Results, reports, documentation | `results/`, `docs/` | Run records and reports from synthetic data; architecture, protocol, capability matrix, research roadmap |

`statcompute/adapters.py` defines the one contract everything is run through.

## Using the engine

```python
from statcompute.engine import CoxPH, LogisticFE

cox = CoxPH("time", "event", features, strata="provider", ties="breslow").fit(frame)
cox.summary()                      # feature, estimate, standard_error, z, p_value

logit = LogisticFE("y", features, "provider").fit(frame)
logit.provider_effects             # provider, gamma, standard_error, rows, events, bounded
```

## Quick start on one machine

```
pip install -e ".[test]"                     # needs Java 17+ for PySpark
pytest -q                                     # 66 tests, about three minutes
python -m statcompute.bench run --plan smoke --data-root /tmp/sc/data --results-root /tmp/sc/results
python -m statcompute.bench report --results-root /tmp/sc/results
```

The tests that compare with pprof_py are skipped unless it is installed
(`pip install git+https://github.com/UM-KevinHe/pprof_py@<commit in REFERENCES.lock>`), and those
that run the R package unless `Rscript` finds `pprof`.

## On Databricks

`docs/databricks-runbook.md` gives the steps. In short: a fixed-size cluster on Runtime 17.3 LTS in
dedicated access mode; open the repository as a Git folder; run `notebooks/databricks/00_environment`,
then the `smoke` plan through `01_generate`, `02_run` and `03_report`, then the larger plans.

## Documents

| Document | Purpose |
|---|---|
| `docs/architecture.md` | Structure, dependency rules, how an algorithm moves from probe to engine |
| `docs/benchmark-protocol.md` | What is measured, under which conditions, and what counts as a fair comparison |
| `docs/capabilities.md` | What each implementation supports, read from its source at the pinned commit |
| `docs/reference-inspection.md` | Baseline functionality and the performance evidence so far |
| `docs/databricks-runbook.md` | Running the benchmarks on the standard cluster |
| `docs/research/` | The research roadmap and its evidence |
| `STATUS.md`, `DECISIONS.md`, `OPEN_ITEMS.md` | State of the work; decisions proposed and taken; what is open |

Licence: MIT, see `LICENSE`.
