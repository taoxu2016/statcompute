# CLAUDE.md: rules for AI assistants working in this repository

- Read `docs/architecture.md` and `docs/benchmark-protocol.md` before a non-trivial change.
- `pprof` (R), `pprof_py` and `pprof_spark` are read-only references. Never copy their source
  into this repository, never change them, never commit or push to them. Read them from a clone;
  install them from an exported copy or from their repository at the commit in
  `statcompute/references/REFERENCES.lock`.
- Evidence only: never state that something passes, matches or is faster without the run that shows
  it, and say where the run happened (sandbox, CI, which cluster).
- Synthetic data only. No row values in logs, messages or results.
- A comparison between implementations states whether it is like-for-like. Differences in
  supported functionality go into `docs/capabilities.md`, not into a footnote.
- Statistical and design decisions are proposed in `DECISIONS.md`; the maintainer takes them.
- The engine is the Python package `statcompute.engine` (D-08). It imports nothing from `lab`,
  `references` or `bench`. A change to an estimator comes with a test against pprof_py.
- End of a round: update `STATUS.md`, `OPEN_ITEMS.md` and `DECISIONS.md`.

Commands: `pytest -q`; `python -m statcompute.bench run --plan smoke --data-root D --results-root R`.
