"""pprof (the R package, one machine): https://github.com/UM-KevinHe/pprof

The adapter brings the stored dataset to the driver, writes it as raw doubles to a temporary
folder, and runs `pprof_r_fit.R` with Rscript: `pp_data()` and then `fit_cox()` (which fits through
survival::coxph) or `fit_logistic_fe()`. Statistics are the R package's; nothing is reimplemented.
Set STATCOMPUTE_RSCRIPT to use an Rscript other than the one on the PATH.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
import time as clock

import numpy as np

from ..adapters import EXACT, UNSUPPORTED, Adapter, FitResult, Support, register
from ..testkit.localdata import footprint_bytes, load_arrays
from ..testkit.specs import COX

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pprof_r_fit.R")
_PROBE = {}


def _rscript() -> str:
    return os.environ.get("STATCOMPUTE_RSCRIPT", "Rscript")


def _probe() -> dict:
    """Versions of R and of pprof as this session's Rscript sees them, or the reason they are missing."""
    key = _rscript()
    if key not in _PROBE:
        if shutil.which(key) is None:
            _PROBE[key] = {"error": f"{key} is not on the PATH"}
        else:
            code = 'suppressMessages(library(pprof)); cat(as.character(packageVersion("pprof")), R.version.string, sep = "|")'
            try:
                done = subprocess.run([key, "--vanilla", "-e", code], capture_output=True, text=True, timeout=300)
                if done.returncode == 0 and "|" in done.stdout:
                    version, r_version = done.stdout.strip().split("|", 1)
                    _PROBE[key] = {"version": version, "r": r_version}
                else:
                    _PROBE[key] = {"error": "the R package pprof does not load: " + done.stderr.strip()[-300:]}
            except Exception as error:
                _PROBE[key] = {"error": f"{type(error).__name__}: {error}"}
    return _PROBE[key]


class PprofR(Adapter):
    name, kind, role = "pprof_r", "driver", "reference"

    def describe(self, spark=None):
        info = {"name": self.name, "kind": self.kind, "role": self.role}
        info.update({k: v for k, v in _probe().items() if k != "error"})
        if "error" in _probe():
            info["note"] = _probe()["error"]
        return info

    def available(self, spark=None) -> Support:
        probe = _probe()
        return Support(UNSUPPORTED, [probe["error"]]) if "error" in probe else Support()

    def support(self, model, data) -> Support:
        if model.family == COX:
            return Support(EXACT, ["fits through survival::coxph, whose stopping rule fit_cox() does not expose: "
                                   "relative change in the log partial likelihood below 1e-9, at most 20 iterations"])
        return Support(EXACT, ["SerBIN on one thread; stops when the largest change in beta is below tol "
                               "(default 1e-5); provider effects are bounded at the median +/- 10"])

    def driver_bytes(self, model, data) -> int:
        # In R: the matrix read, the data frame, pp_data()'s sorted copy and the model's own copies.
        # A guess until measured; every record reports R's own peak (extras.peak_r_mb).
        return footprint_bytes(data.rows, data.features, 5.0)

    def fit(self, spark, dataset, model) -> FitResult:
        spec = dataset.spec
        names = spec.feature_names()
        started = clock.perf_counter()
        others = ["provider", "time", "event"] if model.family == COX else ["provider", "y"]
        X, cols = load_arrays(spark, dataset, names, others)
        loaded = clock.perf_counter()
        folder = tempfile.mkdtemp(prefix="statcompute_r_")
        try:
            with open(os.path.join(folder, "x.bin"), "wb") as f:
                for j in range(X.shape[1]):               # column by column: R matrices are column-major
                    np.ascontiguousarray(X[:, j], dtype="<f8").tofile(f)
            for name in others:
                np.ascontiguousarray(cols[name], dtype="<f8").tofile(os.path.join(folder, f"{name}.bin"))
            rows = X.shape[0]
            del X, cols
            gc.collect()
            exported = clock.perf_counter()
            tol = 1e-12 if model.convergence == "tight" else 1e-5
            command = [_rscript(), "--vanilla", SCRIPT, folder, "cox" if model.family == COX else "logistic",
                       str(rows), str(len(names)), "provider" if model.strata else "none", model.ties,
                       repr(tol), str(model.max_iterations or 10000)]
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError("Rscript failed: " + (done.stderr.strip() or done.stdout.strip())[-1500:])
            with open(os.path.join(folder, "result.json")) as f:
                out = json.load(f)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        seconds = out["seconds"]
        return FitResult(
            names, out["coefficients"], out["standard_errors"], out["log_likelihood"],
            None if out["iterations"] is None else int(out["iterations"]), bool(out["converged"]), out["settings"],
            {"load": loaded - started, "export": exported - loaded, "r_read": seconds["read"],
             "r_prepare": seconds["prepare"], "fit": seconds["fit"]},
            {"peak_r_mb": out["peak_r_mb"], "r_version": out["r_version"], "pprof_version": out["pprof_version"],
             "survival_version": out["survival_version"]},
        )


register("pprof_r", PprofR)
