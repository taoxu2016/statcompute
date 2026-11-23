"""How the package's code reaches the executors (statcompute/engine/dist.py)."""
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])                  # the driver's copy; the workers only get PYTHONPATH, which holds a stale one
    import numpy as np
    import pandas as pd
    from pyspark.sql import SparkSession
    from statcompute.engine import CoxPH, LogisticFE
    spark = (SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    rng = np.random.default_rng(3)
    n = 3000
    data = pd.DataFrame({"id": np.arange(n, dtype=float), "g": rng.integers(0, 12, n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    data["t"] = np.ceil(rng.exponential(np.exp(-0.3 * data.x1)) * 50)
    data["d"] = (rng.random(n) < 0.7).astype(float)
    data["y"] = (rng.random(n) < 1 / (1 + np.exp(-0.5 * data.x1 + 0.2 * data.x2))).astype(float)
    frame = spark.createDataFrame(data).repartition(3)
    with_strata = CoxPH("t", "d", ["x1", "x2"], "g", row_id="id").fit(frame)
    without = CoxPH("t", "d", ["x1", "x2"], None, row_id="id").fit(frame)
    logistic = LogisticFE("y", ["x1", "x2"], "g", row_id="id").fit(frame)
    print("FITTED", with_strata.execution["code_sent"], without.converged, logistic.converged, logistic.providers)
    spark.stop()
""")


def _run(tmp_path, **env):
    stale = tmp_path / "stale" / "statcompute"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "__init__.py").write_text('raise ImportError("a stale statcompute on the executor was imported")\n')
    script = tmp_path / "fit.py"
    script.write_text(SCRIPT)
    environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "STATCOMPUTE_SHIP")}
    environment.update(PYTHONPATH=str(tmp_path / "stale"), **env)
    return subprocess.run([sys.executable, str(script), REPO], cwd=str(tmp_path), env=environment, capture_output=True, text=True, timeout=600)


def test_the_executors_run_the_drivers_code_whatever_copy_of_the_package_they_have(tmp_path):
    """Round 20: the package travels inside the tasks' functions, by value. The workers of this
    session can import only a stale copy that refuses to load; the fits must not touch it."""
    done = _run(tmp_path)
    assert "FITTED by value True True 12" in done.stdout, done.stdout[-600:] + done.stderr[-1500:]
    # the same session with the code sent by reference: the workers import their own copy, which is the stale one.
    # This is what makes the test above mean something, and why an archive needs a new application when the package changes.
    stale = _run(tmp_path, STATCOMPUTE_SHIP="archive")
    assert "FITTED" not in stale.stdout and "a stale statcompute on the executor was imported" in stale.stdout + stale.stderr


def test_an_archive_is_refused_once_the_package_has_changed(tmp_path, monkeypatch):
    """The fallback keeps its guard (round 11): an executor goes on importing the first archive it was given."""
    import types

    from statcompute.engine import dist

    added = []
    context = types.SimpleNamespace(master="spark://somewhere:7077", applicationId="app-under-test", addPyFile=added.append)
    session = types.SimpleNamespace(sparkContext=context)
    monkeypatch.setenv("STATCOMPUTE_SHIP", "archive")
    monkeypatch.setattr(dist.tempfile, "gettempdir", lambda: str(tmp_path))
    assert dist.ship_package(session) == dist.ARCHIVE and len(added) == 1 and added[0].endswith("statcompute.zip")
    assert dist.ship_package(session) == dist.ARCHIVE and len(added) == 1                 # unchanged: nothing is sent again
    monkeypatch.setattr(dist, "_fingerprint", lambda folder: "another")
    with pytest.raises(RuntimeError, match="restart the cluster"):
        dist.ship_package(session)
    monkeypatch.delenv("STATCOMPUTE_SHIP")
    assert dist.ship_package(session) == dist.BY_VALUE and len(added) == 1                # by value there is nothing to refuse


def test_a_tasks_function_carries_functions_and_no_module():
    """A module sent by value would carry its loader and its file's path, which belong to the driver's
    machine: the functions that run on executors reach the kernels through namespaces of functions."""
    import io
    import pickletools

    from pyspark import cloudpickle

    from statcompute.engine import dist
    from statcompute.engine import spark as engine
    from statcompute.lab import adapters as lab
    from statcompute.lab import passes

    assert dist._by_value()
    for function in (engine._cox_partition(4, [0.0] * 4, "breslow", True), engine._logistic_scan(4),
                     engine._time_pass1(4, b"", b"", 0.0, b"", False, False, True), lab.partition_kernel("cox", 4, [0.0] * 4),
                     passes._timed("cox", 4, [0.0] * 4, "breslow", "plain"), passes._timed("logistic_fe", 4, [0.0] * 4, "breslow", "engine"),
                     passes._timed("cox", 4, [0.0] * 4, "breslow", "before")):
        blob = cloudpickle.dumps(function)
        text = io.StringIO()
        pickletools.dis(blob, out=text)
        listing = text.getvalue()
        assert "dynamic_subimport" not in listing and "Loader" not in listing and "ModuleSpec" not in listing
        assert 2_000 < len(blob) < 100_000                 # the kernels travel with it, and little else
    assert callable(engine.kernels.cox_partial) and not hasattr(engine.kernels, "np")
