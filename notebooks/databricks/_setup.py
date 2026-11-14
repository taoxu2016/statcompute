# Databricks notebook source
# MAGIC %md
# MAGIC Shared setup, run by the other notebooks with `%run ./_setup`: puts the repository on the Python path
# MAGIC and defines the folders every notebook uses.

# COMMAND ----------

dbutils.widgets.text("repo_root", "", "Repository root (empty: two folders above this notebook)")
dbutils.widgets.text("data_root", "", "Folder for datasets (a Volume or dbfs:/ path)")
dbutils.widgets.text("results_root", "", "Folder for results (a /Volumes/, /Workspace/ or dbfs:/ path)")
dbutils.widgets.text("pprof_spark_build", "", "pprof_spark build folder, the one holding manifest.json (empty: not used)")
dbutils.widgets.dropdown("install_pprof_py", "no", ["no", "yes"], "Install pprof_py on the driver for this notebook")

# COMMAND ----------

import os
import sys

repo_root = dbutils.widgets.get("repo_root").strip() or os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
if not os.path.isdir(os.path.join(repo_root, "statcompute")):
    raise RuntimeError(f"{repo_root} is not the repository root: set the repo_root widget")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
for name in [m for m in list(sys.modules) if m == "statcompute" or m.startswith("statcompute.")]:
    del sys.modules[name]                      # pick up edits to the repository without restarting Python
import statcompute

# pprof_spark's Python package comes from its own build folder; its JARs must be attached to the cluster.
pprof_spark_build = dbutils.widgets.get("pprof_spark_build").strip()
if pprof_spark_build:
    wrapper = os.path.join(pprof_spark_build, "python")
    if not os.path.isdir(os.path.join(wrapper, "pprof_spark")):
        raise RuntimeError(f"{wrapper} does not hold the pprof_spark Python package: check the pprof_spark_build widget")
    if wrapper not in sys.path:
        sys.path.insert(0, wrapper)

# pprof_py at its pinned commit, from GitHub's archive of that commit, with numba (without numba it runs
# its pure-Python kernels, which would make its timings meaningless). Driver only; nothing is changed upstream.
if dbutils.widgets.get("install_pprof_py") == "yes":
    import importlib
    import subprocess

    try:
        import pprof_py  # noqa: F401
    except Exception:
        from statcompute import references

        commit = references.lock()["references"]["pprof_py"]["commit"]
        for arguments in (["numba", "fast_poibin"],
                          ["--no-deps", f"https://github.com/UM-KevinHe/pprof_py/archive/{commit}.zip"]):
            done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *arguments], capture_output=True, text=True)
            if done.returncode != 0:
                print("pip install", " ".join(arguments), "failed:\n", done.stderr[-1500:])
                break
        importlib.invalidate_caches()

data_root = dbutils.widgets.get("data_root").strip()
results_root = dbutils.widgets.get("results_root").strip()
print(f"statcompute {statcompute.__version__} from {repo_root}\ndata_root = {data_root or '(not set)'}\nresults_root = {results_root or '(not set)'}")
