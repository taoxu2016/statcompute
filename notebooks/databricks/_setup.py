# Databricks notebook source
# MAGIC %md
# MAGIC Shared setup, run by the other notebooks with `%run ./_setup`: puts the repository on the Python path
# MAGIC and defines the folders every notebook uses.

# COMMAND ----------

dbutils.widgets.text("repo_root", "", "Repository root (empty: two folders above this notebook)")
dbutils.widgets.text("data_root", "", "Folder for datasets (a Volume or dbfs:/ path)")
dbutils.widgets.text("results_root", "", "Folder for results (a /Volumes/, /Workspace/ or dbfs:/ path)")

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

data_root = dbutils.widgets.get("data_root").strip()
results_root = dbutils.widgets.get("results_root").strip()
print(f"statcompute {statcompute.__version__} from {repo_root}\ndata_root = {data_root or '(not set)'}\nresults_root = {results_root or '(not set)'}")
