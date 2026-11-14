# Databricks notebook source
# MAGIC %md
# MAGIC # statcompute 00: is this cluster fit for a benchmark?
# MAGIC
# MAGIC Checks the conditions of `docs/benchmark-protocol.md` and prints what every later record will carry.
# MAGIC Nothing is fitted here. Run it once per cluster before `01_generate` and `02_run`.

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

import json

from statcompute import adapters, references
from statcompute.bench import environment
from statcompute.bench.metrics import SparkMetrics

static, moving = environment.static(spark), environment.moving(spark)
dbx = static["databricks"]
problems, notes = [], []
if not (dbx.get("runtime") or "").startswith("17.3"):
    notes.append(f"runtime is {dbx.get('runtime')}; the standard cluster is 17.3 LTS (Apache Spark 4.0.0)")
if (dbx.get("scaling_type") or "").lower().startswith("auto") or (
        dbx.get("min_workers") and dbx.get("max_workers") and dbx["min_workers"] != dbx["max_workers"]):
    notes.append("the cluster autoscales: fix its size if you can (min = max); otherwise set hold_executors in 02_run, "
                 "which asks for full size before each Spark fit. Every record shows the executors attached during its fit")
probe = SparkMetrics(spark)
if probe._get("executors") is not None:
    notes.append("task metrics: from the Spark UI's REST API")
elif probe._store_stages("statcompute-probe") is not None:
    notes.append("task metrics: the REST API is closed; the status store answers through py4j, which is enough")
else:
    problems.append("neither the Spark UI's REST API nor the status store can be read: records would lack task metrics")
if moving.get("executors") not in (None, 10):
    notes.append(f"{moving.get('executors')} executors are attached; the standard cluster has 10")
try:
    spark.sparkContext._jsc.sc()
except Exception as error:
    problems.append(f"py4j access to the JVM is blocked ({type(error).__name__}): use dedicated (single-user) access mode")
if (dbx.get("photon") or "").lower() == "true":
    notes.append("Photon is on: pprof_spark runs typed Dataset code, which Photon does not execute")

# Cluster tags that describe size and scaling, with their real names, so the checks above can be corrected if
# Databricks names them differently. Tags that name people or workspaces are not printed.
wanted = ("worker", "scal", "nodetype", "sparkversion", "photon", "autoterm", "numworkers")
tags = {k.rsplit(".", 1)[-1]: v for k, v in spark.sparkContext.getConf().getAll()
        if "clusterUsageTags" in k and any(w in k.lower() for w in wanted)}
print(json.dumps({"static": dict(static, databricks=dict(static["databricks"], cluster_id="(not printed)")),
                  "now": moving, "cluster_tags": tags}, indent=1, default=str))
print("\nPROBLEMS:" if problems else "\nNo blocking problem found.")
for p in problems:
    print("  -", p)
for n in notes:
    print("  note:", n)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which implementations can run here
# MAGIC `pinned` is the commit in `REFERENCES.lock`. A reference that is installed at another commit still runs;
# MAGIC its records then say `matches_pin: false`.

# COMMAND ----------

pins = references.lock().get("references", {})
rows = []
for name in adapters.names():
    adapter = adapters.get(name)
    described, available = adapter.describe(spark), adapter.available(spark)
    rows.append({"adapter": name, "role": adapter.role, "runs on": adapter.kind,
                 "available": available.status != "unsupported", "why not": "; ".join(available.notes),
                 "found": str(described.get("commit") or described.get("version") or ""),
                 "pinned": str((pins.get(name) or {}).get("commit") or ""),
                 "note": "NO NUMBA: pure-Python kernels, timings meaningless" if name == "pprof_py"
                 and available.runnable and described.get("numba") is None else ""})
display(spark.createDataFrame(rows))
