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
    problems.append("the cluster autoscales: timings need a fixed number of workers")
if SparkMetrics(spark).executors() is None:
    notes.append("the Spark UI's REST API is not reachable from the driver: records will lack task metrics")
try:
    spark.sparkContext._jsc.sc()
except Exception as error:
    problems.append(f"py4j access to the JVM is blocked ({type(error).__name__}): use dedicated (single-user) access mode")
if (dbx.get("photon") or "").lower() == "true":
    notes.append("Photon is on: pprof_spark runs typed Dataset code, which Photon does not execute")

print(json.dumps({"static": static, "now": moving}, indent=1, default=str))
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
                 "found": described.get("commit") or described.get("version"),
                 "pinned": (pins.get(name) or {}).get("commit")})
display(spark.createDataFrame(rows))
