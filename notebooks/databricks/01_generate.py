# Databricks notebook source
# MAGIC %md
# MAGIC # statcompute 01: generate the datasets of a plan
# MAGIC
# MAGIC Writes each dataset once under `data_root` as Parquet with a manifest (rows, events, checksum, the
# MAGIC generating coefficients). A dataset that is already there is left alone. Every implementation later reads
# MAGIC these same tables. At 180 features a row is about 1.5 KB before compression: 5 million rows are about
# MAGIC 7 GB, 50 million about 73 GB.

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dbutils.widgets.text("plan", "standard", "Plan (a name under statcompute/bench/plans, or a path)")
dbutils.widgets.text("max_rows", "5000000", "Largest dataset to generate now (rows)")

# COMMAND ----------

import time

from statcompute.bench import load_plan
from statcompute.testkit import DatasetSpec, materialize

if not data_root:
    raise ValueError("set the data_root widget")
plan = load_plan(dbutils.widgets.get("plan").strip())
max_rows = int(float(dbutils.widgets.get("max_rows")))
rows = []
for entry in plan["datasets"]:
    spec = DatasetSpec.from_dict(entry)
    if spec.rows > max_rows:
        continue
    started = time.perf_counter()
    dataset = materialize(spark, spec, data_root)
    rows.append({"dataset": dataset.id, "rows": dataset.rows, "events": dataset.manifest["events"],
                 "checksum": str(dataset.manifest["checksum"]), "seconds": round(time.perf_counter() - started, 1),
                 "path": dataset.path})
    print(rows[-1])
display(spark.createDataFrame(rows))
