# Databricks notebook source
# MAGIC %md
# MAGIC # statcompute 02: run a plan
# MAGIC
# MAGIC Runs every model of the plan on every chosen implementation and writes one JSON record per run under
# MAGIC `results_root/runs/<plan>/`. A run that fails, is unsupported or would not fit the driver is recorded as
# MAGIC such; nothing is carried from one run to the next. Start with `max_rows` at 5 million.

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dbutils.widgets.text("plan", "standard", "Plan")
dbutils.widgets.text("adapters", "", "Implementations, comma-separated (empty: all in the plan)")
dbutils.widgets.text("max_rows", "5000000", "Largest dataset to run now (rows)")
dbutils.widgets.text("families", "", "Model families, comma-separated (empty: both)")
dbutils.widgets.dropdown("cache_input", "false", ["false", "true"], "Cache the input table before pprof_spark fits")
dbutils.widgets.text("lab_rows_per_partition", "", "Rows per partition for lab_spark (empty: 50000)")
dbutils.widgets.text("hold_executors", "", "On an autoscaling cluster: executors to ask for before each Spark fit (empty: none)")

# COMMAND ----------

import pandas as pd

from statcompute.bench import load_plan, run_plan
from statcompute.bench.results import summary

if not (data_root and results_root):
    raise ValueError("set the data_root and results_root widgets")
split = lambda text: [v.strip() for v in text.split(",") if v.strip()] or None
plan = load_plan(dbutils.widgets.get("plan").strip())
plan.setdefault("options", {})["cache_input"] = dbutils.widgets.get("cache_input") == "true"
if dbutils.widgets.get("lab_rows_per_partition").strip():
    plan["options"]["lab_rows_per_partition"] = int(float(dbutils.widgets.get("lab_rows_per_partition")))
if dbutils.widgets.get("hold_executors").strip():
    plan["options"]["hold_executors"] = int(float(dbutils.widgets.get("hold_executors")))
records = run_plan(
    spark, plan, data_root, results_root,
    only_adapters=split(dbutils.widgets.get("adapters")),
    max_rows=int(float(dbutils.widgets.get("max_rows"))),
    families=split(dbutils.widgets.get("families")),
    on_record=lambda r: print(f"{r['adapter']['name']:15s} {r['dataset']['id']:58s} {r['status']:14s} "
                              f"{'' if not r['timing'] else round(r['timing']['wall_s'], 1)}"),
)
display(pd.DataFrame(summary(records)))
