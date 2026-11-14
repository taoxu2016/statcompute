# Databricks notebook source
# MAGIC %md
# MAGIC # statcompute 03: report
# MAGIC
# MAGIC Reads the records under `results_root`, shows timings and numerical agreement, and writes the same as a
# MAGIC Markdown file under `results_root/reports/`. Send that file and the `runs/` folder back to the repository:
# MAGIC they contain synthetic data's results only.

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dbutils.widgets.text("plan", "", "Plan (empty: every plan under results_root)")
dbutils.widgets.text("reference", "pprof_py", "Implementation the others are compared with")

# COMMAND ----------

import datetime
import os

import pandas as pd

from statcompute.bench.report import render
from statcompute.bench.results import load_records, summary
from statcompute.bench.runner import local_path
from statcompute.validation import compare_records

plan_name = dbutils.widgets.get("plan").strip() or None
reference = dbutils.widgets.get("reference").strip()
records = load_records(results_root, plan_name)
display(pd.DataFrame(summary(records)))
display(pd.DataFrame(compare_records(records, reference)))
folder = os.path.join(local_path(results_root), "reports")
os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, f"{plan_name or 'all'}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}.md")
with open(path, "w") as f:
    f.write(render(records, reference, title=f"Benchmark report: {plan_name or 'all plans'}"))
print("written:", path)
