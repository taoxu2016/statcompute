# Databricks notebook source
# MAGIC %md
# MAGIC # statcompute 03: report
# MAGIC
# MAGIC Reads the records under `results_root`, shows timings and numerical agreement, and writes the same as a
# MAGIC Markdown file under `results_root/reports/`. Send that file and the `runs/` folder back to the repository:
# MAGIC they contain synthetic data's results only.
# MAGIC
# MAGIC A results folder keeps every session of a plan. `versions` narrows the report to the sessions made with
# MAGIC the given versions of this repository (for example `0.20.0`): the report of one session is a fraction of
# MAGIC the size. The last cell shows the report's text in a box to copy from. Do not print the report to copy it:
# MAGIC a cell that prints a text this long drops its middle (the report of 2026-11-22 reached the repository as
# MAGIC its first and last 25,600 characters).

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dbutils.widgets.text("plan", "", "Plan (empty: every plan under results_root)")
dbutils.widgets.text("reference", "pprof_py", "Implementation the others are compared with")
dbutils.widgets.text("versions", "", "Versions of this repository to report, comma-separated (empty: all)")

# COMMAND ----------

import datetime
import os

import pandas as pd

from statcompute.bench.report import render
from statcompute.bench.results import load_records, of_versions, summary
from statcompute.bench.runner import local_path
from statcompute.validation import compare_records

plan_name = dbutils.widgets.get("plan").strip() or None
reference = dbutils.widgets.get("reference").strip()
versions = [v.strip() for v in dbutils.widgets.get("versions").split(",") if v.strip()]
records = of_versions(load_records(results_root, plan_name), versions)
display(pd.DataFrame(summary(records)))
display(pd.DataFrame(compare_records(records, reference)))
folder = os.path.join(local_path(results_root), "reports")
os.makedirs(folder, exist_ok=True)
name = "-".join([plan_name or "all"] + versions + [f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}"])
path = os.path.join(folder, name + ".md")
title = f"Benchmark report: {plan_name or 'all plans'}" + (f", version {', '.join(versions)}" if versions else "")
text = render(records, reference, title=title)        # before the file exists: a failure here must not leave an empty file
with open(path, "w") as f:
    f.write(text)
print(f"written: {path} ({len(text):,} characters, {os.path.getsize(path):,} bytes on disk, {len(records)} records)")

# COMMAND ----------

# The report's text, whole, to copy: click in the box, select all, copy. (Printed, a text of this length loses its middle.)
import html

displayHTML(f"<p>{len(text):,} characters, {len(records)} records. Click in the box, select all, copy.</p>"
            "<textarea readonly style='width:100%;height:320px;font-family:monospace;font-size:11px'>"
            f"{html.escape(text)}</textarea>")
