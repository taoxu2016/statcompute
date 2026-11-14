"""Command line: python -m statcompute.bench run|report ...   (local sessions; on Databricks use
the notebooks under notebooks/databricks, which call the same functions)."""
from __future__ import annotations

import argparse

from .report import render
from .results import load_records
from .runner import load_plan, run_plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m statcompute.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a plan in a local Spark session")
    run.add_argument("--plan", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--results-root", required=True)
    run.add_argument("--adapters", default=None, help="comma-separated subset of the plan's adapters")
    run.add_argument("--max-rows", type=int, default=None)
    run.add_argument("--master", default="local[2]")
    report = sub.add_parser("report", help="render the records under a results folder as Markdown")
    report.add_argument("--results-root", required=True)
    report.add_argument("--plan", default=None)
    report.add_argument("--reference", default="pprof_py")
    report.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.command == "run":
        from pyspark.sql import SparkSession

        spark = (SparkSession.builder.master(args.master).appName("statcompute-bench")
                 .config("spark.sql.shuffle.partitions", "8")
                 .config("spark.ui.showConsoleProgress", "false").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        only = args.adapters.split(",") if args.adapters else None
        run_plan(spark, load_plan(args.plan), args.data_root, args.results_root, only, args.max_rows,
                 on_record=lambda r: print(f"{r['adapter']['name']:15s} {r['dataset']['id']:55s} {r['status']:14s} "
                                           f"{'' if not r['timing'] else round(r['timing']['wall_s'], 2)}"))
        spark.stop()
        return 0
    text = render(load_records(args.results_root, args.plan), args.reference)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
