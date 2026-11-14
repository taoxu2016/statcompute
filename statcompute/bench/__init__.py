"""Scale and performance harness: run every implementation through the same datasets and models,
and record what it took. See docs/benchmark-protocol.md."""
from .runner import load_plan, run_plan

__all__ = ["load_plan", "run_plan"]
