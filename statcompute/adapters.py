"""The contract every implementation is run through, and the registry of adapters.

An adapter wraps one implementation (a read-only reference, a prototype from the lab, or the new
engine) so that the harness can ask three things of it in the same way:

  available(spark)        can it run in this session at all, and if not, why
  support(model, data)    does it fit exactly the requested model: "exact", "deviation" or "unsupported"
  fit(spark, dataset, model)  fit and return a FitResult

Comparisons between two implementations are like-for-like only when both report "exact".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

EXACT, DEVIATION, UNSUPPORTED = "exact", "deviation", "unsupported"


@dataclass
class Support:
    """Whether an implementation fits the requested model, with the reasons when it does not."""
    status: str = EXACT
    notes: List[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return self.status != UNSUPPORTED


@dataclass
class FitResult:
    """What every adapter returns. Lists are aligned with `feature_names`."""
    feature_names: List[str]
    coefficients: List[float]
    standard_errors: Optional[List[float]] = None
    log_likelihood: Optional[float] = None
    iterations: Optional[int] = None
    converged: Optional[bool] = None
    settings: Dict[str, Any] = field(default_factory=dict)   # options actually passed to the implementation
    phases: Dict[str, float] = field(default_factory=dict)   # seconds, measured inside the adapter
    extras: Dict[str, Any] = field(default_factory=dict)


class Adapter:
    """Base class. `kind` is where the arithmetic runs: "driver" (one machine) or "spark" (executors).
    `role` is "reference" (read-only package), "prototype" (lab) or "engine" (the new implementation)."""
    name = "abstract"
    kind = "driver"
    role = "reference"

    def describe(self, spark=None) -> Dict[str, Any]:
        """Name, version and commit of the implementation as found in this session."""
        return {"name": self.name, "kind": self.kind, "role": self.role}

    def available(self, spark=None) -> Support:
        return Support()

    def support(self, model, data) -> Support:
        raise NotImplementedError

    def driver_bytes(self, model, data) -> int:
        """Estimated driver memory the fit needs, in bytes; 0 when rows never reach the driver."""
        return 0

    def fit(self, spark, dataset, model) -> FitResult:
        raise NotImplementedError


_REGISTRY: Dict[str, Callable[[], Adapter]] = {}


def register(name: str, factory: Callable[[], Adapter]) -> None:
    _REGISTRY[name] = factory


def names() -> List[str]:
    _load()
    return sorted(_REGISTRY)


def get(name: str) -> Adapter:
    _load()
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def _load() -> None:
    """Import the modules that register adapters. Importing them never imports the wrapped package."""
    from .references import pprof_py_adapter, pprof_r_adapter, pprof_spark_adapter  # noqa: F401
    from .lab import adapters as _lab_adapters  # noqa: F401
