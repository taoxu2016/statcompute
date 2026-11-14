"""Declarative specifications of a synthetic dataset and of a model to fit to it."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

COX, LOGISTIC_FE = "cox", "logistic_fe"

# The "correlated" design cycles through these prevalences and scales. THRESHOLDS are the normal
# quantiles z with P(Z > z) = prevalence, to nine digits.
PREVALENCES = (0.30, 0.10, 0.03, 0.01)
THRESHOLDS = (0.524400513, 1.281551566, 1.880793608, 2.326347874)
SCALES = (0.1, 1.0, 10.0, 50.0)
GRID_SD = math.sqrt((129 ** 2 - 1) / 12.0) / 64.0   # SD of the uniform grid -1, -63/64, ..., 1


@dataclass(frozen=True)
class DatasetSpec:
    """A synthetic dataset. Every value is a deterministic function of these fields and the row
    identifier, so the same specification gives the same table on any cluster and partitioning.

    family            "cox" (columns time, event) or "logistic_fe" (column y)
    design            "simple": independent features; "correlated": one shared factor, rare
                      indicators and continuous features with standard deviations from 0.1 to 50
    indicator_share   fraction of the features that are 0/1 indicators
    indicator_prevalence   P(indicator = 1) in the simple design
    correlation       loading of every feature on the shared factor in the correlated design
    """
    family: str
    rows: int
    providers: int
    features: int
    design: str = "simple"
    seed: int = 79
    indicator_share: float = 0.5
    indicator_prevalence: float = 0.3
    correlation: float = 0.5
    generator: str = "sc-synth"
    generator_version: int = 1

    def __post_init__(self):
        if self.family not in (COX, LOGISTIC_FE):
            raise ValueError(f"family must be 'cox' or 'logistic_fe', got {self.family!r}")
        if self.design not in ("simple", "correlated"):
            raise ValueError(f"design must be 'simple' or 'correlated', got {self.design!r}")
        if not (1 <= self.providers <= self.rows) or self.features < 1:
            raise ValueError("need 1 <= providers <= rows and features >= 1")
        if not (0.0 <= self.indicator_share <= 1.0 and 0.0 < self.indicator_prevalence < 1.0):
            raise ValueError("indicator_share must lie in [0, 1] and indicator_prevalence in (0, 1)")
        if not (0.0 <= self.correlation < 1.0):
            raise ValueError("correlation must lie in [0, 1)")

    def feature_names(self) -> List[str]:
        return [f"x{k}" for k in range(1, self.features + 1)]

    def is_indicator(self, k: int) -> bool:
        """Whether feature k (1-based) is an indicator; indicators are spread evenly over 1..p."""
        s = self.indicator_share
        return math.floor(k * s + 1e-12) > math.floor((k - 1) * s + 1e-12)

    def ordinal(self, k: int) -> int:
        """Position of feature k (1-based) among the features of its own type, from 0."""
        indicators = math.floor(k * self.indicator_share + 1e-12)
        return (indicators if self.is_indicator(k) else k - indicators) - 1

    def prevalence(self, k: int) -> float:
        """P(x_k = 1) of an indicator; the correlated design cycles through PREVALENCES."""
        if self.design == "simple":
            return self.indicator_prevalence
        return PREVALENCES[self.ordinal(k) % len(PREVALENCES)]

    def threshold(self, k: int) -> float:
        return THRESHOLDS[self.ordinal(k) % len(THRESHOLDS)]

    def scale(self, k: int) -> float:
        """Standard deviation of feature k (1-based)."""
        if self.is_indicator(k):
            q = self.prevalence(k)
            return math.sqrt(q * (1.0 - q))
        return GRID_SD if self.design == "simple" else SCALES[self.ordinal(k) % len(SCALES)]

    def true_beta(self) -> List[float]:
        """Generating coefficients. Simple design: (-1)^k 0.5/sqrt(p). Correlated design: the same
        divided by the feature's standard deviation, so that every feature moves the linear
        predictor by a comparable amount."""
        base = [(-1.0) ** k * 0.5 / math.sqrt(self.features) for k in range(1, self.features + 1)]
        if self.design == "simple":
            return base
        return [b / self.scale(k) for k, b in enumerate(base, start=1)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DatasetSpec":
        return DatasetSpec(**d)

    def dataset_id(self) -> str:
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.family}-n{self.rows}-m{self.providers}-p{self.features}-{self.design}-s{self.seed}-{digest}"


@dataclass(frozen=True)
class ModelSpec:
    """The model every implementation is asked to fit.

    family       "cox": proportional hazards, stratified by provider when strata == "provider";
                 "logistic_fe": logistic regression with one fixed intercept per provider
    ties         Cox tie method: "breslow" or "efron"
    convergence  "default": each implementation's own stopping rule and tolerance;
                 "tight": each implementation's rule at a much tighter tolerance, so that converged
                 estimates can be compared without stopping rules getting in the way
    """
    family: str
    ties: str = "breslow"
    strata: Optional[str] = "provider"
    convergence: str = "default"
    max_iterations: Optional[int] = None

    def __post_init__(self):
        if self.family not in (COX, LOGISTIC_FE):
            raise ValueError(f"family must be 'cox' or 'logistic_fe', got {self.family!r}")
        if self.ties not in ("breslow", "efron"):
            raise ValueError("ties must be 'breslow' or 'efron'")
        if self.strata not in (None, "provider"):
            raise ValueError("strata must be None or 'provider'")
        if self.convergence not in ("default", "tight"):
            raise ValueError("convergence must be 'default' or 'tight'")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelSpec":
        return ModelSpec(**d)

    def label(self) -> str:
        if self.family == COX:
            strata = "stratified" if self.strata else "unstratified"
            return f"cox/{strata}/{self.ties}/{self.convergence}"
        return f"logistic_fe/{self.convergence}"
