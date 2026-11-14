"""Shared test utilities: specifications, deterministic synthetic data, tolerance classes."""
from .specs import COX, LOGISTIC_FE, DatasetSpec, ModelSpec
from .synthetic import Dataset, generate, load, materialize

__all__ = ["COX", "LOGISTIC_FE", "DatasetSpec", "ModelSpec", "Dataset", "generate", "load", "materialize"]
