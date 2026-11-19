"""statcompute: estimator-preserving computation for classical statistical models.

Sub-packages, one per area of the repository (docs/architecture.md):
  testkit      shared synthetic data, model specifications and tolerance classes
  references   adapters for the read-only reference implementations
  lab          experimental algorithms and prototypes
  validation   numerical comparison of fits
  bench        scale and performance harness
  engine       the new implementation: Cox and logistic fixed-effect regression on Spark
"""
__version__ = "0.10.0"
