"""Make this package importable on Spark executors when it was not installed there."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile


def _fingerprint(package_dir: str) -> str:
    digest = hashlib.sha1()
    for folder, _, files in sorted(os.walk(package_dir)):
        for name in sorted(files):
            if name.endswith((".py", ".R", ".json", ".lock")):
                with open(os.path.join(folder, name), "rb") as f:
                    digest.update(name.encode() + f.read())
    return digest.hexdigest()


def ship_package(spark) -> None:
    """Zip the package and add it to the session with addPyFile, once per Spark application.

    The archive and a marker live in one folder per application, so the record survives a reload of
    this module (the notebooks reload it on every run). Spark refuses a second archive of the same
    name from another path, and executors keep importing the first one they were given; so when the
    package has changed since it was shipped, this stops with the reason instead of running old
    code on the executors. Local sessions need nothing.
    """
    sc = spark.sparkContext
    if sc.master.startswith("local"):
        return
    import statcompute

    package_dir = os.path.dirname(os.path.abspath(statcompute.__file__))
    fingerprint = _fingerprint(package_dir)
    folder = os.path.join(tempfile.gettempdir(), f"statcompute-{sc.applicationId}")
    marker = os.path.join(folder, "shipped")
    if os.path.exists(marker):
        with open(marker) as f:
            if f.read().strip() == fingerprint:
                return
        raise RuntimeError(
            "the statcompute package has changed since it was sent to the executors of this Spark application, "
            "and they would go on running the old code: restart the cluster (or detach and re-attach) and run again")
    os.makedirs(folder, exist_ok=True)
    archive = shutil.make_archive(os.path.join(folder, "statcompute"), "zip",
                                  root_dir=os.path.dirname(package_dir), base_dir="statcompute")
    sc.addPyFile(archive)
    with open(marker, "w") as f:
        f.write(fingerprint)
