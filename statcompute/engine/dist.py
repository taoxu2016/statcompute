"""Make this package importable on Spark executors when it was not installed there."""
from __future__ import annotations

import os
import shutil
import tempfile

_SHIPPED = set()


def ship_package(spark) -> None:
    """Zip the package and add it to the session with addPyFile, once per application. Local
    sessions need nothing. A notebook-scoped `%pip install` of the wheel makes this redundant,
    and harmless."""
    sc = spark.sparkContext
    if sc.master.startswith("local") or sc.applicationId in _SHIPPED:
        return
    import statcompute

    root = os.path.dirname(os.path.dirname(os.path.abspath(statcompute.__file__)))
    archive = shutil.make_archive(os.path.join(tempfile.mkdtemp(), "statcompute"), "zip",
                                  root_dir=root, base_dir="statcompute")
    sc.addPyFile(archive)
    _SHIPPED.add(sc.applicationId)
