"""How this package's code reaches the Spark executors.

By value, the default: PySpark's pickler is told to put everything of this package that a task's
function uses inside the function it sends. The executors then need no copy of the package, can
never run an older one than the driver, and nothing has to be restarted when the package changes:
a new driver process is enough. The kernels are plain NumPy modules, which is what makes this cheap.

As an archive, the earlier way and the fallback: the package is zipped and added to the session with
addPyFile. An executor keeps importing the first archive it was given, so a changed package then
needs a new Spark application; `ship_package` refuses to go on and says so.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import types

BY_VALUE, ARCHIVE = "by value", "archive"


def functions_of(module) -> types.SimpleNamespace:
    """A module's own contents (its functions, classes and constants) as a plain namespace.

    The functions that run on the executors reach the kernels through such a namespace and not
    through the module. What travels with a task is then the functions it uses, and never a module
    object: a module sent by value would carry its loader and its file's path, which belong to the
    driver's machine and to whatever imported it there."""
    return types.SimpleNamespace(**{name: value for name, value in vars(module).items()
                                    if not name.startswith("__") and not isinstance(value, types.ModuleType)})


def _fingerprint(package_dir: str) -> str:
    digest = hashlib.sha1()
    for folder, _, files in sorted(os.walk(package_dir)):
        for name in sorted(files):
            if name.endswith((".py", ".R", ".json", ".lock")):
                with open(os.path.join(folder, name), "rb") as f:
                    digest.update(name.encode() + f.read())
    return digest.hexdigest()


def _by_value() -> bool:
    """Tell PySpark's pickler to send this package's functions, classes and modules by value.
    False when it cannot (an older pickler) or when STATCOMPUTE_SHIP=archive asks for the other way."""
    if os.environ.get("STATCOMPUTE_SHIP", "").strip().lower() == ARCHIVE:
        return False
    try:
        from pyspark import cloudpickle

        import statcompute

        cloudpickle.register_pickle_by_value(statcompute)
        return True
    except Exception:
        return False


def ship_package(spark) -> str:
    """Make the package's code reach the executors of this session; returns how: "by value" or
    "archive". Called at the start of every fit; it costs nothing after the first call.

    By value there is nothing to send ahead and nothing to keep in step. With the archive, the
    archive and a marker live in one folder per Spark application, so the record survives a reload
    of this module; when the package has changed since it was shipped, this stops with the reason
    instead of running old code on the executors.
    """
    if _by_value():
        return BY_VALUE
    sc = spark.sparkContext
    if sc.master.startswith("local"):
        return ARCHIVE                                   # local workers import the driver's own copy
    import statcompute

    package_dir = os.path.dirname(os.path.abspath(statcompute.__file__))
    fingerprint = _fingerprint(package_dir)
    folder = os.path.join(tempfile.gettempdir(), f"statcompute-{sc.applicationId}")
    marker = os.path.join(folder, "shipped")
    if os.path.exists(marker):
        with open(marker) as f:
            if f.read().strip() == fingerprint:
                return ARCHIVE
        raise RuntimeError(
            "the statcompute package has changed since it was sent to the executors of this Spark application as an "
            "archive, and they would go on running the old code: restart the cluster and run again")
    os.makedirs(folder, exist_ok=True)
    archive = shutil.make_archive(os.path.join(folder, "statcompute"), "zip",
                                  root_dir=os.path.dirname(package_dir), base_dir="statcompute")
    sc.addPyFile(archive)
    with open(marker, "w") as f:
        f.write(fingerprint)
    return ARCHIVE
