"""Adapters for the read-only reference implementations, and the record of their versions.

Nothing here contains or modifies their source. Each adapter imports the package as installed in
the session and maps the shared specifications onto its public API. REFERENCES.lock at the root of
the repository records the commit every comparison refers to.
"""
import json
import os


def lock() -> dict:
    """The contents of REFERENCES.lock."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "REFERENCES.lock"), os.path.join(here, "..", "..", "REFERENCES.lock")):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return {}
