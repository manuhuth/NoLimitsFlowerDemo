"""Federated NLME estimation with Flower and NoLimitsPy."""

import os
from pathlib import Path

# Flower's per-run runtime environment does not pass PYTHON_JULIAPKG_* through, so a
# client would silently fall back to the venv's own Julia project (registered
# NoLimits, without the federation primitives). Pin the repo's pre-release project
# here, before anything boots Julia. Drop this once the primitives are released.
_JULIA_ENV = Path(__file__).resolve().parents[2] / "julia_env"
if _JULIA_ENV.is_dir():
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(_JULIA_ENV))
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
