"""Federated NLME estimation with Flower and NoLimitsPy."""

import os
from pathlib import Path

# Flower's per-run runtime environment does not pass PYTHON_JULIAPKG_* through, so a
# client would silently fall back to a Julia project of its own (without CSV, which the
# warfarin loader needs). Pin the repo's shared project here, before anything boots Julia.
_JULIA_ENV = Path(__file__).resolve().parents[2] / "julia_env"
if _JULIA_ENV.is_dir():
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(_JULIA_ENV))
    os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
