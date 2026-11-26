"""Access to the `beh.*` parameters.

Values come from the ParamView of the year. If a name is missing from the view (unit tests with
fake views, or a partially loaded parameter set) the central value of `params/behaviour.yaml`
is used, so no number is ever hard-coded in the code.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from ..core.params import PARAMS_DIR


@lru_cache(maxsize=1)
def defaults() -> dict[str, Any]:
    """Read-only table: parameter name -> earliest value in params/behaviour.yaml."""
    data = yaml.safe_load((PARAMS_DIR / "behaviour.yaml").read_text()) or {}
    return {name: spec["values"][min(spec["values"])] for name, spec in data.items()}


def par(p, name: str) -> Any:
    v = p.get(name) if p is not None else None
    if v is None:
        v = defaults()[name]
    return v


def enabled(p) -> bool:
    return float(par(p, "beh.enabled")) != 0.0


def scale(p) -> float:
    return max(0.0, float(par(p, "beh.scale")))
