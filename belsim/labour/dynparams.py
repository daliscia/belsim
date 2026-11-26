"""Access to the `dyn.*` parameters (params/dynamics.yaml), shared by labour, macro and behaviour.

Values come from the ParamView of the year; if a name is missing from the view (unit tests with
fake views) the central value of the YAML file is used, so no number is hard-coded in the code.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from ..core.params import PARAMS_DIR

# origins of an employment difference with the baseline that carry their own productivity / wage
SOURCES = ("unemp_exclusion", "pension_age", "structural_unemployment", "labour_supply",
           "labour_demand", "cyclical")


@lru_cache(maxsize=1)
def defaults() -> dict[str, Any]:
    """Read-only table: parameter name -> earliest value in params/dynamics.yaml."""
    data = yaml.safe_load((PARAMS_DIR / "dynamics.yaml").read_text()) or {}
    return {name: spec["values"][min(spec["values"])] for name, spec in data.items()}


def par(p, name: str) -> Any:
    v = p.get(name) if p is not None else None
    if v is None:
        v = defaults()[name]
    return v
