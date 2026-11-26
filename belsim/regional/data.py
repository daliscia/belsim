"""Read-only regional calibration table (data/base/regional_accounts.csv, Eurostat NUTS1),
produced by `belsim.data.fetch_regional`. Loaded once and cached."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..core.dims import BASE_YEAR, REGIONS

BASE_DIR = Path(__file__).resolve().parents[2] / "data" / "base"


@lru_cache(maxsize=4)
def accounts(year: int = BASE_YEAR) -> MappingProxyType:
    """indicator -> array [region] in `REGIONS` order (units of the table: EUR mn, thousand, %)."""
    path = BASE_DIR / "regional_accounts.csv"
    if not path.exists():
        raise RuntimeError("data/base/regional_accounts.csv missing: run belsim.data.fetch_regional")
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if int(r["year"]) == year and r["region"] in REGIONS:
                rows.setdefault(r["indicator"], {})[r["region"]] = float(r["value"])
    out = {}
    for k, d in rows.items():
        if all(r in d for r in REGIONS):
            a = np.array([d[r] for r in REGIONS])
            a.setflags(write=False)
            out[k] = a
    return MappingProxyType(out)


def shares(indicator: str, year: int = BASE_YEAR) -> np.ndarray:
    """Regional shares (sum to 1 over the three regions; the extra-regio part is left out)."""
    a = accounts(year)[indicator]
    return a / a.sum()
