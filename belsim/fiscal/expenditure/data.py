"""Read-only calibration tables of the expenditure module (data/base/expenditure_*.csv),
loaded once and cached. Produced by `belsim.data.fetch_expenditure`."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from ...core.dims import ENTITIES

BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "base"


def _rows(name: str) -> list[dict[str, str]]:
    path = BASE_DIR / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def lines_2024() -> MappingProxyType:
    """line -> {'value', ESA components..., entity amounts...} (EUR mn, 2024 actual)."""
    out = {}
    for r in _rows("expenditure_lines_2024.csv"):
        out[r["line"]] = MappingProxyType({k: float(v) for k, v in r.items() if k != "line"})
    return MappingProxyType(out)


@lru_cache(maxsize=1)
def transfers_2024() -> MappingProxyType:
    """(from entity, to entity) -> EUR mn (D7 + D9 + D4 between subsectors, 2024 actual)."""
    return MappingProxyType({(r["from"], r["to"]): float(r["value"])
                             for r in _rows("expenditure_transfers_2024.csv")})


@lru_cache(maxsize=1)
def scalars() -> MappingProxyType:
    return MappingProxyType({r["key"]: float(r["value"]) for r in _rows("expenditure_scalars.csv")})


def entity_shares(line: str) -> dict[str, float]:
    row = lines_2024().get(line)
    if row is None:
        return {}
    amounts = {e: max(row[e], 0.0) for e in ENTITIES}
    tot = sum(amounts.values())
    return {e: a / tot for e, a in amounts.items()} if tot > 0 else {}
