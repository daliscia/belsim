"""Government expenditure of Belgian general government (ESA 2010, consolidated S13).

`run(state, prev, p, ctx)` fills `state.expenditure` (line -> EUR mn); the sum of the lines is
total expenditure. `LINES` describes every line (label, group, COFOG/ESA code, entity shares
from the 2024 subsector accounts). See `model.py` for the drivers and `lines.py` for the
mapping of lines to COFOG x ESA cells.
"""
from __future__ import annotations

from ...core.state import LineMeta
from . import calc, data
from .lines import all_defs
from .model import NS, compute, run

__all__ = ["LINES", "run", "compute", "calc", "NS"]


def _build_lines() -> dict[str, LineMeta]:
    return {
        lid: LineMeta(label=d.label, group=d.group, esa=d.esa, entity_shares=data.entity_shares(lid),
                      description=d.description)
        for lid, d in all_defs().items()
    }


LINES: dict[str, LineMeta] = _build_lines()
