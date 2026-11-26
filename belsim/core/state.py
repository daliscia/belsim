"""Simulation state for one year, and the results container.

Units: persons (float), euros per person per year for wages/benefits, EUR MILLION current
prices for every macro and fiscal aggregate. Indices (cpi, health_index, ...) = 1.0 in BASE_YEAR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dims import (
    N_AGE, N_DECILE, N_JOBTYPE, N_REGION, N_SECTOR, N_SEX, N_STATUS,
    REGIONS, SECTORS, SEXES, STATUSES, JOBTYPES,
)


@dataclass
class LineMeta:
    """Metadata for one revenue or expenditure line."""
    label: str
    group: str                       # display group, e.g. "Social contributions", "Pensions"
    esa: str = ""                    # ESA 2010 / COFOG code if any
    entity_shares: dict[str, float] = field(default_factory=dict)  # entity -> share, sums to 1
    description: str = ""


@dataclass
class YearState:
    year: int

    # --- demographics (owner: belsim.demographics) ---
    pop: np.ndarray = field(default_factory=lambda: np.zeros((N_AGE, N_SEX, N_REGION)))
    demo: dict[str, float] = field(default_factory=dict)

    # --- labour (owner: belsim.labour) ---
    status: np.ndarray = field(default_factory=lambda: np.zeros((N_STATUS, N_AGE, N_SEX, N_REGION)))
    emp: np.ndarray = field(default_factory=lambda: np.zeros((N_SECTOR, N_JOBTYPE)))
    wage: np.ndarray = field(default_factory=lambda: np.zeros((N_SECTOR, N_JOBTYPE)))
    wage_dist: np.ndarray = field(default_factory=lambda: np.zeros((N_JOBTYPE, N_DECILE)))
    labour: dict[str, float] = field(default_factory=dict)

    # --- macro (owner: belsim.macro) ---
    va: np.ndarray = field(default_factory=lambda: np.zeros(N_SECTOR))
    macro: dict[str, float] = field(default_factory=dict)

    # --- public finances (owners: belsim.fiscal.*) ---
    expenditure: dict[str, float] = field(default_factory=dict)
    revenue: dict[str, float] = field(default_factory=dict)
    fiscal: dict[str, float] = field(default_factory=dict)

    # --- regional breakdowns: "<metric>.<REGION>" -> value (owners: belsim.regional, revenue) ---
    regional: dict[str, float] = field(default_factory=dict)

    # --- behavioural adjustments vs baseline (owner: belsim.behaviour) ---
    adjust: dict[str, Any] = field(default_factory=dict)

    # --- anything else modules need to carry year to year (namespaced by module) ---
    extra: dict[str, Any] = field(default_factory=dict)

    def series(self) -> dict[str, float]:
        """Flat scalar view of the year: 'section.key' -> value. Used by API and comparisons."""
        out: dict[str, float] = {}
        pop = self.pop
        out["pop.total"] = float(pop.sum())
        for r, name in enumerate(REGIONS):
            out[f"pop.region.{name}"] = float(pop[:, :, r].sum())
        for s, name in enumerate(SEXES):
            out[f"pop.sex.{name}"] = float(pop[:, s, :].sum())
        out["pop.age_0_17"] = float(pop[:18].sum())
        out["pop.age_18_66"] = float(pop[18:67].sum())
        out["pop.age_67_plus"] = float(pop[67:].sum())
        for i, name in enumerate(STATUSES):
            out[f"status.{name}"] = float(self.status[i].sum())
        for i, sec in enumerate(SECTORS):
            out[f"emp.sector.{sec}"] = float(self.emp[i].sum())
            out[f"va.sector.{sec}"] = float(self.va[i])
        for j, jt in enumerate(JOBTYPES):
            out[f"emp.jobtype.{jt}"] = float(self.emp[:, j].sum())
        for section in ("demo", "labour", "macro", "expenditure", "revenue", "fiscal", "regional"):
            for k, v in getattr(self, section).items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    out[f"{section}.{k}"] = float(v)
        return out


@dataclass
class Results:
    name: str
    years: list[int] = field(default_factory=list)
    states: list[YearState] = field(default_factory=list)

    def add(self, state: YearState) -> None:
        self.years.append(state.year)
        self.states.append(state)

    def at(self, year: int) -> YearState:
        return self.states[self.years.index(year)]

    def has(self, year: int) -> bool:
        return year in self.years

    def series(self) -> dict[str, list[float | None]]:
        """'section.key' -> list aligned with self.years (None where a key is missing)."""
        per_year = [s.series() for s in self.states]
        keys: dict[str, None] = {}
        for d in per_year:
            for k in d:
                keys.setdefault(k)
        return {k: [d.get(k) for d in per_year] for k in keys}
