"""Micro-calculators of the expenditure module (pure, vectorised, no state).

`replacement_income(status, prev_gross, p)` is the gross annual benefit of a NEW beneficiary whose
previous gross annual wage was `prev_gross`. All formulas are homogeneous of degree one in
(wage, EUR parameters): the model uses that to apply real welfare adjustments of ceilings and
minima as `W * f(wage / W)`.
"""
from __future__ import annotations

import numpy as np

PENSION_SCHEMES = ("employees", "self_employed", "civil_servants")
# job type (belsim.core.dims.JOBTYPES) -> pension scheme
JOBTYPE_SCHEME = {"emp_private": "employees", "self_employed": "self_employed", "emp_public": "civil_servants"}


def _capped(gross, rate, ceiling, floor):
    return np.maximum(rate * np.minimum(gross, ceiling), floor)


def replacement_income(status: str, prev_gross, p, scheme: str = "employees", career_extra_years: float = 0.0):
    """Gross annual benefit (EUR) of a new beneficiary.

    status: 'unemployed' | 'disabled' | 'retired'. For 'retired', `prev_gross` is the career-average
    (revalued) gross wage and `scheme` selects employees / self_employed / civil_servants.

    - unemployed: rate x wage capped at the wage ceiling, floored at the minimum benefit (average
      over the spell: the rate parameter already averages the degressive periods).
    - disabled:   same structure with the sickness-insurance parameters.
    - assisted:   social assistance (integration income): average amount x generosity factor.
    - retired:    replacement rate x capped wage x average career fraction, floored at the minimum
      pension (pro rata of the career), times the reform factor on new pensions of the scheme
      (`pension.new_pension_factor`: reference-wage period, accrual fractions, assimilated
      periods), times (1 + average bonus/malus). The wage is first multiplied by the share of
      career income that counts for the scheme (`pension.career_income_factor`, 1 except for
      the self-employed whose correction coefficient is being phased out). `career_extra_years`
      adds 1/45 of a career per year (people retiring later than in 2024), up to a full career.
    """
    g = np.asarray(prev_gross, dtype=float)
    if status == "unemployed":
        b = _capped(g, p["unemp.replacement_rate"], p["unemp.wage_ceiling"], p["unemp.min_benefit"])
        return b * p["unemp.generosity_factor"]
    if status == "disabled":
        b = _capped(g, p["disab.replacement_rate"], p["disab.wage_ceiling"], p["disab.min_benefit"])
        return b * p["disab.generosity_factor"]
    if status == "assisted":
        # social assistance: a flat means-tested amount, whatever the previous wage
        return np.full(g.shape, float(p["assist.amount"]) * float(p["assist.generosity_factor"]))
    if status == "retired":
        if scheme not in PENSION_SCHEMES:
            raise ValueError(f"unknown pension scheme '{scheme}'")
        cf = p["pension.career_fraction"][scheme]
        if career_extra_years:
            cf = min(max(cf + career_extra_years / 45.0, 0.0), max(cf, 1.0))
        cif = p.get("pension.career_income_factor")
        if cif and float(cif.get(scheme, 1.0)) != 1.0:
            g = g * float(cif[scheme])
        b = _capped(g, p["pension.replacement_rate"][scheme] * cf, p["pension.wage_ceiling"][scheme],
                    p["pension.min_pension"][scheme] * cf)
        exposure = p["pension.bonus_malus_exposure_years"]      # by scheme (a scalar applies to the three)
        if isinstance(exposure, dict):
            exposure = float(exposure.get(scheme, 0.0))
        return (b * p["pension.new_pension_factor"][scheme]
                * (1.0 + p["pension.bonus_malus"] + p["pension.bonus_malus_per_year"] * exposure))
    raise ValueError(f"unknown status '{status}'")


def pension_accrual_incentive(p) -> float:
    """Extra pension from one additional year of work, as a fraction of the pension (employees
    scheme): one more career year (1/45 of a full career) plus the bonus/malus per year. Used by
    the behaviour module for the retirement-age response (differences vs baseline only)."""
    cf = p["pension.career_fraction"]["employees"]
    return 1.0 / (45.0 * cf) + p["pension.bonus_malus_per_year"]
