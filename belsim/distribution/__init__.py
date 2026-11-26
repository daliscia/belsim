"""Distributional analysis: stylised households, net incomes, winners and losers of a scenario.

    from belsim.distribution import distribution, payslip
    out = distribution(base_results, scen_results, base_pset, scen_pset, 2030)

See `analysis.py` for the method and its limits, `households.py` for the household set and its
weights, `calc.py` for the net income of a household. The tax rules themselves (marital
quotient, supplements for children, child credit, surcharges by region) are those of
`belsim.fiscal.revenue.calc.tax_household`, the calculator behind the aggregate income tax.
"""
from .analysis import distribution, param_view, payslip_grid
from .calc import HOUSEHOLD_PRESETS, BenefitEnv, net_income, payslip
from .households import Household, household_set, weights

__all__ = ["distribution", "payslip", "payslip_grid", "param_view", "net_income", "household_set", "weights",
           "Household", "BenefitEnv", "HOUSEHOLD_PRESETS"]
