"""Regional dimension: economy, labour market and governments of Flanders, Wallonia, Brussels.

`run(state, prev, p, ctx)` (engine step 8b, after `fiscal.balance`) writes
`state.regional["<metric>.<REG>"]` (REG in FLA, WAL, BRU) and the accounts of the federated
entities in `state.fiscal["<item>.<entity>"]` (`ENTITY_DETAIL`). It only distributes national
results: it never changes them. Regional policy enters the national model through the
region-keyed parameters read by labour (`labour.nairu`, `labour.activation_factor`,
`labour.disability_rate_factor`, `labour.public_employment_growth`), macro
(`regional.productivity_growth_wedge`) and expenditure (see `fiscal/expenditure/model.py`).

Units: persons; EUR mn (amounts); EUR per person (`*_per_capita`, `avg_wage`); fractions (rates
and growth); `*_pct_gdp` in % of the GDP of the region, as the national `fiscal.*_pct_gdp`.

Concepts
- Residence: population, labour-market status, household income. Place of work: employment_workplace,
  wage_bill (compensation of employees D1), avg_wage (D1 per employee), GDP (`economy.py`).
- Household income: national disposable income (macro) x share of the region, the share being
  (labour income of residents + cash social benefits by residence) x a factor fitted on the 2024
  regional household accounts (Eurostat nama_10r_2hhinc B6N); primary income likewise (B5N).
- Regional governments: `government.py`. `gov_*` aggregates the federated entities by region:
  Community accounts are split between Brussels and the other region by population served
  (`regional.brussels_dutch_share`), the consolidation item by `regional.consolidation_shares`
  (revenue, expenditure) and population (balance, debt). Local government by region: the 2024
  accounts by region (NBB); afterwards local taxes follow the tax bases of the region (municipal
  PIT surcharges, property tax), other revenue and expenditure the 2024 amount per inhabitant
  x the population of the region; the three regions always sum to the national aggregates.
- `net_fiscal_transfer`: revenue of the federal government and social security raised in the
  region minus their spending there, minus the region's population share of their combined
  balance (so the three regions sum to zero; positive = net contributor). Revenue by region:
  the revenue module's `revenue.<line>.<REG>` when published, else proxies (PIT and social
  contributions: labour income of residents fitted on the 2024 household accounts; consumption
  taxes: disposable income; other: GDP). Spending: social benefits by residence of the
  recipients (`regional.fiscal_incidence_keys`), Financing Act grants by receiving entity, the
  rest (general services, defence, interest, transfers to local government) by population.

No module state: everything carried from year to year is in `state.extra["regional"]`.
"""
from __future__ import annotations

import numpy as np

from ..core.dims import BASE_YEAR, ENTITIES, REGIONS, S
from . import data, economy, government
from .government import ENTITIES4, ENTITY_DETAIL, OTHER

__all__ = ["run", "ENTITY_DETAIL", "ENTITIES4", "OTHER", "METRICS"]

NS = "regional"
FED, SS, RC, LOC = ENTITIES
_EMPLOYED = [S["emp_private"], S["emp_public"], S["self_employed"]]

# metric -> (label, unit); every metric is published as state.regional["<metric>.<REG>"]
METRICS = {
    "population": ("Population (1 January)", "persons"),
    "working_age_population": ("Population aged 15-64", "persons"),
    "employment_residents": ("Employed residents", "persons"),
    "employment_workplace": ("Employment by place of work", "persons"),
    "net_commuting": ("Workplace employment minus employed residents", "persons"),
    "unemployed": ("Unemployed (benefit recipients + excluded job-seekers)", "persons"),
    "unemployment_benefit_recipients": ("Unemployment-benefit recipients", "persons"),
    "unemployment_rate": ("Unemployment rate", "fraction"),
    "employment_rate_20_64": ("Employment rate 20-64", "fraction"),
    "disabled": ("People in invalidity", "persons"),
    "retired": ("Pensioners", "persons"),
    "inactive": ("Other inactive 18-64", "persons"),
    "public_employees": ("General-government employees residing in the region", "persons"),
    "wage_bill": ("Compensation of employees, place of work", "EUR mn"),
    "avg_wage": ("Compensation per employee, place of work", "EUR"),
    "gdp_nominal": ("GDP, current prices", "EUR mn"),
    "gdp_real": ("GDP, 2024 prices", "EUR mn"),
    "gdp_per_capita": ("GDP per inhabitant", "EUR"),
    "gdp_real_growth": ("Real GDP growth", "fraction"),
    "productivity": ("GDP per person employed at the workplace", "EUR"),
    "household_income": ("Household disposable income", "EUR mn"),
    "household_income_per_capita": ("Household disposable income per inhabitant", "EUR"),
    "household_primary_income_per_capita": ("Household primary income per inhabitant", "EUR"),
    "social_benefits": ("Cash social benefits received by residents", "EUR mn"),
    "gov_revenue": ("Federated entities: revenue", "EUR mn"),
    "gov_expenditure": ("Federated entities: expenditure", "EUR mn"),
    "gov_balance": ("Federated entities: balance", "EUR mn"),
    "gov_debt": ("Federated entities: debt", "EUR mn"),
    "gov_balance_pct_gdp": ("Federated entities: balance", "% of regional GDP"),
    "gov_debt_pct_gdp": ("Federated entities: debt", "% of regional GDP"),
    "gov_own_taxes": ("Federated entities: regional taxes", "EUR mn"),
    "transfers_from_federal": ("Financing Act transfers from the federal government", "EUR mn"),
    "local_gov_revenue": ("Local government: revenue", "EUR mn"),
    "local_gov_expenditure": ("Local government: expenditure", "EUR mn"),
    "local_gov_balance": ("Local government: balance", "EUR mn"),
    "local_gov_debt": ("Local government: debt", "EUR mn"),
    "federal_revenue_raised": ("Federal + social-security revenue raised in the region", "EUR mn"),
    "federal_spending": ("Federal + social-security spending in the region", "EUR mn"),
    "net_fiscal_transfer": ("Net contribution through the federal budget and social security", "EUR mn"),
    "net_fiscal_transfer_pct_gdp": ("Net contribution through the federal budget and social security",
                                    "% of regional GDP"),
}
# cash-benefit lines counted in the social benefits received by households
_CASH_BENEFITS = ("pension_employees", "pension_self_employed", "pension_civil_servants", "pension_grapa",
                  "unemployment", "rcc_swt", "invalidity", "sickness_primary", "disability_allowances",
                  "family_allowances", "social_assistance", "labour_market_benefits")


def _sh(a) -> np.ndarray:
    return government._sh(a)


def _profile(ages, costs, n=101) -> np.ndarray:
    out = np.zeros(n)
    bounds = list(ages) + [n]
    for lo, hi, c in zip(bounds[:-1], bounds[1:], costs):
        out[int(lo):int(hi)] = c
    return out


def _economy(state, p) -> dict:
    eco = state.extra.get(economy.NS)
    if eco is None:        # states built without macro.output (unit tests)
        reg = economy.workplace(state, p, economy.calibrate(state, p))
        m = state.macro
        eco = economy.finish(reg, np.ones(3), m.get("gdp_nominal", 0.0), m.get("gdp_real", 0.0),
                             m.get("compensation_employees", 0.0))
    return eco


def run(state, prev, p, ctx=None) -> None:
    px = prev.extra.get(NS) if prev is not None else None
    bs = getattr(ctx, "base_state", None) if ctx is not None else None
    bx = bs.extra.get(NS) if (prev is None and bs is not None) else None
    src = px if px is not None else bx
    st, pop, m, lab, f = state.status, state.pop, state.macro, state.labour, state.fiscal
    eco = _economy(state, p)
    lx = state.extra.get("labour") or {}
    R: dict[str, np.ndarray] = {}

    # ---------------- people ----------------
    pop_r = pop.sum(axis=(0, 1))
    pop_age_r = pop.sum(axis=1)
    by = {k: st[S[k]].sum(axis=(0, 1)) for k in ("unemployed", "disabled", "retired", "emp_private", "emp_public")}
    employed = st[_EMPLOYED].sum(axis=0)
    emp_res = employed.sum(axis=(0, 1))
    excluded = np.asarray(lx.get("excluded_inactive_region", np.zeros(3)), dtype=float)
    unemployed = by["unemployed"] + excluded
    inactive = st[S["inactive"]][18:65].sum(axis=(0, 1))
    R["population"] = pop_r
    R["working_age_population"] = pop[15:65].sum(axis=(0, 1))
    R["employment_residents"] = emp_res
    R["employment_workplace"] = eco["emp_wp"]
    R["net_commuting"] = eco["emp_wp"] - emp_res
    R["unemployed"] = unemployed
    R["unemployment_benefit_recipients"] = by["unemployed"]
    R["unemployment_rate"] = unemployed / np.maximum(emp_res + unemployed, 1.0)
    R["employment_rate_20_64"] = employed[20:65].sum(axis=(0, 1)) / np.maximum(pop[20:65].sum(axis=(0, 1)), 1.0)
    R["disabled"], R["retired"], R["inactive"] = by["disabled"], by["retired"], inactive
    R["public_employees"] = by["emp_public"]

    # ---------------- economy by place of work ----------------
    R["wage_bill"] = eco["d1"]
    R["avg_wage"] = eco["d1"] * 1e6 / np.maximum(eco["employees_wp"], 1.0)
    R["gdp_nominal"], R["gdp_real"] = eco["gdp"], eco["gdp_real"]
    R["gdp_per_capita"] = eco["gdp"] * 1e6 / np.maximum(pop_r, 1.0)
    R["productivity"] = eco["gdp"] * 1e6 / np.maximum(eco["emp_wp"], 1.0)
    if prev is not None and all(f"gdp_real.{r}" in prev.regional for r in REGIONS):
        R["gdp_real_growth"] = eco["gdp_real"] / np.array([prev.regional[f"gdp_real.{r}"] for r in REGIONS]) - 1.0
    else:
        R["gdp_real_growth"] = data.accounts(BASE_YEAR)["gdp_real_growth"] / 100.0

    # ---------------- allocation keys ----------------
    prof_m = _profile(p["health.profile_ages"], p["health.profile_cost"])
    prof_f = prof_m * _profile(p["health.profile_ages"], p["health.profile_female_ratio"])
    amounts = p["family.allowance_per_child"]
    kids = pop[:18].sum(axis=(0, 1)) + float(p["family.student_share"]) * pop[18:25].sum(axis=(0, 1))
    labour_income = (m["compensation_employees"] * _sh(eco["employees_res"])
                     + m["mixed_income"] * _sh(st[S["self_employed"]].sum(axis=(0, 1))))
    K = {
        "pop": pop_r, "population": pop_r, "gdp": eco["gdp"],
        "pupils": pop[6:18].sum(axis=(0, 1)),
        "young_old": pop[:18].sum(axis=(0, 1)) + pop[80:].sum(axis=(0, 1)),
        "retired": by["retired"], "elderly": pop[65:].sum(axis=(0, 1)), "unemployed": by["unemployed"],
        "disabled": by["disabled"], "inactive": inactive, "private_employees": by["emp_private"],
        "public_employees": by["emp_public"],
        "children": kids, "family": kids * np.array([float(amounts[r]) for r in REGIONS]),
        "health_need": prof_m @ pop[:, 0, :] + prof_f @ pop[:, 1, :],
        "ltc_need": _profile(p["health.ltc_profile_ages"], p["health.ltc_profile_cost"]) @ pop_age_r,
    }
    calib = src["calib"] if src is not None else None
    if calib is None:
        li = _sh(labour_income)
        calib = {"pit": data.shares("hh_income_taxes_paid") / li,
                 "ssc": data.shares("hh_social_contributions_paid") / li,
                 "primary": data.shares("hh_primary_income") / li}

    # ---------------- cash benefits by residence, household income ----------------
    inc_keys = p["regional.fiscal_incidence_keys"]
    benefits = np.zeros(3)
    for lid in _CASH_BENEFITS:
        benefits += state.expenditure.get(lid, 0.0) * _sh(K["family"] if lid == "family_allowances"
                                                          else K[inc_keys.get(lid, "population")])
    if "disposable" not in calib:
        calib = dict(calib, disposable=data.shares("hh_disposable_income") / _sh(labour_income + benefits))
    disp = m.get("household_disposable_income", 0.0) * _sh(calib["disposable"] * (labour_income + benefits))
    primary_nat = m["compensation_employees"] + m["mixed_income"] + m.get("household_capital_income", 0.0)
    primary = primary_nat * _sh(calib["primary"] * labour_income)
    K["pit"] = calib["pit"] * labour_income
    K["ssc"] = calib["ssc"] * labour_income
    K["disp"] = disp
    vals = [state.regional.get(f"revenue.pit_federal.{r}") for r in REGIONS]
    if all(v is not None and v > 0 for v in vals):
        K["pit"] = np.array(vals, dtype=float)          # the revenue module's PIT by residence
    R["social_benefits"] = benefits
    R["household_income"] = disp
    R["household_income_per_capita"] = disp * 1e6 / np.maximum(pop_r, 1.0)
    R["household_primary_income_per_capita"] = primary * 1e6 / np.maximum(pop_r, 1.0)

    # ---------------- federated entities ----------------
    gx_prev = {k: px[k] for k in ("key0", "resid", "bal_e")} if px is not None else None
    gx_base = {k: bx[k] for k in ("key0", "resid", "bal_e")} if bx is not None else None
    fiscal, reg, gx = government.compute(state, prev, p, K, gx_prev, gx_base)
    f.update(fiscal)
    R.update(reg)
    R["gov_balance_pct_gdp"] = reg["gov_balance"] / eco["gdp"] * 100.0
    R["gov_debt_pct_gdp"] = reg["gov_debt"] / eco["gdp"] * 100.0

    # ---------------- local government by region ----------------
    # 2024 accounts by region (NBB), then: local taxes follow the tax bases of the region (the
    # revenue module's municipal PIT surcharges and property tax by region), the other revenue
    # (mostly grants) and expenditure follow the 2024 amount per inhabitant x population of the
    # region; debt = last year's debt - balance + the region's share of the stock-flow adjustment.
    from ..fiscal import revenue as _rev_mod
    loc = p["regional.local_government_2024"]
    loc_share = {item: _sh([float(loc[item][r]) for r in REGIONS]) for item in ("revenue", "expenditure", "debt")}
    prop_meta = (getattr(_rev_mod, "LINES", None) or {}).get("property_tax")
    tax_vals = [(state.regional.get(f"revenue.pit_local.{r}"), state.regional.get(f"revenue.property_tax.{r}"))
                for r in REGIONS]
    if all(a is not None and b_ is not None for a, b_ in tax_vals):
        loc_tax = np.array([a + b_ * government._entity_share(prop_meta, LOC, 1.0) for a, b_ in tax_vals])
        loc_tax = np.minimum(loc_tax, f[f"revenue.{LOC}"] * loc_share["revenue"])     # never above the 2024 share
    else:
        loc_tax = np.zeros(3)
    loc0 = src.get("local0") if src is not None else None
    if loc0 is None:
        loc0 = {"pop": pop_r.copy(),
                "other": np.maximum(f[f"revenue.{LOC}"] * loc_share["revenue"] - loc_tax, 0.0),
                "exp": f[f"expenditure.{LOC}"] * loc_share["expenditure"]}
    pop_idx = pop_r / loc0["pop"]
    R["local_gov_revenue"] = loc_tax + (f[f"revenue.{LOC}"] - float(loc_tax.sum())) * _sh(loc0["other"] * pop_idx)
    R["local_gov_expenditure"] = f[f"expenditure.{LOC}"] * _sh(loc0["exp"] * pop_idx)
    R["local_gov_balance"] = R["local_gov_revenue"] - R["local_gov_expenditure"]
    debt_nat = f[f"debt.{LOC}"]
    if prev is not None and all(f"local_gov_debt.{r}" in prev.regional for r in REGIONS):
        d_prev = np.array([prev.regional[f"local_gov_debt.{r}"] for r in REGIONS])
        sfa_loc = debt_nat - float(d_prev.sum()) + float(R["local_gov_balance"].sum())
        R["local_gov_debt"] = d_prev - R["local_gov_balance"] + sfa_loc * _sh(np.abs(d_prev))
    else:
        R["local_gov_debt"] = debt_nat * loc_share["debt"]

    # ---------------- federal budget and social security by region ----------------
    from ..fiscal import expenditure as exp_mod
    try:
        from ..fiscal import revenue as rev_mod
        rev_lines = getattr(rev_mod, "LINES", None) or {}
    except Exception:
        rev_lines = {}
    exp_lines = getattr(exp_mod, "LINES", {}) or {}
    pop_sh = _sh(pop_r)
    own_taxes = p.get("regional.own_taxes_2024")
    rev_fs = np.zeros(3)
    for lid, v in state.revenue.items():
        sh = government._entity_share(rev_lines.get(lid), FED, 0.0) + government._entity_share(rev_lines.get(lid), SS, 0.0)
        if sh and v:
            rev_fs += v * sh * government.revenue_region_shares(state, lid, K, gx["key0"], own_taxes)
    rev_tot = (f[f"own_revenue.{FED}"] + f[f"own_revenue.{SS}"] + f.get(f"entity_residual.{FED}", 0.0)
               + f.get(f"entity_residual.{SS}", 0.0)
               + sum(f.get(f"transfer.{a}_to_{b}", 0.0) for a in (RC, LOC) for b in (FED, SS)))
    rev_fs += (rev_tot - float(rev_fs.sum())) * _sh(eco["gdp"])
    exp_fs = np.zeros(3)
    for lid, v in state.expenditure.items():
        sh = government._entity_share(exp_lines.get(lid), FED, 0.0) + government._entity_share(exp_lines.get(lid), SS, 0.0)
        if sh and v:
            exp_fs += v * sh * _sh(K[inc_keys.get(lid, "population")])
    exp_fs += (f[f"own_expenditure.{FED}"] + f[f"own_expenditure.{SS}"] - float(exp_fs.sum())) * pop_sh
    exp_fs += gx["grant_e"] @ gx["m_er"]
    exp_fs += (f.get(f"transfer.{SS}_to_{RC}", 0.0) + f.get(f"transfer.{FED}_to_{LOC}", 0.0)
               + f.get(f"transfer.{SS}_to_{LOC}", 0.0)) * pop_sh
    raw = rev_fs - exp_fs
    R["federal_revenue_raised"], R["federal_spending"] = rev_fs, exp_fs
    R["net_fiscal_transfer"] = raw - float(raw.sum()) * pop_sh
    R["net_fiscal_transfer_pct_gdp"] = R["net_fiscal_transfer"] / eco["gdp"] * 100.0

    for metric, arr in R.items():
        for i, r in enumerate(REGIONS):
            state.regional[f"{metric}.{r}"] = float(arr[i])
    state.extra[NS] = {"calib": calib, "key0": gx["key0"], "resid": gx["resid"], "bal_e": gx["bal_e"],
                       "local0": loc0}
