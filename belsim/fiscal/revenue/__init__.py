"""Government revenue: every revenue line of Belgian general government (ESA 2010, S13
consolidated). The sum of the lines is total general-government revenue.

Each line = tax base (from labour / macro / expenditure) x legislated parameters
(params/revenue.yaml, through the calculators of `calc.py`) x calibration factor.

Personal income tax and social contributions are the household calculator
(`calc.tax_household`, the same one `belsim.distribution` uses) summed over a population of tax
units: region of residence x tax-unit type (dependants, single parents, sole-income couples) x
24 income groups, for each kind of income, put on the income distribution of the tax returns -
see `taxunits.py`. Head counts and incomes come from
the simulated state; the tax base is aligned on the fiscal statistics in the base year.

Calibration. In the base year the static model of every line is evaluated with whatever bases
the state provides and the factor `actual_2024 / model_2024` is stored in
`state.extra["revenue"]`, then carried unchanged from `prev.extra["revenue"]`. The 2024 actuals
and the entity shares come from `data/base/revenue_lines.csv` (Eurostat gov_10a_main,
gov_10a_taxag and the NBB national tax list, built by `belsim.data.fetch_revenue`).
If a base is missing or zero in the base year the line falls back on nominal GDP as its base.
For the personal income tax the remaining factors stand for what is taxed outside the model's
tax units (non-residents, withholding taxes that are final) and for the timing of the
accounts: the regional and municipal surcharges are recorded when assessed, a year or two
after the income (factors of about 0.9: ESA 2024 municipal receipts 4.1 bn against 4.4 bn
assessed on 2024 incomes and 4.2 bn on 2023 incomes, Statbel), the federal line takes the
counterpart. The model attributes all three levels to the income year.

Observed years and baseline paths (`state.extra["revenue_paths"]`). What the static model of a
line cannot know is carried by a few baseline adjusters, computed in the baseline run and
reused unchanged by every scenario of the same year (so a scenario moves a line only through
its own bases and rates):
  outturn    the baseline reproduces the observed 2025 accounts line by line
             (data/base/revenue_outturn.csv, same sources and mapping as the base year); the
             ratio observed / model of each line then persists, except for the share that
             `rev.outturn_persistence` marks as temporary (assessment timing, one-offs);
  cit_base   taxable corporate profits follow `cit.base_share_path` (share of GDP, from the
             Planning Bureau's company account) instead of the model's profit proxy, which has
             no depreciation dynamics; after the last year of the path the ratio to the proxy
             is held;
  volume     taxed volumes of fuel, tobacco and alcohol follow `excise.volume_growth` instead
             of real consumption of the category;
  lag        personal income tax amounts are indexed by law on the consumer prices of the
             previous year, not on the health index of the year (`legal_params`). The ratio of
             the two is that of the baseline; in a scenario the amounts carry the deviation of
             the health index from the baseline with the same one-year delay (`lag_run`).
Alignment (`rev.fpb_alignment`, switch `rev.fpb_alignment_enabled`). The observed 2025 accounts
are the starting point and are never overridden. From 2026 the change of each category of the
Planning Bureau's account since 2025 (household and indirect taxes by subsector, corporate tax,
capital taxes, contributions; points of GDP) is the Bureau's: dated multipliers by category close
what is left between the model's own projection and that change. They are parameters, not
adjusters - a scenario that switches them off shows their size - and they are computed by
`align.py` (`uv run python -m belsim.fiscal.revenue.align`). The last entry persists, so the
aligned ratios carry on.
The adjusters need the run context of the engine (`ctx.baseline`); without it (unit tests on
synthetic states) they are neutral.

Behaviour. Bases arrive already adjusted for behavioural responses (macro applies
`state.adjust`); nothing is re-applied here. Statically, revenue changes by rate x base:
consumption taxes are levied on the pre-tax value of each category, obtained by removing the
*baseline* taxes (`ctx.base_p`) from `macro["cons.<cat>"]`, so a rate change is not diluted.
Supplies that a scenario takes out of the standard VAT rate keep the compliance gap of
standard-rated supplies (`calc.vat_collected_rate`).
(If macro passes tax changes through to the value of `cons.<cat>`, it can set
`macro["cons_reflects_tax_changes"] = 1` and the scenario's own rates are removed instead.)

Regions. `state.regional` receives, for REG in FLA, WAL, BRU (values sum to the national line):
  revenue.<line>.<REG>               every line of `REGIONAL_LINES`, by region
  revenue_total.<REG>                regional + local own taxes: the lines above x the share of
                                     each line received by regions, communities and local
                                     government (`LINES[line].entity_shares`)
  federal_taxes_by_residence.<REG>   the same lines x the share received by the federal
                                     government and social security
Allocation: personal income tax (three levels), special contribution and social contributions
by residence of the taxpayer, from the population of tax units (employer contributions follow
the residence of the worker); registration duties, inheritance tax and property tax from the
regional bases and rates (`prop.*`); the other lines with the key of
`rev.regional_allocation` (consumption = disposable income of the tax units by residence,
population, vehicles). Regional levers: `pit.regional_surcharge`, `pit.municipal_surcharge`,
`pit.tax_expenditure_*`, `prop.*` (dicts by region) and `rev.regional_factor`.

Inputs consumed
  state.emp [sector, jobtype], state.wage_dist [jobtype, decile], state.status, state.pop
  state.macro: gdp_nominal, cpi, cons.<cat>, corporate_profits_taxable,
      household_capital_income, household_financial_wealth, property_transactions,
      inheritance_flow, gov_consumption (fallback only), investment_housing
  state.expenditure: pension / unemployment / sickness-invalidity benefit lines (found by name;
      fallback: status headcounts x expenditure.calc.replacement_income); investment and the
      lines holding intermediate consumption (non-deductible VAT paid by government itself, so
      a spending cut loses VAT; the baseline lines of the year come from `ctx.base_state`)
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...core.dims import BASE_YEAR, CONS_CATS, ENTITIES, REGIONS, S
from ...core.params import ParamView
from ...core.state import LineMeta
from . import calc, taxunits

DATA = Path(__file__).resolve().parents[3] / "data" / "base"

# line -> (label, group, description). Amounts, ESA codes and entity shares come from the data.
_META: dict[str, tuple[str, str, str]] = {
    "ssc_employer_private": ("Employer contributions, private sector", "Social contributions",
                             "Actual employer social contributions paid by non-government employers: face rate on gross wages less the structural (low-wage) reduction and, from mid-2025, the exemption above the very-high-wage ceiling."),
    "ssc_employer_public": ("Employer contributions, public sector", "Social contributions",
                            "Actual employer social contributions paid by general government on its own wage bill."),
    "ssc_imputed": ("Imputed contributions (civil-servant pensions)", "Social contributions",
                    "Imputed employer contributions, the counterpart of unfunded civil-servant pensions (also recorded as expenditure)."),
    "ssc_employee": ("Employee contributions", "Social contributions",
                     "Personal contributions on wages (13.07 %) less the work bonus for low wages (parts A and B)."),
    "ssc_self_employed": ("Self-employed contributions", "Social contributions",
                          "Contributions of the self-employed (20.5 % / 14.16 % with income ceilings and a minimum)."),
    "ssc_replacement_income": ("Contributions on replacement income", "Social contributions",
                               "Contributions of the non-employed, mainly the 3.55 % health contribution on pensions."),
    "special_ssc": ("Special social security contribution", "Social contributions",
                    "Special contribution of employees by band of taxable income (recorded as an income tax in ESA, received by social security); halved and individualised from income 2028."),
    "pit_federal": ("Personal income tax, federal", "Personal income tax",
                    "Federal personal income tax (payroll withholding, prepayments, assessments, non-residents) after the transfer of the regional share; net of the federal pension-savings relief."),
    "pit_regional": ("Personal income tax, regional surcharges", "Personal income tax",
                     "Regional surcharges (opcentiemen / centimes additionnels) on the reduced state tax, net of regional tax reductions (service vouchers, housing)."),
    "pit_local": ("Personal income tax, municipal surcharges", "Personal income tax",
                  "Municipal surcharges on personal income tax."),
    "cit": ("Corporate income tax", "Corporate income tax",
            "Corporate income tax incl. prepayments, assessments, non-resident companies and withholding tax paid by companies."),
    "vat": ("VAT", "VAT", "Value added tax on household consumption, housing investment, government purchases and inputs of exempt sectors."),
    "excise_fuel": ("Excise on fuel and energy products", "Excise and consumption taxes",
                    "Excise duties on mineral oils and energy products, incl. strategic-stock contributions."),
    "excise_tobacco": ("Excise on tobacco", "Excise and consumption taxes", "Excise duties on tobacco products."),
    "excise_alcohol": ("Excise on alcohol", "Excise and consumption taxes", "Excise duties on beer, wine, spirits and intermediate products."),
    "excise_energy": ("Federal energy levies", "Excise and consumption taxes",
                      "Energy contribution, federal contribution on electricity and gas, heating-oil contributions."),
    "energy_green_levies": ("Green-certificate and regional energy levies", "Excise and consumption taxes",
                            "Regional green-certificate obligations and network surcharges recorded as taxes on energy."),
    "excise_beverages_packaging": ("Excise on soft drinks and coffee, packaging levy", "Excise and consumption taxes",
                                   "Excise on non-alcoholic beverages and coffee, packaging levy, animal-health levies."),
    "insurance_premium_tax": ("Tax on insurance premiums", "Excise and consumption taxes",
                              "Annual tax on insurance contracts (9.25 %, 9.60 % from July 2026) and social-security supplements on car, fire and hospital insurance."),
    "gambling_lottery": ("Gambling taxes and National Lottery", "Excise and consumption taxes",
                         "Regional taxes on gambling and betting, National Lottery monopoly profits and rent."),
    "withholding_tax": ("Withholding tax on interest and dividends", "Capital and wealth taxes",
                        "Withholding tax (30 %) on interest and dividends received by households."),
    "capital_gains_tax": ("Capital gains tax on financial assets", "Capital and wealth taxes",
                          "Tax on realised capital gains on financial assets (10 % from 2026, law of 6 April 2026). Yield follows the official estimate, not calibrated."),
    "securities_account_tax": ("Tax on securities accounts", "Capital and wealth taxes",
                               "Annual tax on securities accounts above 1 million EUR (0.15 %, 0.30 % from 2026)."),
    "stock_exchange_tax": ("Stock-exchange tax", "Capital and wealth taxes", "Tax on stock-exchange transactions (TOB)."),
    "registration_duties": ("Registration duties", "Capital and wealth taxes",
                            "Regional registration duties on real-estate purchases and mortgage duties."),
    "inheritance_gift_tax": ("Inheritance and gift tax", "Capital and wealth taxes", "Regional inheritance and gift taxes."),
    "property_tax": ("Recurrent property tax", "Capital and wealth taxes",
                     "Precompte immobilier / onroerende voorheffing: regional levy plus provincial and municipal surcharges."),
    "vehicle_taxes": ("Vehicle and road taxes", "Other taxes",
                      "Annual road tax, registration tax (BIV/TMC) and kilometre charge, paid by households and companies."),
    "environmental_taxes": ("Emission permits and other environmental taxes", "Other taxes",
                            "Auctioned ETS emission permits, regional taxes on water, waste and manure."),
    "bank_levies": ("Bank and financial-sector levies", "Other taxes",
                    "Annual tax on credit institutions, deposit-guarantee contribution, tax on collective investment and insurance undertakings."),
    "nuclear_levy": ("Nuclear levy", "Other taxes", "Contribution payable by nuclear operators."),
    "pharma_levies": ("Pharmaceutical industry levies", "Other taxes", "Contributions on the turnover of the pharmaceutical industry (to social security)."),
    "other_taxes_production": ("Other taxes on production and products", "Other taxes",
                               "Local business taxes, court and document duties, aviation tax, other federal and regional taxes on production."),
    "other_taxes_income_wealth": ("Other taxes on income, wealth and capital", "Other taxes",
                                  "Local taxes on households, tax on non-profit institutions, long-term-savings tax, tax regularisation."),
    "sales": ("Sales of goods and services", "Non-tax revenue",
              "Market output, output for own final use and payments for non-market output (fees, tuition, hospital and care fees)."),
    "interest_received": ("Interest received", "Non-tax revenue", "Interest received from outside general government."),
    "dividends_received": ("Dividends received", "Non-tax revenue",
                           "Dividends from public holdings (Belfius, BNP Paribas, Proximus, bpost, National Bank, regional and municipal holdings)."),
    "rent_other_property_income": ("Rent and other property income", "Non-tax revenue", "Rent on land and natural resources, other investment income."),
    "eu_transfers_current": ("Current transfers from the EU", "Non-tax revenue",
                             "Current transfers from EU institutions (incl. Recovery and Resilience Facility grants until 2026)."),
    "other_current_transfers": ("Other current transfers received", "Non-tax revenue",
                                "Miscellaneous current transfers from outside general government, other international cooperation, subsidies received."),
    "eu_transfers_capital": ("Capital transfers from the EU", "Non-tax revenue", "Investment grants from EU institutions."),
    "other_capital_transfers": ("Other capital transfers received", "Non-tax revenue", "Investment grants and other capital transfers from outside general government."),
    "uncollectible_taxes": ("Taxes assessed but unlikely to be collected", "Non-tax revenue",
                            "ESA adjustment D.995 (negative): assessed VAT, withholding tax and contributions that will not be collected."),
}

# lines whose base is nominal GDP (no explicit base in the model)
_GDP_LINES = ("environmental_taxes", "nuclear_levy", "pharma_levies", "other_taxes_production",
              "other_taxes_income_wealth", "sales", "interest_received", "rent_other_property_income",
              "eu_transfers_current", "other_current_transfers", "eu_transfers_capital",
              "other_capital_transfers", "uncollectible_taxes")
# lines with no 2024 receipts: yield comes from parameters alone
_UNCALIBRATED = ("capital_gains_tax",)
_VAT_EXTRA_BASES = ("housing_investment", "government", "government_investment", "exempt_sectors")
_BENEFIT_KINDS = {"pension": "retired", "unemployment": "unemployed", "disability": "disabled"}
_BENEFIT_WORDS = {
    "pension": ("pension",),
    "unemployment": ("unemploy", "chomage", "werkloos", "rcc", "swt"),
    "disability": ("disab", "invalid", "sick", "incapac"),
}
# not taxable cash benefits: administration, care in kind, means-tested allowances
_NOT_BENEFIT = ("admin", "care", "allowance", "grapa", "assistance")
_PUBLIC_PENSION_WORDS = ("public", "civil", "statut", "government")
_PIT_LINES = ("pit_federal", "pit_regional", "pit_local")
# split of `pit.reform_other_measures_cost` between the federal and the regional tax; with the
# municipal surcharges that follow (about 6.6 % of both) the three levels sum to the amount
_REFORM_FEDERAL, _REFORM_REGIONAL = 0.732, 0.206
# lines attributed to regions from the population of tax units (residence of the taxpayer)
_UNIT_LINES = ("ssc_employee", "special_ssc", "ssc_employer_private", "ssc_employer_public",
               "ssc_self_employed", "ssc_replacement_income")
# lines attributed with the regional bases and rates of the property taxes
_PROPERTY_LINES = {"registration_duties": calc.property_transaction_tax_by_region,
                   "inheritance_gift_tax": calc.inheritance_tax_by_region,
                   "property_tax": calc.property_tax_by_region}
# lines attributed with a key of `rev.regional_allocation`
_KEY_LINES = ("vat", "excise_fuel", "excise_tobacco", "excise_alcohol", "excise_energy",
              "excise_beverages_packaging", "energy_green_levies", "gambling_lottery", "vehicle_taxes",
              "environmental_taxes", "other_taxes_production", "other_taxes_income_wealth")
REGIONAL_LINES = _PIT_LINES + _UNIT_LINES + tuple(_PROPERTY_LINES) + _KEY_LINES


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, float], dict[str, LineMeta]]:
    """(2024 actuals EUR mn, LINES) from data/base/revenue_lines.csv. Read once, read-only."""
    path = DATA / "revenue_lines.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python -m belsim.data.fetch_revenue`")
    actual: dict[str, float] = {}
    lines: dict[str, LineMeta] = {}
    with path.open() as f:
        rows = {r["line"]: r for r in csv.DictReader(f)}
    for name, (label, group, desc) in _META.items():
        r = rows[name]
        actual[name] = float(r["amount"])
        lines[name] = LineMeta(label=label, group=group, esa=r["esa"],
                               entity_shares={e: float(r[e]) for e in ENTITIES}, description=desc)
    unknown = set(rows) - set(_META)
    if unknown:
        raise ValueError(f"revenue_lines.csv has lines unknown to the model: {sorted(unknown)}")
    return actual, lines


LINES: dict[str, LineMeta] = _load()[1]


@lru_cache(maxsize=1)
def outturn() -> dict[int, dict[str, float]]:
    """year -> observed amount of every line (EUR mn) for the years after the base year that
    are published (data/base/revenue_outturn.csv). Empty if the file is missing."""
    path = DATA / "revenue_outturn.csv"
    out: dict[int, dict[str, float]] = {}
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):
                if r["line"] in _META:
                    out.setdefault(int(r["year"]), {})[r["line"]] = float(r["amount"])
    return out


def actual_2024() -> dict[str, float]:
    """Official 2024 amount of every line (EUR mn)."""
    return dict(_load()[0])


# ---------------------------------------------------------------------------------------------
# bases taken from the expenditure module
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _benefit_kind(key: str) -> tuple[str | None, bool]:
    """(taxable benefit kind of an expenditure line, whether it is a civil-servant pension)."""
    k = key.lower()
    if any(w in k for w in _NOT_BENEFIT):
        return None, False
    for kind, words in _BENEFIT_WORDS.items():
        if any(w in k for w in words):
            return kind, kind == "pension" and any(w in k for w in _PUBLIC_PENSION_WORDS)
    return None, False


def _benefits(state, p) -> tuple[dict[str, tuple[float, float]], float]:
    """kind -> (benefit spending EUR mn, beneficiaries), and civil-servant pension spending."""
    exp = state.expenditure or {}
    totals = dict.fromkeys(_BENEFIT_KINDS, 0.0)
    public_pensions = 0.0
    for key, val in exp.items():
        kind, public = _benefit_kind(key)
        if kind is not None:
            totals[kind] += float(val)
            if public:
                public_pensions += float(val)
    out = {}
    for kind, status in _BENEFIT_KINDS.items():
        count = float(state.status[S[status]].sum())
        total = totals[kind]
        if total <= 0.0 and count > 0.0:
            total = _fallback_benefit(state, p, status) * count / 1e6
        out[kind] = (total, count)
    return out, public_pensions


def _minimum_pension_increment(state, p, pb, base_state) -> tuple[float, float] | None:
    """(pension spending that a scenario adds through the guaranteed minimum pension, EUR mn -
    negative for a cut -, the minimum itself in EUR a year), or None without a change: the share
    `pension.min_pension_spending_share` of each scheme's spending follows
    `pension.min_pension` (expenditure), and that money goes to the pensions at or near the
    minimum, which pay little or no income tax - not to all pensions in proportion. The change
    is measured in real terms (the amounts are indexed: a scenario with other prices has
    another amount in euros without having changed the minimum)."""
    if pb is p or pb is None or base_state is None:
        return None
    index, index_b = float(state.macro.get("health_index", 1.0)), float(base_state.macro.get("health_index", 1.0))
    real = index_b / index if index > 0 and index_b > 0 else 1.0
    get, get_b = getattr(p, "get", None), getattr(pb, "get", None)
    new, old = (get("pension.min_pension") if get else None), (get_b("pension.min_pension") if get_b else None)
    share = (get("pension.min_pension_spending_share") if get else None) or {}
    if not new or not old:
        return None
    exp = state.expenditure or {}
    total = 0.0
    for scheme, k in share.items():
        if not old.get(scheme) or scheme not in new:
            continue
        ratio = round(float(new[scheme]) / float(old[scheme]) * real, 9)
        step = float(k) * (ratio - 1.0)
        if ratio != 1.0 and step > -1.0:
            total += float(exp.get(f"pension_{scheme}", 0.0)) * step / (1.0 + step)
    if total == 0.0:
        return None
    return total, sum(float(v) for v in new.values()) / len(new)


def _fallback_benefit(state, p, status: str) -> float:
    """Average benefit from the expenditure calculator when no expenditure line is found."""
    workers = state.emp.sum(axis=0)
    n = float(workers.sum())
    if n <= 0.0:
        return 0.0
    avg_wage = float((workers * state.wage_dist.mean(axis=1)).sum() / n)
    try:
        from ..expenditure import calc as ecalc
        return float(np.asarray(ecalc.replacement_income(status, avg_wage, p)).mean())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------------------------
# static model of every line (before calibration)
# ---------------------------------------------------------------------------------------------
def _own_share(line: str) -> float:
    """Share of a line received by regions, communities and local government."""
    sh = LINES[line].entity_shares
    return sh["regions_communities"] + sh["local"]


_OWN_SHARE = {line: _own_share(line) for line in REGIONAL_LINES}
_REGION_KEYS = {line: tuple(f"revenue.{line}.{r}" for r in REGIONS) for line in REGIONAL_LINES}
_REGION_KEYS.update({k: tuple(f"{k}.{r}" for r in REGIONS) for k in ("revenue_total", "federal_taxes_by_residence")})


@lru_cache(maxsize=1)
def _purchase_lines() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(expenditure lines holding intermediate consumption next to wages, investment lines),
    from the ESA components in the metadata of the expenditure lines."""
    try:
        from ..expenditure import LINES as lines
    except Exception:
        return (), ()
    return (tuple(k for k, m in lines.items() if "P2" in m.esa),
            tuple(k for k, m in lines.items() if m.esa.strip() == "P51G"))


def _government_purchases(state, p, base_state) -> tuple[float, float]:
    """(current purchases, investment) of general government bearing non-deductible VAT (EUR mn):
    investment is taxed with its own rate mix (`vat.mix` key government_investment - mostly
    works and equipment at the standard rate). From the expenditure
    lines of the year: investment, plus the intermediate consumption inside the lines that mix
    wages and purchases (`vat.gov_intermediate_share` of them; of a change against the baseline,
    `vat.gov_marginal_purchases_share` - a change in volume: the baseline lines are first
    carried to the wage level of the run, so a change of wages alone, e.g. an index jump, is
    not read as fewer purchases). Falls back on a share of government consumption when the
    expenditure lines are not available."""
    exp = state.expenditure or {}
    mixed, invest = _purchase_lines()
    total = sum(float(exp.get(k, 0.0)) for k in mixed)
    if total <= 0.0:
        return float(state.macro.get("gov_consumption", 0.0)) * p["vat.gov_purchases_share"], 0.0
    reference = total
    if base_state is not None and base_state.expenditure:
        w0 = float(base_state.macro.get("wage_index", 1.0))
        ratio = float(state.macro.get("wage_index", w0)) / w0 if w0 > 0 else 1.0
        reference = sum(float(base_state.expenditure.get(k, 0.0)) for k in mixed) * ratio
    return (p["vat.gov_intermediate_share"] * reference + p["vat.gov_marginal_purchases_share"] * (total - reference),
            sum(float(exp.get(k, 0.0)) for k in invest))


def _static(state, p, pb, info: dict | None, pit: dict | None = None, base_state=None,
            paths: dict | None = None) -> tuple[dict, dict, dict]:
    """(model value of every line, lump tax expenditures to subtract after calibration,
    regional detail: line -> [region] model values, or lumps under "lump.<line>").

    p: legislation of the run; pb: legislation used to strip indirect taxes from consumption
    values (baseline); info: base-year reference levels (None while calibrating); pit: structure
    of the population of tax units (default: the one stored in `info`); paths: baseline
    adjusters of the year (`_paths`; None = neutral).
    """
    paths = paths or {}
    volume = paths.get("volume", {})
    m = state.macro
    gdp = float(m["gdp_nominal"])
    gdp0 = info["gdp0"] if info else gdp
    wealth = float(m.get("household_financial_wealth", 0.0))
    wealth0 = info["wealth0"] if info else wealth
    out = dict.fromkeys(_META, 0.0)
    regional: dict[str, np.ndarray] = {}
    for line in _GDP_LINES:
        out[line] = gdp

    # --- labour and replacement income: the household calculator over the population of tax units ---
    benefits, public_pensions = _benefits(state, p)
    increment = _minimum_pension_increment(state, p, pb, base_state)
    if increment:
        benefits = {**benefits, "pension_minimum_increment": increment}
    u = taxunits.evaluate(state, p, benefits, pit if pit is not None else info["pit"])
    for line in _UNIT_LINES:
        regional[line] = u[line] / 1e6
        out[line] = float(regional[line].sum())
    # state tax -> federal tax (after the transfer of the regional share, less the refundable
    # child credit), regional surcharges, municipal surcharges; each region at its own rates
    autonomy = p["pit.autonomy_factor"]
    surcharge, municipal = calc.by_region(p["pit.regional_surcharge"]), calc.by_region(p["pit.municipal_surcharge"])
    state_tax = u["state_tax"]
    gap = _base_gap(state, p, pb, info, gdp / gdp0 if gdp0 > 0 else 1.0) if info else None
    if gap is not None:                          # alternative tax base: the part it leaves unexplained
        state_tax = state_tax + gap * (gdp / gdp0 if gdp0 > 0 else 1.0)
    fed = (1.0 - autonomy) * state_tax / 1e6
    reg = fed * surcharge
    regional["pit_federal"] = fed - u["child_credit"] / 1e6
    regional["pit_regional"] = reg
    # Imputed contributions (D612) are matched one-for-one by imputed compensation (D122) on the
    # expenditure side, which follows the public wage bill. Tie both to the same driver so a
    # change in civil-servant pensions shows fully in the balance.
    wage_bill_public = float(u["wage_bill_public"].sum()) / 1e6
    out["ssc_imputed"] = wage_bill_public if wage_bill_public > 0.0 else public_pensions
    scale = gdp / gdp0 if gdp0 > 0 else 1.0
    regional["lump.pit_regional"] = (calc.by_region(p["pit.tax_expenditure_service_vouchers"])
                                     + calc.by_region(p["pit.tax_expenditure_mortgage"])) * scale
    if not isinstance(p["pit.tax_expenditure_service_vouchers"], dict):      # scalar form: national amounts
        regional["lump.pit_regional"] = (p["pit.tax_expenditure_service_vouchers"] + p["pit.tax_expenditure_mortgage"]) \
            * scale * calc.region_weights(p)
    # measures of the 2026 income-tax reform that the calculator does not hold, at their official
    # cost: on the federal and the regional tax (the municipal surcharges follow below)
    other = float(p.get("pit.reform_other_measures_cost", 0.0) or 0.0) * scale
    if other:
        regional["lump.pit_regional"] = regional["lump.pit_regional"] + _REFORM_REGIONAL * other * calc.region_weights(p)
    lumps = {
        "pit_federal": p["pit.tax_expenditure_pension_savings"] * scale + _REFORM_FEDERAL * other,
        "pit_regional": float(regional["lump.pit_regional"].sum()),
    }
    # municipal surcharges are levied on the federal and regional tax actually due, i.e. after
    # the tax reductions (the calibrated lines less the lump tax expenditures)
    calib = info["calib"] if info else {}
    due = (calib.get("pit_federal", 1.0) * fed - lumps["pit_federal"] * calc.region_weights(p)
           + calib.get("pit_regional", 1.0) * reg - regional["lump.pit_regional"])
    regional["pit_local"] = np.maximum(due, 0.0) * municipal
    for line in _PIT_LINES:
        out[line] = float(regional[line].sum())

    # --- regional allocation keys ---
    pop = state.pop.sum(axis=(0, 1))
    adults = state.pop[18:].sum(axis=(0, 1))
    k = (1.0 - autonomy) * (1.0 + surcharge) * (1.0 + municipal)
    disposable = u["gross"] - u["ssc_employee"] - u["ssc_self_employed"] - u["ssc_replacement_income"] \
        - u["special_ssc"] - k * u["state_tax"] + u["child_credit"]
    keys = {"population": pop, "consumption": disposable,
            "vehicles": adults * calc.by_region(p["rev.vehicle_tax_relative"])}
    alloc, factor = p["rev.regional_allocation"], p["rev.regional_factor"]
    mult = {}
    for line in _KEY_LINES:
        key = keys[alloc[line]]
        tot = float(key.sum())
        key = key / tot if tot > 0 else np.full(len(REGIONS), 1.0 / len(REGIONS))
        if line in factor:                     # regional lever on the part levied by regions / local government
            own = _OWN_SHARE[line]
            key = key * ((1.0 - own) + own * calc.by_region(factor[line]))
        regional[line] = key
        mult[line] = float(key.sum())
    for line in _GDP_LINES:
        if line in mult:
            out[line] = gdp * mult[line]

    # --- companies and capital ---
    out["cit"] = (float(m.get("corporate_profits_taxable", 0.0)) * paths.get("cit_base", 1.0)
                  * calc.cit_effective_rate(p))
    out["withholding_tax"] = float(m.get("household_capital_income", 0.0)) * calc.capital_income_tax_rate(p)
    out["securities_account_tax"] = wealth * p["cap.securities_account_tax_rate"]
    out["stock_exchange_tax"] = wealth * p["cap.tob_rate"]
    out["bank_levies"] = wealth
    growth = wealth / wealth0 if wealth0 > 0 else scale
    out["capital_gains_tax"] = p["cap.gains_tax_rate"] * p["cap.gains_taxable_gains"] * growth
    for line, base in (("registration_duties", "property_transactions"), ("inheritance_gift_tax", "inheritance_flow"),
                       ("property_tax", "cons.housing")):
        regional[line] = _PROPERTY_LINES[line](p)
        out[line] = float(m.get(base, 0.0)) * float(regional[line].sum())
    services = float(m.get("cons.other_services", 0.0))
    out["insurance_premium_tax"] = services * p["rev.insurance_premium_tax_rate"]
    out["gambling_lottery"] = services * mult["gambling_lottery"]
    out["vehicle_taxes"] = float(keys["vehicles"].sum()) * float(m.get("cpi", 1.0)) * mult["vehicle_taxes"]
    out["dividends_received"] = sum(p["rev.dividends"].values()) * scale

    # --- consumption: VAT and excise on the pre-tax value of each category ---
    # the compliance gap is not the same on supplies at the standard rate and at the reduced
    # rates; supplies that a scenario takes out of the standard rate keep it
    # (`calc.vat_collected_rate`, against the baseline rate structure `pb`)
    strip = p if m.get("cons_reflects_tax_changes") else pb
    standard = reduced = collected = 0.0

    def add_vat(key: str, base: float) -> None:
        nonlocal standard, reduced, collected
        s, r = calc.vat_rate_parts(p, key)
        standard += base * s
        reduced += base * r
        collected += base * calc.vat_collected_rate(p, key, pb)

    for c in CONS_CATS:
        value = float(m.get(f"cons.{c}", 0.0))
        pretax = value / ((1.0 + sum(calc.excise_rates(strip, c).values())) * (1.0 + calc.vat_rate(strip, c)))
        excise = calc.excise_rates(p, c)
        vol = volume.get(c, 1.0)                # taxed volume relative to real consumption
        for line, rate in excise.items():
            out[line] += pretax * rate * vol * mult.get(line, 1.0)
        add_vat(c, pretax * (1.0 + sum(excise.values())))
    purchases, public_investment = _government_purchases(state, p, base_state)
    extra = {
        "housing_investment": float(m.get("investment_housing", 0.0)),
        "government": purchases, "government_investment": public_investment,
        "exempt_sectors": gdp * p["vat.exempt_sector_input_share_gdp"],
    }
    for key in _VAT_EXTRA_BASES:
        if key not in p["vat.mix"]:              # older parameter sets: investment with the purchases
            extra["government"] += extra[key]
            continue
        add_vat(key, extra[key] / (1.0 + calc.vat_rate(strip, key)))
    out["vat"] = collected
    if info is None:                              # calibration: keep the structure for information
        regional["info.vat"] = np.array([standard, reduced])
    return out, lumps, regional


def _calibrate(state, pb) -> dict:
    """Base-year calibration: structure of the population of tax units (on the fiscal
    statistics), then factor = (actual + lump) / static model, per line."""
    actual = _load()[0]
    pit = taxunits.calibrate(state, pb, _benefits(state, pb)[0])
    model, lumps, _ = _static(state, pb, pb, None, pit)
    gdp = float(state.macro["gdp_nominal"])
    calib: dict[str, float] = {}
    fallback: list[str] = []
    for line, value in model.items():
        if line in _UNCALIBRATED:
            calib[line] = 1.0
            continue
        target = actual[line] + lumps.get(line, 0.0)
        if not np.isfinite(value) or value <= 1e-9:
            fallback.append(line)       # no usable base in the state: nominal GDP instead
            value = gdp
        calib[line] = target / value
    # municipal surcharges: on the calibrated federal and regional tax (second pass)
    local = _static(state, pb, pb, {"calib": {k: calib[k] for k in ("pit_federal", "pit_regional")}, "gdp0": gdp,
                                    "wealth0": float(state.macro.get("household_financial_wealth", 0.0)),
                                    "pit": pit})[0]["pit_local"]
    if np.isfinite(local) and local > 1e-9 and "pit_local" not in fallback:
        calib["pit_local"] = actual["pit_local"] / local
    return {
        "calib": calib, "gdp_fallback": tuple(fallback), "gdp0": gdp,
        "wealth0": float(state.macro.get("household_financial_wealth", 0.0)), "base_year": state.year,
        "pit": pit,
    }


# ---------------------------------------------------------------------------------------------
# baseline adjusters (see the module docstring)
# ---------------------------------------------------------------------------------------------
_VOLUME_CATS = tuple(c for c, comps in calc.EXCISE_COMPONENTS.items() if any(price for _, _, price in comps))


def _model_volumes(state, pb) -> dict[str, float]:
    """Volume of each excise category in the model's own units: pre-tax value / pre-tax price."""
    out = {}
    for c in _VOLUME_CATS:
        price = pb[calc.EXCISE_COMPONENTS[c][0][2]]
        value = float(state.macro.get(f"cons.{c}", 0.0))
        out[c] = value / ((1.0 + sum(calc.excise_rates(pb, c).values())) * (1.0 + calc.vat_rate(pb, c)) * price)
    return out


def _lag_factor(state, prev, p) -> float:
    """Legal indexation of the personal income tax amounts of the year over the health index
    the parameters are uprated with: consumer prices of the previous year over those of the
    year before the base year (Income Tax Code art. 178)."""
    if prev is None:
        return 1.0
    health = float(state.macro.get("health_index", 1.0))
    cpi = float(prev.macro.get("cpi", 1.0))
    base = float(p.get("pit.indexation_lag_base", 0.0) or 0.0)
    return cpi / (base * health) if base > 0 and health > 0 else 1.0


def _paths(state, prev, p, pb, info: dict, base_state, active: bool = True) -> dict:
    """Baseline adjusters of the year. In a scenario they are those of the baseline of the same
    year; in a baseline run they are built from the parameters and last year's values. `active`
    is False outside an engine run: everything neutral."""
    volumes = _model_volumes(state, pb)
    if not active:
        return {"lag": {}, "model_volumes": volumes}
    shared = (base_state.extra.get("revenue_paths") if base_state is not None else None)
    if shared is not None:
        # scenario: the baseline's adjusters; taxed volumes move with the scenario's consumption
        ratio = {c: (volumes[c] / v if v > 0 else 1.0) for c, v in shared.get("model_volumes", {}).items()}
        out = {**shared, "volume_ratio": ratio}
        # legal indexation with its one-year delay: the amounts of year t carry the deviation of
        # the health index from the baseline of year t-1, not that of year t
        health = dict(shared.get("health", {}))
        h, hb = float(state.macro.get("health_index", 1.0)), health.get(state.year, 0.0)
        own = (prev.extra.get("revenue_paths") if prev is not None else None) or {}
        run = dict(own.get("lag_run", {}))
        dev, dev_prev = (h / hb if hb > 0 else 1.0), own.get("health_dev", 1.0)
        run[state.year] = shared.get("lag", {}).get(state.year, 1.0) * (dev_prev / dev if dev > 0 else 1.0)
        out["lag_run"], out["health_dev"] = run, dev
        return out
    before = (prev.extra.get("revenue_paths") if prev is not None else None) or {}
    lag = {**before.get("lag", {}), state.year: _lag_factor(state, prev, p)}
    out = {"lag": lag, "lag_run": lag, "health_dev": 1.0, "model_volumes": volumes,
           "health": {**before.get("health", {}), state.year: float(state.macro.get("health_index", 1.0))},
           "outturn": before.get("outturn", {}), "outturn_year": before.get("outturn_year")}
    m = state.macro
    gdp, profits = float(m["gdp_nominal"]), float(m.get("corporate_profits_taxable", 0.0))
    # taxable corporate profits: share of GDP on the official path, then a constant ratio to the proxy
    path = p.get("cit.base_share_path") or {}
    index = path.get(state.year, path.get(str(state.year)))
    later = state.year > info.get("base_year", BASE_YEAR)
    if later and index is not None and profits > 0 and info.get("profits0", 0.0) > 0 and info["gdp0"] > 0:
        out["cit_base"] = float(index) * (gdp / info["gdp0"]) / (profits / info["profits0"])
    else:
        out["cit_base"] = before.get("cit_base", 1.0)
    # taxed volumes: official trend over the model's real consumption of the category
    growth = p.get("excise.volume_growth") or {}
    old, old_vol, cum = before.get("volume", {}), before.get("model_volumes", {}), before.get("volume_cum", {})
    out["volume"], out["volume_cum"] = {}, {}
    for c, v in volumes.items():
        g = float(growth.get(c, 0.0)) if later else 0.0
        if later and c in growth and old_vol.get(c, 0.0) > 0 and v > 0:
            out["volume"][c] = old.get(c, 1.0) * (1.0 + g) * old_vol[c] / v
        else:
            out["volume"][c] = old.get(c, 1.0)
        out["volume_cum"][c] = cum.get(c, 1.0) * (1.0 + g)
    return out


def _ets2(p, paths: dict) -> float:
    """Receipts of the second EU emissions trading system (road fuel and heating, from 2028),
    EUR mn: allowance price x covered emissions, which follow the taxed fuel volume."""
    price = float(p.get("rev.ets2_price", 0.0) or 0.0)
    if price <= 0.0:
        return 0.0
    volume = paths.get("volume_cum", {}).get("fuel", 1.0) * paths.get("volume_ratio", {}).get("fuel", 1.0)
    return price * float(p["rev.ets2_emissions"]) * volume


def _persisted(p, paths: dict, year: int) -> dict[str, float]:
    """line -> multiplier carrying the observed outturn: observed / model in the outturn year,
    then the part of that gap that `rev.outturn_persistence` lets persist."""
    factors, y0 = paths.get("outturn") or {}, paths.get("outturn_year")
    if not factors or y0 is None or year <= y0:
        return factors if year == y0 else {}
    keep = p.get("rev.outturn_persistence") or {}
    return {line: 1.0 + float(keep.get(line, 1.0)) * (f - 1.0) for line, f in factors.items()}


# federal = federal government and social security (Entity I)
_ALIGN_SUBSECTOR = {"federal": "federal", "social_security": "federal", "regions_communities": "regions", "local": "local"}


@lru_cache(maxsize=1)
def alignment_categories() -> dict[str, tuple[str, ...]]:
    """Categories of `rev.fpb_alignment` -> revenue lines: the categories of the Planning
    Bureau's general-government account (Tableau D.1), household and indirect taxes by
    subsector - federal government with social security, regions and communities, local
    government (Tableaux D.2, D.3, D.5, D.6; a line goes with the subsector that receives most of it)."""
    direct = _PIT_LINES + ("special_ssc", "withholding_tax", "capital_gains_tax", "securities_account_tax",
                           "vehicle_taxes", "other_taxes_income_wealth")
    out: dict[str, list[str]] = {}
    for group, lines in (("households", direct), ("indirect", [k for k, m in LINES.items() if m.esa.startswith("D2")])):
        for line in lines:
            shares = LINES[line].entity_shares
            out.setdefault(f"{group}_{_ALIGN_SUBSECTOR[max(shares, key=shares.get)]}", []).append(line)
    out["corporate"] = ["cit"]
    out["capital_taxes"] = ["inheritance_gift_tax"]
    out["contributions"] = [k for k in _META if k.startswith("ssc_") and k != "ssc_imputed"]
    return {k: tuple(v) for k, v in out.items()}


def _alignment(p) -> dict[str, float]:
    """line -> multiplier aligning the baseline on the Planning Bureau's outlook
    (`rev.fpb_alignment`, by category; empty when switched off or neutral)."""
    if not p.get("rev.fpb_alignment_enabled", 0):
        return {}
    out = {}
    for category, f in (p.get("rev.fpb_alignment") or {}).items():
        if f != 1.0:
            for line in alignment_categories().get(category, ()):
                out[line] = float(f)
    return out


def _lagged_names(pset) -> tuple[str, ...]:
    """Personal income tax amounts that the law indexes (uprated `pit.` parameters)."""
    cached = getattr(pset, "_revenue_lagged", None)
    if cached is None:
        cached = tuple(n for n, s in pset.specs.items() if n.startswith("pit.") and s.uprate != "none")
        try:
            pset._revenue_lagged = cached
        except Exception:
            pass
    return cached


def _scale(v, k: float):
    if isinstance(v, dict):
        return {a: _scale(b, k) for a, b in v.items()}
    if isinstance(v, (list, tuple)):
        return [_scale(b, k) for b in v]
    return v * k if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def legal_params(p, pset, lag: dict[int, float] | None, scenario: bool = True):
    """`p` with the personal income tax amounts on their legal indexation: the parameters are
    uprated with the health index of the year, the law uses the consumer prices of the previous
    year (`lag`: year -> ratio of the two in the baseline, `state.extra["revenue_paths"]["lag"]`). An amount
    entered for a year (a dated value, or a scenario 'set') is exact in that year and follows
    the legal indexation afterwards. scenario=False ignores the changes of `pset`'s scenario
    (baseline legislation). Returns `p` itself when there is nothing to adjust."""
    year = getattr(p, "year", None)
    if not lag or pset is None or year is None or year not in lag:
        return p
    f = lag[year]
    sets: dict[str, list] = {}
    if scenario:
        for m in pset.scenario.measures:
            for c in m.changes:
                if c.op == "set" and c.param.startswith("pit.") and c.start <= year and (c.end is None or year <= c.end):
                    sets.setdefault(c.param, []).append(c)
    over = {}
    for name in _lagged_names(pset):
        if name not in p:
            continue
        ref = max((y for y in pset.specs[name].values if y <= year), default=BASE_YEAR)
        whole = [c.start for c in sets.get(name, ()) if c.key is None]
        if whole:
            ref = max(ref, max(whole))
        k = f / lag.get(max(ref, BASE_YEAR), 1.0)
        value = p[name]
        keyed = [c for c in sets.get(name, ()) if c.key is not None and c.start >= ref]
        if abs(k - 1.0) < 1e-12 and not keyed:
            continue
        value = _scale(value, k)
        for c in keyed:                          # one element set by the scenario: exact in its year
            try:
                value[c.key] = value[c.key] / k * f / lag.get(max(c.start, BASE_YEAR), 1.0)
            except (KeyError, IndexError, TypeError):
                pass
        over[name] = value
    if not over:
        return p
    return ParamView(year, {**{k: p[k] for k in p.keys()}, **over})


_BASE_FACTOR = "pit.bracket_base_factor"


def _default_base(p):
    """`p` with the tax base of the tax returns (`pit.bracket_base_factor` neutral); `p` itself
    when it already is."""
    if taxunits._bracket_base(p) is None:
        return p
    return ParamView(getattr(p, "year", BASE_YEAR), {**{k: p[k] for k in p.keys()}, _BASE_FACTOR: None})


def _base_gap(state, p, pb, info: dict, scale: float):
    """State tax that the alternative tax base (`pit.bracket_base_factor` of the run `p`) does
    not explain, by region, EUR at base-year scale: the tax on the base of the tax returns less
    the tax on the alternative base, under the baseline legislation `pb`, measured once - in the
    first year the factor is on - and then carried as an amount that grows with nominal GDP and
    does not react to the schedule. Without other measures the receipts of that year are those
    of the default base; a change of rates is costed on the smaller base. None while the factor
    is neutral."""
    if taxunits._bracket_base(p) is None:
        return None
    if info.get("pit_base_gap") is None:
        benefits = _benefits(state, pb)[0]
        alt = ParamView(getattr(pb, "year", BASE_YEAR), {**{k: pb[k] for k in pb.keys()}, _BASE_FACTOR: p[_BASE_FACTOR]})
        full = taxunits.evaluate(state, _default_base(pb), benefits, info["pit"])["state_tax"]
        info["pit_base_gap"] = (full - taxunits.evaluate(state, alt, benefits, info["pit"])["state_tax"]) / max(scale, 1e-12)
    return info["pit_base_gap"]


def run(state, prev, p, ctx) -> None:
    """Fill `state.revenue` (line -> EUR mn) and the regional revenue keys of `state.regional`.
    Idempotent."""
    info = prev.extra.get("revenue") if prev is not None else None
    pb = getattr(ctx, "base_p", None) or p
    if info is None:
        info = _calibrate(state, _default_base(pb))
    if "profits0" not in info:
        info["profits0"] = float(state.macro.get("corporate_profits_taxable", 0.0)) if prev is None else 0.0
    state.extra["revenue"] = info

    base_state = getattr(ctx, "base_state", None)
    engine_run = hasattr(ctx, "baseline")
    paths = _paths(state, prev, p, pb, info, base_state, active=engine_run)
    pset = getattr(ctx, "params", None)
    p_legal = legal_params(p, pset, paths.get("lag_run", paths.get("lag")))
    pb = p_legal if pb is p else legal_params(pb, pset, paths.get("lag"), scenario=False)
    p = p_legal
    model, lumps, regional = _static(state, p, pb, info, base_state=base_state, paths=paths)
    gdp = float(state.macro["gdp_nominal"])
    for line in info["gdp_fallback"]:
        model[line] = gdp
    calib, lf = info["calib"], p["rev.line_factor"]
    raw = {line: float((calib[line] * model[line] - lumps.get(line, 0.0)) * lf.get(line, 1.0)) for line in _META}
    # observed outturn: fixed in the baseline run of the observed year, carried afterwards
    observed = outturn().get(state.year) if base_state is None and engine_run else None
    if observed:
        paths["outturn"] = {line: observed[line] / raw[line] for line in _META
                            if line in observed and abs(raw[line]) > 1e-6 and observed[line] / raw[line] > 0}
        paths["outturn_year"] = state.year
    state.extra["revenue_paths"] = paths
    keep = _persisted(p, paths, state.year)
    align = _alignment(p)
    factor = {line: lf.get(line, 1.0) * keep.get(line, 1.0) * align.get(line, 1.0) for line in _META}
    state.revenue = {line: raw[line] * keep.get(line, 1.0) for line in _META}
    state.revenue["environmental_taxes"] += _ets2(p, paths)
    for line, f in align.items():
        state.revenue[line] *= f
    state.extra["revenue_alignment"] = align        # what was applied (read by `align.compute`)

    # --- by region: every regional line sums to its national line ---
    rev, out = state.revenue, state.regional
    own, federal = np.zeros(len(REGIONS)), np.zeros(len(REGIONS))
    for line in REGIONAL_LINES:
        key = regional[line]
        total = float(key.sum())
        if line == "pit_regional":             # regional reductions are subtracted region by region
            values = (calib[line] * key - regional["lump.pit_regional"]) * factor.get(line, 1.0)
        elif total != 0.0:
            values = key * (rev[line] / total)
        else:
            values = np.full(len(REGIONS), rev[line] / len(REGIONS))
        share = _OWN_SHARE[line]
        own += share * values
        federal += (1.0 - share) * values
        out.update(zip(_REGION_KEYS[line], values.tolist()))
    out.update(zip(_REGION_KEYS["revenue_total"], own.tolist()))
    out.update(zip(_REGION_KEYS["federal_taxes_by_residence"], federal.tolist()))
