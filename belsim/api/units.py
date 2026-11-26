"""Display metadata for series keys and dimensions: unit kinds and human labels.

Unit kinds (the UI formats and converts by kind, never by key name):
  eur_mn       EUR million, current prices  -> switchable: EUR bn / EUR bn 2024 prices / % of GDP
  eur_mn_real  EUR million, already in base-year prices (no further deflating)
  eur          EUR per person per year
  persons      headcount
  rate         fraction (0.21) shown as percent
  pct          already a percentage (e.g. % of GDP)
  index        index, 1.0 in the base year
  years        years (ages, life expectancy)
  number       anything else

`regional.<metric>.<REG>` keys follow the same conventions as the national ones: rates and shares
are fractions, `*_pct_gdp` are percent, headcounts are persons, `*_per_capita` and averages are
EUR per person, everything else is an amount in EUR million.
"""
from __future__ import annotations

import re

from ..core.dims import CONS_CATS, ENTITIES, JOBTYPES, REGIONS, SECTORS, STATUSES

# First matching rule wins.
_RULES: list[tuple[str, str]] = [
    (r".*_pct_gdp(\..*)?$", "pct"),
    (r"^(revenue|expenditure)\..*", "eur_mn"),
    (r"^fiscal\.(net_expenditure_growth)$", "rate"),
    (r"^fiscal\..*", "eur_mn"),
    (r"^va\..*", "eur_mn"),
    (r"^macro\.gdp_real$", "eur_mn_real"),
    (r"^macro\.gdp_per_.*", "eur"),
    (r"^macro\.(cpi_tax_effect|cpi_wage_cost_effect|cpi_demand_effect|output_gap)$", "rate"),
    (r"^macro\.(real_unit_labour_cost|tax_threshold_drift)$", "index"),
    (r"^macro\.(inflation|interest_rate.*|.*_growth|demand_gap|.*_rate|.*_share)$", "rate"),
    (r"^macro\.(cpi|health_index|wage_index|gdp_deflator|.*_index)$", "index"),
    (r"^macro\..*", "eur_mn"),
    (r"^labour\.(wage_bill.*|self_employed_income)$", "eur_mn"),
    (r"^labour\..*_factor(_.*)?$", "number"),
    (r"^labour\.(avg_.*|.*wage.*)$", "eur"),
    (r"^labour\.(.*_rate.*|.*_share)$", "rate"),
    (r"^labour\.hours.*", "number"),
    (r"^labour\..*", "persons"),
    (r"^demo\.life_expectancy.*", "years"),
    (r"^demo\.(tfr|old_age_dependency|.*_ratio|.*_rate)(\..*)?$", "number"),
    (r"^demo\..*", "persons"),
    (r"^(pop|status|emp)\..*", "persons"),
    # regional.<metric>.<REG> (and regional.revenue.<line>.<REG>)
    (r"^regional\.(revenue|expenditure)\..*", "eur_mn"),
    (r"^regional\.[^.]*(_rate|_share|_ratio_gdp|_growth|inflation)(_\d+_\d+|_\d+_plus)?(\..*)?$", "rate"),
    (r"^regional\.[^.]*(_index|cpi)(\..*)?$", "index"),
    (r"^regional\.(life_expectancy|.*_age)(\..*)?$", "years"),
    (r"^regional\.[^.]*(tfr|dependency|_ratio|_factor)(\..*)?$", "number"),
    (r"^regional\.[^.]*(_per_capita|_per_head|_per_person|_per_inhabitant|_per_worker|_per_household|"
     r"avg_|average_|mean_|median_|_wage($|\.|_))", "eur"),
    (r"^regional\.[^.]*gdp_real(\..*)?$", "eur_mn_real"),
    (r"^regional\.(productivity|[^.]*_per_employee)(\..*)?$", "eur"),
    (r"^regional\.[^.]*(_population|_employees|_recipients|net_commuting)(\..*)?$", "persons"),
    (r"^regional\.(population|pop_|employment|employed|employees|unemployed|unemployment|labour_force|"
     r"active|inactive|self_employed|workers|jobs|persons|headcount|commuters|commuting|births|deaths|"
     r"net_migration|migration|students|pupils|pensioners|retired|disabled|beneficiaries|households)"
     r"[^.]*(\..*)?$", "persons"),
    (r"^regional\..*", "eur_mn"),
]
_COMPILED = [(re.compile(p), k) for p, k in _RULES]


_METRIC_UNITS = {"persons": "persons", "fraction": "rate", "rate": "rate", "EUR mn": "eur_mn", "EUR million": "eur_mn",
                 "EUR": "eur", "EUR/year": "eur", "years": "years", "index": "index"}


def _regional_metric_kind(key: str) -> str | None:
    """Unit kind of `regional.<metric>.<REG>` from the regional module's own METRICS table
    ({metric: (label, unit)}), when it has one for that metric."""
    parts = key.split(".")
    if parts[0] != "regional" or len(parts) != 3:
        return None
    try:
        from ..regional import METRICS
        unit = METRICS[parts[1]][1]
    except Exception:
        return None
    if str(unit).startswith("%"):
        return "pct"
    if "gdp_real" in parts[1] and unit == "EUR mn":
        return "eur_mn_real"
    return _METRIC_UNITS.get(unit)


def unit_kind(key: str) -> str:
    if key.startswith("regional."):
        kind = _regional_metric_kind(key)
        if kind:
            return kind
    for rx, kind in _COMPILED:
        if rx.match(key):
            return kind
    return "number"


SECTOR_LABELS = {
    "A": "Agriculture, forestry, fishing",
    "BDE": "Mining, energy, water & waste",
    "C": "Manufacturing",
    "F": "Construction",
    "G": "Wholesale & retail trade",
    "H": "Transport & storage",
    "I": "Accommodation & food services",
    "J": "Information & communication",
    "K": "Finance & insurance",
    "L": "Real estate",
    "M": "Professional, scientific, technical",
    "N": "Administrative & support services",
    "O": "Public administration & defence",
    "P": "Education",
    "Q": "Health & social work",
    "RSTU": "Arts, other services, households",
}
STATUS_LABELS = {
    "child": "Children (0-14)",
    "student": "Students",
    "emp_private": "Private-sector employees",
    "emp_public": "Public-sector employees",
    "self_employed": "Self-employed",
    "unemployed": "Unemployed",
    "disabled": "Long-term sick / disabled",
    "retired": "Retired",
    "inactive": "Other inactive",
}
JOBTYPE_LABELS = {
    "emp_private": "Private employees",
    "emp_public": "Public employees",
    "self_employed": "Self-employed",
}
ENTITY_LABELS = {
    "federal": "Federal government",
    "social_security": "Social security",
    "regions_communities": "Regions & communities",
    "local": "Local authorities",
}
REGION_LABELS = {"FLA": "Flanders", "WAL": "Wallonia", "BRU": "Brussels"}
CONS_LABELS = {
    "food": "Food & non-alcoholic drinks",
    "alcohol": "Alcohol",
    "tobacco": "Tobacco",
    "fuel": "Motor fuels",
    "energy_home": "Home energy",
    "housing": "Housing (rents)",
    "restaurants": "Restaurants & hotels",
    "renovation": "Housing maintenance",
    "other_goods": "Other goods",
    "other_services": "Other services",
}


def _humanise(code: str) -> str:
    return code.replace("_", " ").strip().capitalize()


def dim_labels(lang: str | None = None) -> dict[str, dict[str, str]]:
    """Labels for every dimension member; falls back to a humanised code if dims.py grows.
    With `lang`, translated through `belsim.i18n` (English where a translation is missing)."""
    if lang and lang != "en":
        from ..i18n import translate
        return {dim: {code: translate(f"dim.{dim}", code, lang, label) for code, label in table.items()}
                for dim, table in dim_labels().items()}

    def lab(codes, table):
        return {c: table.get(c, _humanise(c)) for c in codes}
    return {
        "sectors": lab(SECTORS, SECTOR_LABELS),
        "statuses": lab(STATUSES, STATUS_LABELS),
        "jobtypes": lab(JOBTYPES, JOBTYPE_LABELS),
        "entities": lab(ENTITIES, ENTITY_LABELS),
        "regions": lab(REGIONS, REGION_LABELS),
        "cons_cats": lab(CONS_CATS, CONS_LABELS),
        "sexes": {"M": "Men", "F": "Women"},
    }


UNITS_DOC = {
    "eur_mn": "EUR million, current prices",
    "eur_mn_real": "EUR million, 2024 prices",
    "eur": "EUR per person per year",
    "persons": "persons",
    "rate": "fraction (0.21 = 21%)",
    "pct": "percent",
    "index": "index, 2024 = 1.0",
    "years": "years",
    "number": "number",
}
