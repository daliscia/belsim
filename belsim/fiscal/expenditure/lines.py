"""Definition of the expenditure lines: a partition of consolidated general-government
expenditure (ESA 2010, S13) by COFOG function x ESA transaction.

Each (COFOG group, transaction) cell of Eurostat `gov_10a_exp` is assigned to the FIRST
`LineDef` whose selector matches (order matters: specific before general). A few lines are
then carved out of a parent line with an amount from another source (`SPLITS`), e.g. pension
schemes out of "old age + survivors cash benefits". `belsim.data.fetch_expenditure` applies
these definitions and writes `data/base/expenditure_lines_2024.csv`; the model only reads that.
"""
from __future__ import annotations

from dataclasses import dataclass

# ESA transactions that add up to total expenditure (TE) in gov_10a_exp.
ITEMS = ["D1", "P2", "D29_D5_D8", "D3", "D4", "D62", "D632", "D7", "D9", "OP5ANP"]
ALL = tuple(ITEMS)


@dataclass(frozen=True)
class LineDef:
    id: str
    label: str
    group: str                    # display group
    esa: str                      # COFOG group(s) + dominant ESA transaction(s), e.g. "GF01 D1+P2"
    cofog: tuple[str, ...]        # COFOG code prefixes ("GF" = everything)
    items: tuple[str, ...] = ALL  # ESA transactions
    driver: str = "generic"       # which formula in model.py computes the line
    fgroup: str = "general"       # key of pubwage.staffing_factor / exp.other_spending_factor
    description: str = ""


G_PENS, G_SOC, G_HEALTH, G_EDU = "Pensions", "Social protection", "Health", "Education"
G_GEN, G_DEF, G_ORDER, G_ECO = "General public services", "Defence", "Public order & safety", "Economic affairs"
G_SUBS, G_INV, G_INT, G_OTHER = "Subsidies to firms", "Investment & capital transfers", "Interest", "Environment, housing, culture"

# Parent lines (assigned from the COFOG matrix). Order = matching priority.
PARENTS: list[LineDef] = [
    LineDef("interest", "Interest on public debt", G_INT, "D41", ("GF",), ("D4",), "interest",
            description="Debt stock x implicit rate with a maturity structure (see model.py)."),
    LineDef("defence", "Defence", G_DEF, "GF02 D1+P2+P51G", ("GF02",), ALL, "defence", "defence",
            "All defence spending incl. equipment investment; follows defence.target_pct_gdp."),
    LineDef("public_investment", "Public investment (excl. defence)", G_INV, "P51G", ("GF",), ("OP5ANP",),
            "investment", description="Gross capital formation of general government, all functions but defence."),
    LineDef("capital_transfers", "Investment grants & other capital transfers", G_INV, "D9", ("GF",), ("D9",),
            "capital_transfers"),
    LineDef("pensions", "Old-age & survivors pensions", G_PENS, "GF1002+GF1003 D62", ("GF1002", "GF1003"),
            ("D62",), "pensions"),
    LineDef("unemployment", "Unemployment benefits", G_SOC, "GF1005 D62", ("GF1005",), ("D62",), "unemployment",
            description="Full and temporary unemployment benefits; beneficiaries x average benefit."),
    LineDef("sick_disab", "Sickness & disability cash benefits", G_SOC, "GF1001 D62", ("GF1001",), ("D62",),
            "sick_disab"),
    LineDef("long_term_care", "Long-term care (elderly & disability care in kind)", G_HEALTH, "GF1001+GF1002 D632",
            ("GF1001", "GF1002"), ("D632",), "ltc"),
    LineDef("family_allowances", "Family allowances & parental benefits", G_SOC, "GF1004 D62", ("GF1004",),
            ("D62",), "family", description="Regional competence since 2019 (Groeipakket etc.)."),
    LineDef("childcare_family_services", "Childcare & family services", G_SOC, "GF1004 D1+P2+D632", ("GF1004",), ALL,
            "children_services", "social"),
    LineDef("social_assistance", "Social assistance (leefloon / RIS and other aid)", G_SOC, "GF1007 D62",
            ("GF1007",), ("D62",), "assistance"),
    LineDef("social_inclusion_services", "Social inclusion services (CPAS/OCMW, asylum reception)", G_SOC,
            "GF1007 D1+P2+D632", ("GF1007",), ALL, "generic", "social"),
    LineDef("social_housing", "Housing benefits & social housing", G_SOC, "GF1006", ("GF1006",), ALL, "generic",
            "social"),
    LineDef("social_protection_admin", "Social protection administration & other", G_SOC, "GF10 D1+P2", ("GF10",),
            ALL, "generic", "social"),
    LineDef("health_care", "Health care (reimbursements, hospitals)", G_HEALTH, "GF07 D632+D62+D3", ("GF07",),
            ("D632", "D62", "D3"), "health"),
    LineDef("health_admin_prevention", "Health administration, prevention & research", G_HEALTH, "GF07 D1+P2",
            ("GF07",), ALL, "generic", "health"),
    LineDef("educ_primary", "Pre-primary & primary education", G_EDU, "GF0901 D1+P2", ("GF0901",), ALL, "education",
            "education"),
    LineDef("educ_secondary", "Secondary education", G_EDU, "GF0902+GF0903 D1+P2", ("GF0902", "GF0903"), ALL,
            "education", "education"),
    LineDef("educ_tertiary", "Tertiary education", G_EDU, "GF0904 D1+P2", ("GF0904",), ALL, "education", "education"),
    LineDef("educ_other", "Other education (adult, support services, grants)", G_EDU, "GF0905-GF0908 D1+P2", ("GF09",),
            ALL, "education", "education"),
    LineDef("labour_market_benefits", "Labour-market cash benefits (career break, activation)", G_SOC, "GF04 D62",
            ("GF04",), ("D62",), "labour_benefits"),
    LineDef("subsidy_wage", "Wage subsidies (targeted reductions, withholding-tax exemptions)", G_SUBS,
            "GF0401 D3", ("GF0401",), ("D3",), "subsidy_wage"),
    LineDef("subsidy_transport", "Transport subsidies (rail, bus)", G_SUBS, "GF0405 D3", ("GF0405",), ("D3",),
            "subsidy"),
    LineDef("subsidy_energy_environment", "Energy & environmental subsidies", G_SUBS, "GF0403+GF05 D3",
            ("GF0403", "GF05"), ("D3",), "subsidy"),
    LineDef("subsidy_other", "Other subsidies to firms (agriculture, R&D, other)", G_SUBS, "GF04 D3 other",
            ("GF04",), ("D3",), "subsidy"),
    LineDef("transport", "Transport (operation, excl. subsidies & investment)", G_ECO, "GF0405 D1+P2", ("GF0405",), ALL,
            "generic", "economic"),
    LineDef("economic_affairs", "Other economic affairs (employment services, agriculture, R&D)", G_ECO, "GF04 D1+P2",
            ("GF04",), ALL, "generic", "economic"),
    LineDef("development_aid", "Foreign economic aid", G_GEN, "GF0102 D7", ("GF0102",), ALL, "generic", "general"),
    LineDef("research", "Basic research", G_GEN, "GF0104+GF0105 D1+P2", ("GF0104", "GF0105"), ALL, "generic", "general"),
    LineDef("general_admin", "General administration (executive, fiscal, general services)", G_GEN, "GF01 D1+P2",
            ("GF01",), ALL, "generic", "general"),
    LineDef("police_fire", "Police & fire services", G_ORDER, "GF0301+GF0302 D1+P2", ("GF0301", "GF0302"), ALL,
            "generic", "order"),
    LineDef("justice_prisons", "Justice & prisons", G_ORDER, "GF0303-GF0306 D1+P2", ("GF03",), ALL, "generic", "order"),
    LineDef("environment", "Environmental protection (waste, water)", G_OTHER, "GF05 D1+P2", ("GF05",), ALL, "generic",
            "environment"),
    LineDef("housing_community", "Housing & community amenities", G_OTHER, "GF06 D1+P2", ("GF06",), ALL, "generic",
            "housing"),
    LineDef("culture_recreation", "Recreation, culture & religion", G_OTHER, "GF08 D1+P2", ("GF08",), ALL, "generic",
            "culture"),
]

# Lines carved out of a parent line: child id -> (parent id, LineDef). The parent keeps the remainder
# (and for 'pensions' and 'sick_disab' the parent disappears: children cover it fully).
CHILDREN: dict[str, tuple[str, LineDef]] = {
    "pension_employees": ("pensions", LineDef(
        "pension_employees", "Pensions - private-sector employees scheme", G_PENS, "GF1002+GF1003 D62", (), (),
        "pension")),
    "pension_self_employed": ("pensions", LineDef(
        "pension_self_employed", "Pensions - self-employed scheme", G_PENS, "GF1002+GF1003 D62", (), (), "pension")),
    "pension_civil_servants": ("pensions", LineDef(
        "pension_civil_servants", "Pensions - civil servants", G_PENS, "GF1002+GF1003 D62", (), (), "pension")),
    "pension_grapa": ("pensions", LineDef(
        "pension_grapa", "Income guarantee for the elderly (GRAPA/IGO)", G_PENS, "GF1002 D62 means-tested", (), (),
        "grapa")),
    "invalidity": ("sick_disab", LineDef(
        "invalidity", "Invalidity & long-term incapacity benefits", G_SOC, "GF1001 D62", (), (), "invalidity")),
    "disability_allowances": ("sick_disab", LineDef(
        "disability_allowances", "Allowances for persons with disabilities (means-tested)", G_SOC, "GF1001 D62",
        (), (), "disab_allowance")),
    "sickness_primary": ("sick_disab", LineDef(
        "sickness_primary", "Short-term sickness, work accidents & occupational diseases", G_SOC, "GF1001 D62",
        (), (), "sickness")),
    "rcc_swt": ("unemployment", LineDef(
        "rcc_swt", "Unemployment with company supplement (RCC/SWT)", G_SOC, "GF1005 D62", (), (), "rcc")),
    "subsidy_service_vouchers": ("subsidy_wage", LineDef(
        "subsidy_service_vouchers", "Service vouchers (titres-services / dienstencheques)", G_SUBS, "GF0401 D3",
        (), (), "service_vouchers")),
    "eu_contribution": ("general_admin", LineDef(
        "eu_contribution", "EU budget contribution (VAT- and GNI-based own resources)", G_GEN, "D76", (), (),
        "eu")),
}
VANISHING_PARENTS = ("pensions", "sick_disab")


def all_defs() -> dict[str, LineDef]:
    """Final lines, in display order."""
    out: dict[str, LineDef] = {}
    for d in PARENTS:
        if d.id in VANISHING_PARENTS:
            for cid, (par, cd) in CHILDREN.items():
                if par == d.id:
                    out[cid] = cd
            continue
        out[d.id] = d
        for cid, (par, cd) in CHILDREN.items():
            if par == d.id:
                out[cid] = cd
    return out


def assign(cofog: str, item: str) -> str | None:
    """Parent line id of one (4-digit COFOG group, ESA transaction) cell."""
    for d in PARENTS:
        if item in d.items and any(cofog.startswith(c) for c in d.cofog):
            return d.id
    return None
