"""Household types covering the Belgian population, with population weights.

The set is a small synthetic population: each household type (single worker, single parent,
one- and two-earner couples with no, one, two or three and more children, households with three
adults, pensioners, people out of work, students) is repeated over income groups taken from the population of tax units behind the
aggregate income tax (`belsim.fiscal.revenue.taxunits`), so the households and the budget share
one income distribution:

  employees       six groups of the earnings distribution of all employees (`dist.wage_level_shares`:
                  bottom fifth ... top 3 %); a second earner is in one of three groups
                  (`PARTNER_GROUPS`), independently of the first - also in the households of
                  three adults (two earners and a third adult: a pensioner, a worker of the
                  lowest group, someone on an unemployment or invalidity benefit, or an adult
                  without income);
                  the lowest group is less often the only earner of a household than the others
                  (`dist.sole_earner_low_income_ratio`, set on the observed in-work poverty);
  self-employed   four groups of the income distribution of the self-employed
                  (`dist.self_employed_level_shares`), as main earner;
  pensioners      minimum pension, average employee pension, average civil-servant pension, and
                  the top of the pension distribution (`dist.pensioner_high_share`).
Gross incomes are the model's (wages of the national accounts, benefits of the budget); the
income actually taxed is that of the same group of taxpayers in the fiscal statistics
(`ratio`: taxable income over what the payslip rules give - wages hold untaxed components, a
year's income is spread over more taxpayers than there are jobs at one moment, see taxunits).

Weights (number of private households) are built on the population of the income survey the
set is compared with (`silc_frame`, EU-SILC 2025, data/base/distribution_reference.csv): persons
by household type - one adult, two adults, three or more, with and without dependent children,
one or two and three or more children, under and over 65 - and, among them, the people under 65
living in a household with very low work intensity (the households without work of the set).
The EU-LFS household statistics 2024 (data/base/distribution_households.csv) give what the
survey tables do not: one or two earners among the couples in work, the number of children of
working single parents, students living alone. The income groups come on top. Households
without work (singles, couples, families with children, three adults) are split between
unemployment benefit, invalidity benefit and integration income like the people out of work of
the model; the unemployed and long-term sick who do not fit there live with workers (partner of
a one-earner couple, third adult). Where the budget pays a benefit to a share only of the
unemployed (`unemp.coverage_rate` below 1, from 2025), that share of the unemployed households
keeps the benefit per recipient and the others live on the integration income
(`UNCOVERED`). For an analysis the weights are then rescaled
to the head counts of the baseline simulation of the year analysed (`consistent_weights`:
workers, pensioners, unemployed, long-term sick), so the households add up to the population
behind the budget; the same weights serve baseline and scenario: the analysis shows what a
scenario does to given households, not how many people change status.

The households flagged `representative` are the short list shown by default (one typical case
per type, with the identifiers of the former stylised set).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dims import J, REGIONS, S
from ..fiscal.expenditure.calc import replacement_income
from ..fiscal.revenue import calc as rcalc
from ..fiscal.revenue import taxunits
from .calc import BenefitEnv

DATA = Path(__file__).resolve().parents[2] / "data" / "base"
WAGE_LEVELS = ("min_wage", "p25", "median", "p75", "p90", "top")
SELF_LEVELS = ("self_low", "self_median", "self_high", "self_top")
# second earner of a couple: groups of employees made of two wage levels each
RATIO_MIN = 0.2      # lower bound of the ratio taxable income in the tax returns / payslip rules
PARTNER_GROUPS = {"partner_low": ("min_wage", "p25"), "partner_mid": ("median", "p75"), "partner_high": ("p90", "top")}


@dataclass(frozen=True)
class Adult:
    kind: str            # income kind of belsim.distribution.calc (emp_private, pension, assistance, none ...)
    level: str = ""      # income group, or the variant of the benefit


@dataclass(frozen=True)
class Household:
    id: str
    label: str
    group: str
    adults: tuple[Adult, ...]
    children: int = 0
    weight: float = 0.0  # number of households
    representative: bool = True
    base: str = ""       # household type
    share: float = 1.0   # share of the type in this income group
    mix: tuple = ()      # composition of a `third` adult: ((kind, level, share of the households), ...)
    benefit_scale: float = 1.0   # benefit of the main adult (a group of `BENEFIT_GROUPS`) over spending per recipient
    cohab_scale: tuple = ()      # ((kind, benefit at the cohabitant rate over spending per recipient), ...)

    @property
    def persons(self) -> int:
        return len(self.adults) + self.children


def _emp(level):
    return Adult("emp_private", level)


NONE = Adult("none")
PENSIONER = Adult("pension", "average")
# third adult of the households of three adults: a pensioner, a person in work (in the lowest
# income group of employees: part-year and small part-time earnings, typical of young adults in
# their parents' home), a person on an unemployment or invalidity benefit (cohabitant rate) or
# an adult without income, in the shares of `third_adult_mix`; the household carries the average
# of the cases
THIRD = Adult("third", "mix")
THIRD_STATUS = (("pension", "average"), ("unemployment", "cohab"), ("disability", "cohab"), ("none", ""))
THIRD_WORK = ("emp_private", "min_wage")
# household types whose main earner is the only earner
SOLE_TYPES = ("single", "single_parent_1", "single_parent_2", "single_parent_3", "couple_1earner", "couple_1earner_1",
              "couple_1earner_2", "couple_1earner_3", "couple_worker_pensioner")
_LEVEL_LABELS = {"min_wage": "lowest fifth of earnings", "p25": "low wage (P25)", "median": "median wage",
                 "p75": "wage at P75", "p90": "wage at P90", "top": "top wage (top 3%)",
                 "self_low": "low income", "self_median": "median income", "self_high": "high income",
                 "self_top": "top income (top 3%)",
                 "partner_low": "low wage", "partner_mid": "middle wage", "partner_high": "high wage"}
_KIDS = {0: "no children", 1: "1 child", 2: "2 children", 3: "3 or more children"}

# household types with work: base -> (label of the type, group, children, kind of second adult, extra adults,
# representative (main level, partner level), identifier of the representative)
_WORK_TYPES = {
    "single": ("Single worker", "Single workers", 0, None, (), ("median", None)),
    "single_parent_1": ("Single parent", "Working families", 1, None, (), ("p25", None)),
    "single_parent_2": ("Single parent", "Working families", 2, None, (), ("median", None)),
    "single_parent_3": ("Single parent", "Working families", 3, None, (), (None, None)),
    "couple_1earner": ("Couple, one earner", "Working families", 0, "none", (), ("median", None)),
    "couple_1earner_1": ("Couple, one earner", "Working families", 1, "none", (), (None, None)),
    "couple_1earner_2": ("Couple, one earner", "Working families", 2, "none", (), ("median", None)),
    "couple_worker_pensioner": ("Couple, one earner and a pensioner", "Working families", 0, "pension", (), (None, None)),
    "couple_2earners": ("Couple, two earners", "Working families", 0, "work", (), ("median", "partner_low")),
    "couple_2earners_1": ("Couple, two earners", "Working families", 1, "work", (), (None, None)),
    "couple_2earners_2": ("Couple, two earners", "Working families", 2, "work", (), ("median", "partner_low")),
    "couple_1earner_3": ("Couple, one earner", "Working families", 3, "none", (), (None, None)),
    "couple_2earners_3": ("Couple, two earners", "Working families", 3, "work", (), (None, None)),
    "three_adults": ("Three adults, two earners and a third adult", "Working families", 0, "work", (THIRD,), ("median", "partner_low")),
    "three_adults_1": ("Three adults, two earners and a third adult", "Working families", 1, "work", (THIRD,), ("median", "partner_low")),
}
SELF_TYPES = ("single", "couple_1earner", "couple_1earner_1", "couple_1earner_2", "couple_1earner_3", "couple_2earners",
              "couple_2earners_1", "couple_2earners_2", "couple_2earners_3")
# recipients of unemployment and invalidity benefits living alone: three groups of the benefit
# distribution of the tax units (shares of the recipients)
BENEFIT_GROUPS = {"low": 0.40, "mid": 0.40, "high": 0.20}
# partner without work in a one-earner couple: no income, or a benefit at the cohabitant rate
PARTNER_STATUS = ("none", "unemployment", "disability")
_STATUS_LABELS = {"none": "partner without income", "unemployment": "partner on unemployment benefit",
                  "disability": "partner on invalidity benefit"}
# main adult of an unemployed household without unemployment benefit (level of the definitions)
UNCOVERED = "uncovered"
# other identifiers of the former stylised set kept on the closest household
_LEGACY = {("single", "self_median", None): "single_self_employed",
           ("couple_2earners_2", "p90", "partner_mid"): "couple_2earners_2_high"}


def _work_definitions() -> list[tuple]:
    """(id, label, group, adults, children, representative, base, main level, partner level)."""
    out = []
    for base, (label, group, children, second, extra, rep) in _WORK_TYPES.items():
        mains = WAGE_LEVELS + (SELF_LEVELS if base in SELF_TYPES else ())
        for main in mains:
            self_employed = main in SELF_LEVELS
            first = Adult("self_employed", main) if self_employed else _emp(main)
            if second == "work":
                partners = ("partner_mid",) if self_employed else tuple(PARTNER_GROUPS)
            elif second == "same":
                partners = (main,)
            elif second == "none" and not self_employed:
                partners = PARTNER_STATUS
            elif second == "none":
                partners = ("none",)
            else:
                partners = (None,)
            for partner in partners:
                if second == "none":
                    mate = (NONE,) if partner == "none" else (Adult(partner, "cohab"),)
                elif second == "pension":
                    mate = (PENSIONER,)
                else:
                    mate = (_emp(partner),) if partner else ()
                adults = (first,) + mate + extra
                is_rep = (main, partner if second == "work" else None) == rep and partner in (None, "none", main, rep[1])
                if base == "single":
                    is_rep = not self_employed or main == "self_median"
                ident = base if is_rep and base != "single" else \
                    f"{base}__{main}" + (f"__{partner}" if second == "work" or (second == "none" and partner != "none") else "")
                if base == "single" and not self_employed:
                    ident = f"single_{main}"
                legacy = _LEGACY.get((base, main, partner if second == "work" else None))
                if legacy:
                    ident, is_rep = legacy, True
                who = ("self-employed, " if self_employed else "") + _LEVEL_LABELS[main]
                if second == "work":
                    who += " + " + _LEVEL_LABELS[partner]
                elif second == "none":
                    who += ", " + _STATUS_LABELS[partner]
                text = f"{label} ({who}), {_KIDS[children]}" if base != "single" else \
                    (f"Single self-employed, {_LEVEL_LABELS[main]}" if self_employed else f"Single worker, {_LEVEL_LABELS[main]}")
                out.append((ident, text, group, adults, children, is_rep, base, main,
                            partner if second == "work" or (second == "none" and not self_employed) else None))
    return out


# id, label, group, adults, children, representative, base, main level, partner level
ALL_DEFINITIONS: list[tuple] = _work_definitions() + [
    ("unemployed_single__low", "Unemployed single, low benefit", "Out of work", (Adult("unemployment", "low"),), 0, False, "unemployed_single", "low", None),
    ("unemployed_single", "Unemployed single, middle benefit", "Out of work", (Adult("unemployment", "mid"),), 0, True, "unemployed_single", "mid", None),
    ("unemployed_single__high", "Unemployed single, high benefit", "Out of work", (Adult("unemployment", "high"),), 0, False, "unemployed_single", "high", None),
    ("sick_single__low", "Long-term sick single, low invalidity benefit", "Out of work", (Adult("disability", "low"),), 0, False, "sick_single", "low", None),
    ("sick_single", "Long-term sick single, middle invalidity benefit", "Out of work", (Adult("disability", "mid"),), 0, True, "sick_single", "mid", None),
    ("sick_single__high", "Long-term sick single, high invalidity benefit", "Out of work", (Adult("disability", "high"),), 0, False, "sick_single", "high", None),
    ("assistance_single", "Single on integration income", "Out of work", (Adult("assistance", "single"),), 0, True, "assistance_single", None, None),
    ("assistance_family", "Single parent on integration income, 2 children", "Out of work", (Adult("assistance", "family"),), 2, True, "assistance_family", None, None),
    ("unemployed_family", "Unemployed single parent, 2 children", "Out of work", (Adult("unemployment", "mid"),), 2, False, "unemployed_family", "mid", None),
    ("sick_family", "Long-term sick single parent, 2 children", "Out of work", (Adult("disability", "mid"),), 2, False, "sick_family", "mid", None),
    ("unemployed_couple", "Couple out of work, two unemployment benefits", "Out of work", (Adult("unemployment", "mid"), Adult("unemployment", "cohab")), 0, False, "unemployed_couple", "two", None),
    ("unemployed_couple__one", "Couple out of work, one unemployment benefit", "Out of work", (Adult("unemployment", "mid"), NONE), 0, False, "unemployed_couple", "one", None),
    ("sick_couple", "Couple out of work, two invalidity benefits", "Out of work", (Adult("disability", "mid"), Adult("disability", "cohab")), 0, False, "sick_couple", "two", None),
    ("sick_couple__one", "Couple out of work, one invalidity benefit", "Out of work", (Adult("disability", "mid"), NONE), 0, False, "sick_couple", "one", None),
    ("assistance_couple", "Couple on integration income", "Out of work", (Adult("assistance", "family"), NONE), 0, False, "assistance_couple", None, None),
    ("unemployed_couple_2", "Couple out of work, two unemployment benefits, 2 children", "Out of work", (Adult("unemployment", "mid"), Adult("unemployment", "cohab")), 2, False, "unemployed_couple_2", "two", None),
    ("unemployed_couple_2__one", "Couple out of work, one unemployment benefit, 2 children", "Out of work", (Adult("unemployment", "mid"), NONE), 2, False, "unemployed_couple_2", "one", None),
    ("sick_couple_2", "Couple out of work, two invalidity benefits, 2 children", "Out of work", (Adult("disability", "mid"), Adult("disability", "cohab")), 2, False, "sick_couple_2", "two", None),
    ("sick_couple_2__one", "Couple out of work, one invalidity benefit, 2 children", "Out of work", (Adult("disability", "mid"), NONE), 2, False, "sick_couple_2", "one", None),
    ("assistance_couple_2", "Couple on integration income, 2 children", "Out of work", (Adult("assistance", "family"), NONE), 2, False, "assistance_couple_2", None, None),
    ("unemployed_couple_3", "Couple out of work, two unemployment benefits, 3 or more children", "Out of work", (Adult("unemployment", "mid"), Adult("unemployment", "cohab")), 3, False, "unemployed_couple_3", "two", None),
    ("unemployed_couple_3__one", "Couple out of work, one unemployment benefit, 3 or more children", "Out of work", (Adult("unemployment", "mid"), NONE), 3, False, "unemployed_couple_3", "one", None),
    ("sick_couple_3", "Couple out of work, two invalidity benefits, 3 or more children", "Out of work", (Adult("disability", "mid"), Adult("disability", "cohab")), 3, False, "sick_couple_3", "two", None),
    ("sick_couple_3__one", "Couple out of work, one invalidity benefit, 3 or more children", "Out of work", (Adult("disability", "mid"), NONE), 3, False, "sick_couple_3", "one", None),
    ("assistance_couple_3", "Couple on integration income, 3 or more children", "Out of work", (Adult("assistance", "family"), NONE), 3, False, "assistance_couple_3", None, None),
    ("unemployed_couple_1", "Couple out of work, two unemployment benefits, 1 child", "Out of work", (Adult("unemployment", "mid"), Adult("unemployment", "cohab")), 1, False, "unemployed_couple_1", "two", None),
    ("unemployed_couple_1__one", "Couple out of work, one unemployment benefit, 1 child", "Out of work", (Adult("unemployment", "mid"), NONE), 1, False, "unemployed_couple_1", "one", None),
    ("sick_couple_1", "Couple out of work, two invalidity benefits, 1 child", "Out of work", (Adult("disability", "mid"), Adult("disability", "cohab")), 1, False, "sick_couple_1", "two", None),
    ("sick_couple_1__one", "Couple out of work, one invalidity benefit, 1 child", "Out of work", (Adult("disability", "mid"), NONE), 1, False, "sick_couple_1", "one", None),
    ("assistance_couple_1", "Couple on integration income, 1 child", "Out of work", (Adult("assistance", "family"), NONE), 1, False, "assistance_couple_1", None, None),
    ("unemployed_three", "Three adults out of work, two unemployment benefits", "Out of work", (Adult("unemployment", "mid"), Adult("unemployment", "cohab"), NONE), 0, False, "unemployed_three", "two", None),
    ("unemployed_three__one", "Three adults out of work, one unemployment benefit", "Out of work", (Adult("unemployment", "mid"), NONE, NONE), 0, False, "unemployed_three", "one", None),
    ("sick_three", "Three adults out of work, two invalidity benefits", "Out of work", (Adult("disability", "mid"), Adult("disability", "cohab"), NONE), 0, False, "sick_three", "two", None),
    ("sick_three__one", "Three adults out of work, one invalidity benefit", "Out of work", (Adult("disability", "mid"), NONE, NONE), 0, False, "sick_three", "one", None),
    ("assistance_three", "Three adults on integration income", "Out of work", (Adult("assistance", "family"), NONE, NONE), 0, False, "assistance_three", None, None),
    # unemployed people without a benefit (`unemp.coverage_rate` below 1): on the integration income
    ("unemployed_single__no_benefit", "Unemployed single without unemployment benefit, on integration income", "Out of work", (Adult("unemployment", "uncovered_single"),), 0, False, "unemployed_single", UNCOVERED, None),
    ("unemployed_family__no_benefit", "Unemployed single parent without unemployment benefit, on integration income, 2 children", "Out of work", (Adult("unemployment", "uncovered_family"),), 2, False, "unemployed_family", UNCOVERED, None),
    ("unemployed_couple__no_benefit", "Couple out of work without unemployment benefit, on integration income", "Out of work", (Adult("unemployment", "uncovered_family"), NONE), 0, False, "unemployed_couple", UNCOVERED, None),
    ("unemployed_couple_2__no_benefit", "Couple out of work without unemployment benefit, on integration income, 2 children", "Out of work", (Adult("unemployment", "uncovered_family"), NONE), 2, False, "unemployed_couple_2", UNCOVERED, None),
    ("unemployed_couple_3__no_benefit", "Couple out of work without unemployment benefit, on integration income, 3 or more children", "Out of work", (Adult("unemployment", "uncovered_family"), NONE), 3, False, "unemployed_couple_3", UNCOVERED, None),
    ("unemployed_couple_1__no_benefit", "Couple out of work without unemployment benefit, on integration income, 1 child", "Out of work", (Adult("unemployment", "uncovered_family"), NONE), 1, False, "unemployed_couple_1", UNCOVERED, None),
    ("unemployed_three__no_benefit", "Three adults out of work without unemployment benefit, on integration income", "Out of work", (Adult("unemployment", "uncovered_family"), NONE, NONE), 0, False, "unemployed_three", UNCOVERED, None),
    ("pensioner_single_low", "Single pensioner, minimum pension", "Pensioners", (Adult("pension", "low"),), 0, True, "pensioner_single", "low", None),
    ("pensioner_single_avg", "Single pensioner, average employee pension", "Pensioners", (PENSIONER,), 0, True, "pensioner_single", "average", None),
    ("pensioner_single_civil", "Single pensioner, average civil-servant pension", "Pensioners", (Adult("pension", "civil"),), 0, True, "pensioner_single", "civil", None),
    ("pensioner_single_high", "Single pensioner, high pension (top of the distribution)", "Pensioners", (Adult("pension", "high"),), 0, False, "pensioner_single", "high", None),
    ("pensioner_couple_low", "Pensioner couple, one minimum pension at the household rate", "Pensioners", (Adult("pension", "household_low"), NONE), 0, True, "pensioner_couple", "low", None),
    ("pensioner_couple_avg", "Pensioner couple, average + minimum pension", "Pensioners", (PENSIONER, Adult("pension", "low")), 0, True, "pensioner_couple", "average", None),
    ("pensioner_couple_two_avg", "Pensioner couple, two average pensions", "Pensioners", (PENSIONER, PENSIONER), 0, False, "pensioner_couple", "two_average", None),
    ("pensioner_couple_high", "Pensioner couple, high + average pension", "Pensioners", (Adult("pension", "high"), PENSIONER), 0, False, "pensioner_couple", "high", None),
    ("pensioner_couple_civil", "Pensioner couple, civil-servant + minimum pension", "Pensioners", (Adult("pension", "civil"), Adult("pension", "low")), 0, False, "pensioner_couple", "civil", None),
    ("student", "Student living alone with a student job", "Students", (Adult("student", "student"),), 0, True, "student", None, None),
]

# the representative households (short list; their labels are the ones translated)
DEFINITIONS: list[tuple] = [d for d in ALL_DEFINITIONS if d[5]]


# ---------------------------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cells() -> dict:
    df = pd.read_csv(DATA / "distribution_households.csv", dtype={"n_child": str})
    return {(r.hhwkstat, r.n_child, r.phhcomp): float(r.households_ths) * 1000.0 for r in df.itertuples()}


@lru_cache(maxsize=1)
def _reference() -> dict[str, float]:
    df = pd.read_csv(DATA / "distribution_reference.csv")
    return dict(zip(df["indicator"], df["value"].astype(float)))


# household types of the income survey (Eurostat codes): one adult under / over 65, single
# parent, two adults both under 65 / at least one over 65, two adults with 1, 2, 3 or more
# dependent children, three or more adults without / with dependent children
SILC_TYPES = ("A1_LT65", "A1_GE65", "A1_DCH", "A2_2LT65", "A2_GE1_GE65", "A2_DCH1", "A2_DCH2", "A2_DCH_GE3",
              "A_GE3", "A_GE3_DCH")


@lru_cache(maxsize=1)
def silc_frame() -> dict:
    """The population of the income survey by household type (EU-SILC 2025, incomes of 2024,
    data/base/distribution_reference.csv): `persons` (type -> persons in private households:
    share of the population of ilc_lvps02 x the population behind the published poverty count,
    people at risk of poverty / rate, ilc_li03) and `low_work` (type -> persons under 65 living
    in a household with very low work intensity, ilc_lvhl13n; their share of the people under
    65 of the type is `low_work_intensity_rate.household.<type>`)."""
    ref = _reference()
    total = 1000.0 * ref["poverty_persons_ths.household.TOTAL"] / (0.01 * ref["poverty_rate.household.TOTAL"])
    shares = {k: ref[f"population_share.household.{k}"] for k in SILC_TYPES}
    scale = total / sum(shares.values())                        # the published shares are rounded
    return {"total": total, "persons": {k: v * scale for k, v in shares.items()},
            "low_work": {k: 1000.0 * ref.get(f"low_work_intensity_ths.household.{k}", 0.0) for k in SILC_TYPES}}


# households without work by composition: pool -> (adults, children)
JOBLESS_POOLS = {"single": (1, 0), "family": (1, 2), "couple": (2, 0), "couple_1": (2, 1), "couple_2": (2, 2), "couple_3": (2, 3),
                 "three": (3, 0)}
JOBLESS_KINDS = ("unemployed", "sick", "assistance")


def base_weights(p, state0) -> dict[str, float]:
    """Number of households of each type (before the split into income groups). `state0` is the
    model's base-year state (shares of the unemployed, the long-term sick and assistance
    recipients among working-age households without work).

    Persons by household type and the people in households without work come from the income
    survey (`silc_frame`); a type of the set has a fixed size, so its number of households is
    persons / size (families with three or more children are a type with three; three adults
    stand for three or more). The household
    statistics (EU-LFS, `_cells`) split the couples in work into one and two earners and the
    working single parents by number of children, and count the students living alone."""
    c = _cells()
    f = silc_frame()
    persons, low = f["persons"], f["low_work"]

    def cell(work, comp, n="TOTAL"):
        return c.get((work, n, comp), 0.0)

    def split(total, *parts):
        tot = sum(parts)
        return [total * x / tot if tot > 0 else total / len(parts) for x in parts]

    w: dict[str, float] = {}
    # households without work (people under 65 in a household with very low work intensity)
    pools = {"single": low["A1_LT65"], "family": low["A1_DCH"] / 3.0, "couple": low["A2_2LT65"] / 2.0,
             "couple_1": low["A2_DCH1"] / 3.0, "couple_2": (low["A2_DCH2"] + low["A_GE3_DCH"]) / 4.0,
             "couple_3": low["A2_DCH_GE3"] / 5.0,
             "three": low["A_GE3"] / 3.0}
    # one adult
    w["student"] = cell("NWRK", "A1_NCH") - cell("NWRK_X_EDUC", "A1_NCH")
    w["pensioner_single"] = persons["A1_GE65"]
    w["single"] = persons["A1_LT65"] - pools["single"] - w["student"]
    # single parents in work: 1, 2, 3 or more children as in the household statistics
    kids = [cell("WRK", "A1_CH", n) for n in ("1", "2", "GE3")]
    size = sum(k * (2 + i) for i, k in enumerate(kids)) / sum(kids)
    parents = (persons["A1_DCH"] - low["A1_DCH"]) / size
    w["single_parent_1"], w["single_parent_2"], w["single_parent_3"] = split(parents, *kids)
    # two adults without children, at least one of 65 or over: pensioners, but for the couples
    # with a partner under 65 in work (the people under 65 of these households - those in a
    # household with very low work intensity over their share - less those without work)
    rate = 0.01 * _reference().get("low_work_intensity_rate.household.A2_GE1_GE65", 0.0)
    w["couple_worker_pensioner"] = max(low["A2_GE1_GE65"] / rate - low["A2_GE1_GE65"], 0.0) if rate > 0 else 0.0
    w["pensioner_couple"] = persons["A2_GE1_GE65"] / 2.0 - w["couple_worker_pensioner"]
    # two adults under 65 without children: one or two earners
    w["couple_2earners"], w["couple_1earner"] = split(persons["A2_2LT65"] / 2.0 - pools["couple"],
                                                      cell("WRK", "CPL_NCH"), cell("GE1_WRK_A1_NWRK", "CPL_NCH"))
    # two adults with 1, 2, 3 or more children: one or two earners
    for n, key, size in (("1", "A2_DCH1", 3.0), ("2", "A2_DCH2", 4.0), ("GE3", "A2_DCH_GE3", 5.0)):
        k = n[-1]
        w[f"couple_2earners_{k}"], w[f"couple_1earner_{k}"] = split(
            (persons[key] - low[key]) / size, cell("WRK", "CPL_CH", n), cell("GE1_WRK_A1_NWRK", "CPL_CH", n))
    # three or more adults
    w["three_adults"] = persons["A_GE3"] / 3.0 - pools["three"]
    w["three_adults_1"] = (persons["A_GE3_DCH"] - low["A_GE3_DCH"]) / 4.0
    # households without work by main income: in the proportions of the unemployed, the
    # long-term sick and the recipients of integration income of the model
    st = state0.status
    n = {"unemployed": float(st[S["unemployed"]].sum()),
         "sick": float(st[S["disabled"]].sum()),
         "assistance": float(p["assist.takeup_rate"]) * float(st[S["inactive"]][18:65].sum())}
    for kind, pool in pools.items():
        for k, v in n.items():
            w[f"{k}_{kind}"] = pool * v / sum(n.values())
    return {k: max(v, 0.0) for k, v in w.items()}


def _tilt(shares: dict, low: str, ratio: float, sole: float, multi: float, third: float = 0.0) -> tuple[dict, dict]:
    """Shares of the income groups among the main earners who are the only earner of their
    household and among those of the households with two earners: the lowest group `low` is
    `ratio` times as frequent among the former as among all workers; `third` more workers (the
    third adults in work) are all in the lowest group; the overall shares (weights `sole`,
    `multi` and `third`) stay those of `shares`."""
    if sole <= 0 or multi <= 0:
        return dict(shares), dict(shares)
    s = shares[low]
    total = sole + multi + third
    lower = max((s * total - third - multi) / (s * sole), 0.0) if s > 0 else 0.0
    upper = min((s * total - third) / (s * sole), 1.0 / s) if s > 0 else 1.0
    ratio = float(np.clip(ratio, lower, max(upper, lower)))
    rest = (1.0 - ratio * s) / (1.0 - s)
    a = {k: v * (ratio if k == low else rest) for k, v in shares.items()}
    b = {k: (v * total - a[k] * sole - (third if k == low else 0.0)) / multi for k, v in shares.items()}
    return a, b


def _structure(p, state0) -> dict:
    """`shares`: id -> share of its household type; `third`: composition of the third adult of
    the households of three adults (kind -> share).

    Main earners: self-employed in the proportion that gives the model's number of self-employed
    (they are main earners of the types of `SELF_TYPES` only), then the shares of the income
    groups, the lowest group being less frequent among sole earners
    (`dist.sole_earner_low_income_ratio`). In a share of the couples without work both adults
    receive a benefit (`dist.jobless_couple_second_benefit_share`). The unemployed and long-term
    sick who are not in a household without work are partners of one-earner couples (up to
    `dist.one_earner_partner_benefit_share` of them), then third adults of the households of
    three adults next to the pensioners (`dist.three_adults_pensioner_share`) and the third
    adults in work (what the main earners and their partners leave of employment; `third`
    key `work`). `covered`: share of the unemployed households with an unemployment benefit
    (`unemp.coverage_rate`)."""
    bw = base_weights(p, state0)
    emp = state0.emp.sum(axis=0)
    hosts = sum(bw[b] for b in SELF_TYPES)
    self_share = float(np.clip(emp[J["self_employed"]] / hosts, 0.0, 0.6)) if hosts > 0 else 0.0
    wage, self_ = p["dist.wage_level_shares"], p["dist.self_employed_level_shares"]
    partner = {g: sum(wage[x] for x in levels) for g, levels in PARTNER_GROUPS.items()}
    civil = float(p["pension.scheme_mix_2024"]["civil_servants"])
    low, high = p["dist.pensioner_low_share"], float(p["dist.pensioner_high_share"])
    # income groups of sole earners and of the main earners of households with two earners
    ratio = float(p.get("dist.sole_earner_low_income_ratio", 1.0))

    def hosts_of(kinds, self_employed):
        return sum(bw[b] * ((self_share if self_employed else 1.0 - self_share) if b in SELF_TYPES else (0.0 if self_employed else 1.0))
                   for b in _WORK_TYPES if (b in SOLE_TYPES) == (kinds == "sole") and (b in SELF_TYPES or not self_employed))

    # the unemployed and long-term sick who do not fit in the households without work live with
    # workers: partners of one-earner couples first, then third adults
    st = state0.status
    workers = sum(bw[b] * (1 if b in SOLE_TYPES else 2) for b in _WORK_TYPES)
    employed = float(sum(st[S[k]].sum() for k in WORK))
    # third adults: pensioners (`dist.three_adults_pensioner_share`), then the people in work
    # that the main earners and their partners do not account for (adult children in work)
    three_adults = bw["three_adults"] + bw["three_adults_1"]
    pensioner = float(p.get("dist.three_adults_pensioner_share", 1.0))
    earner = float(np.clip((employed - workers) / three_adults, 0.0, 1.0 - pensioner)) if three_adults > 0 else 0.0
    workers += earner * three_adults
    scale = employed / workers if workers > 0 else 1.0     # as `consistent_weights`
    # (the third adults in work are in the lowest income group of employees: `THIRD_WORK`)
    wage_sole, wage_multi = _tilt(wage, "min_wage", ratio, hosts_of("sole", False), hosts_of("multi", False),
                                  earner * three_adults)
    self_sole, self_multi = _tilt(self_, "self_low", ratio, hosts_of("sole", True), hosts_of("multi", True))
    second = float(p.get("dist.jobless_couple_second_benefit_share", 0.0))     # couples out of work with two benefits

    def placed(k):                                   # recipients in the households without work
        return sum(bw[f"{k}_{pool}"] * (1.0 if adults == 1 else 1.0 + second) for pool, (adults, _) in JOBLESS_POOLS.items())

    extra = {"unemployment": max(float(st[S["unemployed"]].sum()) - placed("unemployed"), 0.0),
             "disability": max(float(st[S["disabled"]].sum()) - placed("sick"), 0.0)}
    total = sum(extra.values())
    slots = sum(bw[b] for b in SOLE_TYPES if b.startswith("couple_1earner")) * (1.0 - self_share) * scale
    fill = min(1.0, float(p["dist.one_earner_partner_benefit_share"]) * slots / total) if total > 0 else 0.0
    status = {k: (v * fill / slots if slots > 0 else 0.0) for k, v in extra.items()}
    status["none"] = max(1.0 - sum(status.values()), 0.0)
    three = three_adults * scale
    left = total * (1.0 - fill)
    fill3 = min(1.0, (1.0 - pensioner - earner) * three / left) if left > 0 else 0.0
    third = {"pension": pensioner, "work": earner,
             **{k: (v * (1.0 - fill) * fill3 / three if three > 0 else 0.0) for k, v in extra.items()}}
    third["none"] = max(1.0 - sum(third.values()), 0.0)
    # unemployed households: the share with an unemployment benefit (the others: `UNCOVERED`)
    covered = float(np.clip(p.get("unemp.coverage_rate", 1.0), 0.0, 1.0))
    out = {}
    for ident, _, _, _, _, _, base, main, mate in ALL_DEFINITIONS:
        if main == UNCOVERED:
            share = 1.0 - covered
        elif base in _WORK_TYPES:
            s = self_share if base in SELF_TYPES else 0.0
            sole = base in SOLE_TYPES
            if main in SELF_LEVELS:
                share = s * (self_sole if sole else self_multi)[main]
            else:
                share = (1.0 - s) * (wage_sole if sole else wage_multi)[main]
            if mate in partner and main not in SELF_LEVELS:
                share *= partner[mate]
            elif mate in PARTNER_STATUS:
                share *= status[mate]
        elif base == "pensioner_single":
            rest = (1.0 - civil) * (1.0 - low["single"])
            share = {"civil": civil, "low": (1.0 - civil) * low["single"], "average": rest * (1.0 - high),
                     "high": rest * high}[main]
        elif base == "pensioner_couple":
            rest = (1.0 - civil) * (1.0 - low["couple"])
            two = float(p.get("dist.pensioner_couple_second_average_share", 0.0) or 0.0)
            share = {"civil": civil, "low": (1.0 - civil) * low["couple"], "average": rest * (1.0 - high) * (1.0 - two),
                     "two_average": rest * (1.0 - high) * two, "high": rest * high}[main]
        elif base in ("unemployed_single", "sick_single"):
            share = BENEFIT_GROUPS[main]
        elif main in ("one", "two"):
            share = second if main == "two" else 1.0 - second
        else:
            share = 1.0
        if base.startswith("unemployed_") and main != UNCOVERED:
            share *= covered
        out[ident] = share
    return {"shares": out, "third": third, "covered": covered}


def _shares(p, state0) -> dict[str, float]:
    """id -> share of its household type (see `_structure`)."""
    return _structure(p, state0)["shares"]


def third_adult_mix(p, state0) -> dict[str, float]:
    """Composition of the third adult of the households of three adults: kind (pension, work,
    unemployment, disability, none) -> share of these households."""
    return _structure(p, state0)["third"]


def weights(p, state0) -> dict[str, float]:
    """Number of households of each row of `DEFINITIONS`."""
    bw, shares = base_weights(p, state0), _shares(p, state0)
    return {d[0]: bw[d[6]] * shares[d[0]] for d in ALL_DEFINITIONS}


WORK = ("emp_private", "emp_public", "self_employed")


def consistent_weights(w: dict[str, float], state, third: dict | float = 1.0, institutions: float = 0.0,
                       assistance: float = 0.0) -> dict[str, float]:
    """Weights rescaled to the head counts of a simulated year: households with a worker so that
    the workers of the set equal employment, pensioner households so that pensioners equal the
    retired living in private households (after those living with workers), the unemployed and
    long-term-sick households to the numbers of unemployed and long-term sick. The households on
    integration income grow by `assistance` people (those excluded from unemployment benefits by
    the time limit who claim it, as in the budget). Other households keep their weight.
    `third`: composition of the third adult of the households of three adults (`third_adult_mix`:
    kind -> share; a number is the share of pensioners). `institutions`: share of the retired
    who do not live in a private household (`dist.pensioner_institution_share`)."""
    if not isinstance(third, dict):
        third = {"pension": float(third)}
    st = state.status
    target = {"work": float(sum(st[S[k]].sum() for k in WORK)),
              "pension": float(st[S["retired"]].sum()) * (1.0 - institutions),
              "unemployment": float(st[S["unemployed"]].sum()), "disability": float(st[S["disabled"]].sum())}
    kinds = {d[0]: [a.kind for a in d[3]] for d in ALL_DEFINITIONS}

    def n(i, group):
        count = sum(1 for k in kinds[i] if (k in WORK if group == "work" else k == group))
        return count + (third.get(group, 0.0) if "third" in kinds[i] else 0.0)

    out = dict(w)
    workers = [i for i in kinds if n(i, "work")]
    have = sum(w[i] * n(i, "work") for i in workers)
    if have > 0 and target["work"] > 0:
        for i in workers:
            out[i] = w[i] * target["work"] / have
    for group in ("pension", "unemployment", "disability"):
        own = [i for i in kinds if n(i, group) and i not in workers]
        with_workers = sum(out[i] * n(i, group) for i in workers)
        have = sum(w[i] * n(i, group) for i in own)
        if have > 0 and target[group] > 0:
            for i in own:
                out[i] = w[i] * max(target[group] - with_workers, 0.0) / have
    on_assistance = [i for i in kinds if kinds[i][0] == "assistance"]
    have = sum(w[i] for i in on_assistance)
    if assistance > 0.0 and have > 0.0:
        for i in on_assistance:
            out[i] = w[i] * (1.0 + assistance / have)
    return out


def household_set(p, state0, state=None) -> list[Household]:
    """The households with their weights: the household structure of the income survey
    (`base_weights`; `state0` is the base-year state), rescaled to the head counts of `state`
    if given."""
    w = weights(p, state0)
    structure = _structure(p, state0)
    shares, third = structure["shares"], structure["third"]
    if state is not None:
        excluded = float((state.labour or {}).get("unemp_excluded_to_inactive") or 0.0)
        w = consistent_weights(w, state, third, float(p.get("dist.pensioner_institution_share", 0.0) or 0.0),
                               float(p.get("assist.unemp_spillover", 0.0) or 0.0) * excluded)
    mix = tuple((kind, level, third.get(kind, 0.0)) for kind, level in THIRD_STATUS)
    mix += ((THIRD_WORK[0], THIRD_WORK[1], third.get("work", 0.0)),)
    covered = structure["covered"]
    # benefits per recipient: the budget gives the average over all recipients; those living
    # with a partner or as third adult receive the cohabitant rate (`dist.cohabitant_benefit_ratio`
    # of the level of the others), the average staying that of the budget
    main, cohab = dict.fromkeys(OWN_BENEFIT_LINES, 0.0), dict.fromkeys(OWN_BENEFIT_LINES, 0.0)
    for d in ALL_DEFINITIONS:
        for a in d[3]:
            if a.kind in main:
                (cohab if a.level == "cohab" else main)[a.kind] += w[d[0]]
            elif a is THIRD:
                for kind in main:
                    cohab[kind] += w[d[0]] * third.get(kind, 0.0)
    ratio = p["dist.cohabitant_benefit_ratio"]
    level = {k: ((main[k] + cohab[k]) / (main[k] + float(ratio[k]) * cohab[k]) if main[k] + cohab[k] > 0 else 1.0) for k in main}
    cohab_scale = tuple((k, level[k] * float(ratio[k])) for k in main)

    def scale(d):                                    # of the main adult of a household without work
        kind = d[3][0].kind
        if kind not in level or d[3][0].level not in BENEFIT_GROUPS:
            return 1.0
        return level[kind] / covered if kind == "unemployment" and covered > 0 else level[kind]

    # (a type without households - the unemployed without benefit while all are covered - is left out)
    return [Household(d[0], d[1], d[2], d[3], d[4], w[d[0]], d[5], d[6], shares[d[0]],
                      mix if THIRD in d[3] else (), scale(d), cohab_scale)
            for d in ALL_DEFINITIONS if w[d[0]] > 0.0]


# ---------------------------------------------------------------------------------------------
# income levels from the simulated state
# ---------------------------------------------------------------------------------------------
def wage_levels(state) -> dict[str, float]:
    """Stylised gross annual wage levels from `state.wage_dist` (mean earnings of each tenth of
    workers, part-time included). P25 and P75 are the means of the third and eighth tenth, the
    median the average of the fifth and sixth; P90 extends the quantile function log-linearly
    from P75-P85; the top level is the mean of the top 3 % under a Pareto tail fitted to P90 and
    the mean of the top tenth."""
    d = np.asarray(state.wage_dist[J["emp_private"]], dtype=float)
    s = np.asarray(state.wage_dist[J["self_employed"]], dtype=float)
    p90 = d[8] * (d[8] / d[7]) ** 0.5
    ratio = max(d[9] / p90, 1.0 + 1e-6)              # mean above a threshold / threshold = a/(a-1)
    alpha = ratio / (ratio - 1.0)
    return {"p25": float(d[2]), "median": float(0.5 * (d[4] + d[5])), "p75": float(d[7]), "p90": float(p90),
            "top_alpha": float(alpha), "top_ratio": float(ratio),
            "self_median": float(0.5 * (s[4] + s[5]))}


def benefit_env(state, pop_state=None) -> BenefitEnv:
    """Benefit indices and average pensions of a simulated year (from the expenditure module)."""
    x = (state.extra or {}).get("expenditure") or {}
    pop = (pop_state or state).pop
    kids = np.asarray(pop[:18].sum(axis=(0, 1)), dtype=float)
    shares = {r: (float(kids[i] / kids.sum()) if kids.sum() > 0 else 1.0 / len(REGIONS)) for i, r in enumerate(REGIONS)}
    avg, mix = dict(x.get("pension_avg") or {}), dict(x.get("pension_mix") or {})
    # pension income per pensioner over the average pension per scheme: a pensioner often draws
    # more than one pension (own and survivor's, several schemes), the budget counts them all
    ratio = 1.0
    retired = float(state.status[S["retired"]].sum())
    per_scheme = sum(mix.get(k, 0.0) * avg.get(k, 0.0) for k in avg)
    if retired > 0 and per_scheme > 0:
        from ..fiscal.revenue import _benefits
        spending = _benefits(state, None)[0]["pension"][0] if state.expenditure else 0.0
        if spending > 0:
            ratio = float(np.clip(spending * 1e6 / retired / per_scheme, 1.0, 1.6))
    return BenefitEnv(welfare_index=float(x.get("welfare_index", 1.0)), benefit_index=float(x.get("benefit_index", 1.0)),
                      pension_avg=avg, child_region_shares=shares, pension_income_ratio=ratio)


def wage(level: str, p, wages: dict) -> float:
    """Reference wage levels of the payslip explorer and of earnings-related benefits: minimum
    wage, P25, median, P75, P90 and the mean of the top 3 % of private-sector employees."""
    if level == "min_wage":
        return float(p["dist.min_wage_monthly"] * p["dist.min_wage_months"])
    if level == "top":
        top_share = float(p["dist.wage_level_shares"]["top"])
        threshold = wages["p90"] * (0.10 / top_share) ** (1.0 / wages["top_alpha"])
        return float(threshold * wages["top_ratio"])
    if level == "ref":
        return wage(str(p["dist.benefit_reference_wage"]), p, wages)
    return float(wages[level])


def _group_means(value: np.ndarray, weight: np.ndarray, shares) -> np.ndarray:
    """Mean of `value` (points ordered by rank, with weights) in consecutive groups holding the
    given shares of the total weight (a point that straddles a limit is split)."""
    hi = np.cumsum(weight) / weight.sum()
    lo = hi - weight / weight.sum()
    edges = np.concatenate([[0.0], np.cumsum(shares)])
    edges[-1] = 1.0
    out = np.zeros(len(shares))
    for g in range(len(shares)):
        part = np.clip(np.minimum(hi, edges[g + 1]) - np.maximum(lo, edges[g]), 0.0, None)
        out[g] = float((part * value).sum() / part.sum()) if part.sum() > 0 else 0.0
    return out


OWN_BENEFIT_LINES = {"unemployment": "unemployment", "disability": "invalidity"}


def income_levels(state, p, cal: dict | None = None, ratios: dict | None = None) -> dict:
    """Income of every group used by the households, from the population of tax units of a
    simulated year: `levels` (name -> gross annual income per person: the six groups of
    employees, the three groups of second earners, the four groups of the self-employed,
    `pension_high`) and `ratio` (name -> taxable income of the group in the fiscal distribution
    over the taxable income the payslip rules give; also for `pension_low`, `pension_average`,
    `unemployment`, `disability`). The ratios need the base-year calibration `cal`
    (`state.extra["revenue"]["pit"]`); a scenario evaluation passes the baseline's `ratios`.
    The reference wage levels of `wage_levels` are included."""
    from ..fiscal.revenue import _benefits

    out = wage_levels(state)
    benefits = dict(_benefits(state, p)[0]) if state.expenditure else {}
    # benefits in payment to the unemployed and the long-term sick: their own budget lines (the
    # tax units also hold the sickness benefits of people in work and unemployment with company
    # supplement, which reach the households through `dist.other_benefit_recipients`)
    for kind, line in OWN_BENEFIT_LINES.items():
        if kind in benefits and line in state.expenditure:
            benefits[kind] = (float(state.expenditure[line]), benefits[kind][1])
    # (temporary unemployment is paid to people in work: only full unemployment is counted)
    full = ((state.extra or {}).get("expenditure") or {}).get("unemployment_full")
    if full is not None and "unemployment" in benefits and 0.0 < float(full) <= benefits["unemployment"][0]:
        benefits["unemployment"] = (float(full), benefits["unemployment"][1])
    pop = taxunits.population(state, p, benefits)
    fine = {k: (taxunits.refine(g), float(n.sum())) for k, (n, g) in pop.items()}
    taxable = {}
    if cal is not None and ratios is None:
        rows = taxunits._rows(pop, p, cal)
        for r, t in zip(rows, taxunits._mapped(rows, cal)):
            w = r["w"].sum(axis=(0, 1))
            num, den = taxable.get(r["kind"], (0.0, 0.0))
            taxable[r["kind"]] = (num + (t * r["w"]).sum(axis=(0, 1)), den + w)
        taxable = {k: np.divide(num, den, out=np.zeros_like(num), where=den > 0) for k, (num, den) in taxable.items()}

    def groups(kinds, shares):
        """(gross, taxable in the fiscal distribution) of consecutive groups of the pooled kinds."""
        g = np.concatenate([fine[k][0] for k in kinds])
        w = np.concatenate([fine[k][1] * taxunits.FINE_W for k in kinds])
        order = np.argsort(g, kind="stable")
        gross = _group_means(g[order], w[order], shares)
        if not taxable:
            return gross, None
        t = np.concatenate([taxable.get(k, np.zeros(taxunits.N_FINE)) for k in kinds])
        return gross, _group_means(t[order], w[order], shares)

    wage_s, self_s = p["dist.wage_level_shares"], p["dist.self_employed_level_shares"]
    high = float(p["dist.pensioner_high_share"])
    low = float(p["dist.pensioner_low_share"]["single"])
    spec = {
        "emp": (("emp_private", "emp_public"), WAGE_LEVELS, [wage_s[k] for k in WAGE_LEVELS], "emp_private"),
        "partner": (("emp_private", "emp_public"), tuple(PARTNER_GROUPS),
                    [sum(wage_s[x] for x in v) for v in PARTNER_GROUPS.values()], "emp_private"),
        "self": (("self_employed",), SELF_LEVELS, [self_s[k] for k in SELF_LEVELS], "self_employed"),
        "pension": (("pension",), ("pension_low", "pension_average", "pension_high"),
                    [low, (1.0 - low) * (1.0 - high), (1.0 - low) * high], "pension"),
        "unemployment": (("unemployment",), ("unemployment",), [1.0], "unemployment"),
        "disability": (("disability",), ("disability",), [1.0], "disability"),
        "unemployment_groups": (("unemployment",), tuple(f"unemployment_{g}" for g in BENEFIT_GROUPS),
                                list(BENEFIT_GROUPS.values()), "unemployment"),
        "disability_groups": (("disability",), tuple(f"disability_{g}" for g in BENEFIT_GROUPS),
                              list(BENEFIT_GROUPS.values()), "disability"),
    }
    levels, ratio = {}, dict(ratios or {})
    for kinds, names, shares, kind in spec.values():
        if not all(k in fine and fine[k][1] > 0 for k in kinds):
            continue
        gross, tax = groups(kinds, shares)
        for i, name in enumerate(names):
            levels[name] = float(gross[i])
            if tax is not None:
                rules = float(rcalc.tax_base(gross[i], p, kind)["taxable"][0])
                ratio[name] = float(np.clip(tax[i] / rules, RATIO_MIN, 1.5)) if rules > 0 else 1.0
    if ratios is None and ratio:
        # earnings-weighted average ratio of the groups of employees and of the self-employed
        for key, names, shares in (("mean.emp", WAGE_LEVELS, wage_s), ("mean.self", SELF_LEVELS, self_s)):
            mass = sum(shares[n] * levels[n] for n in names if n in levels and n in ratio)
            if mass > 0:
                ratio[key] = float(sum(shares[n] * levels[n] * ratio[n] for n in names if n in levels and n in ratio) / mass)
    out["levels"], out["ratio"] = levels, ratio
    return out


def benefit(kind: str, variant: str, p, env: BenefitEnv, prev_wage: float = 0.0) -> float:
    """Gross annual benefit of a stylised recipient, with the indices of the simulated year (same
    construction as the expenditure module: W x f(wage / W) x indexation of benefits)."""
    w, b = env.welfare_index, env.benefit_index
    if kind == "unemployment":
        return float(w * replacement_income("unemployed", prev_wage / w, p) * b)
    if kind == "disability":
        return float(w * replacement_income("disabled", prev_wage / w, p) * b)
    if kind == "assistance":
        return float(p["assist.amount"] * p["assist.generosity_factor"] * w * b * p["dist.assist_category_ratio"][variant])
    if kind == "pension":
        level = p["pension.level_factor"]
        if variant in ("low", "household_low"):
            x = float(w * replacement_income("retired", 0.0, p, "employees") * b) * level["employees"]
            return x * (float(p["dist.pension_household_rate_ratio"]) if variant == "household_low" else 1.0)
        scheme = "civil_servants" if variant == "civil" else "employees"
        avg = env.pension_avg.get(scheme, p["pension.avg_pension_2024"][scheme])
        return float(avg) * level[scheme] * env.pension_income_ratio
    raise ValueError(f"no benefit for kind '{kind}'")


def resolve(hh: Household, p, wages: dict, env: BenefitEnv, ref_wages: dict | None = None) -> list[tuple[str, float, float]]:
    """(kind, gross annual income, taxable ratio) of each adult of a household. `wages` comes
    from `income_levels`; `ref_wages` (default `wages`) are the wage levels behind
    earnings-related benefits."""
    out = []
    ref_wages = wages if ref_wages is None else ref_wages
    levels, ratio = wages.get("levels", {}), wages.get("ratio", {})
    paid = ref_wages.get("levels", {})                 # benefits in payment: those of the rules evaluated
    spread = float(p.get("dist.fiscal_dispersion", 0.0) or 0.0)
    for a in hh.adults:
        if a.kind in ("emp_private", "emp_public", "self_employed"):
            # dispersion of the tax returns: the earnings of a group move towards its
            # taxable income in the fiscal statistics relative to the average group (same total
            # earnings); the taxable income itself, hence the tax, is unchanged
            r = ratio.get(a.level, 1.0)
            mean = ratio.get("mean.self" if a.kind == "self_employed" else "mean.emp", 0.0)
            # (a ratio at its lower bound is not informative: that group keeps its earnings)
            k = (r / mean) ** spread if mean > 0 and r > RATIO_MIN + 1e-9 else 1.0
            out.append((a.kind, levels[a.level] * k, r / k))
        elif a.kind == "student":
            out.append((a.kind, float(p["dist.student_job_earnings"]), 1.0))
        elif a.kind == "none":
            out.append((a.kind, 0.0, 1.0))
        elif a.kind == "third":
            # third adult: the average of the cases of `hh.mix` (`resolve_third` gives each one)
            cases = resolve_third(hh, p, wages, env, ref_wages)
            out.append((a.kind, sum(share * gross for share, _, gross, _ in cases), 1.0))
        elif a.kind == "pension":
            name = {"low": "pension_low", "household_low": "pension_low", "high": "pension_high"}.get(a.level, "pension_average")
            gross = paid["pension_high"] if a.level == "high" else benefit(a.kind, a.level, p, env)
            out.append((a.kind, gross, ratio.get(name, 1.0)))
        elif a.kind in ("unemployment", "disability") and a.level in BENEFIT_GROUPS:
            # benefits in payment of the budget (spending per recipient), by group, at the level
            # of the recipients who are not cohabitants; unemployment: over the share of the
            # unemployed who receive a benefit
            out.append((a.kind, paid[f"{a.kind}_{a.level}"] * hh.benefit_scale, ratio.get(f"{a.kind}_{a.level}", 1.0)))
        elif a.kind == "unemployment" and a.level.startswith(UNCOVERED):
            # unemployed without unemployment benefit: the integration income of their situation
            out.append(("assistance", benefit("assistance", a.level.split("_", 1)[1], p, env), 1.0))
        elif a.kind in ("unemployment", "disability") and a.level == "cohab":
            cohab = dict(hh.cohab_scale).get(a.kind, float(p["dist.cohabitant_benefit_ratio"][a.kind]))
            out.append((a.kind, paid[a.kind] * cohab, ratio.get(a.kind, 1.0)))
        else:
            prev = wage("ref", p, ref_wages) if a.kind in ("unemployment", "disability") else 0.0
            out.append((a.kind, benefit(a.kind, a.level, p, env, prev), ratio.get(a.kind, 1.0)))
    return out


def resolve_third(hh: Household, p, wages: dict, env: BenefitEnv, ref_wages: dict | None = None) -> list[tuple[float, str, float, float]]:
    """(share of the households, kind, gross annual income, taxable ratio) of each case of the
    third adult of a household of three adults (`Household.mix`)."""
    out = []
    for kind, level, share in hh.mix:
        if kind == "none":
            out.append((share, kind, 0.0, 1.0))
        else:
            k, gross, ratio = resolve(replace(hh, adults=(Adult(kind, level),), mix=()), p, wages, env, ref_wages)[0]
            out.append((share, k, gross, ratio))
    return out
