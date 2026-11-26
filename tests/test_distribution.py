"""Distributional analysis: household set, household tax rules, scenario comparison, payslip."""
import time

import numpy as np
import pytest

from belsim.core import engine
from belsim.core.dims import REGIONS, S
from belsim.core.params import Change, Measure, ParamSet, Scenario
from belsim.distribution import analysis, calc, distribution, household_set, net_income, param_view, payslip
from belsim.distribution import households as hh_mod
from belsim.fiscal.revenue import calc as rcalc

END = 2030


@pytest.fixture(scope="module")
def pset():
    return ParamSet.load()


@pytest.fixture(scope="module")
def base(pset):
    return engine.run(None, end=END, params=pset)


@pytest.fixture(scope="module")
def runner(pset, base):
    cache = {}

    def run(*changes, year=END):
        key = tuple((c.param, repr(c.value), c.op, c.start, c.key) for c in changes)
        if key not in cache:
            sc = Scenario(name="test", measures=[Measure(name="m", changes=list(changes))] if changes else [])
            cache[key] = (sc, engine.run(sc, end=END, baseline=base, params=pset))
        sc, res = cache[key]
        return distribution(base, res, pset.with_scenario(None), pset.with_scenario(sc), year)

    return run


def by_id(out):
    return {h["id"]: h for h in out["households_all"]}


def levels_of(base, p, year):
    return hh_mod.income_levels(base.at(year), p, base.at(2024).extra["revenue"]["pit"])


# ------------------------------------------------------------------ household set

def test_weights_cover_the_population(pset, base):
    p = param_view(pset, base, 2024)
    hhs = household_set(p, base.at(2024))
    assert 250 <= len(hhs) <= 300 and len({h.id for h in hhs}) == len(hhs)
    assert 20 <= sum(h.representative for h in hhs) <= 30
    assert all(h.weight > 0 for h in hhs)
    for base_type in {h.base for h in hhs}:                    # the income groups of a type add up
        assert sum(h.share for h in hhs if h.base == base_type) == pytest.approx(1.0)
    n_households = sum(h.weight for h in hhs)
    n_persons = sum(h.weight * h.persons for h in hhs)
    # persons: the population of the income survey; households: within 2 % of the household
    # statistics (5.2 million), which the weights are not built on
    frame = hh_mod.silc_frame()
    assert n_persons == pytest.approx(frame["total"], rel=1e-9) and 11.5e6 < n_persons < 11.8e6
    assert n_households == pytest.approx(hh_mod._cells()[("TOTAL", "TOTAL", "TOTAL")], rel=0.02)
    assert n_persons < float(base.at(2024).pop.sum())
    # dependent children of the survey: the under-18s and the students of 18 to 24 living with
    # their parents (the children and the students of the model: 2.92 million)
    children = sum(h.weight * h.children for h in hhs)
    st = base.at(2024).status
    assert children == pytest.approx(float(st[S["child"]].sum() + st[S["student"]].sum()), rel=0.02)
    # persons by household type are those of the survey
    by_type = {}
    for h in hhs:
        key = analysis.person_groups(h)
        name = next(k for k in analysis.POVERTY_HOUSEHOLD_TYPES if k in key and k != "aged_65_plus")
        by_type[name] = by_type.get(name, 0.0) + h.weight * h.persons
    survey = {"single": ("A1_LT65", "A1_GE65"), "single_parent": ("A1_DCH",), "couple_no_children": ("A2_2LT65", "A2_GE1_GE65"),
              "couple_with_children": ("A2_DCH1", "A2_DCH2", "A2_DCH_GE3", "A_GE3_DCH"), "three_adults_no_children": ("A_GE3",)}
    for name, codes in survey.items():
        assert by_type[name] == pytest.approx(sum(frame["persons"][c] for c in codes), rel=1e-6), name
    groups = {h.group for h in hhs}
    assert {"Single workers", "Working families", "Out of work", "Pensioners", "Students"} <= groups


def test_wage_levels_follow_the_simulated_distribution(pset, base):
    for year in (2024, END):
        p = param_view(pset, base, year)
        w = hh_mod.wage_levels(base.at(year))
        levels = [hh_mod.wage(k, p, w) for k in hh_mod.WAGE_LEVELS]
        assert levels == sorted(levels)
        d = base.at(year).wage_dist[0]
        assert d[4] <= levels[2] <= d[5] and levels[4] > d[8] and levels[5] > d[9]
    w0, w1 = hh_mod.wage_levels(base.at(2024)), hh_mod.wage_levels(base.at(END))
    assert w1["median"] > w0["median"]


def test_weights_follow_the_simulated_head_counts(pset, base):
    """For an analysis the households add up to the population behind the budget of that year."""
    for year in (2024, END):
        st = base.at(year)
        hhs = household_set(param_view(pset, base, year), base.at(2024), st)

        def persons(kinds):
            # (the third adult of the households of three adults counts in each kind for its share)
            return sum(h.weight * sum(analysis.adult_counts(h).get(k, 0.0) for k in kinds) for h in hhs)

        p = param_view(pset, base, year)
        share = p["dist.three_adults_pensioner_share"]
        mix = hh_mod.third_adult_mix(p, base.at(2024))
        assert mix["pension"] == share and sum(mix.values()) == pytest.approx(1.0) and min(mix.values()) >= 0
        assert persons(hh_mod.WORK) == pytest.approx(sum(float(st.status[S[k]].sum()) for k in hh_mod.WORK))
        # pensioners: the retired living in private households
        assert persons(("pension",)) == pytest.approx(float(st.status[S["retired"]].sum()) * (1 - p["dist.pensioner_institution_share"]))
        assert persons(("unemployment",)) == pytest.approx(float(st.status[S["unemployed"]].sum()))
        assert persons(("disability",)) == pytest.approx(float(st.status[S["disabled"]].sum()))
        assert 4.9e6 < sum(h.weight for h in hhs) < 5.9e6
        assert sum(h.weight * h.persons for h in hhs) == pytest.approx(float(st.pop.sum()), rel=0.03)
    # the base-year set needs no scaling of the households in work, without work or on a benefit:
    # the third adults in work take up what is left of employment
    st = base.at(2024)
    p = param_view(pset, base, 2024)
    frame, scaled = household_set(p, st), household_set(p, st, st)
    for a, b in zip(frame, scaled):
        if not a.base.startswith("pensioner"):
            assert b.weight == pytest.approx(a.weight, rel=1e-6), a.id
        else:
            assert b.weight == pytest.approx(a.weight, rel=0.08), a.id


# ------------------------------------------------------------------ one tax model: households and budget

PIT = ("pit_federal", "pit_regional", "pit_local")


def household_tax(pset, base, year):
    """Weighted sum of the income tax of the households (EUR mn)."""
    st = base.at(year)
    p = param_view(pset, base, year)
    wages, env = levels_of(base, p, year), hh_mod.benefit_env(st)
    hhs = household_set(p, base.at(2024), st)
    return sum(h.weight * analysis.evaluate(h, p, wages, env, rates=False)["income_tax"] for h in hhs) / 1e6


def test_household_tax_is_consistent_with_aggregate_income_tax(pset, base, runner):
    """Same calculator, same income distribution, two populations: the households (types x
    income groups of the tax units, weighted) and the grid of tax units behind the budget.
    Stated tolerance: the households carry 88-110 % of the aggregate personal income tax (90 %
    in 2024 and in 2030; it was 71-74 % with 26 households at fixed wage levels) and 95-112 %
    of personal social contributions (106 % in 2024, 109 % in 2030: the observed fall of the
    contributions of the self-employed in 2025 is in the budget line and the households treat
    every employee as a private-sector one; it was 110-112 % while the contributions of the
    self-employed followed the rules on national-accounts income instead of declared income)."""
    for year in (2024, END):
        aggregate = sum(base.at(year).revenue[k] for k in PIT)
        ratio = household_tax(pset, base, year) / aggregate
        assert 0.88 < ratio < 1.10, (year, ratio)
        out = runner(year=year)
        cover = out["meta"]["coverage"]
        # the coverage also counts the income tax on the benefits taken from the budget lines
        # (sickness and career-break benefits of people in work), about 1 % of the tax
        on_benefits = sum(h["weight"] * h["baseline"]["other_benefits_tax"] for h in out["households_all"]) / 1e6 / aggregate
        assert 0.0 < on_benefits < 0.02
        assert cover["income_tax"] == pytest.approx(ratio + on_benefits, abs=2e-3)
        assert 0.95 < cover["social_contributions"] < 1.12, (year, cover)
    # wages of the employees in the set against the wage bill of the model
    st, p = base.at(2024), param_view(pset, base, 2024)
    wages, env = levels_of(base, p, 2024), hh_mod.benefit_env(st)
    paid = sum(h.weight * g for h in household_set(p, st, st)
               for k, g, _ in hh_mod.resolve(h, p, wages, env) if k == "emp_private") / 1e6
    assert paid == pytest.approx(st.labour["wage_bill_private"] + st.labour["wage_bill_public"], rel=0.05)


def test_income_distribution_is_close_to_observed(runner):
    """Stated tolerances against EU-SILC 2025, incomes of 2024 (data/base/distribution_reference.csv):
    at-risk-of-poverty rate within 1 point (10.1 % against 10.9 %), mean equivalised income within
    1 % (33,889 against 33,873 EUR) and median within 2.5 % (30,617 against 31,299 EUR: on the
    household structure of the survey the families with children and the pensioner couples of the
    set are 4-5 % short of the survey's median for their type, the households of three adults
    above it), Gini within 0.01 (0.240 against 0.234), S80/S20 within 0.2 (3.30 against 3.25),
    income share of each tenth within 1 point (top tenth 19.6 % against 19.9 %), rates at 40, 50
    and 70 % of the median within 1.5 points (1.3, 3.7 and 20.3 % against 1.8, 5.1 and 20.5 %)."""
    out = runner(year=2024)
    ind = out["indicators"]
    assert ind["gini"]["baseline"] == pytest.approx(ind["gini"]["observed"]["value"], abs=0.01)
    assert ind["s80s20"]["baseline"] == pytest.approx(ind["s80s20"]["observed"]["value"], abs=0.2)
    assert ind["poverty_rate"]["observed"]["value"] == pytest.approx(0.109)
    assert ind["poverty_rate"]["baseline"] == pytest.approx(0.109, abs=0.01)
    ref = analysis.reference_indicators()
    assert ind["poverty_rate"]["line_baseline"] / 0.6 == pytest.approx(ref["median_equivalised_income_eur"]["value"], rel=0.025)
    rows = out["households_all"]
    mean = sum(h["weight"] * h["persons"] * h["baseline"]["equivalised_net"] for h in rows) / sum(h["weight"] * h["persons"] for h in rows)
    assert mean == pytest.approx(33873.0, rel=0.01)                 # Eurostat ilc_di03, mean equivalised net income 2025
    for d in out["deciles"]:
        assert d["income_share"] == pytest.approx(d["income_share_observed"]["value"], abs=0.01), d["decile"]
    for pct, row in out["poverty"]["other_lines"].items():
        assert row["observed"]["value"] == pytest.approx(0.01 * ref[f"poverty_rate_line{pct}.age.TOTAL"]["value"])
        assert row["baseline"] == pytest.approx(row["observed"]["value"], abs=0.015), pct
    rates = [out["poverty"]["other_lines"][k]["baseline"] for k in ("40", "50")] + [ind["poverty_rate"]["baseline"],
                                                                                    out["poverty"]["other_lines"]["70"]["baseline"]]
    assert rates == sorted(rates)
    cov = out["meta"]["coverage"]
    assert 0.90 <= cov["income_tax"] <= 1.10 and 0.95 <= cov["social_contributions"] <= 1.08


# observed at-risk-of-poverty rates (Eurostat EU-SILC 2025, incomes of 2024) and the tolerance of
# the set, in points: 3 everywhere but for one group with a known limit (stated next to it);
# the rate of the set in 2024 is in the comment
POVERTY_TOLERANCES = {
    # ilc_li03 / ilc_li02
    "single": (0.144, 0.03),                    # 16.2 %
    "single_parent": (0.207, 0.03),             # 22.6 %
    "couple_no_children": (0.076, 0.03),        # 7.0 %
    "couple_with_children": (0.105, 0.03),      # 8.4 %: households of three adults with children too well off
    "three_adults_no_children": (0.038, 0.03),  # 1.9 %
    "aged_65_plus": (0.093, 0.03),              # 10.2 %
    # ilc_li04 (most frequent activity status, people of 18 and over)
    "employed": (0.041, 0.03),                  # 4.6 %
    "employee": (0.029, 0.03),                  # 3.4 %
    "self_employed": (0.118, 0.03),             # 9.9 %
    "unemployed": (0.404, 0.03),                # 38.5 %
    "retired": (0.085, 0.03),                   # 7.7 %
    "other_inactive": (0.220, 0.04),            # 25.4 %: every long-term sick person of a group at the budget average
}


def test_poverty_by_household_type_and_activity_status(runner):
    """The at-risk-of-poverty rate of the main household types and activity statuses is within the
    stated tolerance of the rate published by Eurostat (on the household statistics alone it was
    17.6 % against 14.4 % for single people, 23.9 % against 20.7 % for single parents, 11.9 %
    against 7.6 % for couples without children, 0.4 % against 3.8 % for three or more adults,
    15.9 % against 11.8 % for the self-employed), and the published rate is shown next to it."""
    out = runner(year=2024)
    pov = out["poverty"]
    rows = {r["id"]: r for k in ("by_household_type", "by_activity_status") for r in pov[k]}
    assert [r["id"] for r in pov["by_household_type"]] == list(analysis.POVERTY_HOUSEHOLD_TYPES)
    assert [r["id"] for r in pov["by_activity_status"]] == list(analysis.POVERTY_ACTIVITY_STATUSES)
    assert set(rows) == set(POVERTY_TOLERANCES)
    for key, (observed, tolerance) in POVERTY_TOLERANCES.items():
        r = rows[key]
        assert set(r) >= {"id", "label", "persons", "population_share", "baseline", "scenario", "delta", "observed"}
        assert r["observed"]["value"] == pytest.approx(observed, abs=1e-9) and r["observed"]["year"] == 2025, key
        assert r["baseline"] == pytest.approx(observed, abs=tolerance), key
        assert r["scenario"] == r["baseline"] and r["delta"] == 0.0
    # the order of the groups is that of the survey
    assert rows["unemployed"]["baseline"] > rows["other_inactive"]["baseline"] > rows["retired"]["baseline"] > rows["employee"]["baseline"]
    assert rows["self_employed"]["baseline"] > rows["employee"]["baseline"]
    assert rows["single_parent"]["baseline"] > rows["single"]["baseline"] > rows["couple_no_children"]["baseline"]
    # the household types cover everybody once, the statuses the adults
    types = [r for r in pov["by_household_type"] if r["id"] != "aged_65_plus"]
    assert sum(r["persons"] for r in types) == pytest.approx(out["meta"]["persons_total"], abs=10)
    assert sum(r["population_share"] for r in types) == pytest.approx(1.0, abs=1e-3)
    total = sum(r["baseline"] * r["persons"] for r in types) / sum(r["persons"] for r in types)
    assert total == pytest.approx(out["indicators"]["poverty_rate"]["baseline"], abs=2e-4)
    adults = sum(h["weight"] * h["adults"] for h in out["households_all"])
    assert sum(rows[k]["persons"] for k in ("employed", "unemployed", "retired", "other_inactive")) == pytest.approx(adults, abs=10)
    assert rows["employed"]["persons"] == pytest.approx(rows["employee"]["persons"] + rows["self_employed"]["persons"], abs=2)
    # median equivalised income of the groups, next to the published one (ilc_di04, ilc_di05)
    for key in ("single", "couple_no_children", "couple_with_children", "employed", "unemployed", "retired"):
        assert rows[key]["median_income"] == pytest.approx(rows[key]["median_income_observed"]["value"], rel=0.08), key


def test_poverty_breakdown_follows_a_scenario(runner):
    """A higher integration income lowers poverty where its recipients are, a higher pension among
    the people of 65 and over; nothing moves without a measure."""
    out = runner(Change(param="assist.generosity_factor", value=1.25, start=2027))
    rows = {r["id"]: r for k in ("by_household_type", "by_activity_status") for r in out["poverty"][k]}
    assert out["indicators"]["poverty_rate"]["delta"] < -0.002
    assert rows["other_inactive"]["delta"] < -0.01 and rows["single_parent"]["delta"] < -0.005
    assert abs(rows["employee"]["delta"]) < 0.003
    for r in rows.values():
        assert r["delta"] == pytest.approx(r["scenario"] - r["baseline"], abs=1.1e-4)
    out = runner()
    assert all(r["delta"] == 0.0 for k in ("by_household_type", "by_activity_status") for r in out["poverty"][k])


def test_household_structure_follows_the_statistics(pset, base):
    """Who lives with whom: the households without work are the people living in a household
    with very low work intensity of the income survey (not scaled up to hold every benefit
    recipient); the unemployed and long-term sick who do not fit there live with workers; third
    adults are in work as far as employment requires; low earners are less often sole earners,
    without changing the number of workers of each income group."""
    p = param_view(pset, base, 2024)
    st = base.at(2024)
    hhs = household_set(p, st, st)
    low = hh_mod.silc_frame()["low_work"]
    w = {}
    for h in hhs:
        w[h.base] = w.get(h.base, 0.0) + h.weight * h.persons
    jobless = lambda pool: sum(w[f"{k}_{pool}"] for k in hh_mod.JOBLESS_KINDS)
    assert jobless("single") == pytest.approx(low["A1_LT65"], rel=1e-6)                 # 283 thousand people
    assert jobless("couple") == pytest.approx(low["A2_2LT65"], rel=1e-6)
    assert jobless("family") == pytest.approx(low["A1_DCH"], rel=1e-6)
    assert jobless("three") == pytest.approx(low["A_GE3"], rel=1e-6)
    assert sum(jobless(k) for k in ("couple_1", "couple_2", "couple_3")) == pytest.approx(
        sum(low[k] for k in ("A2_DCH1", "A2_DCH2", "A2_DCH_GE3", "A_GE3_DCH")), rel=1e-6)
    # couples of a pensioner and a worker: the people under 65 of the survey's couples with
    # someone of 65 or over, less those without work (92 thousand)
    assert w["couple_worker_pensioner"] / 2 == pytest.approx(91_500, rel=0.02)
    # third adults: a pensioner, a worker, a benefit recipient or an adult without income
    mix = hh_mod.third_adult_mix(p, st)
    assert set(mix) == {"pension", "work", "unemployment", "disability", "none"}
    assert mix["unemployment"] > 0 and mix["disability"] > 0 and mix["none"] >= 0 and 0.2 < mix["work"] < 0.6
    three = [h for h in hhs if h.base.startswith("three_adults")]
    assert three
    for h in three:
        got = {}
        for kind, _, share in h.mix:
            got[kind] = got.get(kind, 0.0) + share
        assert got.pop("emp_private") == pytest.approx(mix["work"])
        assert got == {k: v for k, v in mix.items() if k != "work"}
    assert all(not h.mix for h in hhs if not h.base.startswith("three_adults"))
    # adults without income of their own: about as many as the other inactive of working age
    none = sum(h.weight * analysis.adult_counts(h).get("none", 0.0) for h in hhs)
    assert none == pytest.approx(float(st.status[S["inactive"]].sum()), rel=0.10)
    # low earners: less often sole earners, same number in total (third adults in work included)
    ratio = p["dist.sole_earner_low_income_ratio"]
    assert 0 < ratio < 1
    shares = p["dist.wage_level_shares"]
    def mains(bases, kind, level=None):
        return sum(h.weight for h in hhs if h.base in bases and h.adults[0].kind == kind and level in (None, h.adults[0].level))
    multi = tuple(b for b in hh_mod._WORK_TYPES if b not in hh_mod.SOLE_TYPES)
    sole, both = hh_mod.SOLE_TYPES, hh_mod.SOLE_TYPES + multi
    assert mains(sole, "emp_private", "min_wage") / mains(sole, "emp_private") == pytest.approx(ratio * shares["min_wage"])
    third = sum(h.weight for h in three) * mix["work"]
    assert (mains(both, "emp_private", "min_wage") + third) / (mains(both, "emp_private") + third) == pytest.approx(shares["min_wage"])
    self_shares = p["dist.self_employed_level_shares"]
    assert mains(hh_mod.SELF_TYPES, "self_employed", "self_low") / mains(hh_mod.SELF_TYPES, "self_employed") == pytest.approx(self_shares["self_low"])
    assert mains(("single",), "self_employed", "self_low") / mains(("single",), "self_employed") == pytest.approx(ratio * self_shares["self_low"])
    # benefits per recipient: cohabitants and the others average to the spending of the budget
    for kind, cohab in dict(hhs[0].cohab_scale).items():
        assert cohab <= 1.0 + 1e-12 and cohab >= p["dist.cohabitant_benefit_ratio"][kind] - 1e-12


def test_unemployed_without_benefit_and_excluded(pset, base, runner):
    """From 2026 a quarter of the unemployed receive no benefit (`unemp.coverage_rate`): the set
    shows them as households on the integration income next to those who keep the benefit per
    recipient of the budget - not every unemployed household with a benefit cut by a quarter
    (which put 63 % of the unemployed under the poverty line in 2027-2030); the people excluded by
    the time limit who claim the integration income are households on it."""
    st0 = base.at(2024)
    assert not [h for h in household_set(param_view(pset, base, 2024), st0, st0) if "__no_benefit" in h.id]
    year = 2028
    p, st = param_view(pset, base, year), base.at(year)
    covered = p["unemp.coverage_rate"]
    assert covered < 1
    hhs = household_set(p, st0, st)
    wages, env = levels_of(base, p, year), hh_mod.benefit_env(st)
    single = {h.id: h for h in hhs if h.base == "unemployed_single"}
    without = single["unemployed_single__no_benefit"]
    assert without.weight == pytest.approx((1 - covered) * sum(h.weight for h in single.values()))
    assert hh_mod.resolve(without, p, wages, env)[0][0] == "assistance"
    assert analysis.person_groups(without)["unemployed"] == 1.0
    # those with a benefit: spending on full unemployment per unemployed person, over the share covered
    paid = sum(h.weight * hh_mod.resolve(h, p, wages, env)[0][1] for h in single.values() if h is not without)
    per_person = st.extra["expenditure"]["unemployment_full"] * 1e6 / float(st.status[S["unemployed"]].sum())
    assert paid / sum(h.weight for h in single.values()) == pytest.approx(per_person, rel=0.01)
    # households on integration income: those of 2024 plus the excluded who claim it
    count = lambda hs: sum(h.weight for h in hs if h.adults[0].kind == "assistance")
    excluded = st.labour["unemp_excluded_to_inactive"]
    assert excluded > 50_000
    assert count(hhs) == pytest.approx(count(household_set(param_view(pset, base, 2024), st0, st0)) + p["assist.unemp_spillover"] * excluded)
    for y in (2027, END):
        rows = {r["id"]: r for r in runner(year=y)["poverty"]["by_activity_status"]}
        assert 0.25 < rows["unemployed"]["baseline"] < 0.50, y


def test_budget_benefits_reach_the_households(pset, base, runner):
    """The cash benefits of the budget that the household rules do not produce are in the income
    of the households: the lines of `dist.other_benefit_recipients` add up to the budget (social
    assistance: together with the integration income of the rules), family allowances to those of
    all entitled children."""
    out = runner(year=2024)
    st, p = base.at(2024), param_view(pset, base, 2024)
    rows = out["households_all"]
    lines = p["dist.other_benefit_recipients"]
    assert set(lines) <= set(st.expenditure)
    other = sum(h["weight"] * h["baseline"]["other_benefits"] for h in rows) / 1e6
    rules = sum(h["weight"] * (h["baseline"]["benefits"] - h["baseline"]["other_benefits"] - h["baseline"]["family_allowance"])
                for h in rows if h["type"].startswith("assistance")) / 1e6
    assert other + rules == pytest.approx(sum(st.expenditure[k] for k in lines), rel=0.01)
    family = sum(h["weight"] * h["baseline"]["family_allowance"] for h in rows) / 1e6
    amounts = p["family.allowance_per_child"]
    entitled = sum((float(st.pop[:18, :, r].sum()) + p["family.student_share"] * float(st.pop[18:25, :, r].sum())) * amounts[name]
                   for r, name in enumerate(REGIONS)) / 1e6
    assert family == pytest.approx(entitled, rel=0.01)
    assert 0.75 * st.expenditure["family_allowances"] < family <= st.expenditure["family_allowances"]
    # all cash benefits of the households against the cash benefit lines of the budget
    # (but for the pensions of people in institutions and temporary unemployment, paid to people in work)
    benefits = sum(h["weight"] * h["baseline"]["benefits"] for h in rows) / 1e6
    budget = sum(v for k, v in st.expenditure.items() if k.startswith("pension_") or k in (
        "unemployment", "rcc_swt", "invalidity", "disability_allowances", "sickness_primary", "family_allowances",
        "social_assistance", "labour_market_benefits"))
    pensions = sum(v for k, v in st.expenditure.items() if k.startswith("pension_") and k != "pension_grapa")
    outside = p["dist.pensioner_institution_share"] * pensions + st.expenditure["unemployment"] - st.extra["expenditure"]["unemployment_full"]
    assert benefits == pytest.approx(budget - outside, rel=0.05)
    paid = sum(h["weight"] * (h["baseline"]["benefits"] - h["baseline"]["other_benefits"] - h["baseline"]["family_allowance"])
               for h in rows if h["type"].startswith("pensioner")) / 1e6
    assert 0.9 * (1 - p["dist.pensioner_institution_share"]) * pensions < paid
    for h in rows:
        b = h["baseline"]
        assert b["other_benefits"] >= 0 and 0 <= b["other_benefits_tax"] <= 0.6 * b["other_benefits"] + 1e-9
        if h["type"] == "single":
            assert 0 < b["other_benefits"] < 2500           # sickness and career-break benefits per worker


def test_within_group_dispersion():
    """Spreading the people of a mass point around its income: same mean, a point just above the
    line holds people below it, no spread gives the plain count."""
    y, mass = np.array([10_000.0, 20_000.0, 40_000.0]), np.array([1.0, 2.0, 1.0])
    assert list(analysis.share_below(y, 15_000.0)) == [1.0, 0.0, 0.0]
    soft = analysis.share_below(y, 19_000.0, 0.2)
    assert soft[0] > 0.99 and 0.3 < soft[1] < 0.5 and soft[2] < 1e-3
    mixed = analysis.share_below(y, 19_000.0, np.array([0.0, 0.2, 0.0]))
    assert list(mixed[[0, 2]]) == [1.0, 0.0] and mixed[1] == pytest.approx(soft[1])
    ys, ms = analysis.spread(y, mass, 0.2)
    assert ms.sum() == pytest.approx(mass.sum()) and (ys * ms).sum() == pytest.approx((y * mass).sum())
    assert analysis.gini(ys, ms) > analysis.gini(y, mass)
    same = analysis.spread(y, mass, 0.0)
    assert same[0] is not None and list(same[0]) == list(y) and list(same[1]) == list(mass)
    rate, line = analysis.poverty_rate(y, mass, 0.6)
    assert (rate, line) == (0.25, 12_000.0)


def test_reference_tables_are_complete():
    """The observed values the set is compared with were fetched (Eurostat EU-SILC 2025)."""
    ref = analysis.reference_indicators()
    for spec in (analysis.POVERTY_HOUSEHOLD_TYPES, analysis.POVERTY_ACTIVITY_STATUSES):
        for _, row in spec.values():
            assert row in ref and ref[row]["year"] == 2025 and 0 < ref[row]["value"] < 100, row
    for row in ("poverty_rate_line40.age.TOTAL", "poverty_rate_line50.age.TOTAL", "poverty_rate_line70.age.TOTAL",
                "inwork_poverty_rate.household.A1", "inwork_poverty_rate.activity.NSAL", "poverty_rate.work_intensity.WI0-02",
                "median_equivalised_income_eur.activity.RET", "population_share.household.A1", "poverty_rate.tenure.RENT"):
        assert row in ref, row
    assert ref["poverty_rate.household.TOTAL"]["value"] == ref["poverty_rate"]["value"] == 10.9


def test_short_list_is_part_of_the_full_set(runner):
    out = runner(year=2024)
    # the short list is part of the full one
    ids = {h["id"] for h in out["households_all"]}
    assert all(h["id"] in ids and h["representative"] for h in out["households"])
    assert 20 <= len(out["households"]) <= 30 and len(out["households_all"]) == len(ids)


def test_broad_reform_costs_the_same_for_households_and_budget(pset, base, runner):
    """A reform that reaches every taxpayer alike (tax-free allowance + 1,000 EUR) costs the same,
    within 15 %, summed over the stylised households and in the aggregate income tax."""
    changes = (Change("pit.basic_allowance", 1000, 2027, op="add"), Change("beh.enabled", 0, 2025))
    out = runner(*changes)
    sc = Scenario(name="t", measures=[Measure(name="m", changes=list(changes))])
    res = engine.run(sc, end=END, baseline=base, params=pset)
    households = sum(h["weight"] * h["delta"]["by_cause"]["direct_taxes"] for h in out["households_all"]) / 1e6
    budget = -sum(res.at(END).revenue[k] - base.at(END).revenue[k] for k in PIT)
    assert 1_500 < budget < 3_200
    assert households == pytest.approx(budget, rel=0.15)


def test_aggregate_income_tax_needs_no_large_calibration(pset, base):
    """Population of tax units + household rules + tax base of the fiscal statistics: the
    residual calibration of the three income-tax lines is within 0.85-1.15 (it was 0.73-0.87),
    and the regions' own taxes are published and add up."""
    st = base.at(2024)
    info = st.extra["revenue"]
    assert 0.95 < info["calib"]["pit_federal"] < 1.10
    assert 0.90 < info["calib"]["pit_regional"] < 1.05
    assert 0.85 < info["calib"]["pit_local"] < 1.00       # ESA records municipal surcharges with a lag
    pit = info["pit"]
    assert 0.85 < pit["share"]["wages"] < 1.0 and 0.7 < pit["share"]["self_employed"] < 0.95
    assert np.all(np.abs(pit["tax_rel"] - 1.0) < 0.08)        # regional distributions of the tax returns
    assert 1.0 < pit["persons"] < 1.1
    for year in (2024, END):
        s = base.at(year)
        for line in ("pit_federal", "pit_regional", "pit_local", "vat", "property_tax", "ssc_employee"):
            assert sum(s.regional[f"revenue.{line}.{r}"] for r in REGIONS) == pytest.approx(s.revenue[line], rel=1e-9)
        fed = np.array([s.regional[f"revenue.pit_federal.{r}"] for r in REGIONS])
        assert fed / fed.sum() == pytest.approx([0.642, 0.269, 0.088], abs=0.02)      # Statbel, income 2024
        assert all(s.regional[f"revenue_total.{r}"] > 0 and s.regional[f"federal_taxes_by_residence.{r}"] > 0 for r in REGIONS)


# ------------------------------------------------------------------ household rules

def test_single_without_children_equals_revenue_calculator(pset, base):
    p = param_view(pset, base, 2027)
    g = np.linspace(0, 200000, 41)
    for kind in rcalc.KINDS:
        t = rcalc.tax_on_income(g, p, kind)
        r = net_income(p, [(kind, g)])
        assert np.array_equal(r["net"], t["net"])
        assert np.array_equal(r["income_tax"], t["pit"])
        assert np.array_equal(r["employer_cost"], t["employer_cost"])


def test_level_split_matches_revenue_calculator(pset, base):
    p = param_view(pset, base, 2027)
    t = rcalc.tax_on_income(np.array([60000.0]), p, "emp_private")
    f, r, m = calc.level_shares(p)
    state = t["pit_federal"][0] / f
    assert t["pit_regional"][0] == pytest.approx(r * state)
    assert t["pit_local"][0] == pytest.approx(m * state)


def test_standard_single_worker_agrees_with_tax_on_income(pset, base, runner):
    """A single worker of an income group: the payslip of the group's gross wage - scaled to the
    dispersion of the tax returns (`dist.fiscal_dispersion`) - taxed on the taxable income of
    the group in the fiscal distribution, plus the items of the survey income concept (capital
    income and its withholding tax, recurrent property tax)."""
    out = runner()
    p = param_view(pset, base, END)
    w = levels_of(base, p, END)
    for level in ("p25", "median", "p90"):
        h = by_id(out)[f"single_{level}"]
        ratio = w["ratio"][level]
        k = (ratio / w["ratio"]["mean.emp"]) ** p["dist.fiscal_dispersion"]
        gross = w["levels"][level] * k
        t = rcalc.tax_on_income(gross, p, "emp_private")
        r = net_income(p, [("emp_private", gross, ratio / k)])
        assert 0.6 < ratio < 1.1 and 0.6 < k < 1.3
        b = h["baseline"]
        extra_tax = b["capital_tax"] + b["property_tax"] + b["other_benefits_tax"]
        extra_income = b["capital_income"] + b["other_benefits"]      # (sickness and career-break benefits of the budget)
        assert b["gross"] - extra_income == pytest.approx(float(t["gross"][0]), abs=0.03)
        assert b["net"] - extra_income + extra_tax == pytest.approx(float(r["net"][0]), abs=0.04)
        assert b["income_tax"] - extra_tax == pytest.approx(float(r["income_tax"][0]), abs=0.04)
        assert 0 < b["other_benefits"] < 0.05 * b["net"] and 0 < b["other_benefits_tax"] < 0.6 * b["other_benefits"]
        assert 0 < b["capital_income"] < 0.1 * b["net"] and 0 < b["property_tax"] < 0.05 * b["net"]
        assert float(r["taxable_income"][0]) == pytest.approx(ratio / k * float(t["taxable_income"][0]))
        assert float(r["ssc_employee"][0]) <= float(t["ssc_employee"][0]) + 1e-6     # only the special contribution differs


def test_marital_quotient(pset, base):
    p = param_view(pset, base, 2024)
    g = 60000.0
    single = net_income(p, [("emp_private", g)])
    couple = net_income(p, [("emp_private", g), ("none", 0.0)])
    taxable = float(single["taxable_income"][0])
    expected = min(p["pit.marital_quotient_rate"] * taxable, p["pit.marital_quotient_max"]["work"])
    assert float(couple["marital_quotient"][0]) == pytest.approx(expected)
    assert float(couple["income_tax"][0]) < float(single["income_tax"][0])
    # (the special contribution has a household scale of its own: a few hundred euros at most)
    assert float(couple["ssc_employee"][0]) == pytest.approx(float(single["ssc_employee"][0]), abs=400)
    # two similar incomes: no transfer, two singles
    two = net_income(p, [("emp_private", g), ("emp_private", g)])
    assert float(two["marital_quotient"][0]) == 0.0
    # (but for the special contribution, which has one ceiling for the household until 2027)
    assert float(two["net"][0]) == pytest.approx(2 * float(single["net"][0]), abs=800)
    assert float(two["net"][0]) >= 2 * float(single["net"][0]) - 1e-6
    # the legislated cut of the maximum makes the quotient worth less in 2030 than in 2024
    p30 = param_view(pset, base, END)
    assert p30["pit.marital_quotient_max"]["work"] < p["pit.marital_quotient_max"]["work"]


def test_child_supplements_and_refundable_credit(pset, base):
    p = param_view(pset, base, 2024)
    g = 60000.0
    k = sum(calc.level_shares(p))
    r0 = net_income(p, [("emp_private", g)])
    r1 = net_income(p, [("emp_private", g), ("emp_private", g)], children=1)
    two_singles = 2 * float(r0["income_tax"][0])
    rate0 = p["pit.bracket_rates"][0]
    assert p["pit.basic_allowance"] + calc.child_supplement(1, p) < p["pit.bracket_thresholds"][0]
    # supplements are valued on the special allowance scale: 25 % up to its first limit, 30 % above
    a_thr, a_rates = rcalc.allowance_scale(p)
    value = lambda extra: float(rcalc.progressive_tax(p["pit.basic_allowance"] + extra, a_thr, a_rates)
                                - rcalc.progressive_tax(p["pit.basic_allowance"], a_thr, a_rates))
    assert value(calc.child_supplement(1, p)) > rate0 * calc.child_supplement(1, p)
    assert two_singles - float(r1["income_tax"][0]) == pytest.approx(value(calc.child_supplement(1, p)) * k)
    # single parent: child supplement + single-parent supplement
    sp = net_income(p, [("emp_private", g)], children=1)
    extra = calc.child_supplement(1, p) + p["pit.single_parent_allowance"]
    assert float(r0["income_tax"][0]) - float(sp["income_tax"][0]) == pytest.approx(value(extra) * k)
    # no taxable income: refundable credit, capped per child
    poor = net_income(p, [("assistance", 15000.0)], children=2)
    cap = 2 * p["pit.child_credit_max_per_child"]
    assert float(poor["child_credit"][0]) == pytest.approx(min(rate0 * calc.child_supplement(2, p), cap))
    assert float(poor["income_tax"][0]) == pytest.approx(-float(poor["child_credit"][0]))
    # more children: larger supplements, beyond the table too
    s = [calc.child_supplement(n, p) for n in range(0, 7)]
    assert s[0] == 0 and all(b > a for a, b in zip(s, s[1:]))
    assert s[6] - s[5] == pytest.approx(p["pit.child_allowance_extra_per_child"])


def test_family_allowance_by_region(pset, base):
    p = param_view(pset, base, 2024)
    amounts = p["family.allowance_per_child"]
    for region in ("FLA", "WAL", "BRU"):
        # above the income ceiling of the social supplement: the average less its average cost;
        # under the ceiling: the supplement on top
        sup, share = p["dist.family_social_supplement"], p["dist.family_supplement_child_share"]
        r = net_income(p, [("emp_private", 40000.0)], children=2, region=region)
        assert float(r["family_allowance"][0]) == pytest.approx(2 * (amounts[region] - share * sup["amount"]))
        low = net_income(p, [("emp_private", 30000.0)], children=2, region=region)
        assert float(low["family_allowance"][0]) - float(r["family_allowance"][0]) == pytest.approx(2 * sup["amount"])
    env = hh_mod.benefit_env(base.at(2024))
    avg = calc.family_allowance_per_child(p, env)
    assert min(amounts.values()) < avg < max(amounts.values())


# ------------------------------------------------------------------ scenario comparison

def test_empty_scenario_changes_nothing(runner):
    for year in (2024, 2027, END):
        out = runner(year=year)
        for h in out["households_all"]:
            assert h["delta"]["eur"] == 0.0 and h["delta"]["pct"] == 0.0 and h["delta"]["nominal_eur"] == 0.0
            assert all(v == 0.0 for v in h["delta"]["by_cause"].values())
            assert h["baseline"] == h["scenario"]
        for d in out["deciles"]:
            assert d["delta_eur"] == 0.0 and d["delta_pct"] == 0.0
            assert d["gainers_share"] == 0.0 and d["losers_share"] == 0.0
        for ind in out["indicators"].values():
            assert ind["baseline"] == ind["scenario"] and ind["delta"] == 0.0


def test_output_is_consistent(runner):
    out = runner(Change(param="vat.rate_standard", value=0.22, start=2027))
    assert "not a microsimulation" in out["meta"]["method"]
    assert len(out["deciles"]) == 10
    persons = [d["persons"] for d in out["deciles"]]
    assert max(persons) - min(persons) <= 1
    assert sum(persons) == pytest.approx(out["meta"]["persons_total"], abs=10)
    assert sum(d["households"] for d in out["deciles"]) == pytest.approx(out["meta"]["households_total"], abs=10)
    eq = [d["baseline_equivalised_income"] for d in out["deciles"]]
    assert eq == sorted(eq)
    for h in out["households_all"]:
        b = h["baseline"]
        assert b["net"] == pytest.approx(b["gross"] + b["family_allowance"] - b["ssc_employee"] - b["income_tax"], abs=0.03)
        assert b["net_after_indirect"] == pytest.approx(b["net"] - b["indirect_taxes"], abs=0.02)
        assert 0 < b["indirect_taxes"] < 0.25 * b["net"]
        assert sum(h["delta"]["by_cause"].values()) == pytest.approx(h["delta"]["eur"], abs=0.03)
        assert h["delta"]["eur"] == pytest.approx(h["scenario"]["net_real"] - b["net"], abs=0.02)
        for k in ("average", "marginal", "participation"):
            v = h["rates"][k]
            assert v is None or -0.5 < v < 1.5
    ind = out["indicators"]
    assert 0.05 < ind["gini"]["baseline"] < 0.45 and 1.5 < ind["s80s20"]["baseline"] < 8
    assert 0.0 <= ind["poverty_rate"]["baseline"] < 0.4
    assert ind["gini"]["observed"]["value"] == pytest.approx(0.234, abs=0.03)       # EU-SILC 2025, incomes of 2024


def test_higher_tax_free_allowance_helps_workers_by_the_expected_amount(pset, base, runner):
    change = Change(param="pit.basic_allowance", value=2000, op="add", start=2027)
    out = runner(change)
    hh = by_id(out)
    sc = Scenario(name="s", measures=[Measure(name="m", changes=[change])])
    pb = param_view(pset, base, END)
    ps = pset.with_scenario(sc).at(END, analysis.index_fn(base))      # the rule change alone
    a0, a1 = pb["pit.basic_allowance"], ps["pit.basic_allowance"]
    assert a1 - a0 >= 2000 - 1e-6                                      # 2027 euros
    thr, rates = pb["pit.bracket_thresholds"], pb["pit.bracket_rates"]
    expected = float(rcalc.progressive_tax(a1, thr, rates) - rcalc.progressive_tax(a0, thr, rates)) * sum(calc.level_shares(pb))
    # (the scenario's own price index is a little lower, and the brackets with it: a few percent less)
    for i in ("single_median", "single_p75", "single_p90", "single_top", "single_self_employed"):
        assert hh[i]["delta"]["by_cause"]["direct_taxes"] == pytest.approx(expected, rel=0.06)
        # (at the very top the second-round fall of gross wages outweighs a lump-sum tax cut)
        assert hh[i]["delta"]["eur"] > 0 or i == "single_top"
    # two earners gain twice, people without taxable income gain nothing from the rule
    # (more when the partner's income is below the allowance: the unused part moves to the other
    # partner, where the special scale of the allowance values it at 30 or 40 %)
    assert 1.76 * expected < hh["couple_2earners"]["delta"]["by_cause"]["direct_taxes"] < 2.6 * expected
    for i in ("assistance_single", "assistance_family", "pensioner_single_low"):
        # (only the second-round change of the property tax they bear)
        assert hh[i]["delta"]["by_cause"]["direct_taxes"] == pytest.approx(0.0, abs=5.0)
    # households without taxable income only feel the (small, negative) second-round effects
    assert out["summary"]["losers_share"] < 0.25 and out["summary"]["gainers_share"] > 0.7
    assert all(d["delta_eur"] > 0 for d in out["deciles"])


def test_vat_rise_weighs_more_on_low_incomes(runner):
    out = runner(Change(param="vat.rate_standard", value=0.22, start=2027), year=2027)
    assert out["meta"]["prices"]["cpi_scenario"] > out["meta"]["prices"]["cpi_baseline"]
    for h in out["households_all"]:
        assert h["delta"]["by_cause"]["indirect_taxes_prices"] < 0
        assert h["scenario"]["indirect_taxes"] > h["baseline"]["indirect_taxes"]
    burden = [100 * d["by_cause"]["indirect_taxes_prices"] / d["baseline_income"] for d in out["deciles"]]
    assert all(b < 0 for b in burden)
    assert burden[0] < burden[4] < burden[9]                       # % of income: larger loss at the bottom
    assert np.mean(burden[:3]) < 1.3 * np.mean(burden[-3:])        # at least 30 % heavier
    # incomes are indexed only partly in the first year: everyone loses purchasing power, the
    # bottom half more than the top
    pct = [d["delta_pct"] for d in out["deciles"]]
    assert np.mean(pct[:5]) < np.mean(pct[5:]) and np.mean(pct[:5]) < 0


def test_family_allowance_increase_only_helps_households_with_children(runner):
    out0 = runner()
    out = runner(Change(param="family.allowance_per_child", value=1.10, op="mul", start=2027))
    base_rows = by_id(out0)
    for h in out["households_all"]:
        ben = h["delta"]["by_cause"]["benefits"]
        if h["children"] > 0:
            # the parameter is the average amount per child: the social supplement on top of it
            # (or its average cost, for families above the income ceiling) does not move
            expected = 0.10 * base_rows[h["id"]]["baseline"]["family_allowance"]
            assert 0.75 * expected < ben < 1.12 * expected and h["delta"]["eur"] > 0.6 * expected
        else:
            assert ben == pytest.approx(0.0, abs=25.0)                 # second-round indexation only
            assert abs(h["delta"]["pct"]) < 0.1                    # only second-round macro effects
    with_children = [h for h in out["households_all"] if h["children"] > 0]
    assert with_children and all(h["delta"]["eur"] > 0 for h in with_children)


def test_pension_and_benefit_levers_reach_the_right_households(runner):
    out = runner(Change(param="assist.generosity_factor", value=1.10, start=2027))
    hh = by_id(out)
    for i in ("assistance_single", "assistance_family"):
        assert hh[i]["delta"]["by_cause"]["benefits"] > 0.05 * hh[i]["baseline"]["net"]
    assert hh["pensioner_single_avg"]["delta"]["by_cause"]["benefits"] == pytest.approx(0.0, abs=3.0)
    assert out["indicators"]["gini"]["scenario"] < out["indicators"]["gini"]["baseline"]


def test_distribution_is_fast(pset, base, runner):
    runner(Change(param="vat.rate_standard", value=0.22, start=2027))      # simulation cached
    t = time.perf_counter()
    runner(Change(param="vat.rate_standard", value=0.22, start=2027), year=2029)
    assert time.perf_counter() - t < 0.5


# ------------------------------------------------------------------ indicators

def test_inequality_statistics():
    y, w = np.array([10.0, 10.0, 10.0]), np.array([1.0, 2.0, 3.0])
    assert analysis.gini(y, w) == 0.0
    assert analysis.gini(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)
    m = analysis.quantile_membership(np.array([3.0, 1.0, 2.0]), np.array([2.0, 2.0, 6.0]), 5)
    assert np.allclose(m.sum(axis=1), 2.0) and np.allclose(m.sum(axis=0), [2.0, 2.0, 6.0])
    assert m[0, 1] == 2.0 and m[4, 0] == 2.0
    assert analysis.s80s20(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.ones(5)) == pytest.approx(5.0)
    rate, line = analysis.poverty_rate(np.array([1.0, 10.0, 10.0, 12.0]), np.ones(4), 0.6)
    assert line == pytest.approx(6.0) and rate == pytest.approx(0.25)


# ------------------------------------------------------------------ payslip

def test_payslip_breakdown(pset, base):
    p = param_view(pset, base, 2024)
    g = np.linspace(0, 150000, 31)
    r = payslip(p, g, "emp_private", "single")
    t = rcalc.tax_on_income(g, p, "emp_private")
    assert np.array_equal(r["net"], t["net"]) and np.array_equal(r["employer_cost"], t["employer_cost"])
    assert np.allclose(r["tax_federal"] + r["tax_regional"] + r["tax_local"], t["pit"])
    assert np.all((r["marginal_rate"] > -0.05) & (r["marginal_rate"] < 1.0))
    assert r["marginal_rate"][-1] == pytest.approx(1 - (1 - p["ssc.employee_rate"]) * (1 - p["pit.bracket_rates"][-1] * sum(calc.level_shares(p))), abs=0.02)
    fam = payslip(p, g, "emp_private", {"adults": 2, "children": 2, "region": "BRU"})
    assert np.all(fam["net"] > r["net"])
    assert np.allclose(fam["net"], fam["gross"] - fam["ssc_employee"] - fam["income_tax"] + fam["family_allowance"])
    sup, share = p["dist.family_social_supplement"], p["dist.family_supplement_child_share"]
    assert fam["family_allowance"][0] == pytest.approx(2 * (p["family.allowance_per_child"]["BRU"] + (1 - share) * sup["amount"]))
    assert fam["family_allowance"][-1] == pytest.approx(2 * (p["family.allowance_per_child"]["BRU"] - share * sup["amount"]))
    two = payslip(p, g, "emp_private", "couple_two_earners", median_wage=50000.0)
    assert two["household"]["partner_gross"] == 50000.0
    assert np.all(two["net"] > two["net_earner_only"])
    with pytest.raises(ValueError):
        payslip(p, g, "emp_private", "no_such_household")
    with pytest.raises(ValueError):
        payslip(p, g, "wizard", "single")
