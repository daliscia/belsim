"""Regression guards on the baseline projection (spending by scheme, labour market, entities).

Companion of test_validation_tolerances.py for the long-run references: pensions by scheme,
acute and long-term care, employment, balances by entity, debt. Tolerances are wider than the
current gaps; references and sources are in data/reference/validation.yaml, except the accounts
of the federated entities and of each level of government (Federal Planning Bureau outlook of
June 2026, Tableaux 13-16 and D.2-D.6, quoted here and in params/expenditure.yaml) which that
file does not hold yet.
"""
import pytest

from belsim import validation as V
from belsim.core import engine
from belsim.core.dims import REGIONS
from belsim.core.params import Change, Measure, Scenario


@pytest.fixture(scope="module")
def base():
    return engine.run_baseline()


@pytest.fixture(scope="module")
def refs():
    return {e["id"]: e for e in V.load()["baseline"]}


def gap(base, refs, id_, when):
    entry = refs[id_]
    values = {str(k): float(v) for k, v in entry["values"].items()}
    return V.baseline_value(base, when, entry["model"]) - values[str(when)]


# (reference id, year or period, absolute tolerance in the unit of the reference)
LEVELS = [
    # pensions by scheme, % of GDP (Study Committee on Ageing 2026)
    *[("sca2026_pensions", y, 0.2) for y in (2025, 2030, 2031, 2040, 2050)], ("sca2026_pensions", 2070, 0.4),
    ("sca2026_pensions_change", "2025-2031", 0.15),
    *[("sca2026_pensions_employees", y, 0.4) for y in (2025, 2030, 2040, 2050, 2070)],
    *[("sca2026_pensions_self_employed", y, 0.15) for y in (2025, 2030, 2040, 2050, 2070)],
    *[("sca2026_pensions_public", y, 0.3) for y in (2025, 2030, 2040, 2050, 2070)],
    ("sca2026_pensions_change", "2025-2070", 0.4),
    # health: acute care (the model line is 0.2 point wider in 2025), long-term care, total
    *[("sca2026_health_acute", y, 0.4) for y in (2025, 2030, 2040, 2050, 2070)],
    *[("sca2026_long_term_care", y, 0.25) for y in (2025, 2030, 2040, 2050, 2070)],
    *[("sca2026_health_total", y, 0.4) for y in (2025, 2030, 2031, 2040, 2050, 2070)],
    ("sca2026_health_acute_change", "2031-2050", 0.25), ("sca2026_health_acute_change", "2050-2070", 0.3),
    ("sca2026_long_term_care_change", "2025-2070", 0.25),
    # other benefits
    *[("sca2026_incapacity", y, 0.3) for y in (2025, 2030, 2040, 2050, 2070)],
    *[("sca2026_unemployment", y, 0.2) for y in (2025, 2030, 2040, 2050, 2070)],
    # labour market
    *[("fpb2026_employment_rate_20_64", y, 1.3) for y in range(2025, 2032)],            # points
    ("sca2026_employment_growth", "2025-2031", 0.15), ("sca2026_employment_growth", "2025-2070", 0.15),
    ("sca2026_gdp_growth", "2025-2031", 0.2), ("sca2026_gdp_growth", "2025-2070", 0.2),
    *[("fpb2026_gdp_growth", y, 0.45) for y in range(2025, 2032)],
    # population and migration
    *[("sca2026_net_migration", y, 3.0) for y in (2031, 2050, 2070)],                   # thousand
    *[("pop2026_18_66", y, 110) for y in (2025, 2030, 2040, 2050, 2070)],
    # entities and debt, % of GDP (Planning Bureau outlook of June 2026)
    # Entity I: gaps 0.15 / -0.27 / 0.17 / 0.04 / 0.12 / 0.26 in 2025-2030, 0.45 in 2031 - the
    # general-government balance is 0.35 point better than the Bureau's in 2031 (revenue kept at
    # the observed 2025 level, expenditure 0.2 point lower) and that sits at the federal level
    *[("fpb2026_balance_entity1", y, 0.35) for y in range(2025, 2031)],
    ("fpb2026_balance_entity1", 2031, 0.55),
    *[("fpb2026_balance_regions", y, 0.3) for y in range(2025, 2032)],
    *[("fpb2026_balance_local", y, 0.15) for y in (2025, 2027, 2029, 2031)],
    *[("fpb2026_debt", y, 2.0) for y in range(2025, 2032)],
    *[("fpb2026_expenditure", y, 0.45) for y in range(2025, 2032)],
    *[("fpb2026_employment", y, 25) for y in range(2025, 2032)],                        # thousand
]


@pytest.mark.parametrize("id_,when,tol", LEVELS, ids=lambda v: str(v))
def test_baseline_tracks_official_long_run_projection(base, refs, id_, when, tol):
    g = gap(base, refs, id_, when)
    assert abs(g) <= tol, f"{id_} {when}: gap {g:+.3f} (tolerance {tol})"


def test_base_year_lines_are_the_2024_accounts(base):
    from belsim.fiscal.expenditure import data
    e = base.at(2024).expenditure
    for line, row in data.lines_2024().items():
        assert e[line] == pytest.approx(row["value"], rel=1e-9), line
    assert sum(e.values()) == pytest.approx(data.scalars()["te_2024"], rel=1e-9)


def test_expenditure_2025_is_close_to_the_outturn(base):
    from belsim.fiscal.expenditure import data
    assert sum(base.at(2025).expenditure.values()) == pytest.approx(data.scalars()["te_2025"], rel=0.01)
    assert base.at(2025).fiscal["debt"] == pytest.approx(data.scalars()["debt_2025"], rel=0.01)


def test_employment_gains_follow_the_committee(base, refs):
    """Change of the employment rate (18-66) and of employment since 2025: the levels differ by
    concept (administrative data against the model's survey-based 2024), the changes compare."""
    rate, emp = refs["sca2026_employment_rate_18_66"], refs["sca2026_employment"]
    for year, tol_rate, tol_emp in ((2030, 0.9, 60), (2040, 1.6, 110), (2050, 1.5, 110), (2070, 1.4, 150)):
        d_rate = (V.baseline_value(base, year, rate["model"]) - V.baseline_value(base, 2025, rate["model"])
                  - (rate["values"][year] - rate["values"][2025]))
        d_emp = (V.baseline_value(base, year, emp["model"]) - V.baseline_value(base, 2025, emp["model"])
                 - (emp["values"][year] - emp["values"][2025]))
        assert abs(d_rate) <= tol_rate, (year, d_rate)
        assert abs(d_emp) <= tol_emp, (year, d_emp)


# Federal Planning Bureau, Perspectives economiques 2026-2031 (June 2026): GDP (Tableau A.2, EUR bn),
# balance (Tableau D.4) and expenditure (Tableaux 13-16) of the federated entities, EUR bn
FPB_GDP = {2024: 620.272, 2025: 642.012, 2027: 687.370, 2029: 733.587, 2031: 778.846}
FPB_ENTITY_BALANCE = {
    "flemish_community": {2025: -3.9, 2027: -0.9, 2029: -2.0, 2031: -1.4},
    "french_community": {2025: -1.5, 2027: -1.4, 2029: -1.2, 2031: -1.3},
    "walloon_region": {2025: -2.7, 2027: -1.6, 2029: -1.6, 2031: -1.6},
    "brussels_region": {2025: -0.9, 2027: -0.9, 2029: -0.8, 2031: -0.9},
}
FPB_ENTITY_EXPENDITURE = {
    "flemish_community": {2024: 73.2, 2025: 75.1, 2027: 77.7, 2029: 83.1, 2031: 86.9},
    "french_community": {2024: 26.5, 2025: 27.2, 2027: 28.9, 2029: 30.1, 2031: 31.6},
    "walloon_region": {2024: 20.1, 2025: 21.2, 2027: 21.0, 2029: 21.8, 2031: 22.7},
    "brussels_region": {2024: 9.3, 2025: 9.2, 2027: 9.5, 2029: 9.7, 2031: 10.1},
}
# expenditure of each level of government, EUR bn (Tableaux D.2, D.6, D.3, D.5; incl. transfers paid)
FPB_LEVEL_EXPENDITURE = {
    "social_security": {2024: 135.438, 2027: 151.531, 2029: 162.100, 2031: 174.316},
    "regions_communities": {2024: 124.997, 2027: 132.615, 2029: 139.772, 2031: 146.175},
    "local": {2024: 44.397, 2027: 47.516, 2029: 50.154, 2031: 52.452},
}


@pytest.mark.parametrize("entity", list(FPB_ENTITY_BALANCE))
def test_federated_entities_track_the_official_outlook(base, entity):
    """Balance within 0.3 point of GDP and change of expenditure since 2024 within 0.2 point: the
    paths are spending paths of the entity (entity.*_factor), not an allocation. (The balance of
    the Flemish Community in 2027 carries a one-off Financing Act settlement the model spreads.)"""
    for year, bn in FPB_ENTITY_BALANCE[entity].items():
        official = 100.0 * bn / FPB_GDP[year]
        model = base.at(year).fiscal[f"balance_pct_gdp.{entity}"]
        assert abs(model - official) <= 0.3, (entity, year, model, official)
    ratio = lambda y: 100.0 * base.at(y).fiscal[f"expenditure.{entity}"] / base.at(y).macro["gdp_nominal"]
    off = FPB_ENTITY_EXPENDITURE[entity]
    for year in (2025, 2027, 2029, 2031):
        d_off = 100.0 * (off[year] / FPB_GDP[year] - off[2024] / FPB_GDP[2024])
        assert abs(ratio(year) - ratio(2024) - d_off) <= 0.2, (entity, year, ratio(year) - ratio(2024), d_off)


@pytest.mark.parametrize("level", list(FPB_LEVEL_EXPENDITURE))
def test_expenditure_by_level_of_government_tracks_the_outlook(base, level):
    ratio = lambda y: 100.0 * base.at(y).fiscal[f"expenditure.{level}"] / base.at(y).macro["gdp_nominal"]
    off = FPB_LEVEL_EXPENDITURE[level]
    for year in (2027, 2029, 2031):
        d_off = 100.0 * (off[year] / FPB_GDP[year] - off[2024] / FPB_GDP[2024])
        assert abs(ratio(year) - ratio(2024) - d_off) <= 0.3, (level, year, ratio(year) - ratio(2024), d_off)


def test_entity_paths_are_spending_of_the_entity_that_decides_them(base):
    """Without the budget paths of the entities, expenditure and the deficit are higher; a path of
    one entity changes the lines, the total and that entity's account by the same amount."""
    names = ("staff_cost", "purchases", "subsidies", "current_transfers", "capital_transfers", "investment")
    one = {k: 1.0 for k in ("federal", "social_security", "flemish_community", "walloon_region",
                            "french_community", "brussels_region", "local")}
    scen = engine.run(Scenario(name="s", measures=[Measure("m", [
        *[Change(f"entity.{n}_factor", one, 2025) for n in names], Change("beh.enabled", 0, 2024)])]),
        end=2031, baseline=base)
    a, b = base.at(2029), scen.at(2029)
    g = a.macro["gdp_nominal"]
    assert 0.5 < 100 * (b.fiscal["expenditure_total"] - a.fiscal["expenditure_total"]) / g < 1.3
    assert b.fiscal["balance.regions_communities"] < a.fiscal["balance.regions_communities"] - 2500
    scen = engine.run(Scenario(name="s", measures=[Measure("m", [
        Change("entity.subsidies_factor", 0.9, 2027, op="mul", key="walloon_region"), Change("beh.enabled", 0, 2024)])]),
        end=2029, baseline=base)
    b = scen.at(2029)
    saved = a.fiscal["expenditure_total"] - b.fiscal["expenditure_total"]
    assert saved > 50
    assert a.fiscal["expenditure.walloon_region"] - b.fiscal["expenditure.walloon_region"] == pytest.approx(saved, rel=0.05)
    for e in ("flemish_community", "french_community", "brussels_region"):
        assert abs(a.fiscal[f"expenditure.{e}"] - b.fiscal[f"expenditure.{e}"]) < 0.05 * saved
    # a national lever on a line keeps its full effect on top of the paths
    scen = engine.run(Scenario(name="s", measures=[Measure("m", [
        Change("invest.public_pct_gdp", 0.5, 2028, op="add"), Change("beh.enabled", 0, 2024)])]), end=2029, baseline=base)
    b = scen.at(2029)
    assert b.expenditure["public_investment"] - a.expenditure["public_investment"] == pytest.approx(
        0.005 * b.macro["gdp_nominal"], rel=0.02)


def test_pensions_do_not_dip_when_the_legal_age_reaches_67(base):
    """Planning Bureau outlook, Tableau D.7: private-sector pensions 2024 = 100 -> 121.5 (2029),
    126.8 (2030), 131.6 (2031); Study Committee on Ageing: 11.5% of GDP in 2030 and 2031."""
    lines = ("pension_employees", "pension_self_employed")
    v = {y: sum(base.at(y).expenditure[k] for k in lines) for y in (2024, 2029, 2030, 2031)}
    for year, official in ((2029, 121.5), (2030, 126.8), (2031, 131.6)):
        assert 100 * v[year] / v[2024] == pytest.approx(official, abs=2.0)
    pens = lambda y: 100 * sum(x for k, x in base.at(y).expenditure.items() if k.startswith("pension_")) \
        / base.at(y).macro["gdp_nominal"]
    assert pens(2031) > pens(2029) - 0.08
    # the step from 66 to 67 does not move the 65-year-olds
    from belsim.core.dims import S
    share = lambda y, a: base.at(y).status[S["retired"]][a].sum() / base.at(y).pop[a].sum()
    assert share(2031, 65) == pytest.approx(share(2029, 65), abs=0.04)
    assert share(2031, 66) < share(2029, 66) - 0.2


def test_pension_reform_costing_follows_report_13299(base):
    """Bonus-malus with the retirement response: -228 / -463 / -677 / -641 / -794 mn in 2027-2031
    (2024 prices, table 2). The abolished capital bonus is a separate lever."""
    pens = lambda r, y: sum(v for k, v in r.at(y).expenditure.items() if k.startswith("pension_"))
    ref = engine.run(Scenario(name="r", measures=[Measure("m", [Change("pension.bonus_malus_per_year", 0.0, 2027)])]),
                     end=2031, baseline=base)
    d = {y: (pens(base, y) - pens(ref, y)) / base.at(y).macro["health_index"] for y in range(2027, 2032)}
    assert sum(d.values()) == pytest.approx(-2803, rel=0.25)
    for year, official in ((2027, -228), (2029, -677), (2031, -794)):
        assert d[year] == pytest.approx(official, rel=0.4), (year, d[year])
    # capital bonus of the law of April 2024 as it would have matured: about 0.1% of GDP
    bonus = engine.run(Scenario(name="b", measures=[Measure("m", [
        Change("pension.capital_bonus", 4600, 2029), Change("beh.enabled", 0, 2024)])]), end=2031, baseline=base)
    cost = 100 * (pens(bonus, 2031) - pens(base, 2031)) / base.at(2031).macro["gdp_nominal"]
    assert 0.06 < cost < 0.12


def test_health_norm_costing_against_its_own_baseline(base):
    """DC2024 (Open Vld 402): a 1.5% norm against need-driven growth of 3.3% a year (the PROMES
    projection of February 2024) saves 1 782 mn in 2027 and 3 253 mn in 2029 at 2024 prices."""
    def run(norm):
        return engine.run(Scenario(name="s", measures=[Measure("m", [
            Change("health.growth_norm", norm, 2025), Change("health.excess_cost_growth", 0.017, 2025),
            Change("beh.enabled", 0, 2024)])]), end=2029, baseline=base)
    ref, cut = run(0.033), run(0.015)
    saving = {y: (ref.at(y).expenditure["health_care"] - cut.at(y).expenditure["health_care"])
              / base.at(y).macro["health_index"] for y in (2027, 2029)}
    assert saving[2027] == pytest.approx(1782, rel=0.2)
    assert saving[2029] == pytest.approx(3253, rel=0.15)


def test_target_group_subsidies_move_regional_employment(base):
    scen = engine.run(Scenario(name="s", measures=[Measure("m", [
        Change("subsidy.target_group_factor", 0.5, 2027, key="WAL")])]), end=2032, baseline=base)
    a, b = base.at(2032), scen.at(2032)
    d = {r: b.regional[f"employment_residents.{r}"] - a.regional[f"employment_residents.{r}"] for r in REGIONS}
    assert -6000 < d["WAL"] < -1500 and abs(d["FLA"]) < 0.4 * abs(d["WAL"])
    assert b.expenditure["subsidy_wage"] < a.expenditure["subsidy_wage"] - 300


def test_financing_act_transfers_come_from_the_entity_rules(base):
    """The federal transfer is the sum of the mechanisms by entity; a region gaining tax base
    gains PIT allocation and loses solidarity (the others' shares fall)."""
    for year in (2024, 2029, 2040):
        f = base.at(year).fiscal
        by_entity = sum(f[f"federal_grants.{e}"] for e in FPB_ENTITY_BALANCE)
        assert by_entity == pytest.approx(f["transfer.federal_to_regions_communities"], rel=1e-9)
    f24, f29 = base.at(2024).fiscal, base.at(2029).fiscal
    g24, g29 = base.at(2024).macro["gdp_nominal"], base.at(2029).macro["gdp_nominal"]
    recv = lambda f: sum(f[f"transfer.{a}_to_regions_communities"] for a in ("federal", "social_security", "local"))
    # transfers received by the communities and regions: 10.5% of GDP in 2024, 10.3% in 2029 (FPB Tableau 12)
    assert 100 * recv(f24) / g24 == pytest.approx(10.5, abs=0.15)
    assert 100 * recv(f29) / g29 == pytest.approx(10.3, abs=0.25)
    scen = engine.run(Scenario(name="s", measures=[Measure("m", [
        Change("regional.productivity_growth_wedge", 0.005, 2027, key="WAL"), Change("beh.enabled", 0, 2024)])]),
        end=2040, baseline=base)
    a, b = base.at(2040), scen.at(2040)
    assert b.regional["gdp_nominal.WAL"] > a.regional["gdp_nominal.WAL"] * 1.03
    assert sum(b.regional[f"gdp_nominal.{r}"] for r in REGIONS) == pytest.approx(b.macro["gdp_nominal"], rel=1e-9)


def test_pension_age_65_costing(base):
    """DC2024 (PVDA-PTB 301): keeping the legal age at 65 costs EUR 2.2 bn of pensions in 2029 at
    2024 prices (EUR 1.7 bn in 2027)."""
    scen = engine.run(Scenario(name="s", measures=[Measure("m", [
        Change("labour.legal_pension_age", 65, 2025, end=2029), Change("beh.enabled", 0, 2024)])]),
        end=2029, baseline=base)
    pens = lambda r, y: sum(v for k, v in r.at(y).expenditure.items() if k.startswith("pension_"))
    cost = {y: (pens(scen, y) - pens(base, y)) / base.at(y).macro["cpi"] for y in (2027, 2029)}
    assert cost[2029] == pytest.approx(2232, rel=0.3)
    assert cost[2027] == pytest.approx(1663, rel=0.45)
    # people who neither work nor may retire yet wait on another benefit: fewer pensioners than workers lost
    from belsim.core.dims import S
    more_retired = scen.at(2029).status[S["retired"]].sum() - base.at(2029).status[S["retired"]].sum()
    fewer_workers = base.at(2029).labour["employment"] - scen.at(2029).labour["employment"]
    assert more_retired > fewer_workers > 10_000
