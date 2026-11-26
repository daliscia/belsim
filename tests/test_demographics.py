"""Demographics module: base-year data, accounting identities, validation against EUROPOP2023."""
import time

import numpy as np
import pandas as pd
import pytest

from belsim import demographics as demo
from belsim.core.dims import BASE_YEAR, END_YEAR, N_AGE, N_REGION, N_SEX, REGIONS
from belsim.core.params import Change, Measure, Scenario
from belsim.core.state import YearState

REQUIRED = ["births", "deaths", "net_migration", "immigration", "emigration", "tfr",
            "life_expectancy_M", "life_expectancy_F", "old_age_dependency"]


def scenario(*changes: Change) -> Scenario:
    return Scenario(name="test", measures=[Measure(name="m", changes=list(changes))])


@pytest.fixture(scope="module")
def base():
    return demo.project()


@pytest.fixture(scope="module")
def observed():
    return pd.read_csv(demo.DATA / "demographics_observed.csv").set_index(["year", "region"])


@pytest.fixture(scope="module")
def europop():
    return demo.reference()


def test_base_year_is_observed_population(base, observed):
    pop = base.at(BASE_YEAR).pop
    assert pop.shape == (N_AGE, N_SEX, N_REGION)
    assert pop.sum() == pytest.approx(observed.loc[(2024, "BE"), "population_1jan"], abs=0.5)
    assert 11.7e6 < pop.sum() < 11.9e6
    for i, r in enumerate(REGIONS):
        assert pop[:, :, i].sum() == pytest.approx(observed.loc[(2024, r), "population_1jan"], abs=0.5)


def test_base_year_flows_match_observed(base, observed):
    d = base.at(BASE_YEAR).demo
    o = observed.loc[(2024, "BE")]
    assert d["births"] == pytest.approx(o["births"], rel=1e-6)
    assert d["deaths"] == pytest.approx(o["deaths"], rel=1e-4)
    assert d["net_migration"] == pytest.approx(o["net_migration_incl_adjustment"], rel=1e-9)
    assert d["tfr"] == pytest.approx(1.444, abs=0.005)            # Eurostat demo_r_frate2, BE 2024
    assert d["life_expectancy_M"] == pytest.approx(80.3, abs=0.3)  # Eurostat demo_mlifetable 2024
    assert d["life_expectancy_F"] == pytest.approx(84.4, abs=0.3)


def test_first_projected_years_close_to_observed(base, observed):
    # 1 January 2025 and 2026 are already published; the model must land on them.
    for year in (2025, 2026):
        assert base.at(year).pop.sum() == pytest.approx(observed.loc[(year, "BE"), "population_1jan"], rel=1e-3)
    for i, r in enumerate(REGIONS):
        assert base.at(2025).pop[:, :, i].sum() == pytest.approx(
            observed.loc[(2025, r), "population_1jan"], rel=3e-3)


def test_required_keys_every_year(base):
    assert base.years == list(range(BASE_YEAR, END_YEAR + 1))
    for s in base.states:
        for k in REQUIRED:
            assert k in s.demo and np.isfinite(s.demo[k]), (s.year, k)
        assert s.demo["old_age_dependency"] == pytest.approx(s.pop[67:].sum() / s.pop[18:67].sum())


def test_cohort_accounting_identity(base):
    for a, b in zip(base.states[:-1], base.states[1:]):
        d = a.demo
        expected = a.pop.sum() + d["births"] - d["deaths"] + d["net_migration"]
        assert b.pop.sum() == pytest.approx(expected, rel=1e-12, abs=1e-3)
        assert d["net_migration"] == pytest.approx(d["immigration"] - d["emigration"], abs=1e-6)


def test_cell_level_accounting(base):
    """Cohort by cohort: next = survivors + immigrants - emigrants + net internal moves."""
    a, b = base.at(2040), base.at(2041)
    x = a.extra["demographics"]
    surv = a.pop - x["deaths"]
    aged = np.zeros_like(surv)
    aged[1:] = surv[:-1]
    aged[-1] += surv[-1]
    aged[0] = x["births"] - x["deaths_newborn"]
    np.testing.assert_allclose(b.pop, aged + x["immigration"] - x["emigration"] + x["internal_net"],
                               rtol=1e-12, atol=1e-6)
    # inter-regional moves cancel out nationally, cell by cell
    np.testing.assert_allclose(x["internal_net"].sum(axis=2), 0.0, atol=1e-6)


def test_no_negative_cells(base):
    for s in base.states:
        assert s.pop.min() >= 0.0, s.year
        assert np.isfinite(s.pop).all()


def test_baseline_tracks_europop_total(base, europop):
    # EUROPOP2023 has lower net migration after 2050 than the national projection now followed
    for year, tol in ((2030, 0.02), (2040, 0.02), (2050, 0.02), (2070, 0.035)):
        ours = base.at(year).pop.sum()
        ref = europop.loc[year, "population_1jan"]
        assert abs(ours / ref - 1.0) < tol, (year, ours, ref)
    # FPB-Statbel population projection 2025-2080 (30 June; SCA annual report 2026, annex 1)
    for year, official in ((2030, 12_036e3), (2040, 12_336e3), (2050, 12_601e3), (2070, 12_960e3)):
        assert abs(base.at(year).pop.sum() / official - 1.0) < 0.015, year


def test_baseline_tracks_europop_67_plus(base, europop):
    for year in (2030, 2040, 2050, 2070):
        ours = base.at(year).pop[67:].sum()
        ref = europop.loc[year, "pop_67_plus"]
        assert abs(ours / ref - 1.0) < 0.03, (year, ours, ref)
        share = ours / base.at(year).pop.sum()
        assert abs(share - ref / europop.loc[year, "population_1jan"]) < 0.01


def test_baseline_fertility_and_migration_follow_reference(base, europop):
    for year in (2035, 2050, 2070):
        assert base.at(year).demo["tfr"] == pytest.approx(europop.loc[year, "tfr"], abs=0.01)
    # net migration: EUROPOP2023 to 2030, then the national projection (FPB-Statbel 2025-2080)
    assert base.at(2028).demo["net_migration"] == pytest.approx(europop.loc[2028, "net_migration"], rel=1e-9)
    for year, official in ((2031, 34_400), (2050, 36_900), (2070, 35_600)):
        assert base.at(year).demo["net_migration"] == pytest.approx(official, rel=1e-9)
    assert base.at(2070).demo["life_expectancy_M"] > base.at(2024).demo["life_expectancy_M"] + 3


def test_higher_net_migration_raises_population(base):
    scen = demo.project(scenario=scenario(Change("demo.net_migration", 20_000, 2030, op="add")))
    np.testing.assert_array_equal(scen.at(2030).pop, base.at(2030).pop)   # flows of 2030 -> pop 2031
    assert scen.at(2031).pop.sum() == pytest.approx(base.at(2031).pop.sum() + 20_000, abs=1.0)
    assert scen.at(2030).demo["net_migration"] == pytest.approx(base.at(2030).demo["net_migration"] + 20_000)
    gain = scen.at(2070).pop.sum() - base.at(2070).pop.sum()
    assert gain > 20_000 * 40 * 0.8          # cumulated migrants (minus deaths, plus their children)
    assert scen.at(2070).demo["old_age_dependency"] < base.at(2070).demo["old_age_dependency"]


def test_migration_factors(base):
    more_in = demo.project(scenario=scenario(Change("demo.immigration_factor", 1.1, 2030)), end=2040)
    assert more_in.at(2030).demo["immigration"] == pytest.approx(1.1 * base.at(2030).demo["immigration"])
    assert more_in.at(2040).pop.sum() > base.at(2040).pop.sum()
    more_out = demo.project(scenario=scenario(Change("demo.emigration_factor", 1.2, 2030)), end=2040)
    assert more_out.at(2030).demo["emigration"] == pytest.approx(1.2 * base.at(2030).demo["emigration"], rel=1e-3)
    assert more_out.at(2030).demo["immigration"] == pytest.approx(base.at(2030).demo["immigration"])
    assert more_out.at(2040).pop.sum() < base.at(2040).pop.sum()


def test_migration_factors_by_region(base):
    reg = {r: i for i, r in enumerate(REGIONS)}
    # a region-keyed dict of ones is the baseline, bit for bit
    same = demo.project(scenario=scenario(Change("demo.immigration_factor", {"FLA": 1.0, "WAL": 1.0, "BRU": 1.0}, 2030),
                                          Change("demo.emigration_factor", 1.0, 2030),
                                          Change("demo.internal_migration_factor", 1.0, 2030)), end=2035)
    np.testing.assert_array_equal(same.at(2035).pop, base.at(2035).pop)
    # more immigrants settling in Brussels only
    bru = demo.project(scenario=scenario(Change("demo.immigration_factor", 1.2, 2030, key="BRU")), end=2040)
    imm_b, imm_s = (r.at(2030).extra["demographics"]["immigration"].sum(axis=(0, 1)) for r in (base, bru))
    assert imm_s[reg["BRU"]] == pytest.approx(1.2 * imm_b[reg["BRU"]])
    assert imm_s[reg["FLA"]] == pytest.approx(imm_b[reg["FLA"]])
    assert imm_s[reg["WAL"]] == pytest.approx(imm_b[reg["WAL"]])
    assert bru.at(2040).pop[:, :, reg["BRU"]].sum() > base.at(2040).pop[:, :, reg["BRU"]].sum()
    # fewer emigrants from Wallonia only
    wal = demo.project(scenario=scenario(Change("demo.emigration_factor", 0.8, 2030, key="WAL")), end=2032)
    emi_b, emi_s = (r.at(2030).extra["demographics"]["emigration"].sum(axis=(0, 1)) for r in (base, wal))
    assert emi_s[reg["WAL"]] == pytest.approx(0.8 * emi_b[reg["WAL"]], rel=1e-3)
    assert emi_s[reg["FLA"]] == pytest.approx(emi_b[reg["FLA"]])
    # fewer people leaving Brussels for the two other regions: Brussels gains, the others lose
    stay = demo.project(scenario=scenario(Change("demo.internal_migration_factor", 0.5, 2030, key="BRU")), end=2040)
    assert stay.at(2030).demo["net_internal_migration.BRU"] > base.at(2030).demo["net_internal_migration.BRU"] + 5_000
    assert stay.at(2030).demo["net_internal_migration.FLA"] < base.at(2030).demo["net_internal_migration.FLA"]
    assert stay.at(2040).pop.sum() == pytest.approx(base.at(2040).pop.sum(), rel=2e-3)
    # the scalar form still applies to the three regions
    all_in = demo.project(scenario=scenario(Change("demo.immigration_factor", 1.1, 2030)), end=2031)
    by_key = demo.project(scenario=scenario(*[Change("demo.immigration_factor", 1.1, 2030, key=r) for r in REGIONS]),
                          end=2031)
    np.testing.assert_allclose(all_in.at(2031).pop, by_key.at(2031).pop, rtol=1e-12)


def test_fertility_levers(base):
    scen = demo.project(scenario=scenario(Change("demo.tfr_target", 2.1, 2025)))
    assert scen.at(2070).demo["tfr"] == pytest.approx(2.1, abs=0.02)
    assert base.at(2024).demo["tfr"] < scen.at(2040).demo["tfr"] < 2.1
    assert scen.at(2070).pop[:18].sum() > 1.15 * base.at(2070).pop[:18].sum()
    # people aged 60+ in 2070 were born before the change; they differ only through migration
    # (net migration is held at the parameter, so extra emigrants are matched by immigrants)
    np.testing.assert_allclose(scen.at(2070).pop[60:].sum(), base.at(2070).pop[60:].sum(), rtol=0.01)
    bru = demo.project(scenario=scenario(Change("demo.fertility_factor", 1.2, 2030, key="BRU")), end=2032)
    assert bru.at(2030).demo["tfr.BRU"] == pytest.approx(1.2 * base.at(2030).demo["tfr.BRU"])
    assert bru.at(2030).demo["tfr.FLA"] == pytest.approx(base.at(2030).demo["tfr.FLA"])


def test_mortality_lever(base):
    frozen = demo.project(scenario=scenario(Change("demo.mortality_improvement", 0.0, 2025)))
    assert frozen.at(2070).demo["life_expectancy_F"] == pytest.approx(base.at(2024).demo["life_expectancy_F"], abs=0.05)
    assert frozen.at(2070).pop[67:].sum() < base.at(2070).pop[67:].sum()
    # changing the pace later does not undo past improvement (no jump in the level)
    late = demo.project(scenario=scenario(Change("demo.mortality_improvement", 0.0, 2050)))
    assert late.at(2060).demo["life_expectancy_M"] == pytest.approx(base.at(2050).demo["life_expectancy_M"], abs=0.2)


def test_internal_migration_lever(base):
    none = demo.project(scenario=scenario(Change("demo.internal_migration_factor", 0.0, 2025)), end=2040)
    bru = REGIONS.index("BRU")
    assert base.at(2030).demo["net_internal_migration.BRU"] < -10_000     # Brussels loses residents
    assert none.at(2030).demo["net_internal_migration.BRU"] == 0.0
    assert none.at(2040).pop[:, :, bru].sum() > base.at(2040).pop[:, :, bru].sum()


def test_runs_without_carried_state(base):
    """A `prev` built by someone else (no extra) still works: last year's step is recomputed."""
    pset = demo.load_params()
    prev = YearState(year=2040)
    prev.pop = base.at(2040).pop.copy()
    st = YearState(year=2041)
    demo.run(st, prev, pset.at(2041, None), None)
    assert st.pop.sum() == pytest.approx(base.at(2041).pop.sum(), rel=2e-4)


def test_deterministic_and_fast(base):
    t0 = time.perf_counter()
    again = demo.project()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2, elapsed
    for a, b in zip(base.states, again.states):
        np.testing.assert_array_equal(a.pop, b.pop)
    assert not demo.tables().pop0.flags.writeable


def test_assumptions_are_flagged():
    specs = demo.load_params().specs
    assert all(s.source for s in specs.values())
    assert specs["demo.tfr_convergence_year"].source.startswith("ASSUMPTION")
    assert specs["demo.net_migration"].values[2030] == demo.reference().loc[2030, "net_migration"]
    assert specs["demo.net_migration"].values[2070] == 35600      # FPB-Statbel projection 2025-2080
    assert specs["demo.tfr_target"].values[2024] == pytest.approx(demo.reference().loc[2070, "tfr"])
