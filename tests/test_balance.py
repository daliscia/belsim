"""Tests of belsim.fiscal.balance on a synthetic economy (see test_expenditure_world)."""
import numpy as np
import pytest

import test_expenditure_world as w
from belsim.core.dims import ENTITIES
from belsim.core.params import Change, Measure, Scenario
from belsim.fiscal.expenditure import data

SC = data.scalars()
CONTRACT_KEYS = ["revenue_total", "expenditure_total", "primary_expenditure", "interest", "balance",
                 "primary_balance", "debt", "balance_pct_gdp", "primary_balance_pct_gdp", "debt_pct_gdp",
                 "revenue_pct_gdp", "expenditure_pct_gdp", "net_expenditure_growth", "structural_balance_pct_gdp"]


@pytest.fixture(scope="module")
def base():
    return w.simulate(2035)


def scen(*changes):
    return Scenario(name="s", measures=[Measure("m", list(changes))])


def test_contract_keys(base):
    for s in base.states:
        for k in CONTRACT_KEYS:
            assert np.isfinite(s.fiscal[k]), k
        for e in ENTITIES:
            for pre in ("balance", "revenue", "expenditure", "debt"):
                assert np.isfinite(s.fiscal[f"{pre}.{e}"])


def test_totals_and_identities(base):
    for s in base.states:
        f = s.fiscal
        assert f["revenue_total"] == pytest.approx(sum(s.revenue.values()))
        assert f["expenditure_total"] == pytest.approx(sum(s.expenditure.values()))
        assert f["balance"] == pytest.approx(f["revenue_total"] - f["expenditure_total"])
        assert f["primary_balance"] == pytest.approx(f["balance"] + f["interest"])
        assert f["primary_expenditure"] == pytest.approx(f["expenditure_total"] - f["interest"])
        assert f["debt_pct_gdp"] == pytest.approx(f["debt"] / s.macro["gdp_nominal"] * 100)


def test_entities_sum_to_general_government_and_transfers_net_out(base):
    for s in base.states:
        f = s.fiscal
        assert sum(f[f"balance.{e}"] for e in ENTITIES) == pytest.approx(f["balance"], abs=1e-6)
        # transfers are revenue of one entity and expenditure of another: they net to zero
        extra_rev = sum(f[f"revenue.{e}"] for e in ENTITIES) - f["revenue_total"]
        extra_exp = sum(f[f"expenditure.{e}"] for e in ENTITIES) - f["expenditure_total"]
        assert extra_rev == pytest.approx(extra_exp, abs=1e-6)
        assert extra_rev == pytest.approx(f["transfers_total"], abs=1e-6)
        flows = sum(v for k, v in f.items() if k.startswith("transfer.") and "_to_" in k)
        assert flows == pytest.approx(f["transfers_total"], abs=1e-6)
        assert sum(f[f"entity_residual.{e}"] for e in ENTITIES) == pytest.approx(0.0, abs=1e-6)
        assert sum(f[f"own_revenue.{e}"] for e in ENTITIES) == pytest.approx(f["revenue_total"])
        assert sum(f[f"own_expenditure.{e}"] for e in ENTITIES) == pytest.approx(f["expenditure_total"])


def test_2024_matches_accounts(base):
    f = base.at(2024).fiscal
    assert f["balance"] == pytest.approx(SC["balance_2024"], abs=1.0)
    assert f["debt"] == pytest.approx(SC["debt_2024"])
    assert 103.5 < f["debt_pct_gdp"] < 105.0
    assert -4.6 < f["balance_pct_gdp"] < -4.2
    for e in ENTITIES:
        assert f[f"balance.{e}"] == pytest.approx(SC[f"balance_{e}_2024"], abs=1.0)
        assert f[f"debt.{e}"] == pytest.approx(SC[f"debt_{e}_2024"])
        # non-consolidated subsector totals (incl. transfers) close to the published ones
        assert f[f"expenditure.{e}"] == pytest.approx(SC[f"te_{e}_2024"], rel=0.02)
    assert f["transfer.federal_to_regions_communities"] == pytest.approx(57018.1, abs=1.0)
    assert f["transfer.ss_balancing_grant"] == pytest.approx(0.0, abs=1e-6)
    assert 0.03 < f["net_expenditure_growth"] < 0.08


def test_2024_debt_and_entities_robust_to_revenue_level():
    """Even if the revenue module misses the 2024 total, debt is actual and entities add up."""
    res = w.simulate(2026, rev_factor=0.97)
    f = res.at(2024).fiscal
    assert f["debt"] == pytest.approx(SC["debt_2024"])
    assert sum(f[f"balance.{e}"] for e in ENTITIES) == pytest.approx(f["balance"], abs=1e-6)
    assert f["balance"] < SC["balance_2024"] - 5000


def test_debt_recursion(base):
    pset = w.load_params()
    for prev, s in zip(base.states[:-1], base.states[1:]):
        f = s.fiscal
        # the baseline carries the 2025 outturn and the official adjustments to 2031
        sfa_pct = pset.at(s.year, None)["debt.stock_flow_adjustment_pct_gdp"]
        assert f["stock_flow_adjustment"] == pytest.approx(sfa_pct / 100.0 * s.macro["gdp_nominal"], abs=1e-6)
        assert (sfa_pct == 0.0) == (s.year not in (2025, 2026, 2027, 2028, 2031))
        assert f["debt"] == pytest.approx(prev.fiscal["debt"] - f["balance"] + f["stock_flow_adjustment"])
        if sfa_pct == 0.0:
            for e in ENTITIES:
                assert f[f"debt.{e}"] == pytest.approx(prev.fiscal[f"debt.{e}"] - f[f"balance.{e}"])
        assert f["debt"] == pytest.approx(sum(f[f"debt.{e}"] for e in ENTITIES) + f["debt.intragov_consolidation"])
    sfa = w.simulate(2027, scenario=scen(Change("debt.stock_flow_adjustment_pct_gdp", -1.0, 2026, op="add", end=2026)), baseline=base)
    g = sfa.at(2026).macro["gdp_nominal"]
    assert sfa.at(2026).fiscal["debt"] == pytest.approx(base.at(2026).fiscal["debt"] - 0.01 * g, rel=1e-9)
    assert sfa.at(2027).expenditure["interest"] < base.at(2027).expenditure["interest"]


def test_social_security_balancing_grant(base):
    # baseline: federal government absorbs the drift of the social-security balance
    b24 = base.at(2024).fiscal["balance.social_security"] / base.at(2024).macro["gdp_nominal"]
    for s in base.states:
        assert s.fiscal["balance.social_security"] / s.macro["gdp_nominal"] == pytest.approx(b24, abs=1e-9)
    assert base.at(2035).fiscal["transfer.ss_balancing_grant"] > 0          # ageing costs
    off = w.simulate(2035, scenario=scen(Change("transfer.ss_balancing_share", 0.0, 2026)), baseline=base)
    fo, fb = off.at(2035).fiscal, base.at(2035).fiscal
    assert fo["balance.social_security"] < fb["balance.social_security"] - 1000
    assert fo["balance.federal"] > fb["balance.federal"] + 1000
    assert fo["balance"] == pytest.approx(fb["balance"])                    # only a reallocation
    assert fo["debt"] == pytest.approx(fb["debt"])


def test_transfer_levers(base):
    s = w.simulate(2028, scenario=scen(Change("transfer.adjustment", -1000.0, 2027, key="fed_to_regions")), baseline=base)
    fs, fb = s.at(2027).fiscal, base.at(2027).fiscal
    cut = fb["transfer.federal_to_regions_communities"] - fs["transfer.federal_to_regions_communities"]
    assert cut == pytest.approx(1000.0, rel=1e-9)
    assert fs["balance.federal"] - fb["balance.federal"] == pytest.approx(cut)
    assert fs["balance.regions_communities"] - fb["balance.regions_communities"] == pytest.approx(-cut)
    assert fs["balance"] == pytest.approx(fb["balance"])


def test_structural_balance_and_cycle():
    base = w.simulate(2026)
    low = w.simulate(2026, gap=-0.02)
    for s in low.states:
        f = s.fiscal
        assert f["structural_balance_pct_gdp"] == pytest.approx(f["balance_pct_gdp"] + 0.6 * 2.0)
    for s in base.states:
        assert s.fiscal["structural_balance_pct_gdp"] == pytest.approx(s.fiscal["balance_pct_gdp"])


def test_net_expenditure_growth_nets_out_revenue_measures(base):
    more_rev = w.simulate(2027, scenario=Scenario(name="x"), baseline=base, rev_factor=1.01)
    for y in (2024, 2025, 2026):
        fs, fb = more_rev.at(y).fiscal, base.at(y).fiscal
        if y == 2024:    # the measure enters in 2024: growth is lower by measure / net expenditure
            drm = fs["revenue_total"] - fb["revenue_total"]
            assert fb["net_expenditure_growth"] - fs["net_expenditure_growth"] == pytest.approx(
                drm / (SC["te_2023"] - SC["interest_2023"] - SC["eu_funded_2023"]), rel=1e-6)
        else:            # afterwards only the (small) increment counts
            assert abs(fs["net_expenditure_growth"] - fb["net_expenditure_growth"]) < 0.002
    # spending measure raises the indicator in the year it starts
    spend = w.simulate(2027, scenario=scen(Change("invest.public_pct_gdp", 0.5, 2026, op="add")), baseline=base)
    assert spend.at(2026).fiscal["net_expenditure_growth"] > base.at(2026).fiscal["net_expenditure_growth"] + 0.005
    assert spend.at(2027).fiscal["net_expenditure_growth"] == pytest.approx(
        base.at(2027).fiscal["net_expenditure_growth"], abs=1e-3)


def test_revenue_lines_without_entity_shares_use_default_split(base):
    # the synthetic revenue keys may or may not exist in revenue.LINES: either way entities add up
    f = base.at(2030).fiscal
    assert all(f[f"own_revenue.{e}"] > 0 for e in ENTITIES)
