"""Revenue module: aggregate lines on a synthetic 2024 state (independent of the other modules)."""
import time

import numpy as np
import pytest

from belsim.core.dims import CONS_CATS, ENTITIES, REGIONS, S
from belsim.core.params import Change, Measure, ParamSet, Scenario
from belsim.fiscal import revenue
from belsim.fiscal.revenue import calc

from test_revenue_fixtures import GDP_2024, ctx, make_state, params

OFFICIAL_TOTAL_2024 = 308_088.3   # Eurostat gov_10a_main, BE, S13, TR, 2024 (EUR mn)


@pytest.fixture(scope="module")
def pset():
    return ParamSet.load()


@pytest.fixture(scope="module")
def p(pset):
    return pset.at(2024)


@pytest.fixture()
def base(p):
    st = make_state()
    revenue.run(st, None, p, ctx())
    return st


def reform(pset, *changes, year=2024):
    return pset.with_scenario(Scenario("r", [Measure("m", list(changes))])).at(year)


def run_reform(pset, p, *changes):
    st = make_state()
    revenue.run(st, None, reform(pset, *changes), ctx(base_p=p))
    return st


def test_lines_metadata():
    assert len(revenue.LINES) >= 40
    for name, meta in revenue.LINES.items():
        assert meta.label and meta.group and meta.esa, name
        assert set(meta.entity_shares) == set(ENTITIES)
        assert abs(sum(meta.entity_shares.values()) - 1.0) < 1e-4, name
        assert min(meta.entity_shares.values()) >= 0.0
    assert revenue.LINES["vat"].entity_shares["federal"] == 1.0
    assert revenue.LINES["ssc_employee"].entity_shares["social_security"] > 0.95
    assert revenue.LINES["inheritance_gift_tax"].entity_shares["regions_communities"] == 1.0
    assert revenue.LINES["property_tax"].entity_shares["local"] > 0.9


def test_2024_total_matches_accounts(base):
    assert set(base.revenue) == set(revenue.LINES)
    total = sum(base.revenue.values())
    assert total == pytest.approx(OFFICIAL_TOTAL_2024, abs=1.0)
    assert 0.49 < total / GDP_2024 < 0.515
    actual = revenue.actual_2024()
    for line, v in base.revenue.items():
        assert v == pytest.approx(actual[line], abs=1e-6), line
    # magnitudes of the big blocks (EUR bn)
    r = base.revenue
    assert 90_000 < sum(v for k, v in r.items() if k.startswith("ssc_")) < 100_000
    assert 60_000 < r["pit_federal"] + r["pit_regional"] + r["pit_local"] < 72_000
    assert 35_000 < r["vat"] < 40_000 and 25_000 < r["cit"] < 29_000


def test_model_is_close_to_accounts_before_calibration(base):
    """Fed with official-like bases, the uncalibrated model of the main lines lands within
    a plausible range of the accounts (calibration factors not far from 1)."""
    calib = base.extra["revenue"]["calib"]
    for line in ("ssc_employer_private", "ssc_employer_public", "ssc_employee", "ssc_self_employed",
                 "cit", "vat", "withholding_tax",
                 "registration_duties", "inheritance_gift_tax", "excise_tobacco", "dividends_received"):
        assert 0.6 < calib[line] < 1.5, (line, calib[line])
    # personal income tax: household rules over the population of tax units, tax base aligned on
    # the fiscal statistics -> no large residual (was 0.73-0.87 with single-person payslips)
    for line in ("pit_federal", "pit_regional", "pit_local"):
        assert 0.85 < calib[line] < 1.15, (line, calib[line])
    assert base.extra["revenue"]["gdp_fallback"] == ()


def test_idempotent_and_fast(p, base):
    first = dict(base.revenue)
    revenue.run(base, None, p, ctx())                 # base-year path (calibrates again)
    assert base.revenue == first
    nxt = make_state(year=2025)
    revenue.run(nxt, base, p, ctx())
    again, regional = dict(nxt.revenue), dict(nxt.regional)
    best = 1.0
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(10):
            revenue.run(nxt, base, p, ctx())
        best = min(best, (time.perf_counter() - t) / 10)
    assert best < 0.005                                # about 2 ms a year
    assert nxt.revenue == again and nxt.regional == regional


def test_calibration_carried_forward(p, base):
    nxt = make_state(year=2025, scale=1.04)
    revenue.run(nxt, base, p, ctx())
    assert nxt.extra["revenue"] is base.extra["revenue"]
    ratio = sum(nxt.revenue.values()) / sum(base.revenue.values())
    assert 1.0 < ratio < 1.05


def test_vat_one_point_standard_rate(pset, p, base):
    st = run_reform(pset, p, Change("vat.rate_standard", 0.01, 2024, op="add"))
    d = st.revenue["vat"] - base.revenue["vat"]
    assert 1_300 < d < 1_800            # known costings: about 1.5 bn EUR per point
    others = sum(st.revenue.values()) - sum(base.revenue.values()) - d
    assert abs(others) < 1e-6


@pytest.mark.parametrize("key,line", [
    ("corporate_profits_taxable", "cit"),
    ("household_capital_income", "withholding_tax"),
    ("property_transactions", "registration_duties"),
    ("inheritance_flow", "inheritance_gift_tax"),
    ("cons.tobacco", "excise_tobacco"),
    ("cons.fuel", "excise_fuel"),
    ("cons.housing", "property_tax"),
    ("household_financial_wealth", "securities_account_tax"),
])
def test_doubling_a_base_doubles_its_line(p, base, key, line):
    st = make_state()
    st.macro[key] *= 2.0
    revenue.run(st, base, p, ctx())
    assert st.revenue[line] == pytest.approx(2.0 * base.revenue[line], rel=1e-9)


def test_doubling_consumption_doubles_vat_from_households(p, base):
    st = make_state()
    for k in ("investment_housing", "gov_consumption", "gdp_nominal"):
        st.macro[k] *= 2.0
    for c in CONS_CATS:
        st.macro[f"cons.{c}"] *= 2.0
    revenue.run(st, base, p, ctx())
    assert st.revenue["vat"] == pytest.approx(2.0 * base.revenue["vat"], rel=1e-9)


def test_doubling_employment_doubles_wage_taxes(p, base):
    st = make_state()
    st.emp *= 2.0
    revenue.run(st, base, p, ctx())
    for line in ("ssc_employer_private", "ssc_employer_public", "ssc_employee", "ssc_self_employed", "special_ssc"):
        assert st.revenue[line] == pytest.approx(2.0 * base.revenue[line], rel=1e-9)
    assert st.revenue["pit_local"] > 1.7 * base.revenue["pit_local"]


@pytest.mark.parametrize("param,line", [
    ("cit.rate", "cit"), ("cap.withholding_rate", "withholding_tax"),
    ("cap.securities_account_tax_rate", "securities_account_tax"), ("excise.fuel_duty", "excise_fuel"),
    ("excise.tobacco_duty", "excise_tobacco"), ("rev.insurance_premium_tax_rate", "insurance_premium_tax"),
])
def test_rate_change_is_static_rate_times_base(pset, p, base, param, line):
    """+10 % on a rate with unchanged bases: revenue of the line rises by the change of the
    effective rate (exactly 10 % for single-rate taxes)."""
    st = run_reform(pset, p, Change(param, 1.10, 2024, op="mul"))
    ratio = st.revenue[line] / base.revenue[line]
    if param == "cit.rate":
        assert ratio == pytest.approx(calc.cit_effective_rate(reform(pset, Change(param, 1.10, 2024, op="mul")))
                                      / calc.cit_effective_rate(p), rel=1e-9)
        assert 1.08 < ratio < 1.10
    elif param == "cap.withholding_rate":
        assert 1.08 < ratio < 1.10
    else:
        assert ratio == pytest.approx(1.10, rel=1e-9)


def test_excise_duty_increase_also_raises_vat(pset, p, base):
    st = run_reform(pset, p, Change("excise.fuel_duty", 0.10, 2024, op="add"))
    assert st.revenue["excise_fuel"] == pytest.approx(base.revenue["excise_fuel"] * (p["excise.fuel_duty"] + 0.10) / p["excise.fuel_duty"], rel=1e-9)
    assert st.revenue["vat"] > base.revenue["vat"]


def test_pit_reforms_move_the_right_entities(pset, p, base):
    st = run_reform(pset, p, Change("pit.municipal_surcharge", 0.01, 2024, op="add"))     # every region
    rate = p["pit.municipal_surcharge"]["FLA"]            # the synthetic population lives in one region
    assert st.revenue["pit_local"] == pytest.approx(base.revenue["pit_local"] * (rate + 0.01) / rate, rel=1e-9)
    assert st.revenue["pit_federal"] == pytest.approx(base.revenue["pit_federal"])

    st = run_reform(pset, p, Change("pit.basic_allowance", 1_000, 2024, op="add"))
    cost = sum(base.revenue[k] - st.revenue[k] for k in ("pit_federal", "pit_regional", "pit_local"))
    assert 1_600 < cost < 2_800      # official costing: about 2.2 bn per 1,000 EUR (DC2024)

    st = run_reform(pset, p, Change("pit.tax_expenditure_pension_savings", 0, 2024))
    assert st.revenue["pit_federal"] - base.revenue["pit_federal"] == pytest.approx(927.0)


def test_household_rules_are_in_the_aggregate(pset, p, base):
    """The aggregate income tax comes from the household calculator: supplements for children,
    the marital quotient and the child credit move the budget."""
    pit = lambda st: sum(st.revenue[k] for k in ("pit_federal", "pit_regional", "pit_local"))
    st = run_reform(pset, p, Change("pit.child_allowance_supplement", 0.0, 2024, op="mul"),
                    Change("pit.child_allowance_extra_per_child", 0.0, 2024))
    assert 1_000 < pit(st) - pit(base) < 4_000           # about 1.35 mn returns with dependants
    st = run_reform(pset, p, Change("pit.marital_quotient_max", 0, 2024, key="work"),
                    Change("pit.marital_quotient_max", 0, 2024, key="pension"))
    assert 200 < pit(st) - pit(base) < 800               # official cost: 498 mn (AY 2024)
    st = run_reform(pset, p, Change("pit.child_credit_max_per_child", 0, 2024))
    assert 0 < st.revenue["pit_federal"] - base.revenue["pit_federal"] < 500
    assert st.revenue["pit_regional"] == pytest.approx(base.revenue["pit_regional"])
    info = base.extra["revenue"]["pit"]
    assert 0.8 < info["share"]["wages"] < 1.0 and 1.0 <= info["persons"] < 1.15
    assert np.allclose(info["units_work"].sum(axis=1), 1.0) and np.allclose(info["units_pension"].sum(axis=1), 1.0)


# ------------------------------------------------------------------ regions

def regional_state(year=2024):
    """Synthetic state with the population spread over the three regions."""
    st = make_state(year=year)
    share = np.array([0.58, 0.32, 0.10])
    st.status = st.status[:, :, :, :1] * share
    st.pop = st.pop.sum(axis=2, keepdims=True) * share
    return st


def regional_run(pset, p, *changes):
    st = regional_state()
    revenue.run(st, None, reform(pset, *changes) if changes else p, ctx(base_p=p))
    return st


def test_regional_revenue_sums_to_national(pset, p):
    st = regional_run(pset, p)
    assert sum(st.revenue.values()) == pytest.approx(OFFICIAL_TOTAL_2024, abs=1.0)
    for line in revenue.REGIONAL_LINES:
        parts = [st.regional[f"revenue.{line}.{r}"] for r in REGIONS]
        assert sum(parts) == pytest.approx(st.revenue[line], rel=1e-9, abs=1e-6), line
    assert {"pit_regional", "pit_local", "pit_federal", "vat", "ssc_employee", "registration_duties",
            "inheritance_gift_tax", "property_tax", "vehicle_taxes"} <= set(revenue.REGIONAL_LINES)
    own = sum(st.regional[f"revenue_total.{r}"] for r in REGIONS)
    fed = sum(st.regional[f"federal_taxes_by_residence.{r}"] for r in REGIONS)
    assert own + fed == pytest.approx(sum(st.revenue[k] for k in revenue.REGIONAL_LINES))
    assert own > st.revenue["pit_regional"] + st.revenue["pit_local"] + st.revenue["property_tax"]
    # the base-year regional split of the state tax is that of the fiscal statistics (Statbel 2024)
    fed_pit = np.array([st.regional[f"revenue.pit_federal.{r}"] for r in REGIONS])
    assert fed_pit / fed_pit.sum() == pytest.approx([0.642, 0.269, 0.088], abs=0.01)


def test_regional_surcharge_moves_one_region(pset, p):
    base = regional_run(pset, p)
    st = regional_run(pset, p, Change("pit.regional_surcharge", 0.02, 2024, op="add", key="WAL"))
    key = "revenue.pit_regional.{}".format
    for r in ("FLA", "BRU"):
        assert st.regional[key(r)] == pytest.approx(base.regional[key(r)], rel=1e-12)
    assert st.regional[key("WAL")] > 1.04 * base.regional[key("WAL")]
    d = st.revenue["pit_regional"] - base.revenue["pit_regional"]
    assert d == pytest.approx(st.regional[key("WAL")] - base.regional[key("WAL")])
    assert st.revenue["pit_federal"] == pytest.approx(base.revenue["pit_federal"], rel=1e-12)
    assert st.regional["revenue.pit_local.FLA"] == pytest.approx(base.regional["revenue.pit_local.FLA"], rel=1e-12)
    # other regional levers address one region with a single Change
    for change, line, region in (
            (Change("pit.municipal_surcharge", 0.01, 2024, op="add", key="BRU"), "pit_local", "BRU"),
            (Change("pit.tax_expenditure_mortgage", 0, 2024, key="FLA"), "pit_regional", "FLA"),
            (Change("rev.regional_factor", 1.2, 2024, key="vehicle_taxes.WAL"), "vehicle_taxes", "WAL"),
            (Change("prop.registration_rate", 0.15, 2024, key="BRU"), "registration_duties", "BRU"),
            (Change("prop.inheritance_rate", 0.10, 2024, key="FLA"), "inheritance_gift_tax", "FLA"),
            (Change("prop.property_tax_rate", 0.05, 2024, key="FLA"), "property_tax", "FLA")):
        st = regional_run(pset, p, change)
        for r in REGIONS:
            a, b = st.regional[f"revenue.{line}.{r}"], base.regional[f"revenue.{line}.{r}"]
            assert (a > b * 1.001) if r == region else a == pytest.approx(b, rel=1e-9), (line, r)
        assert st.revenue[line] - base.revenue[line] == pytest.approx(
            st.regional[f"revenue.{line}.{region}"] - base.regional[f"revenue.{line}.{region}"])
    st = regional_run(pset, p, Change("rev.regional_factor", 1.2, 2024, key="vehicle_taxes.WAL"))
    own = revenue.LINES["vehicle_taxes"].entity_shares
    assert st.regional["revenue.vehicle_taxes.WAL"] == pytest.approx(
        base.regional["revenue.vehicle_taxes.WAL"] * (1 + 0.2 * (own["regions_communities"] + own["local"])))


def test_equal_regional_values_equal_scalar(pset, p):
    """A national scalar and a regional dict with the same value in every region give the same
    revenue, and regional rates only redistribute when they differ."""
    scalar = regional_run(pset, p, Change("pit.regional_surcharge", 0.33, 2024), Change("pit.municipal_surcharge", 0.07, 2024))
    same = regional_run(pset, p, Change("pit.regional_surcharge", {"FLA": 0.33, "WAL": 0.33, "BRU": 0.33}, 2024),
                        Change("pit.municipal_surcharge", {"FLA": 0.07, "WAL": 0.07, "BRU": 0.07}, 2024))
    for line in ("pit_federal", "pit_regional", "pit_local"):
        assert same.revenue[line] == pytest.approx(scalar.revenue[line], rel=1e-12)
    # 2024 is calibrated on the accounts whatever the rates; the rates show in the regional split
    wal = lambda st: st.regional["revenue.pit_local.WAL"] / st.regional["revenue.pit_local.FLA"]
    assert wal(regional_run(pset, p)) > 1.05 * wal(same)        # Walloon municipalities levy more than Flemish ones
    q = pset.at(2024)
    assert calc.surcharge_rates(reform(pset, Change("pit.regional_surcharge", 0.33, 2024)))[0] == 0.33
    s_avg, m_avg = calc.surcharge_rates(q)
    assert q["pit.regional_surcharge"]["BRU"] < s_avg < q["pit.regional_surcharge"]["FLA"]
    assert calc.surcharge_rates(q, "BRU") == (q["pit.regional_surcharge"]["BRU"], q["pit.municipal_surcharge"]["BRU"])


def test_ssc_reforms(pset, p, base):
    st = run_reform(pset, p, Change("ssc.employer_rate", -0.01, 2024, op="add"))
    cost = base.revenue["ssc_employer_private"] - st.revenue["ssc_employer_private"]
    assert 1_000 < cost < 2_000       # 1 point on about 180 bn EUR of private wages, calibrated
    st = run_reform(pset, p, Change("ssc.workbonus_a_max_monthly", 1.2, 2024, op="mul"))
    assert st.revenue["ssc_employee"] < base.revenue["ssc_employee"]
    # lower contributions raise taxable income; the fiscal work bonus offsets only part of it
    assert abs(st.revenue["pit_federal"] - base.revenue["pit_federal"]) < 0.01 * base.revenue["pit_federal"]


def test_dated_baseline_changes(pset, base):
    """Legislated reforms in the baseline (same synthetic bases every year, nothing indexed)."""
    def at(year):
        st = make_state(year=year)
        # legislation alone: without the alignment on the Planning Bureau's outlook
        revenue.run(st, base, reform(pset, Change("rev.fpb_alignment_enabled", 0, 2024), year=year), ctx())
        return st.revenue
    r24, r25, r26, r27, r29, r30 = (at(y) for y in (2024, 2025, 2026, 2027, 2029, 2030))
    # capital gains tax from 2026: official 120 mn net in 2026, about 600 mn at cruising speed (2032)
    assert r24["capital_gains_tax"] == 0.0 and r25["capital_gains_tax"] == 0.0
    assert 100 < r26["capital_gains_tax"] < 250 and 250 < r29["capital_gains_tax"] < 400
    assert 400 < at(2032)["capital_gains_tax"] < 600
    # current transfers received on the Planning Bureau's path (share of GDP relative to 2025)
    assert r29["eu_transfers_current"] == pytest.approx(0.794 * r24["eu_transfers_current"])
    # registration duties: Flanders 2 % and Wallonia 3 % from 2025, Flanders back to 3 % in 2027
    assert r25["registration_duties"] < 0.95 * r24["registration_duties"]
    assert r25["registration_duties"] < r27["registration_duties"] < r24["registration_duties"]
    # securities accounts tax doubled, bank tax and insurance tax raised (programme law of 30 May 2026)
    assert r26["securities_account_tax"] == pytest.approx(2.0 * r24["securities_account_tax"])
    assert r26["bank_levies"] == pytest.approx(1.09 * r24["bank_levies"])
    assert r27["insurance_premium_tax"] == pytest.approx(r24["insurance_premium_tax"] * 0.096 / 0.0925)
    assert r27["withholding_tax"] > r26["withholding_tax"] > r24["withholding_tax"]
    # hotels at 12 % from March 2026: official yield 158 mn a year
    assert 100 < r27["vat"] - r25["vat"] < 260
    # personal income tax reform (law of 15 July 2026): allowance 10,570 -> 15,600 by 2030
    pit = lambda r: r["pit_federal"] + r["pit_regional"] + r["pit_local"]
    assert pit(r30) < pit(r29) < pit(r26) < pit(r25)
    assert 4_000 < pit(r25) - pit(r30) < 10_000      # nothing is indexed here: larger than the official 5 bn
    assert r30["special_ssc"] < 0.7 * r24["special_ssc"]
    # Walloon inheritance tax cut from 2028
    assert r29["inheritance_gift_tax"] < 0.95 * r24["inheritance_gift_tax"]


def test_line_factor_lever(pset, p, base):
    st = run_reform(pset, p, Change("rev.line_factor", 1.25, 2024, key="vehicle_taxes"))
    assert st.revenue["vehicle_taxes"] == pytest.approx(1.25 * base.revenue["vehicle_taxes"])
    st = run_reform(pset, p, Change("rev.dividends", 0, 2024, key="belfius"))
    assert st.revenue["dividends_received"] == pytest.approx(base.revenue["dividends_received"] * (1 - 440 / 1991), rel=1e-6)


def test_robust_to_crude_or_missing_bases(p):
    """Missing optional bases fall back on GDP and the 2024 total is still the accounts."""
    st = make_state()
    for k in ("household_financial_wealth", "investment_housing", "inheritance_flow"):
        st.macro.pop(k)
    st.expenditure = {}
    st.wage_dist *= 0.7
    revenue.run(st, None, p, ctx())
    assert sum(st.revenue.values()) == pytest.approx(OFFICIAL_TOTAL_2024, abs=1.0)
    assert set(st.extra["revenue"]["gdp_fallback"]) == {"securities_account_tax", "stock_exchange_tax",
                                                       "bank_levies", "inheritance_gift_tax"}
    assert all(np.isfinite(v) for v in st.revenue.values())


def test_every_revenue_param_has_a_source(pset):
    specs = [s for s in pset.specs.values() if s.file == "revenue.yaml"]
    assert len(specs) > 60
    for s in specs:
        assert s.source and s.description and s.unit, s.name
        assert s.name.split(".")[0] in {"pit", "ssc", "cit", "vat", "excise", "cap", "prop", "rev"}
        if s.unit == "rate" or s.unit == "share":
            assert s.uprate == "none", s.name


def test_kind_of_tax_unit_follows_income(pset, p, base):
    """Tax returns: the single income of a joint return is much higher than an individual
    return's. The mix of tax-unit types by income group keeps the number of units of each kind."""
    from belsim.fiscal.revenue import taxunits
    cal = base.extra["revenue"]["pit"]
    st = make_state()
    benefits = revenue._benefits(st, p)[0]
    rows = taxunits._rows(taxunits.population(st, p, benefits), p, cal)
    units = taxunits._units(cal, rows)
    onehot = (taxunits.FAMILY[:, None] == np.arange(3)[None, :]).astype(float)
    work = [k for k, r in enumerate(rows) if r["kind"] != "pension"]
    w = np.stack([rows[k]["w"] for k in work])                       # [row, region, 1, group]
    fam = np.stack([np.einsum("rtg,tf->rfg", units[k], onehot) for k in work])   # [row, region, family, group]
    live = w.sum(axis=(0, 2, 3)) > 0                                 # the synthetic state has one region
    got = (fam * w).sum(axis=(0, 3))[live] / w.sum(axis=(0, 2, 3))[live][:, None]
    np.testing.assert_allclose(got, (cal["units_work"] @ onehot)[live], atol=2e-4)
    for k in work:
        np.testing.assert_allclose(units[k].sum(axis=1), 1.0, atol=1e-9)
    sole = fam[:, :, taxunits.FAMILIES.index("sole"), :]
    assert (sole[..., -4:] * w[:, :, 0, -4:]).sum() / w[..., -4:].sum() > 2.0 * (sole[..., :8] * w[:, :, 0, :8]).sum() / w[..., :8].sum()
    off = make_state()
    revenue.run(off, None, reform(pset, Change("pit.tax_unit_type_follows_income", 0, 2024)), ctx())
    assert "family_tilt" not in off.extra["revenue"]["pit"]
    assert sum(off.revenue.values()) == pytest.approx(sum(base.revenue.values()), abs=1e-6)


def test_alternative_bracket_base(pset, p, base):
    """`pit.bracket_base_factor`: the receipts of the year are unchanged, a cut of the 45 % rate
    is costed on the smaller base, the 40 % rate on the same one."""
    alt = Change("pit.bracket_base_factor", [1.0, 1.0, 0.88, 0.88], 2024)
    pit = lambda st: sum(st.revenue[k] for k in ("pit_federal", "pit_regional", "pit_local"))
    ref = run_reform(pset, p, alt)
    assert pit(ref) == pytest.approx(pit(base), rel=1e-9)
    assert sum(ref.revenue.values()) == pytest.approx(sum(base.revenue.values()), rel=1e-9)
    cut45, cut40 = Change("pit.bracket_rates", 0.40, 2024, key="2"), Change("pit.bracket_rates", 0.30, 2024, key="1")
    default45, default40 = pit(base) - pit(run_reform(pset, p, cut45)), pit(base) - pit(run_reform(pset, p, cut40))
    assert pit(ref) - pit(run_reform(pset, p, alt, cut45)) == pytest.approx(0.88 * default45, rel=0.02)
    assert pit(ref) - pit(run_reform(pset, p, alt, cut40)) == pytest.approx(default40, rel=0.02)
