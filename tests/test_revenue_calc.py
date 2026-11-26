"""Micro-calculators of the revenue module: payslip sanity, vectorisation, monotonicity."""
import numpy as np
import pytest

from belsim.core.dims import CONS_CATS
from belsim.core.params import Change, Measure, ParamSet, Scenario
from belsim.fiscal.revenue import calc


@pytest.fixture(scope="module")
def pset():
    return ParamSet.load()


@pytest.fixture(scope="module")
def p(pset):
    return pset.at(2024)


def test_median_full_time_payslip(p):
    r = calc.tax_on_income(45_000.0, p, "emp_private")
    net, cost = float(r["net"][0]), float(r["employer_cost"][0])
    assert 29_000 < net < 31_000
    wedge = 1.0 - net / cost
    # OECD Taxing Wages reports 52-53 % at the (higher) average full-time wage; at 45k the
    # legislated scales give 48-50 %.
    assert 0.47 < wedge < 0.53
    assert float(r["ssc_employee"][0] - r["special_ssc"][0]) == pytest.approx(0.1307 * 45_000)
    assert 0.26 < float(r["ssc_employer"][0]) / 45_000 < 0.29
    # identities
    assert net == pytest.approx(45_000 - float(r["ssc_employee"][0]) - float(r["pit"][0]))
    assert cost == pytest.approx(45_000 + float(r["ssc_employer"][0]))
    assert float(r["pit"][0]) == pytest.approx(float(r["pit_federal"][0] + r["pit_regional"][0] + r["pit_local"][0]))
    # regional PIT is about a quarter of the tax, municipal about 7 %
    assert 0.22 < float(r["pit_regional"][0] / r["pit"][0]) < 0.27
    assert 0.06 < float(r["pit_local"][0] / r["pit"][0]) < 0.075
    # wedge rises with the wage and reaches the OECD figure around the average full-time wage
    hi = calc.tax_on_income(60_000.0, p, "emp_private")
    assert 0.51 < 1.0 - float(hi["net"][0] / hi["employer_cost"][0]) < 0.55


def test_work_bonus_kicks_in_for_low_wages(p):
    low, mid = (calc.tax_on_income(g, p, "emp_private") for g in (30_000.0, 50_000.0))
    rate = lambda r: float((r["ssc_employee"][0] - r["special_ssc"][0]) / r["gross"][0])
    assert rate(mid) == pytest.approx(0.1307)
    assert rate(low) < 0.06
    # NSSO formula: parts A + B in full below the part-B limit (2024 averages 116.9 + 157.7)
    assert calc.work_bonus(np.array([28_000.0]), p)[0] == pytest.approx(12 * (116.9 + 157.7))
    assert calc.work_bonus(np.array([50_000.0]), p)[0] == 0.0
    # between the two limits only part A is left, in full at the part-A limit
    a, b = calc.work_bonus_parts(np.array([2_693.8 * 13.92]), p)
    assert a[0] == pytest.approx(12 * 116.9) and b[0] == pytest.approx(0.0, abs=1e-6)
    # structural reduction lowers the employer rate for low wages
    er = lambda r: float(r["ssc_employer"][0] / r["gross"][0])
    assert er(low) < er(mid) - 0.04


def test_marginal_rate_at_the_top(p):
    a, b = (calc.tax_on_income(g, p, "emp_private") for g in (150_000.0, 151_000.0))
    marginal = 1.0 - float(b["net"][0] - a["net"][0]) / 1_000
    assert 0.58 < marginal < 0.62      # 13.07 % + 50 % x (1 + surcharges) on the rest


@pytest.mark.parametrize("kind", calc.KINDS)
def test_vectorised_and_monotone(p, kind):
    g = np.linspace(0.0, 200_000.0, 2001)
    r = calc.tax_on_income(g, p, kind)
    for k in ("gross", "employer_cost", "ssc_employer", "ssc_employee", "pit", "net"):
        assert r[k].shape == g.shape and np.isfinite(r[k]).all(), k
    assert (r["net"] >= -1e-9).all() and (r["pit"] >= 0).all() and (r["ssc_employee"] >= 0).all()
    for k in ("net", "pit", "employer_cost"):
        assert (np.diff(r[k]) >= -1e-6).all(), (kind, k)
    np.testing.assert_allclose(r["net"], g - r["ssc_employee"] - r["pit"])
    np.testing.assert_allclose(r["employer_cost"], g + r["ssc_employer"])
    # same result element by element, and 2-D input keeps its shape
    one = calc.tax_on_income(g[777], p, kind)
    assert float(one["net"][0]) == pytest.approx(float(r["net"][777]))
    assert calc.tax_on_income(g.reshape(3, 667), p, kind)["pit"].shape == (3, 667)


def test_kinds(p):
    se = calc.tax_on_income(np.array([40_000.0, 200_000.0]), p, "self_employed")
    assert float(se["ssc_employee"][0]) == pytest.approx(0.205 * 40_000)
    assert float(se["ssc_employee"][1]) == pytest.approx(0.205 * 72_810.95 + 0.1416 * (107_300.30 - 72_810.95))
    assert (se["ssc_employer"] == 0).all()

    pen = calc.tax_on_income(np.array([18_000.0, 30_000.0]), p, "pension")
    assert float(pen["pit"][0]) == 0.0 and float(pen["ssc_employee"][0]) == 0.0   # modest pension: no tax
    assert float(pen["ssc_employee"][1]) == pytest.approx(0.0355 * 30_000)
    wage = calc.tax_on_income(30_000.0, p, "emp_private")
    # a pension is taxed less than... the same taxable income without the pension reduction
    assert float(pen["pit"][1]) < float(calc.tax_on_income(30_000.0, p, "self_employed")["pit"][0]) + 2_000

    un = calc.tax_on_income(np.array([17_000.0]), p, "unemployment")
    assert float(un["pit"][0]) == 0.0 and float(un["net"][0]) == 17_000.0
    dis = calc.tax_on_income(np.array([25_000.0]), p, "disability")
    assert float(dis["pit"][0]) < float(calc.tax_on_income(np.array([25_000.0]), p, "unemployment")["pit"][0])
    assert float(wage["employer_cost"][0]) > 30_000

    car = calc.tax_on_income(60_000.0, p, "emp_private", company_car=True)
    no_car = calc.tax_on_income(60_000.0, p, "emp_private")
    assert 700 < float(car["pit"][0] - no_car["pit"][0]) < 1_300

    with pytest.raises(ValueError):
        calc.tax_on_income(1.0, p, "student")


def test_rate_calculators(p):
    rates = calc.consumption_tax_rates(p)
    assert set(rates) == set(CONS_CATS)
    assert rates["housing"] == 0.0
    assert 0.06 <= rates["food"] < 0.10
    assert 0.17 < rates["other_goods"] < 0.21
    assert 1.0 < rates["fuel"] < 1.3 and 7.5 < rates["tobacco"] < 9.5 and 0.3 < rates["alcohol"] < 0.7
    assert 0.21 < calc.cit_effective_rate(p) < 0.25
    assert 0.20 < calc.capital_income_tax_rate(p) < 0.30
    assert 0.05 < calc.property_transaction_tax_rate(p) < 0.125


def test_calculators_follow_reforms(pset, p):
    def with_change(*c):
        return pset.with_scenario(Scenario("r", [Measure("m", list(c))])).at(2024)

    q = with_change(Change("vat.rate_standard", 0.22, 2024))
    r0, r1 = calc.consumption_tax_rates(p), calc.consumption_tax_rates(q)
    assert r1["alcohol"] > r0["alcohol"] and r1["housing"] == 0.0
    assert r1["other_goods"] - r0["other_goods"] == pytest.approx(0.01 * p["vat.mix"]["other_goods"][0])
    q = with_change(Change("vat.mix", [1.0, 0.0, 0.0, 0.0], 2024, key="restaurants"))
    assert calc.consumption_tax_rates(q)["restaurants"] == pytest.approx(0.21)
    q = with_change(Change("cit.rate", 0.20, 2024))
    assert calc.cit_effective_rate(q) < calc.cit_effective_rate(p) - 0.03
    q = with_change(Change("cap.withholding_rate", 0.35, 2024))
    full = 1.0 - p["cap.income_share_exempt"] - p["cap.income_share_reduced"]
    assert calc.capital_income_tax_rate(q) == pytest.approx(calc.capital_income_tax_rate(p) + 0.05 * full)
    q = with_change(Change("pit.bracket_rates", 0.55, 2024, key="3"))
    assert float(calc.tax_on_income(120_000.0, q, "emp_private")["pit"][0]) > float(
        calc.tax_on_income(120_000.0, p, "emp_private")["pit"][0])
    q = with_change(Change("pit.basic_allowance", 15_000, 2024))
    d = calc.tax_on_income(45_000.0, q, "emp_private")["net"][0] - calc.tax_on_income(45_000.0, p, "emp_private")["net"][0]
    assert 1_300 < float(d) < 1_500     # (550 x 25 % + 3,880 x 30 %: special allowance scale) x (1 + surcharges)


def test_thresholds_are_indexed(pset):
    index = lambda kind, year: 1.0 if (year <= 2024 or kind == "wage_link") else 1.10
    p25, p30 = pset.at(2025, index), pset.at(2030, index)
    assert p25["pit.basic_allowance"] == pytest.approx(10_570 * 1.10)
    assert p30["pit.basic_allowance"] - p30["pit.allowance_neutralised_replacement"] == 15_600 - 3_396
    assert p30["pit.bracket_thresholds"][0] == pytest.approx(15_820 * 1.10)
    assert p30["pit.bracket_rates"] == [0.25, 0.40, 0.45, 0.50]
    # with wages and thresholds both up 10 %, average tax rates are unchanged (no bracket creep)
    a = calc.tax_on_income(45_000.0, pset.at(2024), "emp_private")
    b = calc.tax_on_income(49_500.0, p25, "emp_private")
    assert float(b["pit"][0] / 49_500) == pytest.approx(float(a["pit"][0] / 45_000), rel=1e-6)


def test_special_contribution_scale(pset, p):
    """Annual scale of the law of 30 March 1994 on taxable income; halved and individualised in 2028."""
    x = np.array([18_000.0, 21_070.96, 37_344.0, 40_997.26, 70_000.0])
    np.testing.assert_allclose(calc.special_ssc(x, p), [0.0, 123.95, 335.50, 481.96, 731.28], atol=0.02)
    p28 = pset.at(2028)
    np.testing.assert_allclose(calc.special_ssc(np.array([19_000.0, 21_070.96, 50_000.0]), p28),
                               [0.0, 98.37, 365.64], atol=0.02)


def test_employer_contributions(pset, p):
    """Structural reduction R = 0.14 (S0 - S) + gamma (S2 - S); ceiling for very high wages from mid-2025."""
    g = np.array([26_000.0, 36_000.0, 50_000.0, 1_000_000.0])
    er = calc.employer_ssc_private(g, p)
    assert er[1] == pytest.approx(0.277 * 36_000 - 0.14 * (43_622.58 - 36_000))
    assert er[0] == pytest.approx(0.277 * 26_000 - 0.14 * (43_622.58 - 26_000) - 0.40 * (27_196.51 - 26_000))
    # below the full-time minimum wage: part-time work at that wage - the reduction of a full-time
    # minimum wage times the fraction worked, so contributions are not wiped out
    # (ssc.fte_floor_wage; 0 switches the rule off)
    floor = 24_576.0
    fte = pset.with_scenario(Scenario(name="s", measures=[Measure(name="m", changes=[Change("ssc.fte_floor_wage", floor, 2024)])])).at(2024)
    half = calc.employer_ssc_private(np.array([0.5 * floor]), fte)[0]
    assert half == pytest.approx(0.5 * calc.employer_ssc_private(np.array([floor]), fte)[0])
    assert 0.05 < half / (0.5 * floor) < 0.20
    off = pset.with_scenario(Scenario(name="s", measures=[Measure(name="m", changes=[Change("ssc.fte_floor_wage", 0, 2024)])])).at(2024)
    assert calc.employer_ssc_private(np.array([0.5 * floor]), off)[0] == pytest.approx(0.027 * 0.5 * floor)
    assert er[2] == pytest.approx(0.277 * 50_000) and er[3] == pytest.approx(277_000)
    p26 = pset.at(2026)
    capped = calc.employer_ssc_private(np.array([1_000_000.0]), p26)[0]
    assert capped == pytest.approx(0.027 * 1e6 + 0.25 * 350_268)
    # 2025 reform: the very-low-wage component is wider (S2 37,440) and flatter (0.15)
    p25 = pset.at(2025)
    assert calc.employer_ssc_private(np.array([30_000.0]), p25)[0] < calc.employer_ssc_private(np.array([30_000.0]), p)[0]


def test_income_tax_reform_path(pset, p):
    """Law of 15 July 2026: higher allowance for earned income only, stronger fiscal work bonus,
    smaller then no reduction for unemployment benefits, high pensions lose the reduction."""
    assert [pset.at(y)["pit.basic_allowance"] for y in (2025, 2026, 2029, 2030)] == [10_570, 11_550, 14_450, 15_600]
    p26, p30 = pset.at(2026), pset.at(2030)
    net = lambda q, kind, g=45_000.0: float(calc.tax_on_income(g, q, kind)["net"][0])
    gain = net(p30, "emp_private") - net(p, "emp_private")      # plain view: nothing indexed
    assert gain == pytest.approx(0.25 * (15_600 - 10_570) * sum(calc.level_shares(p)) + (731.28 - 365.64), rel=1e-3)
    assert sum(calc.level_shares(p)) == pytest.approx((1 - 0.24957) * 1.332 * 1.073, rel=2e-3)
    assert net(p30, "self_employed") > net(p, "self_employed") + 1_200
    # neutralised for replacement income: same tax as with the pre-reform allowance at 2030 prices
    old = pset.with_scenario(Scenario("r", [Measure("m", [
        Change("pit.basic_allowance", 15_600 - 3_396, 2024), Change("pit.allowance_special_scale", 0, 2024)])])).at(2024)
    assert net(p30, "pension", 25_000.0) == pytest.approx(net(old, "pension", 25_000.0))
    assert net(p30, "disability", 25_000.0) == pytest.approx(net(old, "disability", 25_000.0))
    assert net(p26, "unemployment", 20_000.0) < net(p, "unemployment", 20_000.0) - 1_000
    assert net(p30, "unemployment", 20_000.0) < net(p26, "unemployment", 20_000.0)
    assert net(p26, "pension", 80_000.0) < net(p, "pension", 80_000.0) - 700
    low = lambda q: float(calc.tax_on_income(28_000.0, q, "emp_private")["pit"][0])
    assert low(pset.at(2028)) < low(p26) < low(pset.at(2025))


# ------------------------------------------------------------------ household rules (tax units)

def test_tax_on_income_exposes_allowance_and_reductions(p):
    g = np.array([20_000.0, 45_000.0, 90_000.0])
    r = calc.tax_on_income(g, p, "emp_private")
    assert np.all(r["allowance"] == p["pit.basic_allowance"]) and r["reduction"][0] > 0 and r["reduction"][2] == 0
    np.testing.assert_allclose(r["state_tax"], calc.state_tax(r["taxable_income"], r["allowance"], r["reduction"], p))
    f, reg, mun = calc.level_shares(p)
    np.testing.assert_allclose(r["pit_federal"], f * r["state_tax"])
    np.testing.assert_allclose(r["pit"], (f + reg + mun) * r["state_tax"])
    pen = calc.tax_on_income(30_000.0, p, "pension")
    assert float(pen["reduction"][0]) == pytest.approx(p["pit.reduction_replacement"]["pension"], rel=0.05)
    assert (r["child_credit"] == 0).all() and (r["marital_quotient"] == 0).all()


def test_household_argument_and_companion(p):
    g = np.linspace(0.0, 150_000.0, 31)
    single = calc.tax_on_income(g, p, "emp_private")
    k = sum(calc.level_shares(p))
    rate0 = p["pit.bracket_rates"][0]
    # dependent children: supplement at the bottom of the schedule, stacked on the basic allowance
    kid = calc.tax_on_income(g, p, "emp_private", household={"children": 1, "single_parent": 0})
    assert p["pit.basic_allowance"] + calc.child_allowance(1, p) < p["pit.bracket_thresholds"][0]
    # special allowance scale (art. 134): 25 % up to its first limit, 30 % above
    limit, rate1 = p["pit.allowance_scale_thresholds"][0], p["pit.allowance_second_band_rate"]
    at_25 = limit - p["pit.basic_allowance"]
    value = rate0 * at_25 + rate1 * (float(calc.child_allowance(1, p)) - at_25)
    assert float(single["pit"][-1] - kid["pit"][-1]) == pytest.approx(value * k)
    three = calc.tax_on_income(g, p, "emp_private", household={"children": 3, "single_parent": 0})
    assert float(single["pit"][-1] - three["pit"][-1]) > 0.3 * float(calc.child_allowance(3, p)) * k   # partly at 40 %
    assert float(kid["allowance"][0]) == pytest.approx(p["pit.basic_allowance"] + float(calc.child_allowance(1, p)))
    # single parent: one more supplement; no tax to reduce -> refundable credit, capped per child
    sp = calc.tax_on_income(g, p, "emp_private", household={"children": 2})
    assert float(sp["child_credit"][0]) == pytest.approx(min(rate0 * float(calc.child_allowance(2, p)), 2 * p["pit.child_credit_max_per_child"]))
    assert float(sp["pit"][0]) == pytest.approx(-float(sp["child_credit"][0])) and float(sp["child_credit"][-1]) == 0.0
    assert np.all(np.diff(sp["net"]) > 0)
    # vectorised over the number of children
    kids = np.arange(0, 8)[:, None]
    grid = calc.tax_household(p, [("emp_private", g)], kids, single_parent=0.0)
    assert grid["income_tax"].shape == (8, 31)
    np.testing.assert_allclose(grid["income_tax"][0], single["pit"])
    np.testing.assert_allclose(grid["income_tax"][1], kid["pit"])
    assert np.all(np.diff(grid["income_tax"], axis=0) <= 1e-9)
    s = calc.child_allowance(np.arange(0, 8), p)
    assert s[0] == 0 and s[6] - s[5] == pytest.approx(p["pit.child_allowance_extra_per_child"])
    with pytest.raises(ValueError):
        calc.tax_household(p, [])


def test_marital_quotient_and_allowance_transfer(pset, p):
    g = 60_000.0
    single = calc.tax_on_income(g, p, "emp_private")
    couple = calc.tax_household(p, [("emp_private", g), ("none", 0.0)])
    t = float(single["taxable_income"][0])
    q = min(p["pit.marital_quotient_rate"] * t, p["pit.marital_quotient_max"]["work"])
    assert float(couple["marital_quotient"][0]) == pytest.approx(q)
    assert float(calc.marital_quotient(t, 0.0, p, "work")) == pytest.approx(q)
    a, b = couple["adults"]
    assert float(a["taxable_income"][0]) == pytest.approx(t - q) and float(b["taxable_income"][0]) == pytest.approx(q)
    assert float(couple["state_tax"][0]) < float(single["state_tax"][0])
    # not taxed jointly (de facto cohabitants): no quotient, no transfer
    apart = calc.tax_household(p, [("emp_private", g), ("none", 0.0)], joint=0.0)
    assert float(apart["income_tax"][0]) == pytest.approx(float(single["pit"][0]))
    # pensioner couples keep a higher maximum during the phase-down, none from 2045
    p30, p45 = pset.at(2030), pset.at(2045)
    assert p30["pit.marital_quotient_max"]["work"] < p30["pit.marital_quotient_max"]["pension"] < p["pit.marital_quotient_max"]["pension"]
    pens = calc.tax_household(p45, [("pension", 40_000.0), ("none", 0.0)])
    assert float(pens["marital_quotient"][0]) == 0.0
    # 2030: the quotient (6,730) is below the partner's allowance (15,600): the unused part of
    # that allowance goes to the earner, so the couple pays less than the earner alone
    one = calc.tax_on_income(g, p30, "emp_private")
    two = calc.tax_household(p30, [("emp_private", g), ("none", 0.0)])
    unused = p30["pit.basic_allowance"] - float(two["marital_quotient"][0])
    assert unused > 8_000 and float(two["adults"][1]["state_tax"][0]) == 0.0
    assert float(two["adults"][0]["allowance"][0]) == pytest.approx(p30["pit.basic_allowance"] + unused)
    gain = float(one["state_tax"][0] - two["state_tax"][0])
    assert gain > 0.40 * float(two["marital_quotient"][0]) + 0.25 * unused - 1.0


def test_regional_surcharges_in_the_payslip(pset, p):
    g = 50_000.0
    avg = calc.tax_on_income(g, p, "emp_private")
    by = {r: calc.tax_on_income(g, p, "emp_private", household={"region": r}) for r in ("FLA", "WAL", "BRU")}
    assert float(by["BRU"]["pit_regional"][0]) < float(by["FLA"]["pit_regional"][0]) == pytest.approx(float(by["WAL"]["pit_regional"][0]))
    assert float(by["BRU"]["pit_local"][0]) < float(by["FLA"]["pit_local"][0]) < float(by["WAL"]["pit_local"][0])
    assert float(by["BRU"]["pit"][0]) < float(avg["pit"][0]) < float(by["WAL"]["pit"][0])
    assert all(float(v["pit_federal"][0]) == pytest.approx(float(avg["pit_federal"][0])) for v in by.values())
    w = calc.region_weights(p)
    assert sum(w[i] * float(by[r]["pit"][0]) for i, r in enumerate(("FLA", "WAL", "BRU"))) == pytest.approx(float(avg["pit"][0]))
    # one region, one Change
    q = pset.with_scenario(Scenario("r", [Measure("m", [Change("pit.regional_surcharge", 0.30, 2024, key="WAL")])])).at(2024)
    assert float(calc.tax_on_income(g, q, "emp_private", household={"region": "FLA"})["pit"][0]) == float(by["FLA"]["pit"][0])
    assert float(calc.tax_on_income(g, q, "emp_private", household={"region": "WAL"})["pit"][0]) < float(by["WAL"]["pit"][0])
    assert float(calc.tax_on_income(g, q, "emp_private")["pit"][0]) < float(avg["pit"][0])


def test_special_contribution_of_a_household(pset, p):
    """Until income 2027 spouses and legal cohabitants owe one contribution on their joint
    income, on a scale of their own (law of 30 March 1994 art. 108); from 2028 each partner pays
    on his or her own income."""
    x = np.array([18_592.02, 21_070.96, 60_181.95, 70_000.0, 81_944.0, 120_000.0])
    np.testing.assert_allclose(calc.special_ssc(x, p, joint=True), [0.0, 123.95, 632.39, 632.39, 731.28, 731.28], atol=0.02)
    p28 = pset.at(2028)
    assert calc.special_joint_scale(p28) is None
    np.testing.assert_allclose(calc.special_ssc(x, p28, joint=True), calc.special_ssc(x, p28))

    two = [("emp_private", 70_000.0), ("emp_private", 70_000.0)]
    alone = float(calc.tax_on_income(70_000.0, p, "emp_private")["special_ssc"][0])
    couple = calc.tax_household(p, two)
    joint_income = float(couple["adults"][0]["taxable_income"][0] + couple["adults"][1]["taxable_income"][0])
    assert float(couple["special_ssc"][0]) == pytest.approx(float(calc.special_ssc(joint_income, p, joint=True)))
    assert float(couple["special_ssc"][0]) == pytest.approx(731.28) and 2.0 * alone > 1_000   # one maximum, not two
    # at modest wages the joint scale asks more of a couple than of two single persons
    modest = [("emp_private", 35_000.0), ("emp_private", 35_000.0)]
    assert float(calc.tax_household(p, modest)["special_ssc"][0]) > 2.0 * float(calc.tax_on_income(35_000.0, p, "emp_private")["special_ssc"][0])
    # the household's contribution is carried by the employees; the total of contributions adds up
    paid = sum(float(a["ssc_employee"][0]) for a in couple["adults"])
    assert paid == pytest.approx(float(couple["ssc_employee"][0]))
    # partners who are not taxed jointly, and everybody from 2028, pay on their own income
    assert float(calc.tax_household(p, two, joint=0.0)["special_ssc"][0]) == pytest.approx(2.0 * alone)
    alone28 = float(calc.tax_on_income(70_000.0, p28, "emp_private")["special_ssc"][0])
    assert float(calc.tax_household(p28, two)["special_ssc"][0]) == pytest.approx(2.0 * alone28)
    # an employee with a partner on a pension: the joint income counts, the employee pays
    mixed = calc.tax_household(p, [("emp_private", 30_000.0), ("pension", 20_000.0)])
    assert float(mixed["special_ssc"][0]) > float(calc.tax_on_income(30_000.0, p, "emp_private")["special_ssc"][0])
    assert float(mixed["adults"][1]["ssc_employee"][0]) == pytest.approx(float(calc.pension_health_contribution(20_000.0, p)))


def test_vat_compliance_gap_stays_with_supplies_leaving_the_standard_rate(pset, p):
    """The gap is on standard-rated supplies; a base moved from 21 % to 12 % keeps it, a base
    moved from 6 % to 12 % stays without, a base moved up to 21 % takes it."""
    gap = p["vat.compliance_gap"]
    s, r = calc.vat_rate_parts(p, "housing_investment")
    assert calc.vat_collected_rate(p, "housing_investment") == pytest.approx(s * (1.0 - gap) + r)
    assert calc.vat_collected_rate(p, "housing_investment", p) == calc.vat_collected_rate(p, "housing_investment")
    mix = list(p["vat.mix"]["housing_investment"])
    new_housing = pset.with_scenario(Scenario("r", [Measure("m", [Change(
        "vat.mix", [0.0, mix[0] + mix[1], mix[2], mix[3]], 2024, key="housing_investment")])])).at(2024)
    lost = calc.vat_collected_rate(p, "housing_investment", p) - calc.vat_collected_rate(new_housing, "housing_investment", p)
    assert lost == pytest.approx(mix[0] * (0.21 - 0.12) * (1.0 - gap))
    assert calc.vat_collected_rate(new_housing, "housing_investment") == pytest.approx(calc.vat_rate(new_housing, "housing_investment"))
    rest = list(p["vat.mix"]["restaurants"])
    hotels = pset.with_scenario(Scenario("r", [Measure("m", [Change(
        "vat.mix", [rest[0], rest[1] + 0.1, rest[2] - 0.1, rest[3]], 2024, key="restaurants")])])).at(2024)
    gain = calc.vat_collected_rate(hotels, "restaurants", p) - calc.vat_collected_rate(p, "restaurants", p)
    assert gain == pytest.approx(0.1 * (0.12 - 0.06))
    up = pset.with_scenario(Scenario("r", [Measure("m", [Change(
        "vat.mix", [rest[0] + 0.1, rest[1], rest[2] - 0.1, rest[3]], 2024, key="restaurants")])])).at(2024)
    gain = calc.vat_collected_rate(up, "restaurants", p) - calc.vat_collected_rate(p, "restaurants", p)
    assert gain == pytest.approx(0.1 * (0.21 * (1.0 - gap) - 0.06))


def test_bracket_base_factor_scales_the_income_taxed_in_a_bracket(p):
    x = np.array([20_000.0, 40_000.0, 90_000.0])
    thr, rates = p["pit.bracket_thresholds"], p["pit.bracket_rates"]
    full = calc.state_tax(x, 0.0, 0.0, p)
    alt = calc.state_tax(x, 0.0, 0.0, p, (1.0, 1.0, 0.88, 0.88))
    upper = rates[2] * np.clip(x - thr[1], 0.0, thr[2] - thr[1]) + rates[3] * np.maximum(x - thr[2], 0.0)
    np.testing.assert_allclose(full - alt, 0.12 * upper)
    np.testing.assert_allclose(calc.state_tax(x, 0.0, 0.0, p, (1.0, 1.0, 1.0, 1.0)), full)
