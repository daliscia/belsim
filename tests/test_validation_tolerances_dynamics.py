"""Regression guards for the dynamic response of the economy to measures: Federal Planning Bureau
HERMES variants (WP 4 DC2024, WP 10-18) and the evaluations of the 2015 index jump (National
Bank, Planning Bureau). References and sources are in data/reference/validation.yaml.

Six references are compared like for like, with a measure that differs from the one in
validation.yaml (see `like_for_like`): tolerances here are for those measures. The National
Bank's larger GDP effects are checked with the documented "National Bank" value of the
competitiveness lever (NBB_VARIANT), not with the central calibration.
"""
from functools import lru_cache

import pytest
import yaml

from belsim import validation as V
from belsim.core import engine
from belsim.core.params import Change, Measure, ParamSet, Scenario

YEAR = 2027                      # first year of every variant
END = 2033                       # last simulated year


# ----------------------------------------------------------------------------- like for like
# 1. HERMES employee contributions: the 0.5% of GDP is the contribution itself (table 15: social
#    contributions +2,971 mn, household income tax -1,035 mn in year t); validation.yaml sizes
#    the rate on the primary balance after the fall in income tax, i.e. about twice the shock.
EMPLOYEE_SSC = [{"param": "ssc.employee_rate", "value": 0.0172, "op": "add", "start": YEAR}]

# 2. Index jump of 2015, with the timing of the evaluations (Chamber DOC 54 0913/001, Planning
#    Bureau report, section 2.1, "according to the modalities of the National Bank"): the index
#    that drives private wages is 0.2% lower in the first year, 1.4% in the second and 2% from the
#    third; the one that drives public pay and social benefits (pivot index) 0.7% lower in the
#    first year and 2% from the second. Not one whole year of indexation (3.1% in the 2027
#    baseline). Benefits: changes of the baseline factor (0.9 in 2028). Public pay has no
#    indexation lever of its own: it follows the wage index, and the faster loss is given with
#    its wage drift (-0.5 point in 2027, given back in 2029-2030).
INDEX_JUMP = [
    {"param": "labour.indexation_factor", "value": 0.937, "start": 2027, "end": 2027},
    {"param": "labour.indexation_factor", "value": 0.547, "start": 2028, "end": 2028},
    {"param": "labour.indexation_factor", "value": 0.70, "start": 2029, "end": 2029},
    {"param": "exp.benefit_indexation_factor", "value": -0.226, "op": "add", "start": 2027, "end": 2027},
    {"param": "exp.benefit_indexation_factor", "value": -0.674, "op": "add", "start": 2028, "end": 2028},
    {"param": "labour.wage_drift", "key": "emp_public", "value": -0.005, "op": "add", "start": 2027, "end": 2027},
    {"param": "labour.wage_drift", "key": "emp_public", "value": -0.001, "op": "add", "start": 2028, "end": 2028},
    {"param": "labour.wage_drift", "key": "emp_public", "value": 0.005, "op": "add", "start": 2029, "end": 2029},
    {"param": "labour.wage_drift", "key": "emp_public", "value": 0.001, "op": "add", "start": 2030, "end": 2030},
]

# 3. HERMES VAT variants: the whole 0.5% of GDP is VAT on private consumption (tables 10-11).
#    All rates are scaled by m and the rows of the three non-household bases of vat.mix are
#    divided by m, so that their average rate is unchanged; m gives a static 0.5% of GDP.
NON_HOUSEHOLD_VAT_BASES = ("housing_investment", "government", "exempt_sectors")


def vat_on_consumption(m: float) -> list[dict]:
    mix = ParamSet.load().value("vat.mix", YEAR)
    out = [{"param": f"vat.rate_{r}", "value": m, "op": "mul", "start": YEAR}
           for r in ("standard", "intermediate", "reduced")]
    for key in NON_HOUSEHOLD_VAT_BASES:
        taxed = [round(x / m, 6) for x in mix[key][:3]]
        out.append({"param": "vat.mix", "key": key, "value": taxed + [round(1.0 - sum(taxed), 6)], "start": YEAR})
    return out


@lru_cache(maxsize=1)
def vat_multiplier(target: float = 0.5) -> float:
    """Rate multiplier whose static yield is `target` % of GDP in the first year."""
    base = engine.run(None, end=YEAR)
    b = base.at(YEAR)

    def size(m: float) -> float:
        changes = [Change(**c) for c in vat_on_consumption(m)] + [Change("beh.enabled", 0, YEAR - 1)]
        s = engine.run(Scenario(name="s", measures=[Measure(name="m", changes=changes)]), end=YEAR,
                       baseline=base).at(YEAR)
        return (s.fiscal["primary_balance"] / s.macro["gdp_nominal"]
                - b.fiscal["primary_balance"] / b.macro["gdp_nominal"]) * 100.0

    lo, hi = 1.0, 1.5
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if size(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


# 4. HERMES social benefits: the 0.5% of GDP is the gross cut in benefits (table 9: social
#    benefits -2,999 mn in year t, household income tax -547 mn because pensions are taxable);
#    validation.yaml sizes the cut on the primary balance after tax, i.e. 0.63% of GDP gross.
def benefits_cut(k: float) -> list[dict]:
    return [{"param": "pension.level_factor", "value": round(1.0 - 0.0274 * k, 6), "op": "mul", "start": YEAR},
            {"param": "family.allowance_per_child", "value": round(1.0 - 0.2115 * k, 6), "op": "mul", "start": YEAR}]


@lru_cache(maxsize=1)
def benefits_scale(target: float = -0.5) -> float:
    """Scale of the validation.yaml benefit cut whose static effect on primary expenditure is
    `target` % of GDP in the first year."""
    base = engine.run(None, end=YEAR)
    b = base.at(YEAR)

    def size(k: float) -> float:
        changes = [Change(**c) for c in benefits_cut(k)] + [Change("beh.enabled", 0, YEAR - 1)]
        s = engine.run(Scenario(name="s", measures=[Measure(name="m", changes=changes)]), end=YEAR,
                       baseline=base).at(YEAR)
        return (s.fiscal["primary_expenditure"] - b.fiscal["primary_expenditure"]) / b.macro["gdp_nominal"] * 100.0

    lo, hi = 0.3, 1.5
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if size(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


# 5. HERMES employer-contribution cuts of 0.5% of GDP (WP 10-18), sized with employer
#    contributions assessed on full-time-equivalent wages (ssc.fte_floor_wage on): a general cut
#    of 2.12 points of the basic rate; a cut targeted at low wages through the slope of the
#    structural reduction (the bottom four wage deciles; their contributions fall by 60-70%).
EMPLOYER_SSC_CUT_GENERAL = [{"param": "ssc.employer_rate", "value": -0.02124, "op": "add", "start": YEAR}]
EMPLOYER_SSC_CUT_LOW_WAGE = [{"param": "ssc.structural_reduction_slope", "value": 0.7804, "op": "add", "start": YEAR}]

# 6. "National Bank" variant: the Bank's quarterly model has a much stronger export response to
#    labour costs than HERMES. Its evaluations are compared with the value of the competitiveness
#    lever documented in params/dynamics.yaml, not with the central (Planning Bureau) one.
NBB_VARIANT = {"param": "dyn.wage_cost.net_export_output_elasticity", "value": 0.30, "start": YEAR - 1}


def like_for_like() -> dict[str, list[dict]]:
    m = vat_multiplier()
    return {
        "hermes_social_benefits_cut": benefits_cut(benefits_scale()),
        "hermes_employee_ssc_increase": EMPLOYEE_SSC,
        "index_jump_2015_fpb": INDEX_JUMP, "index_jump_2015_nbb": INDEX_JUMP,
        "index_jump_2015_nbb_path": INDEX_JUMP, "index_jump_2015_fpb_path": INDEX_JUMP,
        "hermes_vat_increase": vat_on_consumption(m), "hermes_vat_increase_prices": vat_on_consumption(m),
        "hermes2018_vat_cut": vat_on_consumption(2.0 - m),
        "hermes2018_employer_ssc_cut_general": EMPLOYER_SSC_CUT_GENERAL,
        "hermes2018_employer_ssc_cut_general_prices": EMPLOYER_SSC_CUT_GENERAL,
        "hermes2018_employer_ssc_cut_low_wage": EMPLOYER_SSC_CUT_LOW_WAGE,
    }


# 7. Planning Bureau evaluation of the 2015 index jump: jobs in thousands of 2016-2020, when
#    domestic employment was 4.67 / 4.80 / 4.88 million (Eurostat nama_10_pe, EMP_DC); the model
#    years 2028-2032 have 5.2 to 5.3 million. Compared in % of employment (reference scaled to the
#    baseline employment of the model year). Real disposable income: the model's own flows after
#    tax (macro.household_income_after_tax), not the primary-income proxy of the national accounts.
OBSERVED_EMPLOYMENT = {2016: 4665.3, 2018: 4804.5, 2020: 4875.1}        # thousand persons
RDI = {"num": ["macro.household_income_after_tax"], "den": ["macro.cpi"], "mode": "pct"}


def like_for_like_targets(entry: dict) -> None:
    if entry["id"] != "index_jump_2015_fpb_path":
        return
    base = engine.run(None, end=END)
    for t in entry["targets"]:
        if t["label"].startswith("employment ") and t["ref_year"] in OBSERVED_EMPLOYMENT:
            year = t.get("model_year", t["ref_year"])
            t["ref_value"] = t["ref_value"] * base.at(year).labour["employment"] / 1e3 / OBSERVED_EMPLOYMENT[t["ref_year"]]
        if t["label"].startswith("real household disposable income"):
            t["model"] = dict(RDI)


# (reference id, target label prefix, relative tolerance)
MEASURES = [
    # public spending: GDP path, jobs, budget after feedback
    ("hermes_public_investment_cut", "balance, year t", 0.15),
    ("hermes_public_investment_cut", "balance, t+2", 0.15),
    ("hermes_public_investment_cut", "balance, t+4", 0.15),
    ("hermes_public_consumption_cut", "balance, year t", 0.15),
    ("hermes_public_consumption_cut", "balance, t+2", 0.15),
    ("hermes_public_consumption_cut", "balance, t+4", 0.15),
    ("hermes_social_benefits_cut", "balance, year t", 0.15),            # gross cut of 0.5% of GDP
    ("hermes_social_benefits_cut", "balance, t+2", 0.15),
    ("hermes_social_benefits_cut", "balance, t+4", 0.15),
    ("hermes2018_public_consumption_multiplier", "balance, year t", 0.15),
    ("hermes2018_public_consumption_multiplier", "balance, t+4", 0.15),
    ("hermes_public_investment_cut", "real GDP, year t", 0.35),
    ("hermes_public_investment_cut", "real GDP, t+4", 0.35),            # public capital, persistence
    ("hermes_public_investment_cut", "employment, year t", 0.3),
    ("hermes_public_investment_cut", "employment, t+4", 0.5),
    ("hermes_public_consumption_cut", "real GDP, t+4", 0.2),
    ("hermes_public_consumption_cut", "employment, t+4", 0.5),
    ("hermes_social_benefits_cut", "real GDP, year t", 0.4),
    ("hermes_social_benefits_cut", "real GDP, t+4", 0.45),
    ("hermes2018_public_consumption_multiplier", "real GDP, year t", 0.3),
    ("hermes2018_public_consumption_multiplier", "real GDP, t+4", 0.45),
    # employer contributions: jobs, prices, labour cost, budget after feedback
    ("hermes_employer_ssc_increase", "real GDP, year t", 0.3),
    ("hermes_employer_ssc_increase", "real GDP, t+4", 0.35),
    ("hermes_employer_ssc_increase", "employment, year t", 0.25),
    ("hermes_employer_ssc_increase", "employment, t+2", 0.2),
    ("hermes_employer_ssc_increase", "employment, t+4", 0.15),
    ("hermes_employer_ssc_increase", "balance, year t", 0.3),
    ("hermes_employer_ssc_increase", "balance, t+2", 0.3),
    ("hermes_employer_ssc_increase", "balance, t+4", 0.3),
    ("hermes_employer_ssc_increase_prices", "consumer prices, year t", 0.45),
    ("hermes_employer_ssc_increase_prices", "consumer prices, t+2", 0.35),
    ("hermes_employer_ssc_increase_prices", "consumer prices, t+4", 0.3),
    ("hermes_employer_ssc_increase_prices", "hourly labour cost incl. employer contributions, year t", 0.15),
    ("hermes_employer_ssc_increase_prices", "hourly labour cost incl. employer contributions, t+2", 0.15),
    ("hermes_employer_ssc_increase_prices", "hourly labour cost incl. employer contributions, t+4", 0.15),
    ("hermes2018_employer_ssc_cut_general", "real GDP, t+4", 0.2),
    ("hermes2018_employer_ssc_cut_general", "employment, year t", 0.25),
    ("hermes2018_employer_ssc_cut_general", "employment, t+2", 0.2),
    ("hermes2018_employer_ssc_cut_general", "employment, t+4", 0.2),
    ("hermes2018_employer_ssc_cut_general", "balance, t+4", 0.2),
    ("hermes2018_employer_ssc_cut_general_prices", "consumer prices, year t", 0.45),
    ("hermes2018_employer_ssc_cut_general_prices", "consumer prices, t+2", 0.4),
    ("hermes2018_employer_ssc_cut_general_prices", "consumer prices, t+4", 0.45),
    ("hermes2018_employer_ssc_cut_general_prices", "hourly labour cost incl. employer contributions, year t", 0.3),
    ("hermes2018_employer_ssc_cut_general_prices", "hourly labour cost incl. employer contributions, t+4", 0.3),
    ("hermes2018_employer_ssc_cut_low_wage", "real GDP, t+4", 0.25),
    ("hermes2018_employer_ssc_cut_low_wage", "employment, year t", 0.35),
    ("hermes2018_employer_ssc_cut_low_wage", "employment, t+2", 0.2),
    ("hermes2018_employer_ssc_cut_low_wage", "employment, t+4", 0.2),
    ("employer_ssc_cut_2016_nbb", "balance", 0.3),
    ("hermes2018_employer_ssc_cut_low_wage", "balance, t+4", 0.15),     # jobs placed in the wage deciles whose cost was cut (was 0.5)
    # taxes on workers: jobs lost through bargained wages, small first-year effect
    ("hermes_employee_ssc_increase", "real GDP, year t", 0.4),
    ("hermes_employee_ssc_increase", "real GDP, t+4", 0.2),
    ("hermes_employee_ssc_increase", "employment, t+4", 0.35),
    ("hermes_employee_ssc_increase", "balance, year t", 0.4),
    ("hermes_employee_ssc_increase", "balance, t+4", 0.5),
    ("hermes_pit_increase", "real GDP, year t", 0.3),
    ("hermes_pit_increase", "real GDP, t+4", 0.3),                     # hours response absent from HERMES
    ("hermes_pit_increase", "employment, t+4", 0.3),
    ("hermes2018_vat_cut", "real GDP, t+4", 0.4),
    ("hermes_pit_increase", "balance, year t", 0.25),
    ("hermes_pit_increase", "balance, t+4", 0.25),
    # VAT on private consumption: prices, indexation, jobs, budget
    ("hermes_vat_increase", "real GDP, year t", 0.2),
    ("hermes_vat_increase", "real GDP, t+4", 0.25),     # household real income in the demand block (was 0.55)
    ("hermes_vat_increase", "employment, year t", 0.5),
    ("hermes_vat_increase", "employment, t+4", 0.3),
    ("hermes_vat_increase", "consumer prices, year t", 0.35),
    ("hermes_vat_increase", "balance, year t", 0.2),
    ("hermes_vat_increase", "balance, t+2", 0.4),     # ratio to GDP at baseline tax rates; like for like: test_vat_rise_balance_...
    ("hermes_vat_increase", "balance, t+4", 0.4),
    ("hermes_vat_increase_prices", "consumer prices, t+2", 0.4),
    ("hermes_vat_increase_prices", "consumer prices, t+4", 0.45),
    ("hermes_vat_increase_prices", "nominal hourly labour cost, year t", 0.25),
    ("hermes_vat_increase_prices", "nominal hourly labour cost, t+2", 0.3),
    ("hermes_vat_increase_prices", "nominal hourly labour cost, t+4", 0.25),
    ("hermes_vat_increase_prices", "indirect taxes, year t", 0.2),
    ("hermes_vat_increase_prices", "primary expenditure, year t", 0.2),
    ("hermes_vat_increase_prices", "primary expenditure, t+4", 0.2),
    ("hermes2018_vat_cut", "real GDP, year t", 0.2),
    ("hermes2018_vat_cut", "employment, year t", 0.35),
    ("hermes2018_vat_cut", "employment, t+4", 0.25),
    ("hermes2018_vat_cut", "balance, year t", 0.3),
    ("hermes2018_vat_cut", "consumer prices, year t", 0.45),
    ("hermes2018_vat_cut", "consumer prices, t+4", 0.45),
    ("hermes2018_vat_cut", "nominal hourly labour cost, year t", 0.2),
    # index jump: gradual wage moderation, jobs per point of moderation, prices, budget
    ("index_jump_2015_fpb", "total employment", 0.3),
    ("index_jump_2015_fpb", "nominal labour cost", 0.25),
    ("index_jump_2015_fpb", "government revenue", 0.2),
    ("index_jump_2015_fpb", "government expenditure", 0.2),
    ("index_jump_2015_fpb", "balance", 0.45),     # in points of GDP: the ratio also loses 0.05 with the lower nominal GDP
    ("index_jump_2015_nbb", "private-sector employment", 0.3),
    ("index_jump_2015_nbb", "nominal hourly labour cost", 0.3),
    ("index_jump_2015_nbb", "consumer prices", 0.5),
    ("index_jump_2015_nbb_path", "nominal hourly labour cost, year 1", 0.4),
    ("index_jump_2015_nbb_path", "nominal hourly labour cost, year 2", 0.3),
    ("index_jump_2015_nbb_path", "nominal hourly labour cost, year 3", 0.3),
    ("index_jump_2015_nbb_path", "nominal hourly labour cost, year 4", 0.3),
    ("index_jump_2015_nbb_path", "consumer prices, year 2", 0.55),
    ("index_jump_2015_nbb_path", "consumer prices, year 3", 0.55),
    ("index_jump_2015_nbb_path", "consumer prices, year 4", 0.55),
    ("index_jump_2015_fpb_path", "employment 2016", 0.25),     # in % of employment, see like_for_like_targets
    ("index_jump_2015_fpb_path", "employment 2018", 0.2),
    ("index_jump_2015_fpb_path", "employment 2020", 0.15),
    ("index_jump_2015_fpb_path", "real household disposable income 2020", 0.5),     # income after tax from the model's flows: -0.5% against -0.9%
    ("index_jump_2015_fpb_path", "balance 2016", 0.3),
    ("index_jump_2015_fpb_path", "balance 2018", 0.3),
    ("index_jump_2015_fpb_path", "balance 2020", 0.3),
]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    data = V.load()
    ids = {m[0] for m in MEASURES}
    over = like_for_like()
    data["baseline"] = []
    data["measures"] = [e for e in data["measures"] if e["id"] in ids]
    for entry in data["measures"]:
        if entry["id"] in over:
            entry["measure"] = {"changes": over[entry["id"]]}
        entry["targets"] = [t for t in entry["targets"] if t.get("model_year", t["ref_year"]) <= END]
        like_for_like_targets(entry)
    path = tmp_path_factory.mktemp("validation") / "dynamics.yaml"
    path.write_text(yaml.safe_dump(data))
    return V.run_validation(path, end=END)


@pytest.fixture(scope="module")
def nbb_report(tmp_path_factory):
    """The National Bank's evaluations run with the "National Bank" value of the lever."""
    data = V.load()
    over = like_for_like()
    data["baseline"] = []
    data["measures"] = [e for e in data["measures"] if e["id"] in ("index_jump_2015_nbb", "employer_ssc_cut_2016_nbb")]
    for entry in data["measures"]:
        changes = over.get(entry["id"], entry["measure"]["changes"])
        entry["measure"] = {"changes": list(changes) + [NBB_VARIANT]}
    path = tmp_path_factory.mktemp("validation") / "nbb.yaml"
    path.write_text(yaml.safe_dump(data))
    return V.run_validation(path, end=END)


def _measure(report, id_, label_start):
    for r in report.measures:
        if r.id == id_ and r.label.startswith(label_start):
            return r
    raise AssertionError(f"no measure row {id_} {label_start}")


def test_like_for_like_measures_have_the_reference_size(report):
    assert not report.problems, report.problems
    for id_ in ("hermes_vat_increase", "hermes2018_vat_cut", "hermes_employer_ssc_increase",
                "hermes2018_employer_ssc_cut_general", "hermes2018_employer_ssc_cut_low_wage"):
        r = _measure(report, id_, "sizing")
        assert r.static == pytest.approx(r.ref, rel=0.03), id_
    # employee contributions: 0.5% of GDP of contributions, about half of it net of income tax
    r = _measure(report, "hermes_employee_ssc_increase", "sizing")
    assert 0.2 < r.static < 0.35
    # benefits: 0.5% of GDP gross, less after the income tax and contributions on pensions
    r = _measure(report, "hermes_social_benefits_cut", "sizing")
    assert 0.3 < r.static < 0.45


def test_national_bank_variant_reproduces_the_bank_evaluations(nbb_report):
    """With the documented lever value the GDP effects of the index jump (+0.5%) and of the 2016
    contribution cut (+0.3%) are of the Bank's order of magnitude; the central calibration
    (Planning Bureau) gives about zero for the first, as HERMES does."""
    assert not nbb_report.problems, nbb_report.problems
    for id_, tol in (("index_jump_2015_nbb", 0.35), ("employer_ssc_cut_2016_nbb", 0.5)):
        r = _measure(nbb_report, id_, "real GDP")
        assert abs(r.compared - r.ref) <= tol * abs(r.ref), f"{id_}: model {r.compared:.3f} vs Bank {r.ref:.3f}"
    r = _measure(nbb_report, "index_jump_2015_nbb", "balance")
    assert 0 < r.compared <= 2.0 * r.ref          # right sign and order of magnitude


def test_pension_malus_postpones_retirement_as_in_the_planning_bureau_evaluation():
    """Planning Bureau report 13299, table 2: pension spending 228 / 463 / 677 / 641 / 794 EUR mn
    lower in 2027-2031 (2024 prices), nearly all of it postponed retirement. At about EUR 2.8 bn
    per year of average postponement (some 125,000 new pensions a year of EUR 22,000) that is
    0.08 / 0.17 / 0.24 / 0.23 / 0.28 year. The guard is on the behavioural response in years (the
    pension amounts belong to the expenditure module)."""
    entry = next(e for e in V.load()["measures"] if e["id"] == "pension_bonus_malus_2026")
    base = engine.run(None, end=2031)

    def shift(key):
        m = Measure.from_dict({"name": key, **entry[key]})
        run = engine.run(Scenario(name=key, measures=[m]), end=2031, baseline=base)
        return {y: float(run.at(y).adjust.get("retirement_shift", 0.0)) for y in range(2027, 2032)}

    with_malus, without = shift("measure"), shift("reference_measure")
    years = {y: with_malus[y] - without[y] for y in with_malus}
    implied = {2027: 0.08, 2028: 0.17, 2029: 0.24, 2030: 0.23, 2031: 0.28}
    for y, ref in implied.items():
        assert abs(years[y] - ref) <= 0.4 * ref, f"{y}: {years[y]:.3f} year vs {ref:.2f} implied"
    assert years[2031] < 0.35                      # doubling the malus adds little (saturation)


def test_vat_rise_balance_in_percent_of_gdp_at_market_prices():
    """HERMES VAT variant: balance in % of GDP at market prices, i.e. with the scenario's own
    indirect taxes in the denominator (macro["gdp_nominal_market_prices"]), as in the reference
    (+0.31 / +0.15 / +0.15 point at t / t+2 / t+4)."""
    base = engine.run(None, end=2031)
    changes = [Change(**c) for c in vat_on_consumption(vat_multiplier())]
    scen = engine.run(Scenario(name="vat", measures=[Measure(name="vat", changes=changes)]), end=2031, baseline=base)
    for year, ref, tol in ((2027, 0.31, 0.15), (2029, 0.15, 0.25), (2031, 0.15, 0.2)):
        s, b = scen.at(year), base.at(year)
        assert s.macro["gdp_nominal_market_prices"] > s.macro["gdp_nominal"] * 1.004
        assert b.macro["gdp_nominal_market_prices"] == b.macro["gdp_nominal"]
        gain = (s.fiscal["balance"] / s.macro["gdp_nominal_market_prices"]
                - b.fiscal["balance"] / b.macro["gdp_nominal"]) * 100.0
        assert abs(gain - ref) <= tol * ref, f"{year}: model {gain:.3f} vs official {ref:.3f}"


def test_vat_rise_lowers_real_household_income_and_consumption_demand():
    """HERMES VAT variant (WP 4 DC2024, table 10): real disposable income of households -0.64%
    after four years (private consumption -0.55%). The model's income after tax, deflated with
    consumer prices: benefits are indexed, wages lose part of the indexation again, incomes that
    are not indexed (self-employed, capital) lose the whole price rise."""
    base = engine.run(None, end=2031)
    changes = [Change(**c) for c in vat_on_consumption(vat_multiplier())]
    scen = engine.run(Scenario(name="vat", measures=[Measure(name="vat", changes=changes)]), end=2031, baseline=base)
    s, b = scen.at(2031), base.at(2031)
    k = "household_income_after_tax"
    rdi = ((s.macro[k] / s.macro["cpi"]) / (b.macro[k] / b.macro["cpi"]) - 1.0) * 100.0
    assert abs(rdi - (-0.64)) <= 0.25 * 0.64, f"real disposable income {rdi:.2f}% vs -0.64%"
    real = lambda st, v: v / st.macro["cpi"]
    assert real(s, s.labour["self_employed_income"]) / real(b, b.labour["self_employed_income"]) < 0.992
    assert s.extra["behaviour"]["demand"]["d_household_income_real"] < 0.0


def test_jobs_of_a_low_wage_contribution_cut_are_in_the_low_wage_deciles():
    """Jobs by wage decile (labour.scenario_jobs_private_by_decile): zeros in the baseline, sum
    equal to the difference in private employment, concentrated where employer cost was cut;
    wage deciles, average wage and wage bill consistent with that allocation."""
    import numpy as np
    from belsim.core.dims import J
    base = engine.run(None, end=2031)
    assert all(v == 0.0 for y in (2024, 2031) for v in base.at(y).labour["scenario_jobs_private_by_decile"])
    for changes, low in ((EMPLOYER_SSC_CUT_LOW_WAGE, True), (EMPLOYER_SSC_CUT_GENERAL, False)):
        scen = engine.run(Scenario(name="c", measures=[Measure(name="c", changes=[Change(**c) for c in changes])]),
                          end=2031, baseline=base)
        s, b = scen.at(2031), base.at(2031)
        jobs = np.array(s.labour["scenario_jobs_private_by_decile"])
        d_priv = s.labour["employees_private"] - b.labour["employees_private"]
        assert jobs.sum() == pytest.approx(d_priv, rel=1e-9)
        j = J["emp_private"]
        assert s.wage_dist[j].mean() == pytest.approx(s.labour["avg_wage_private"], rel=1e-9)
        assert s.labour["wage_bill_private"] == pytest.approx(s.labour["avg_wage_private"] * s.labour["employees_private"] / 1e6, rel=1e-9)
        x = s.extra["labour"]
        heads = b.labour["employees_private"] / 10.0 + x["jobs_by_decile"][j]
        assert (heads * x["decile_wage_standard"][j]).sum() / 1e6 == pytest.approx(s.labour["wage_bill_private"], rel=1e-6)
        if low:
            assert jobs[:4].sum() > 1.2 * d_priv and jobs[4:].sum() < 0.0     # more low-wage jobs, some replace others
        else:
            assert abs(jobs[:4].sum() / d_priv - 0.4) < 0.1                   # spread like existing jobs


def test_index_jump_budget_gain_follows_the_planning_bureau_path():
    """Planning Bureau evaluation of the 2015 index jump (Chamber DOC 54 0913/001, table 1):
    budget balance +365 / +794 / +655 / +779 / +894 EUR mn in the first five years, i.e. 0.09 /
    0.18 / 0.15 / 0.17 / 0.19% of GDP; revenue -0.12 / -0.56 / -0.70 / -0.75 / -0.78% and
    expenditure -0.20 / -0.74 / -0.85 / -0.91 / -0.96%. Amounts over the baseline GDP of the
    year, as in the source (the balance ratio itself also moves with nominal GDP)."""
    base = engine.run(None, end=2031)
    scen = engine.run(Scenario(name="jump", measures=[Measure(name="jump", changes=[Change(**c) for c in INDEX_JUMP])]),
                      end=2031, baseline=base)
    official = {2027: (0.09, -0.12, -0.20), 2028: (0.18, -0.56, -0.74), 2029: (0.15, -0.70, -0.85),
                2030: (0.17, -0.75, -0.91), 2031: (0.19, -0.78, -0.96)}
    for year, (bal, rev, exp) in official.items():
        s, b = scen.at(year), base.at(year)
        gdp = b.macro["gdp_nominal"]
        d = {k: (s.fiscal[k] - b.fiscal[k]) / gdp * 100.0 for k in ("balance", "revenue_total", "expenditure_total")}
        assert abs(d["balance"] - bal) <= 0.05, f"{year}: balance {d['balance']:.3f} vs {bal:.2f}"
        assert abs(d["revenue_total"] - rev) <= 0.2 * abs(rev) + 0.02, f"{year}: revenue {d['revenue_total']:.3f} vs {rev:.2f}"
        assert abs(d["expenditure_total"] - exp) <= 0.2 * abs(exp) + 0.02, f"{year}: expenditure {d['expenditure_total']:.3f} vs {exp:.2f}"
        if year >= 2028:
            assert 0.15 <= d["balance"] <= 0.25, f"{year}: balance {d['balance']:.3f}"


def test_full_time_equivalent_floor_leaves_the_baseline_unchanged():
    """Employer contributions assessed on full-time-equivalent wages (ssc.fte_floor_wage on):
    the bottom wage deciles pay contributions in the payslip calculator, the calibrated baseline
    revenue does not move."""
    from belsim.core.dims import J
    from belsim.fiscal.revenue import calc

    pset = ParamSet.load()
    assert pset.value("ssc.fte_floor_wage", 2024) > 20000
    on = engine.run(None, end=2031)
    off = engine.run(None, end=2031, params=pset.with_scenario(Scenario(name="off", measures=[
        Measure(name="off", changes=[Change("ssc.fte_floor_wage", 0, 2024)])])))
    for year in (2024, 2027, 2031):
        assert on.at(year).revenue == pytest.approx(off.at(year).revenue, rel=1e-9)
    grid = on.at(2024).wage_dist[J["emp_private"]]
    rate = calc.employer_ssc_private(grid, pset.at(2024, lambda kind, y: 1.0)) / grid
    assert 0.08 < rate[0] < 0.16 and rate[0] < rate[3] < rate[-1]


@pytest.mark.parametrize("id_,label,tol", MEASURES, ids=lambda v: str(v))
def test_dynamic_response_matches_official_evaluation(report, id_, label, tol):
    r = _measure(report, id_, label)
    assert r.compared is not None, r.status
    assert abs(r.compared - r.ref) <= tol * abs(r.ref), (
        f"{id_} / {r.label}: model {r.compared:.3f} vs official {r.ref:.3f}")
