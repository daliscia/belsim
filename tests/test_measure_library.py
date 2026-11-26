"""Every prebuilt measure loads, runs to 2040 and moves the budget in the expected direction."""
from pathlib import Path

import pytest
import yaml

from belsim.core import engine
from belsim.core.params import Measure, Scenario

LIB = Path(__file__).resolve().parents[1] / "belsim" / "measures" / "library" / "measures.yaml"
MEASURES = [Measure.from_dict(m) for m in yaml.safe_load(LIB.read_text())["measures"]]

# expected sign of the change in the general-government balance in 2040 (None = no expectation)
SIGN = {
    "VAT standard rate 21% -> 22%": 1, "VAT reduced rate 6% -> 9%": 1,
    "Income tax - tax-free allowance +EUR 2,000": -1, "Income tax - top rate 50% -> 55%": 1,
    "Corporate tax 25% -> 20%": -1, "Employer social contributions -2 points": -1,
    "Capital gains tax 10% -> 20%": 1, "Withholding tax 30% -> 35%": 1,
    "Fuel excise +10 cents per litre": 1, "Tobacco excise +25%": 1,
    "Legal pension age 68 from 2035": 1, "Keep legal pension age at 66": -1,
    "Pension malus 5% per year of early retirement": 1, "Restore civil-servant pension perequation": -1,
    "Abolish the unemployment benefit time limit": -1, "Health-care growth norm 2%": 1,
    "Defence 3.5% of GDP by 2035": -1, "Public investment +1% of GDP": -1,
    "Public employment -1% a year for 5 years": 1, "Cut subsidies to firms by 10%": 1,
    "Welfare envelope restored from 2027": -1, "Net migration +20,000 a year": 1,
    "Productivity growth +0.25 point": None, "Interest rates +1 point": -1,
    "Tax thresholds follow prices only (fiscal drag)": 1, "Index jump in 2027": None, "Index jump of 2% over 2027-2029": None,
    "Static costing (no behavioural responses)": None,
    # regional measures (one region or Community)
    "Wallonia: structural unemployment -1 point by 2032": 1, "Brussels: activation effort +25%": 1,
    "Wallonia: invalidity -10% (return to work)": 1,
    "Flanders: +0.25 point regional productivity growth": 1, "Wallonia: family allowance +10%": -1,
    "Brussels: public employment -1% a year for 5 years": 1,
    "Flanders: public investment +0.5% of regional GDP": -1,
    "French Community: education cost per pupil -5%": 1, "Flanders: elderly-care spending +10%": -1,
    "Brussels: service-voucher subsidy -20%": 1,
    # regional migration measures
    "Flanders: international immigration +20%": 1,
    "Brussels: a quarter fewer residents leave for the other regions": None,
}


@pytest.fixture(scope="module")
def base():
    return engine.run(None, end=2040)


def test_library_names_are_unique_and_covered():
    names = [m.name for m in MEASURES]
    assert len(names) == len(set(names))
    assert set(names) == set(SIGN)


@pytest.mark.parametrize("measure", MEASURES, ids=lambda m: m.name)
def test_measure_runs(measure, base):
    scen = engine.run(Scenario(name=measure.name, measures=[measure]), end=2040, baseline=base)
    d = scen.at(2040).fiscal["balance"] - base.at(2040).fiscal["balance"]
    gdp = base.at(2040).macro["gdp_nominal"]
    assert abs(d) / gdp < 0.05, "implausibly large budget effect"
    sign = SIGN[measure.name]
    if sign:
        assert d * sign > 0, f"balance moved by {d:.0f} EUR mn, expected sign {sign:+d}"
