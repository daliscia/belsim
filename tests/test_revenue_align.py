"""`rev.fpb_alignment` against what `belsim.fiscal.revenue.align` computes on the current
baseline, and the rule it stands for: the observed 2025 accounts are the starting point, only
the change since 2025 is the Planning Bureau's.

If the first test fails, the baseline tax bases have moved: run
`uv run python -m belsim.fiscal.revenue.align --write` and check the validation tolerances."""
import pytest

from belsim.core import engine
from belsim.core.params import ParamSet
from belsim.fiscal.revenue import LINES, align, alignment_categories, outturn


@pytest.fixture(scope="module")
def base():
    return engine.run(None, end=2031)


def test_stored_multipliers_are_those_of_the_script(base):
    pset = ParamSet.load()
    computed = align.compute(base)
    assert sorted(computed) == list(range(2026, 2032))
    for year, factors in computed.items():
        stored = pset.value(align.PARAM, year)
        assert set(stored) == set(factors)
        for category, value in factors.items():
            assert stored[category] == pytest.approx(value, abs=2e-4), (year, category, stored[category], value)
    for year in (2024, 2025):
        assert all(v == 1.0 for v in pset.value(align.PARAM, year).values())
    assert pset.value(align.PARAM, 2040) == pset.value(align.PARAM, 2031)      # the 2031 position persists
    # the printed rows are what the parameter file holds
    assert align.rows(computed) in align.PARAMS_FILE.read_text()


def test_change_since_the_observed_2025_is_the_planning_bureau_s(base):
    """Every aligned category: model ratio(t) - model ratio(2025) = Bureau ratio(t) - Bureau
    ratio(2025), the model's 2025 being the outturn."""
    cats = alignment_categories()
    observed = outturn()[2025]
    for line, value in observed.items():
        assert base.at(2025).revenue[line] == pytest.approx(value, abs=0.05), line
    assert not base.at(2025).extra["revenue_alignment"]
    gdp = lambda y: base.at(y).macro["gdp_nominal"]

    def ratio(year, lines, entities=None):
        share = (lambda k: 1.0) if entities is None else (lambda k: sum(LINES[k].entity_shares[e] for e in entities))
        return 100.0 * sum(base.at(year).revenue[k] * share(k) for k in lines) / gdp(year)

    for year in range(2026, 2032):
        for category in align.SINGLE:
            official = 100.0 * align.official_change(category, ("general_government",), year)
            model = ratio(year, cats[category]) - ratio(2025, cats[category])
            assert model == pytest.approx(official, abs=0.003), (category, year)
        for group in align.GROUPS:
            lines = [k for s in align.SUBSECTORS for k in cats[f"{group}_{s}"]]
            for entities in align.SUBSECTORS.values():
                official = 100.0 * align.official_change(group, entities, year)
                model = ratio(year, lines, entities) - ratio(2025, lines, entities)
                assert model == pytest.approx(official, abs=0.004), (group, entities, year)


def test_outlook_table_is_complete():
    data = align.outlook()
    assert all(sorted(v) == list(range(2024, 2032)) for v in data.values())
    assert data[("gdp", "general_government")][2025] == 642_012
    # Tableau D.1: indirect taxes of general government = the four subsectors
    for year, total in ((2025, 74_760), (2031, 90_803)):
        assert sum(data[("indirect", e)][year] for e in ("federal", "regions_communities", "local", "social_security")) == pytest.approx(total, abs=2)
