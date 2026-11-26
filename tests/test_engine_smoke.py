"""End-to-end smoke test of the engine and accounting identities. Must always pass."""
import numpy as np
from belsim.core import engine
from belsim.core.params import Change, Measure, Scenario


def test_baseline_runs_and_identities_hold():
    res = engine.run(None, end=2030)
    assert res.years == list(range(2024, 2031))
    for s in res.states:
        assert np.isfinite(s.pop).all() and s.pop.min() >= 0
        np.testing.assert_allclose(s.status.sum(axis=0), s.pop, rtol=1e-6)
        f = s.fiscal
        assert abs(f["balance"] - (sum(s.revenue.values()) - sum(s.expenditure.values()))) < 1e-3
        assert 9e6 < s.pop.sum() < 15e6
        assert 4e5 < s.macro["gdp_nominal"] < 2e6


def test_empty_scenario_equals_baseline():
    base = engine.run(None, end=2028)
    scen = engine.run(Scenario(name="empty"), end=2028, baseline=base)
    d = engine.compare(base, scen)
    assert max(abs(v) for vals in d.values() for v in vals if v is not None) < 1e-6
