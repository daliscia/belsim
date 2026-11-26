"""Place of the jobs a scenario adds or removes in the wage distribution (labour._jobs_by_decile)."""
import numpy as np
import pytest

from belsim import labour
from belsim.labour.dynparams import SOURCES

REL = np.array([0.35, 0.55, 0.68, 0.78, 0.87, 0.96, 1.07, 1.22, 1.45, 2.07])
REL = REL / REL.mean()


def test_tilted_profile_has_the_relative_wage_of_the_origin():
    flat = (1.0,) * 10
    for rho in (0.55, 0.8, 1.0, 1.2):
        w = np.array(labour._tilted_profile(rho, tuple(REL), flat))
        assert w.sum() == pytest.approx(1.0) and (w > 0).all()
        assert float(w @ REL) == pytest.approx(rho, abs=1e-9)
    assert labour._tilted_profile(1.0, tuple(REL), flat) == pytest.approx(np.full(10, 0.1))
    low = np.array(labour._tilted_profile(0.55, tuple(REL), flat))
    assert low[0] > low[4] > low[9]


def test_recut_keeps_the_mean_and_is_the_identity_for_equal_groups():
    level = REL * 50_000.0
    assert labour._recut(np.full(10, 100.0), level) == pytest.approx(level)
    mass = np.full(10, 100.0)
    mass[:3] += 30.0
    cut = labour._recut(mass, level)
    assert cut.mean() == pytest.approx(float(mass @ level) / mass.sum())
    assert (cut <= level + 1e-9).all() and cut[5] < level[5]


def test_jobs_by_decile_sum_to_the_head_count_difference_and_follow_the_cost_shift():
    steps = {k: np.zeros(3) for k in SOURCES}
    base_steps = {k: np.zeros(3) for k in SOURCES}
    steps["labour_demand"] = np.array([30_000.0, 0.0, 0.0])
    steps["unemp_exclusion"] = np.array([5_000.0, 0.0, 500.0])
    totals = np.array([3_040_000.0, 900_000.0, 800_500.0])
    base_totals = [3_000_000.0, 900_000.0, 800_000.0]
    rel = np.array([REL, REL, REL])
    shift = np.array([0.02, 0.02, 0.01, 0.0, -0.01, -0.01, -0.01, -0.01, -0.005, -0.005])
    jobs, lost = labour._jobs_by_decile(steps, base_steps, totals, base_totals, rel, shift, None)
    assert jobs.sum(axis=1) == pytest.approx([40_000.0, 0.0, 500.0])
    assert jobs[0, 0] > 3.0 * jobs[0, 9] and jobs[0, 4] < jobs[0, 3]    # jobs move to the deciles whose cost fell
    assert lost[0] == pytest.approx(float((jobs[0] * (1.0 - REL)).sum()))
    none, lost0 = labour._jobs_by_decile(base_steps, base_steps, np.array(base_totals), base_totals, rel, None, None)
    assert not none.any() and not lost0.any()
    no_base, _ = labour._jobs_by_decile(steps, None, totals, None, rel, shift, None)
    assert not no_base.any()
