"""Regression guards: where the model has been checked against official figures, keep it there.

Tolerances are deliberately wider than the current gaps; they catch a broken calibration, not
small drifts. References and sources are in data/reference/validation.yaml.
"""
import pytest

from belsim import validation as V


@pytest.fixture(scope="module")
def report():
    return V.run_validation(end=2035)


def _baseline(report, id_, when):
    for r in report.baseline:
        if r.id == id_ and r.when == str(when):
            return r
    raise AssertionError(f"no baseline row {id_} {when}")


def _measure(report, id_, label_start):
    for r in report.measures:
        if r.id == id_ and r.label.startswith(label_start):
            return r
    raise AssertionError(f"no measure row {id_} {label_start}")


# (reference id, year, absolute tolerance)
BASELINE = [
    # % of GDP. The model keeps the observed 2025 revenue level (0.07 to 0.16 point above the
    # Bureau's path in 2026-2031) and its expenditure is 0.2 point below the Bureau's in 2031:
    # gaps 0.04 / -0.07 / -0.04 / -0.03 / 0.01 in 2025-2029, 0.14 in 2030, 0.35 in 2031
    *[("fpb2026_balance", y, 0.2) for y in range(2025, 2030)],
    ("fpb2026_balance", 2030, 0.25),
    ("fpb2026_balance", 2031, 0.45),
    *[("fpb2026_interest", y, 0.3) for y in range(2025, 2032)],
    *[("fpb2026_unemployment_rate", y, 0.5) for y in range(2025, 2032)],
    *[("fpb2026_employment", y, 80) for y in range(2025, 2032)],        # thousand persons
    *[("fpb2026_inflation", y, 0.2) for y in range(2025, 2032)],
    ("sca2026_pensions", 2030, 0.5), ("sca2026_health_total", 2030, 0.4),
    ("fpb2026_population", 2031, 100),
]


@pytest.mark.parametrize("id_,when,tol", BASELINE, ids=lambda v: str(v))
def test_baseline_tracks_official_projection(report, id_, when, tol):
    r = _baseline(report, id_, when)
    assert r.model is not None, r.status
    assert abs(r.gap) <= tol, f"{id_} {when}: model {r.model:.2f} vs official {r.ref:.2f}"


# (reference id, target label prefix, relative tolerance)
MEASURES = [
    ("hermes_public_consumption_cut", "real GDP, year t", 0.35),
    ("hermes_public_consumption_cut", "real GDP, t+4", 0.45),
    ("hermes_social_benefits_cut", "real GDP, year t", 0.6),
    ("hermes2018_employer_ssc_cut_general", "employment, t+2", 0.35),
    ("hermes2018_employer_ssc_cut_general", "employment, t+4", 0.35),
    ("hermes_employer_ssc_increase", "employment, t+4", 0.35),
    ("dc2024_abolish_special_ssc", "special contribution", 0.3),
    ("dc2024_pit_45_to_40", "PIT revenue", 0.4),
    ("securities_tax_doubling", "additional yield", 0.3),
    ("unemployment_time_limit", "people excluded", 0.3),
    ("index_jump_2015_nbb", "consumer prices", 0.4),
    ("index_jump_2015_fpb", "total employment", 0.4),
]


@pytest.mark.parametrize("id_,label,tol", MEASURES, ids=lambda v: str(v))
def test_measure_matches_official_costing(report, id_, label, tol):
    r = _measure(report, id_, label)
    assert r.compared is not None, r.status
    assert abs(r.compared - r.ref) <= tol * abs(r.ref), (
        f"{id_} / {r.label}: model {r.compared:.2f} vs official {r.ref:.2f}")
