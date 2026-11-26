"""Structural tests of the validation layer. No closeness to the references is asserted here:
tolerances are added once the model has been tuned against data/reference/validation.yaml."""
import math

import pytest
import yaml

from belsim import validation as V
from belsim.core import engine
from belsim.core.params import ParamSet, Scenario

DATA = V.load()
BASELINE = DATA["baseline"]
MEASURES = DATA["measures"]


@pytest.fixture(scope="module")
def base():
    return engine.run(None)


@pytest.fixture(scope="module")
def report(base):
    return V.run_validation(base=base)


def test_yaml_has_references():
    assert BASELINE and MEASURES
    ids = [e["id"] for e in BASELINE + MEASURES]
    assert len(ids) == len(set(ids)), "duplicate reference ids"


@pytest.mark.parametrize("entry", BASELINE + MEASURES, ids=lambda e: e["id"])
def test_reference_is_documented(entry):
    assert entry.get("description")
    src = entry.get("source") or {}
    assert src.get("institution") and src.get("title") and src.get("date")
    assert str(entry.get("url", "")).startswith("http")
    assert entry.get("confidence") in V.CONFIDENCE


@pytest.mark.parametrize("entry", BASELINE, ids=lambda e: e["id"])
def test_baseline_entry_shape(entry):
    assert entry.get("unit")
    assert entry.get("values")
    for when, value in entry["values"].items():
        assert isinstance(when, int) or V._PERIOD.match(str(when))
        assert isinstance(value, (int, float)) and math.isfinite(value)


@pytest.mark.parametrize("entry", MEASURES, ids=lambda e: e["id"])
def test_measure_entry_shape(entry):
    assert entry.get("targets")
    for t in entry["targets"]:
        assert t.get("label") and t.get("ref_unit")
        assert isinstance(t["ref_value"], (int, float)) and math.isfinite(t["ref_value"])
        assert isinstance(t["ref_year"], int)
        assert t.get("variant", "unspecified") in V.VARIANTS
        if entry.get("measure"):
            assert t.get("model", {}).get("num"), "target of a modelled measure needs a model spec"
            assert V.BASE_YEAR < int(t.get("model_year", t["ref_year"])) <= V.END_YEAR


@pytest.mark.parametrize("entry", [e for e in MEASURES if e.get("measure")], ids=lambda e: e["id"])
def test_measure_uses_existing_params(entry):
    assert V.missing_params(entry) == []
    measure = V.measure_of(entry)
    assert measure.changes
    # the parameter set accepts the measure (ops, names) and resolves it in its first year
    pset = ParamSet.load().with_scenario(Scenario(name=entry["id"], measures=[measure]))
    for c in measure.changes:
        pset.value(c.param, c.start)


def test_report_runs(report, tmp_path):
    assert len(report.baseline) == sum(len(e["values"]) for e in BASELINE)
    assert len(report.measures) == sum(len(e["targets"]) for e in MEASURES)
    assert not [p for p in report.problems if "run failed" in p]
    modelled = [r for r in report.baseline if r.status == "ok"]
    assert modelled and all(math.isfinite(r.model) for r in modelled)
    costed = [r for r in report.measures if r.status == "ok"]
    assert costed and all(math.isfinite(r.behavioural) for r in costed)
    text = V.render(report)
    assert "## Baseline projection" in text and "## Measure costings" in text
    path = V.write_report(report, tmp_path / "report.md")
    assert path.read_text() == text


def test_every_series_in_the_yaml_exists(report):
    unknown = [p for p in report.problems if "unknown series" in p or "no series matches" in p]
    assert unknown == []


def test_unknown_parameter_or_series_is_reported_not_raised(base, tmp_path):
    src = {"institution": "x", "title": "x", "date": "2025"}
    bad = {
        "baseline": [{"id": "b", "description": "d", "unit": "u", "values": {2030: 1.0},
                      "source": src, "url": "http://x", "confidence": "low",
                      "model": {"num": ["fiscal.does_not_exist"]}}],
        "measures": [
            {"id": "m1", "description": "d", "source": src, "url": "http://x", "confidence": "low",
             "measure": {"changes": [{"param": "no.such_param", "value": 1, "start": 2027}]},
             "targets": [{"label": "t", "ref_value": 1.0, "ref_unit": "EUR mn", "ref_year": 2028,
                          "model": {"num": ["fiscal.balance"]}}]},
            {"id": "m2", "description": "d", "source": src, "url": "http://x", "confidence": "low",
             "measure": {"changes": [{"param": "vat.rate_standard", "value": 0.22, "start": 2027}]},
             "targets": [{"label": "ok", "ref_value": 1.0, "ref_unit": "EUR mn", "ref_year": 2028,
                          "variant": "static", "model": {"num": ["revenue.vat"]}},
                         {"label": "bad", "ref_value": 1.0, "ref_unit": "EUR mn", "ref_year": 2028,
                          "model": {"num": ["revenue.nope"]}}]},
        ],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    rep = V.run_validation(path, base=base)
    assert rep.baseline[0].model is None and "unknown series" in rep.baseline[0].status
    m1, ok, bad_row = rep.measures
    assert "missing parameter: no.such_param" in m1.status and m1.behavioural is None
    assert ok.status == "ok" and ok.static > 0 and ok.gap is not None
    assert "unknown series" in bad_row.status
    assert len(rep.problems) == 3
    assert "no.such_param" in V.render(rep)
