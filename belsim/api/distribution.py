"""Distributional-analysis endpoints (registered in app.py with `app.include_router(router)`).

    POST /api/distribution     {scenario, year}                          winners and losers
                               (`households`: the representative cases, one per household
                               type; `households_all`: every type in every income group - the
                               rows behind the deciles, the indicators and the summary;
                               `poverty`: at-risk-of-poverty rate `by_household_type` and
                               `by_activity_status` - rows {id, label, persons, population_share,
                               baseline, scenario, delta, observed, median_income,
                               median_income_observed} - and `other_lines`: the baseline rate at
                               40, 50 and 70 % of the median with the observed one; `observed`
                               is {value, year, source} from EU-SILC or null)
    POST /api/payslip          {scenario, year, kind, household, gross_min, gross_max, points}
    GET  /api/payslip/options  kinds and household options of the payslip explorer

`scenario` is the same JSON as the body of /api/run (name, measures, optional options.end), so
the simulation behind a call is the cached one of the dashboard. Errors are `ApiError`s, turned
into `{"error": ...}` by the application.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError

from ..core import dims
from . import service as svc

router = APIRouter()

KIND_LABELS = {
    "emp_private": "Private-sector employee", "emp_public": "Public-sector employee",
    "self_employed": "Self-employed", "pension": "Pension", "unemployment": "Unemployment benefit",
    "disability": "Sickness / invalidity benefit",
}
REGION_LABELS = {"average": "Belgium (average)", "FLA": "Flanders", "WAL": "Wallonia", "BRU": "Brussels"}
MAX_POINTS = 400

_cache = svc._LRU(48)        # (params signature, scenario id, request) -> response


class DistributionIn(BaseModel):
    scenario: dict[str, Any] = Field(default_factory=dict)
    year: int = 2030


class PayslipIn(BaseModel):
    scenario: dict[str, Any] = Field(default_factory=dict)
    year: int = 2030
    kind: str = "emp_private"
    household: Any = "single"
    gross_min: float = 0.0
    gross_max: float = 150000.0
    points: int = 76


def _runs(scenario: dict, year: int):
    """(scenario id, baseline results, scenario results, baseline ParamSet, scenario ParamSet)."""
    from .app import ScenarioIn     # imported here: app.py imports this module

    try:
        body = ScenarioIn.model_validate(scenario or {})
    except ValidationError as e:
        msgs = [f"scenario.{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise svc.ApiError(422, "invalid request: " + "; ".join(msgs)) from e
    scen = body.model_dump(exclude={"options"})
    end = svc.clamp_end(body.options.end)
    if not (dims.BASE_YEAR <= year <= end):
        raise svc.ApiError(400, f"year must be between {dims.BASE_YEAR} and {end}")
    sid, base, res = svc.run_scenario(scen, end)
    pset = svc.load_params()
    scenario_obj = svc.to_scenario(svc.normalise_scenario(scen))
    return sid, base, res, pset.with_scenario(None), pset.with_scenario(scenario_obj)


def _guard(fn, what: str):
    try:
        return fn()
    except svc.ApiError:
        raise
    except ValueError as e:
        raise svc.ApiError(422, str(e)) from e
    except Exception as e:
        raise svc.ApiError(500, f"{what} failed: {type(e).__name__}: {e}") from e


@router.post("/api/distribution")
def distribution(body: DistributionIn):
    from ..distribution import analysis

    sid, base, res, base_pset, scen_pset = _runs(body.scenario, body.year)
    key = (svc.params_signature(), sid, "distribution", body.year)
    out = _cache.get(key)
    if out is None:
        out = svc.clean(_guard(lambda: analysis.distribution(base, res, base_pset, scen_pset, body.year),
                               "distributional analysis"))
        _cache.put(key, out)
    return {"id": sid, **out}


@router.post("/api/payslip")
def payslip(body: PayslipIn):
    from ..distribution import analysis
    from ..fiscal.revenue import calc as rcalc

    if body.kind not in rcalc.KINDS:
        raise svc.ApiError(422, f"unknown kind '{body.kind}' (expected one of {list(rcalc.KINDS)})")
    if not (2 <= body.points <= MAX_POINTS):
        raise svc.ApiError(422, f"points must be between 2 and {MAX_POINTS}")
    if not (0 <= body.gross_min < body.gross_max <= 1e7):
        raise svc.ApiError(422, "need 0 <= gross_min < gross_max")
    sid, base, res, base_pset, scen_pset = _runs(body.scenario, body.year)
    out = _guard(lambda: analysis.payslip_grid(base, res, base_pset, scen_pset, body.year, body.kind,
                                               body.household, body.gross_min, body.gross_max, body.points),
                 "payslip")
    return {"id": sid, **svc.clean(out)}


@router.get("/api/payslip/options")
def payslip_options():
    from ..distribution.calc import HOUSEHOLD_PRESETS
    from ..fiscal.revenue import calc as rcalc

    return {
        "kinds": [{"id": k, "label": KIND_LABELS.get(k, k)} for k in rcalc.KINDS],
        "households": [{"id": k, "label": v["label"], "adults": v["adults"], "children": v["children"]}
                       for k, v in HOUSEHOLD_PRESETS.items()],
        "household_fields": {
            "adults": [1, 2], "children": {"min": 0, "max": 12},
            "partner_kind": ["none", *rcalc.KINDS],
            "partner_gross": "EUR per year, or 'median' for the median private-sector wage of the year",
            "region": [{"id": k, "label": v} for k, v in REGION_LABELS.items()],
            "preset": list(HOUSEHOLD_PRESETS),
        },
        "defaults": {"kind": "emp_private", "household": "single", "gross_min": 0, "gross_max": 150000,
                     "points": 76, "year": 2030},
        "max_points": MAX_POINTS,
    }
