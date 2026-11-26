"""FastAPI application: JSON API under /api, static dashboard (web/) at /."""
from __future__ import annotations

import contextlib
import re
import threading
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core import dims
from ..i18n import LANGS, pick_lang
from ..i18n.labels import Labeller
from . import parties, public, version
from . import service as svc
from .units import UNITS_DOC, dim_labels, unit_kind

WEB_DIR = svc.ROOT / "web"

# Headline keys used for the per-measure breakdown of the impact table.
IMPACT_KEYS = [
    "fiscal.revenue_total", "fiscal.expenditure_total", "fiscal.balance", "fiscal.debt",
    "fiscal.balance_pct_gdp", "fiscal.debt_pct_gdp", "macro.gdp_nominal", "macro.gdp_real",
    "macro.cpi", "labour.employment",
]


class ChangeIn(BaseModel):
    param: str
    value: Any = None
    start: int
    op: str = "set"
    end: int | None = None
    phase_in_years: int = 0
    key: str | int | None = None


class MeasureIn(BaseModel):
    name: str = "measure"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True
    changes: list[ChangeIn] = Field(default_factory=list)


class OptionsIn(BaseModel):
    end: int | None = None
    per_measure: bool = False


class ScenarioIn(BaseModel):
    name: str = "scenario"
    description: str = ""
    measures: list[MeasureIn] = Field(default_factory=list)
    options: OptionsIn = Field(default_factory=OptionsIn)


class BatchIn(BaseModel):
    scenarios: list[ScenarioIn] = Field(default_factory=list, max_length=12)
    options: OptionsIn = Field(default_factory=OptionsIn)


def request_lang(request: Request, lang: str | None = Query(None, description="en, fr or nl")) -> str:
    """Language of the model-provided text: `?lang=`, else Accept-Language, else English."""
    return pick_lang(lang, request.headers.get("accept-language"))


def _translated_distribution_routes(app: FastAPI) -> None:
    """/api/distribution and /api/payslip/options in the requested language. The endpoints
    belong to `distribution.py`; these routes are registered first and call them, so that file
    needs no knowledge of languages. If its functions are renamed, the plain router answers."""
    from . import distribution as D

    dist_fn, opts_fn = getattr(D, "distribution", None), getattr(D, "payslip_options", None)
    body_model = getattr(D, "DistributionIn", None)
    if callable(dist_fn) and body_model is not None:
        def distribution(body, lang=Depends(request_lang)):
            return svc.localise_distribution(dist_fn(body), lang)
        # annotations as objects: this module postpones them, and `body_model` is a local name
        distribution.__annotations__ = {"body": body_model, "lang": str}
        app.post("/api/distribution")(distribution)
    if callable(opts_fn):
        @app.get("/api/payslip/options")
        def payslip_options(lang: str = Depends(request_lang)):
            return svc.localise_payslip_options(opts_fn(), lang)


def _refuse_if_read_only() -> None:
    """On a public deployment (`BELSIM_PUBLIC=1`, or `BELSIM_READ_ONLY=1` alone) the server is
    shared by every visitor and writes nothing to disk: saving and deleting are refused. The
    dashboard then keeps saved scenarios in the browser and still exports and imports files."""
    if public.read_only():
        raise svc.ApiError(403, "saving on the server is disabled on this public server; "
                                "scenarios are kept in your browser and can be exported to a file")


_ABS_PATH = re.compile(r"/(?:[\w.\-]+/){2,}[\w.\-]+")


def public_message(status: int, message: str) -> str:
    """What a visitor of a public server may read: no exception text for a server error, no
    file path in any message."""
    if status >= 500:
        return "the server could not compute this; try again later"
    return _ABS_PATH.sub("<path>", message.replace(str(svc.ROOT), "<root>"))


def _warm_up() -> None:
    """Public mode, at start-up: the baseline, the texts and the slow reports, so that the first
    visitor does not wait for them. Failures are left to the requests that need the result."""
    for step in (svc.baseline, svc.baseline_series, lambda: svc.describe_params("en"),
                 version.build_info, svc.validation_start,
                 lambda: svc.library_effects(svc.max_end(), "en"),
                 lambda: parties.parties_compare(svc.max_end(), "en")):
        try:
            step()
        except Exception:
            pass


def create_app() -> FastAPI:
    is_public = public.is_public()
    lim = public.limits()
    docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if is_public else {}

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if is_public:       # at start-up, not at import: a test client without `with` skips it
            threading.Thread(target=_warm_up, name="belsim-warm-up", daemon=True).start()
        yield

    app = FastAPI(title="belsim", description="Belgium economic and public-finance simulator",
                  lifespan=lifespan, **docs)
    try:
        _translated_distribution_routes(app)
    except Exception:  # pragma: no cover - distribution module under construction
        pass
    from .distribution import router as distribution_router; app.include_router(distribution_router)  # noqa: E702 - /api/distribution, /api/payslip

    @app.exception_handler(svc.ApiError)
    def _api_error(request: Request, exc: svc.ApiError):
        message = public_message(exc.status, exc.message) if is_public else exc.message
        return JSONResponse(status_code=exc.status, content={"error": message})

    @app.exception_handler(RequestValidationError)
    def _bad_request(request: Request, exc: RequestValidationError):
        msgs = [f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "invalid request: " + "; ".join(msgs)})

    if is_public:
        @app.exception_handler(Exception)
        def _unexpected(request: Request, exc: Exception):
            return JSONResponse(status_code=500, content={"error": public_message(500, "")})


    def bounded(scen: dict) -> dict:
        """The scenario, refused when it is larger than the public limits."""
        if is_public:
            problem = public.check_scenario_size(scen, lim)
            if problem:
                raise svc.ApiError(413, problem)
        return scen

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        """Liveness: answers without touching the model."""
        return {"status": "ok", "public": is_public}

    @app.get("/api/version")
    def api_version():
        """Commit, date, numbers of parameters, assumptions and tests of this build."""
        import os
        info = version.build_info(count_tests="PYTEST_CURRENT_TEST" not in os.environ)
        return svc.clean({**info, "public": is_public, "base_year": dims.BASE_YEAR, "end_year": dims.END_YEAR,
                          "signature": svc.params_signature()})

    @app.get("/whitepaper", include_in_schema=False)
    def whitepaper():
        page = WEB_DIR / "whitepaper.html"
        if not page.is_file():
            raise svc.ApiError(404, "the methodology page is not installed")
        return FileResponse(page, media_type="text/html")

    # Endpoints are plain `def`: FastAPI runs them in its threadpool, so a simulation never
    # blocks the event loop.

    @app.get("/api/meta")
    def meta(lang: str = Depends(request_lang)):
        try:
            from ..fiscal import expenditure, revenue
            rev = svc.line_meta_localised(revenue.LINES, "revenue", lang)
            exp = svc.line_meta_localised(expenditure.LINES, "expenditure", lang)
        except Exception as e:
            raise svc.ApiError(500, f"cannot import fiscal line metadata: {type(e).__name__}: {e}")
        keys, years = [], list(range(dims.BASE_YEAR, svc.max_end() + 1))
        try:
            keys, years = list(svc.baseline_series()), svc.baseline().years
        except svc.ApiError:
            pass
        labeller = Labeller(lang, None, rev, exp)
        # the lines are already in `lang`: the labeller must not look them up a second time
        labeller.line = lambda side, lid: (labeller.lines[side].get(lid) or {}).get("label")  # type: ignore[method-assign]
        labels = dim_labels(lang)
        detail = {eid: {**info, "label": labeller.dim("entities", eid)} for eid, info in labeller.detail.items()}
        labels["entities"] = {**labels["entities"], **{eid: d["label"] for eid, d in detail.items()}}
        return {
            "base_year": dims.BASE_YEAR,
            "end_year": svc.max_end(),
            "lang": lang,
            # what the dashboard must know about the server: where "Save" stores a scenario
            "mode": {"public": is_public, "save": "browser" if public.read_only() else "server",
                     "max_batch": lim.max_batch if is_public else 12},
            "langs": list(LANGS),
            "dims": {
                "years": years, "ages": dims.AGES, "sexes": dims.SEXES, "regions": dims.REGIONS,
                "sectors": dims.SECTORS, "statuses": dims.STATUSES,
                "employed_statuses": dims.EMPLOYED_STATUSES, "jobtypes": dims.JOBTYPES,
                "entities": dims.ENTITIES, "entity_esa": dims.ENTITY_ESA,
                "cons_cats": dims.CONS_CATS, "deciles": list(range(1, dims.N_DECILE + 1)),
            },
            "labels": labels,
            # federated entities below a subsector: {entity id: {label, parent}} ({} until the
            # regional model publishes ENTITY_DETAIL)
            "entity_detail": detail,
            "revenue_lines": rev,
            "expenditure_lines": exp,
            "units": {k: unit_kind(k) for k in keys},
            "unit_kinds": UNITS_DOC,
            "series_labels": labeller.labels(keys),
            "regional_metrics": labeller.regional_metrics(keys),
            "params_signature": svc.params_signature(),
        }

    @app.get("/api/params")
    def params(lang: str = Depends(request_lang)):
        return svc.describe_params(lang)

    @app.get("/api/baseline")
    def baseline():
        res = svc.baseline()
        return {"years": res.years, "series": svc.baseline_series(), "signature": svc.params_signature()}

    @app.post("/api/run")
    def run(body: ScenarioIn):
        scen = bounded(body.model_dump(exclude={"options"}))
        end = svc.clamp_end(body.options.end)
        sid, base, res = svc.run_scenario(scen, end)
        b = svc.baseline_series(end)
        s = svc.series_of(res)
        out = {"id": sid, "name": body.name, "years": res.years,
               "baseline": {"series": b}, "scenario": {"series": s},
               "delta": {"series": svc.delta_of(b, s)}}
        if body.options.per_measure:
            per = []
            for m in scen["measures"]:
                if m.get("enabled") is False:
                    continue
                _, _, r1 = svc.run_scenario({"name": m["name"], "measures": [m]}, end)
                d = svc.delta_of(b, svc.series_of(r1))
                s1 = svc.series_of(r1)
                keep = [k for k in IMPACT_KEYS if k in d] + [k for k in d if svc.REGIONAL_HEADLINE.match(k)]
                per.append({"name": m["name"],
                            "delta": {k: d[k] for k in keep},
                            "scenario": {k: s1[k] for k in keep if k in s1}})
            out["per_measure"] = per
        return out

    @app.post("/api/run_batch")
    def run_batch(body: BatchIn):
        """Headline series of several scenarios at once (for side-by-side comparison)."""
        end = svc.clamp_end(body.options.end)
        if is_public and len(body.scenarios) > lim.max_batch:
            raise svc.ApiError(413, f"at most {lim.max_batch} scenarios per request on this server")
        return svc.run_batch([bounded(sc.model_dump(exclude={"options"})) for sc in body.scenarios], end)

    @app.post("/api/run_band")
    def run_band(body: ScenarioIn):
        """The scenario with low / central / high behavioural response, and static."""
        return svc.run_band(bounded(body.model_dump(exclude={"options"})), svc.clamp_end(body.options.end))

    @app.get("/api/measures/effects")
    def measure_effects(end: int | None = None, lang: str = Depends(request_lang)):
        """Each prebuilt measure simulated alone: headline changes against the baseline."""
        return svc.library_effects(svc.clamp_end(end), lang)

    @app.get("/api/validation")
    def validation(wait: bool | None = Query(None, description="0: answer at once, 202 while computing"),
                   lang: str = Depends(request_lang)):
        """Model against official reference figures (about a minute the first time, then
        cached). By default the request waits for the result; with `wait=0`, and always in public
        mode, it answers at once: the report, or 202 `{"status": "computing"}` while a background
        thread computes it."""
        if (is_public or wait is False) and not svc.validation_ready():
            svc.validation_start()
            state = svc.validation_status()
            if state["status"] == "ready":
                return svc.validation_report(lang)
            if state["status"] == "error":
                raise svc.ApiError(500, f"validation failed: {state.get('error')}")
            return JSONResponse(status_code=202, content={**state, "status": "computing"},
                                headers={"Retry-After": "5"})
        return svc.validation_report(lang)

    @app.post("/api/validate")
    def validate(body: ScenarioIn):
        scen = svc.normalise_scenario(bounded(body.model_dump(exclude={"options"})))
        return {"problems": svc.validate_scenario(scen, svc.load_params(), dims.END_YEAR)}

    @app.get("/api/arrays")
    def arrays(run: str = "baseline", year: int = Query(dims.BASE_YEAR)):
        return svc.arrays_for(svc.results_for(run), year)

    @app.get("/api/pyramid")
    def pyramid(run: str = "baseline"):
        return svc.pyramid_for(svc.results_for(run))

    @app.get("/api/measures")
    def measures(lang: str = Depends(request_lang)):
        return svc.measure_library(lang)

    @app.get("/api/scenarios")
    def scenarios(kind: str | None = Query(None, description="user (default), party or all"),
                  lang: str = Depends(request_lang)):
        """Saved scenarios (`kind=user`, the default), the prebuilt party programmes of the 2024
        costing (`kind=party`: metadata, coverage, measures with status and source, runnable
        scenario) or both (`kind=all`: the party list under `parties`)."""
        if kind not in (None, "", "user", "party", "all"):
            raise svc.ApiError(400, "kind must be user, party or all")
        if kind == "party":
            return parties.list_parties(lang)
        out = svc.list_scenarios()
        if kind == "all":
            out["parties"] = parties.list_parties(lang)
        return out

    @app.get("/api/parties/compare")
    def parties_compare(end: int | None = None, lang: str = Depends(request_lang)):
        """Headline series of every party's package, with the Bureau's published results."""
        return parties.parties_compare(svc.clamp_end(end), lang)

    @app.get("/api/parties/distribution")
    def parties_distribution(year: int = Query(2031), lang: str = Depends(request_lang)):
        """Disposable income by decile for every party's package (slow the first time)."""
        return parties.parties_distribution(year, lang)

    @app.get("/api/parties/{pid}")
    def party(pid: str, lang: str = Depends(request_lang)):
        return svc.clean(parties.party_summary(parties.get_party(pid), lang))

    @app.get("/api/parties/{pid}/effects")
    def party_effects(pid: str, end: int | None = None, lang: str = Depends(request_lang)):
        """Each measure of the party simulated alone: static and behavioural effects."""
        return parties.party_effects(pid, end, lang)

    @app.post("/api/scenarios")
    def save(body: ScenarioIn):
        _refuse_if_read_only()
        if not body.name.strip():
            raise svc.ApiError(400, "scenario needs a name")
        return svc.save_scenario(body.model_dump(exclude={"options"}))

    @app.delete("/api/scenarios/{slug}")
    def delete(slug: str):
        _refuse_if_read_only()
        svc.delete_scenario(slug)
        return {"deleted": slug}

    @app.get("/api/{rest:path}")
    def not_found(rest: str):
        raise svc.ApiError(404, f"no such endpoint: /api/{rest}")

    if is_public:
        from starlette.middleware.gzip import GZipMiddleware
        app.add_middleware(GZipMiddleware, minimum_size=2048)
        app.add_middleware(public.PublicGuard, lim=lim)     # outermost: limits, headers, caching
    else:
        @app.middleware("http")
        async def no_stale_assets(request: Request, call_next):
            resp = await call_next(request)
            if not request.url.path.startswith("/api/") and not request.url.path.startswith("/vendor/"):
                resp.headers["Cache-Control"] = "no-cache"
            return resp

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
