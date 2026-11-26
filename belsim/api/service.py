"""Simulation service behind the HTTP API: caching, validation, serialisation.

The engine's own `run_baseline` lru_cache is deliberately not used: the baseline here is keyed
on the parameter files' signature so that editing `params/*.yaml` invalidates it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core import params as core_params
from ..core.dims import BASE_YEAR, END_YEAR, REGIONS
from ..i18n import translate
from ..core.params import Change, Measure, ParamSet, Scenario
from ..core.state import Results

try:  # the engine owns the index names (it gained "wage_link" after this module was written)
    from ..core.engine import INDEX_KEYS as _ENGINE_INDEX_KEYS
except Exception:  # pragma: no cover - engine import broken by work in progress
    _ENGINE_INDEX_KEYS = {}

ROOT = Path(__file__).resolve().parents[2]
INDEX_KEYS = {"cpi": "cpi", "health_index": "health_index", "wages": "wage_index",
              "wage_link": "tax_threshold_drift", **_ENGINE_INDEX_KEYS}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def params_dir() -> Path:
    return Path(os.environ.get("BELSIM_PARAMS_DIR") or core_params.PARAMS_DIR)


def scenarios_dir() -> Path:
    return Path(os.environ.get("BELSIM_SCENARIOS_DIR") or ROOT / "scenarios")


def library_dir() -> Path:
    return ROOT / "belsim" / "measures" / "library"


def params_signature() -> str:
    d = params_dir()
    parts = []
    if d.is_dir():
        for p in sorted(d.glob("*.yaml")):
            st = p.stat()
            parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- serialisation

def clean(v: Any) -> Any:
    """Make a value JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def series_of(res: Results) -> dict[str, list[float | None]]:
    return clean(res.series())


def delta_of(base: dict[str, list], scen: dict[str, list]) -> dict[str, list]:
    out = {}
    for k, vb in scen.items():
        va = base.get(k)
        if va is None:
            continue
        out[k] = [(y - x) if (x is not None and y is not None) else None for x, y in zip(va, vb)]
    return out


# ---------------------------------------------------------------- caches

class _LRU:
    def __init__(self, size: int):
        self.size, self.data, self.lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)

    def clear(self):
        with self.lock:
            self.data.clear()


_pset_cache = _LRU(2)
_base_cache = _LRU(6)       # (sig, end) -> Results
_base_series = _LRU(6)      # (sig, end) -> series dict
_run_cache = _LRU(24)       # (sig, scenario id) -> dict(results, scenario, end)
_ids = _LRU(64)             # scenario id -> (scenario dict, end), to re-run after eviction
_base_lock = threading.Lock()


def clear_caches() -> None:
    for c in (_pset_cache, _base_cache, _base_series, _run_cache, _ids, _head_cache, _valid_cache):
        c.clear()


def load_params() -> ParamSet:
    sig = params_signature()
    ps = _pset_cache.get(sig)
    if ps is None:
        try:
            ps = ParamSet.load(params_dir())
        except Exception as e:  # malformed YAML written by a module in progress
            raise ApiError(500, f"cannot load parameters: {e}") from e
        _pset_cache.put(sig, ps)
    return ps


def _engine_run(*args, **kwargs) -> Results:
    from ..core import engine  # imported lazily so a broken module surfaces as an API error
    return engine.run(*args, **kwargs)


def max_end() -> int:
    """Last year served: the model's horizon, or the cap of the public mode (`BELSIM_MAX_END`)."""
    from . import public
    return public.limits().max_end if public.is_public() else END_YEAR


def clamp_end(end: int | None) -> int:
    top = max_end()
    if end is None:
        return top
    if not (BASE_YEAR < end <= top):
        raise ApiError(400, f"options.end must be between {BASE_YEAR + 1} and {top}")
    return int(end)


def baseline(end: int | None = None) -> Results:
    end = max_end() if end is None else end
    key = (params_signature(), end)
    res = _base_cache.get(key)
    if res is None:
        with _base_lock:
            res = _base_cache.get(key)
            if res is None:
                try:
                    res = _engine_run(None, end=end, params=load_params())
                except ApiError:
                    raise
                except Exception as e:
                    raise ApiError(500, f"baseline simulation failed: {type(e).__name__}: {e}") from e
                _base_cache.put(key, res)
    return res


def baseline_series(end: int | None = None) -> dict:
    end = max_end() if end is None else end
    key = (params_signature(), end)
    s = _base_series.get(key)
    if s is None:
        s = series_of(baseline(end))
        _base_series.put(key, s)
    return s


# ---------------------------------------------------------------- scenarios

def scenario_id(scen: dict, end: int) -> str:
    canon = json.dumps({"measures": scen.get("measures", []), "end": end}, sort_keys=True, default=str)
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_scenario(scen: dict, pset: ParamSet, end: int) -> list[str]:
    """Human-readable problems with a scenario dict (empty list = valid)."""
    problems: list[str] = []
    for m in scen.get("measures", []) or []:
        mname = m.get("name", "?")
        for i, c in enumerate(m.get("changes", []) or []):
            where = f"measure '{mname}', change {i + 1}"
            name = c.get("param")
            if name not in pset.specs:
                problems.append(f"{where}: unknown parameter '{name}'")
                continue
            op = c.get("op", "set")
            if op not in ("set", "add", "mul"):
                problems.append(f"{where}: bad op '{op}' (use set, add or mul)")
                continue
            start, stop = c.get("start"), c.get("end")
            if not isinstance(start, int) or isinstance(start, bool):
                problems.append(f"{where}: start must be a year")
                continue
            if stop is not None and stop < start:
                problems.append(f"{where}: end year {stop} is before start year {start}")
            if (c.get("phase_in_years") or 0) < 0:
                problems.append(f"{where}: phase_in_years cannot be negative")
            value = c.get("value")
            if value is None:
                problems.append(f"{where}: value is missing")
                continue
            current = pset.with_scenario(None).value(name, max(start, BASE_YEAR), None)
            key = c.get("key")
            if key is not None:
                if not isinstance(current, (list, dict)):
                    problems.append(f"{where}: '{name}' is a scalar, it has no key '{key}'")
                    continue
                try:
                    current = core_params._get_key(current, str(key))
                except Exception:
                    problems.append(f"{where}: '{name}' has no element '{key}'")
                    continue
            if op in ("add", "mul") and not _is_num(value) and not isinstance(value, (list, dict)):
                problems.append(f"{where}: {op} needs a numeric value, got {value!r}")
            elif _is_num(current) and not _is_num(value) and not isinstance(current, bool):
                problems.append(f"{where}: '{name}' is numeric, got {value!r}")
            elif isinstance(current, list) and isinstance(value, list) and len(value) != len(current):
                problems.append(f"{where}: '{name}' has {len(current)} elements, got {len(value)}")
            elif isinstance(current, (list, dict)) and op == "set" and key is None and not isinstance(value, type(current)):
                problems.append(f"{where}: '{name}' is a {type(current).__name__}; give a full "
                                f"{type(current).__name__} or address one element with 'key'")
    return problems


def to_scenario(scen: dict) -> Scenario:
    try:
        return Scenario(
            name=scen.get("name") or "scenario",
            description=scen.get("description", "") or "",
            measures=[
                Measure(
                    name=m.get("name") or f"measure {i + 1}",
                    description=m.get("description", "") or "",
                    tags=list(m.get("tags", []) or []),
                    changes=[Change(**{k: v for k, v in c.items() if k in Change.__dataclass_fields__})
                             for c in (m.get("changes") or [])],
                )
                for i, m in enumerate(scen.get("measures") or [])
            ],
        )
    except TypeError as e:
        raise ApiError(400, f"malformed scenario: {e}") from e


def normalise_scenario(scen: dict) -> dict:
    """Canonical plain-dict form (drops disabled measures and unknown fields)."""
    out = {"name": scen.get("name") or "scenario", "description": scen.get("description") or "", "measures": []}
    for m in scen.get("measures") or []:
        if m.get("enabled") is False:
            continue
        changes = []
        for c in m.get("changes") or []:
            ch = {"param": c.get("param"), "value": c.get("value"), "start": c.get("start"),
                  "op": c.get("op") or "set"}
            if c.get("end") is not None:
                ch["end"] = c["end"]
            if c.get("phase_in_years"):
                ch["phase_in_years"] = c["phase_in_years"]
            if c.get("key") not in (None, ""):
                ch["key"] = str(c["key"])
            changes.append(ch)
        mm = {"name": m.get("name") or "measure", "description": m.get("description") or "", "changes": changes}
        if m.get("tags"):
            mm["tags"] = list(m["tags"])
        out["measures"].append(mm)
    return out


def run_scenario(scen: dict, end: int = END_YEAR) -> tuple[str, Results, Results]:
    """Returns (scenario id, baseline results, scenario results); cached by scenario hash."""
    scen = normalise_scenario(scen)
    sid = scenario_id(scen, end)
    sig = params_signature()
    base = baseline(end)
    hit = _run_cache.get((sig, sid))
    if hit is not None:
        return sid, base, hit
    pset = load_params()
    problems = validate_scenario(scen, pset, end)
    if problems:
        raise ApiError(422, "; ".join(problems))
    scenario = to_scenario(scen)
    try:
        res = _engine_run(scenario, end=end, baseline=base, params=pset)
    except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, ArithmeticError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        raise ApiError(422, f"scenario could not be simulated: {type(e).__name__}: {msg}") from e
    except Exception as e:
        raise ApiError(500, f"scenario simulation failed: {type(e).__name__}: {e}") from e
    _run_cache.put((sig, sid), res)
    _ids.put(sid, (scen, end))
    return sid, base, res


def results_for(run: str) -> Results:
    """Resolve the `run` query value: 'baseline' or a scenario id returned by /api/run."""
    if run in ("", "baseline", None):
        return baseline()
    known = _ids.get(run)
    if known is None:
        raise ApiError(404, f"unknown run '{run}': POST the scenario to /api/run first")
    scen, end = known
    return run_scenario(scen, end)[2]


# ---------------------------------------------------------------- parameters

def _baseline_index(base: Results):
    def index(kind: str, year: int) -> float:
        if year < BASE_YEAR or not base.has(year):
            return 1.0
        return base.at(year).macro.get(INDEX_KEYS.get(kind, kind), 1.0)
    return index


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if _is_num(v):
        return "scalar"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return "text"


def describe_params(lang: str = "en") -> dict:
    pset = load_params().with_scenario(None)
    try:
        base = baseline()
        years, index = base.years, _baseline_index(base)
    except ApiError:
        years, index = list(range(BASE_YEAR, END_YEAR + 1)), None
    out = []
    # a regional or community competence: one value per region or per federated entity
    from ..i18n.labels import entity_detail
    territories = set(REGIONS) | set(entity_detail())
    for d in pset.describe():
        name = d["name"]
        try:
            current = pset.value(name, BASE_YEAR, index)
        except Exception:
            current = None
        kind = _kind(current)
        item = dict(d)
        item["values"] = {str(y): v for y, v in sorted(d["values"].items())}
        item["kind"] = kind
        item["group"] = name.split(".", 1)[0] if "." in name else "other"
        # the convention of params/*.yaml is the upper-case marker (a source may also be
        # "Derived, partly ASSUMPTION"); the lower-case word in a citation does not count
        item["assumption"] = "ASSUMPTION" in str(d.get("source", ""))
        item["current"] = current
        # a regional competence: one value per region, so a measure can target one region
        item["regional"] = isinstance(current, dict) and bool(current) and set(map(str, current)) <= territories
        if lang != "en":
            item["description"] = translate("param", name, lang, item.get("description") or "")
            item["unit"] = translate("unit", item.get("unit"), lang, item.get("unit") or "")
        if kind in ("scalar", "bool"):
            series = []
            for y in years:
                try:
                    series.append(pset.value(name, y, index))
                except Exception:
                    series.append(None)
            item["baseline"] = series
        out.append(clean(item))
    groups: dict[str, dict] = {}
    for it in out:
        g = groups.setdefault(it["group"], {"group": it["group"], "count": 0, "assumptions": 0, "files": []})
        g["count"] += 1
        g["assumptions"] += int(it["assumption"])
        if it["file"] not in g["files"]:
            g["files"].append(it["file"])
    return {"years": years, "params": out, "groups": list(groups.values()), "lang": lang,
            "files": sorted({it["file"] for it in out}), "signature": params_signature()}


# ---------------------------------------------------------------- arrays

def arrays_for(res: Results, year: int) -> dict:
    if not res.has(year):
        raise ApiError(404, f"year {year} is not in this run ({res.years[0]}-{res.years[-1]})")
    s = res.at(year)
    r = lambda a, nd=1: np.round(np.asarray(a, dtype=float), nd)
    return clean({
        "year": year,
        "pop": r(s.pop.sum(axis=2)),                               # [age][sex]
        "pop_region": r(np.moveaxis(s.pop, 2, 0)),                 # [region][age][sex]
        "status_age": r(s.status.sum(axis=(2, 3))),                # [status][age]
        "status_age_sex": r(s.status.sum(axis=3)),                 # [status][age][sex]
        "emp": r(s.emp),                                           # [sector][jobtype]
        "wage": r(s.wage),                                         # [sector][jobtype]
        "wage_dist": r(s.wage_dist),                               # [jobtype][decile]
        "va": r(s.va, 3),                                          # [sector]
    })


def pyramid_for(res: Results) -> dict:
    """Population by age and sex for every year, total and by region (for the scrubbable pyramid)."""
    pop = np.stack([s.pop for s in res.states])                    # [year, age, sex, region]
    return clean({
        "years": res.years,
        "pop": np.round(pop.sum(axis=3)),                          # [year][age][sex]
        "pop_region": np.round(np.moveaxis(pop, 3, 0)),            # [region][year][age][sex]
    })


# ---------------------------------------------------------------- measure library & saved scenarios

def _read_doc(path: Path) -> Any:
    text = path.read_text()
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def _measures_in(doc: Any) -> tuple[list[dict], dict | None]:
    """Accepts a scenario {name, measures:[..]}, one measure {name, changes:[..]}, a list of
    measures, or {measures:[..]}. Returns (measures, scenario or None)."""
    if isinstance(doc, list):
        return [m for m in doc if isinstance(m, dict) and "changes" in m], None
    if isinstance(doc, dict):
        if "changes" in doc:
            return [doc], None
        if isinstance(doc.get("measures"), list):
            ms = [m for m in doc["measures"] if isinstance(m, dict)]
            scen = doc if doc.get("name") else None
            return ms, scen
    return [], None


def _doc_files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml", ".json"))


def _measure_regions(m: dict) -> list[str]:
    """Regions a measure addresses through region-keyed changes (`key` is a region code)."""
    from ..i18n.labels import entity_detail
    detail = entity_detail()
    found = {str(c.get("key")).split(".")[0] for c in m.get("changes") or [] if c.get("key") is not None}
    found |= {str(t) for t in m.get("tags") or []}
    for e in list(found):
        found |= set(detail.get(e, {}).get("regions") or [])
    return [r for r in REGIONS if r in found]


def localise_measure(m: dict, lang: str) -> dict:
    """Display name, description and tag labels of a library measure in `lang`. `name_en` stays
    the stable English name (ids and the 'already in the scenario' test rely on it)."""
    en = m.get("name_en") or m.get("name") or ""
    out = {**m, "name_en": en, "regions": _measure_regions(m),
           "tag_labels": {t: translate("tag", t, lang, t) for t in m.get("tags") or []}}
    if lang != "en":
        out["name"] = translate("measure.name", en, lang, en)
        out["description"] = translate("measure.description", en, lang, m.get("description") or "")
    return out


def measure_library(lang: str = "en") -> dict:
    lib = _measure_library()
    return {**lib, "lang": lang, "measures": [localise_measure(m, lang) for m in lib["measures"]],
            "scenarios": [{**sc, "name_en": sc.get("name"),
                           "name": translate("measure.name", sc.get("name"), lang, sc.get("name")),
                           "description": translate("measure.description", sc.get("name"), lang, sc.get("description") or ""),
                           "measures": [localise_measure(m, lang) for m in sc.get("measures") or []]}
                          for sc in lib["scenarios"]]}


def _measure_library() -> dict:
    try:
        specs = load_params().specs
    except ApiError:
        specs = {}
    measures, scenarios, errors = [], [], []
    for source, d in (("scenarios", scenarios_dir()), ("library", library_dir())):
        for path in _doc_files(d):
            rel = f"{source}/{path.name}"
            try:
                ms, scen = _measures_in(_read_doc(path))
            except Exception as e:
                errors.append({"file": rel, "error": str(e)})
                continue
            norm = normalise_scenario({"measures": ms})["measures"]
            for raw, m in zip(ms, norm):
                missing = sorted({c["param"] for c in m["changes"] if c["param"] not in specs})
                measures.append({**m, "tags": list(raw.get("tags", []) or []), "file": rel,
                                 "id": f"{rel}#{m['name']}", "unknown_params": missing})
            if scen is not None:
                scenarios.append({"name": scen.get("name"), "description": scen.get("description", ""),
                                  "file": rel, "measures": norm})
    return clean({"measures": measures, "scenarios": scenarios, "errors": errors})


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:60] or "scenario"


def user_dir() -> Path:
    return scenarios_dir() / "user"


def save_scenario(scen: dict) -> dict:
    full = normalise_scenario({**scen, "measures": [{**m, "enabled": True} for m in scen.get("measures") or []]})
    problems = []
    try:
        problems = validate_scenario(full, load_params(), END_YEAR)
    except ApiError:
        pass
    d = user_dir()
    d.mkdir(parents=True, exist_ok=True)
    slug = _slug(full["name"])
    path = d / f"{slug}.yaml"
    path.write_text(yaml.safe_dump(full, sort_keys=False, allow_unicode=True))
    return {"slug": slug, "file": f"scenarios/user/{path.name}", "scenario": full, "problems": problems}


def list_scenarios() -> dict:
    out, errors = [], []
    from . import public
    if public.read_only():      # shared server: nothing is stored there, so nothing is listed
        return {"scenarios": [], "errors": []}
    for path in _doc_files(user_dir()):
        try:
            doc = _read_doc(path)
            if not isinstance(doc, dict):
                raise ValueError("not a scenario mapping")
            out.append({"slug": path.stem, "kind": "user", "file": f"scenarios/user/{path.name}",
                        "modified": path.stat().st_mtime, "scenario": normalise_scenario(doc)})
        except Exception as e:
            errors.append({"file": f"scenarios/user/{path.name}", "error": str(e)})
    return clean({"scenarios": out, "errors": errors})


def delete_scenario(slug: str) -> None:
    if _slug(slug) != slug:
        raise ApiError(400, "bad scenario name")
    for ext in (".yaml", ".yml", ".json"):
        p = user_dir() / f"{slug}{ext}"
        if p.is_file():
            p.unlink()
            return
    raise ApiError(404, f"no saved scenario '{slug}'")


def line_meta(lines: dict) -> dict:
    return clean({k: asdict(v) for k, v in lines.items()})


# ---------------------------------------------------------------- headline results, batches

# Keys returned by the compact endpoints (comparison of scenarios, ranking of library measures).
HEADLINE_KEYS = [
    "fiscal.revenue_total", "fiscal.expenditure_total", "fiscal.balance", "fiscal.primary_balance",
    "fiscal.interest", "fiscal.debt", "fiscal.balance_pct_gdp", "fiscal.debt_pct_gdp",
    "fiscal.revenue_pct_gdp", "fiscal.expenditure_pct_gdp", "macro.gdp_nominal", "macro.gdp_real",
    "macro.cpi", "labour.employment", "labour.employment_rate_20_64", "labour.unemployment_rate",
]
# Regional series kept in the compact endpoints when the model publishes them (per-region
# breakdown of the Compare and Balance views): every `regional.<metric>.<REG>` about employment,
# GDP or a balance, and the balance / revenue / expenditure / debt of each entity.
REGIONAL_HEADLINE = re.compile(r"^regional\.[a-z0-9_]*(employ|gdp|balance|population)[a-z0-9_]*\.[A-Z]{3}$")
REGIONAL_ENTITY_HEADLINE = re.compile(r"^fiscal\.(balance|revenue|expenditure|debt)\.[a-z_]+$")


def headline_keys(series: dict) -> list[str]:
    """HEADLINE_KEYS present in `series`, plus the regional and per-entity headline series."""
    keys = [k for k in HEADLINE_KEYS if k in series]
    keys += sorted(k for k in series if REGIONAL_HEADLINE.match(k) or REGIONAL_ENTITY_HEADLINE.match(k))
    return keys


_head_cache = _LRU(512)     # (sig, scenario id) -> {key: [values]}  (small: no Results kept)


def headline(scen: dict, end: int = END_YEAR) -> tuple[str, dict[str, list]]:
    """Scenario id and its headline series; cached without keeping the full run in memory."""
    norm = normalise_scenario(scen)
    key = (params_signature(), scenario_id(norm, end))
    hit = _head_cache.get(key)
    if hit is None:
        sid, _, res = run_scenario(norm, end)
        s = series_of(res)
        hit = {k: s[k] for k in headline_keys(s)}
        _head_cache.put(key, hit)
    return key[1], hit


def run_batch(scens: list[dict], end: int = END_YEAR) -> dict:
    """Headline series of several scenarios. One bad scenario does not fail the others."""
    out = []
    for scen in scens:
        item: dict[str, Any] = {"name": scen.get("name") or "scenario"}
        try:
            item["id"], item["series"] = headline(scen, end)
        except ApiError as e:
            if e.status >= 500:
                raise
            item["error"] = e.message
        out.append(item)
    base = baseline_series(end)
    keys = headline_keys(base)
    return {"years": baseline(end).years, "keys": keys,
            "baseline": {k: base[k] for k in keys}, "scenarios": out}


def library_effects(end: int = END_YEAR, lang: str = "en") -> dict:
    """Every prebuilt measure simulated on its own against the baseline (headline deltas).
    Names and descriptions are served in `lang`; the simulation always uses the English source."""
    base = baseline_series(end)
    out = []
    for m in _measure_library()["measures"]:
        item = {k: m.get(k) for k in ("id", "name", "description", "tags", "file", "changes")}
        if m.get("unknown_params"):
            item["error"] = "unknown parameters: " + ", ".join(m["unknown_params"])
        else:
            try:
                _, s = headline({"name": m["name"], "measures": [m]}, end)
                item["delta"] = delta_of(base, s)
            except ApiError as e:
                item["error"] = e.message
        out.append(localise_measure(item, lang))
    return clean({"years": baseline(end).years, "keys": [k for k in headline_keys(base)],
                  "measures": out, "signature": params_signature(), "lang": lang})


# ---------------------------------------------------------------- behavioural-response band

BAND_SCALE_PARAM, BAND_SWITCH_PARAM = "beh.scale", "beh.enabled"
BAND_VARIANTS: dict[str, dict] = {
    "low": {"label": "Low behavioural response", "scale": 0.5},
    "central": {"label": "Central", "scale": 1.0},
    "high": {"label": "High behavioural response", "scale": 1.5},
    "static": {"label": "Static (no behavioural response)", "scale": 0.0},
}


def _band_measure(variant: str) -> dict | None:
    start = BASE_YEAR + 1
    if variant == "central":
        return None
    if variant == "static":
        change = {"param": BAND_SWITCH_PARAM, "value": 0, "start": start, "op": "set"}
    else:
        change = {"param": BAND_SCALE_PARAM, "value": BAND_VARIANTS[variant]["scale"], "start": start, "op": "mul"}
    return {"name": f"[sensitivity] {BAND_VARIANTS[variant]['label']}", "changes": [change]}


def run_band(scen: dict, end: int = END_YEAR) -> dict:
    """The scenario under low / central / high behavioural response and without any (static)."""
    specs = load_params().specs
    missing = [p for p in (BAND_SCALE_PARAM, BAND_SWITCH_PARAM) if p not in specs]
    if missing:
        raise ApiError(422, "the model has no parameter " + ", ".join(missing) + ": no sensitivity band")
    scen = normalise_scenario(scen)
    runs, ids = {}, {}
    for variant in BAND_VARIANTS:
        extra = _band_measure(variant)
        s = {**scen, "measures": scen["measures"] + ([extra] if extra else [])}
        sid, _, res = run_scenario(s, end)
        runs[variant] = {"series": series_of(res)}
        ids[variant] = sid
    return {"id": ids["central"], "ids": ids, "years": baseline(end).years, "runs": runs,
            "variants": BAND_VARIANTS,
            "note": "Range of behavioural responses (every elasticity and multiplier scaled by 0.5 "
                    "and 1.5), not a statistical confidence interval."}


# ---------------------------------------------------------------- validation against official figures

_valid_cache = _LRU(2)
_valid_lock = threading.Lock()


def validation_file() -> Path:
    from .. import validation as V
    return Path(os.environ.get("BELSIM_VALIDATION_FILE") or V.REFERENCE_FILE)


def _source_of(entry: dict) -> dict:
    src = entry.get("source") or {}
    if isinstance(src, str):
        src = {"title": src}
    return {"institution": src.get("institution", ""), "title": src.get("title", ""),
            "date": str(src.get("date", "") or ""), "url": entry.get("url", "") or ""}


def validation_report(lang: str = "en") -> dict:
    """The validation report with descriptions, target labels and units in `lang`."""
    rep = _validation_report()
    if lang == "en":
        return rep

    def row(r: dict) -> dict:
        out = {**r, "description": translate("validation", r.get("id"), lang, r.get("description") or ""),
               "unit": translate("unit", r.get("unit"), lang, r.get("unit") or "")}
        if r.get("label"):
            out["label"] = translate("target", r["label"], lang, r["label"])
        return out

    meta = dict(rep.get("meta") or {})
    if isinstance(meta.get("conventions"), str):
        meta["conventions"] = translate("text", meta["conventions"].strip(), lang, meta["conventions"])
    return {**rep, "lang": lang, "meta": meta,
            "baseline": [row(r) for r in rep["baseline"]], "measures": [row(r) for r in rep["measures"]],
            "not_found": [{**n, "description": translate("validation", n.get("id"), lang, n.get("description") or "")}
                          if isinstance(n, dict) else n for n in rep.get("not_found") or []]}


def _validation_report() -> dict:
    """`belsim.validation.run_validation()` as JSON; cached on the params and reference files."""
    import time
    from .. import validation as V

    path = validation_file()
    if not path.is_file():
        raise ApiError(404, f"no validation reference file at {path}")
    st = path.stat()
    key = (params_signature(), str(path), st.st_mtime_ns, st.st_size)
    hit = _valid_cache.get(key)
    if hit is not None:
        return hit
    with _valid_lock:
        hit = _valid_cache.get(key)
        if hit is not None:
            return hit
        t0 = time.time()
        try:
            data = V.load(path)
            rep = V.run_validation(path, base=baseline(END_YEAR))
        except ApiError:
            raise
        except Exception as e:
            raise ApiError(500, f"validation failed: {type(e).__name__}: {e}") from e
        entries = {e.get("id"): e for e in list(data["baseline"]) + list(data["measures"])}

        def common(row) -> dict:
            e = entries.get(row.id) or {}
            return {"id": row.id, "confidence": row.confidence, "status": row.status, "ref": row.ref,
                    "gap": row.gap, "gap_pct": row.gap_pct, "unit": row.unit,
                    "source": _source_of(e), "note": e.get("note", "") or ""}

        base_rows = [{**common(r), "description": r.description, "when": r.when, "model": r.model}
                     for r in rep.baseline]
        meas_rows = []
        targets = {(e.get("id"), t.get("label", "")): t for e in data["measures"] for t in e.get("targets") or []}
        for r in rep.measures:
            e = entries.get(r.id) or {}
            t = targets.get((r.id, r.label)) or {}
            meas_rows.append({**common(r), "description": e.get("description", ""), "label": r.label,
                              "variant": r.variant, "ref_year": r.ref_year, "model_year": r.model_year,
                              "static": r.static, "behavioural": r.behavioural, "model": r.compared,
                              "target_note": t.get("note", "") or "",
                              "measure": e.get("measure") if not e.get("reference_measure") else None})
        hit = clean({
            "baseline": base_rows, "measures": meas_rows, "problems": rep.problems,
            "not_found": rep.not_found, "meta": data.get("meta") or {},
            "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            "seconds": round(time.time() - t0, 1), "signature": params_signature(),
        })
        _valid_cache.put(key, hit)
        return hit


# The report takes about a minute the first time. `validation_start` computes it in a background
# thread (at start-up in public mode, or when a request asks not to wait) and
# `validation_status` tells a client what to show meanwhile.
_valid_job: dict[str, Any] = {"thread": None, "started": None, "error": None, "failed_at": None}
_valid_job_lock = threading.Lock()
VALIDATION_RETRY_SECONDS = 300      # after a failure, not before this long
VALIDATION_TYPICAL_SECONDS = 60


def validation_ready() -> bool:
    path = validation_file()
    if not path.is_file():
        return False
    st = path.stat()
    return _valid_cache.get((params_signature(), str(path), st.st_mtime_ns, st.st_size)) is not None


def validation_start() -> None:
    import time

    with _valid_job_lock:
        t = _valid_job["thread"]
        if (t is not None and t.is_alive()) or validation_ready():
            return
        if _valid_job["failed_at"] and time.time() - _valid_job["failed_at"] < VALIDATION_RETRY_SECONDS:
            return

        def work():
            try:
                _validation_report()
                _valid_job["error"], _valid_job["failed_at"] = None, None
            except Exception as e:      # shown as a state, never raised in a thread
                _valid_job["error"] = e.message if isinstance(e, ApiError) else type(e).__name__
                _valid_job["failed_at"] = time.time()

        _valid_job["started"] = time.time()
        _valid_job["thread"] = threading.Thread(target=work, name="belsim-validation", daemon=True)
        _valid_job["thread"].start()


def validation_status() -> dict:
    """{"status": "ready" | "computing" | "error" | "idle", ...} without computing anything."""
    import time

    if validation_ready():
        return {"status": "ready"}
    t = _valid_job["thread"]
    if t is not None and t.is_alive():
        return {"status": "computing", "elapsed": round(time.time() - (_valid_job["started"] or time.time()), 1),
                "typical_seconds": VALIDATION_TYPICAL_SECONDS}
    if _valid_job["error"]:
        return {"status": "error", "error": _valid_job["error"]}
    return {"status": "idle"}


# ---------------------------------------------------------------- translated metadata

def line_meta_localised(lines: dict, side: str, lang: str) -> dict:
    """LINES metadata with label, group and description in `lang`; `group_id` keeps the English
    group name as a stable identifier."""
    out = line_meta(lines)
    for lid, meta in out.items():
        meta["group_id"] = meta.get("group") or ""
        if lang != "en":
            meta["label"] = translate("line.label", f"{side}.{lid}", lang, meta.get("label") or lid)
            meta["group"] = translate("group", meta["group_id"], lang, meta["group_id"])
            meta["description"] = translate("line.description", f"{side}.{lid}", lang, meta.get("description") or "")
    return out


def localise_distribution(out: Any, lang: str) -> Any:
    """Household labels, groups and the method notes of a /api/distribution answer in `lang`
    (a copy: the answer may be a cached object)."""
    if lang == "en" or not isinstance(out, dict):
        return out
    out = dict(out)
    hh = out.get("households")
    if isinstance(hh, list):
        out["households"] = [
            {**h, "label": translate("household", h.get("id"), lang, h.get("label") or ""),
             "group_id": h.get("group"),
             "group": translate("household_group", h.get("group"), lang, h.get("group") or "")}
            if isinstance(h, dict) else h for h in hh]
    pov = out.get("poverty")
    if isinstance(pov, dict):       # poverty by household type and by activity status
        pov = dict(pov)
        for key in ("by_household_type", "by_activity_status"):
            if isinstance(pov.get(key), list):
                pov[key] = [{**r, "label": translate("text", r.get("label"), lang, r.get("label") or "")}
                            if isinstance(r, dict) else r for r in pov[key]]
        out["poverty"] = pov
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta = dict(meta)
        tx = lambda v: translate("text", v, lang, v) if isinstance(v, str) else v
        for key in ("method", "income_concept"):
            meta[key] = tx(meta.get(key))
        if isinstance(meta.get("caveats"), (list, tuple)):
            meta["caveats"] = [tx(c) for c in meta["caveats"]]
        elif isinstance(meta.get("caveats"), str):
            meta["caveats"] = tx(meta["caveats"])
        if isinstance(meta.get("causes"), dict):
            meta["causes"] = {k: tx(v) for k, v in meta["causes"].items()}
        out["meta"] = {k: v for k, v in meta.items() if v is not None}
    return out


def localise_payslip_options(out: Any, lang: str) -> Any:
    if lang == "en" or not isinstance(out, dict):
        return out
    out = dict(out)

    def labelled(items, kind):
        if not isinstance(items, list):
            return items
        return [{**o, "label": translate(kind, o.get("id"), lang, o.get("label") or "")} if isinstance(o, dict) else o
                for o in items]

    out["kinds"] = labelled(out.get("kinds"), "dim.payslip_kinds")
    out["households"] = labelled(out.get("households"), "preset")
    fields = out.get("household_fields")
    if isinstance(fields, dict):
        out["household_fields"] = {**fields, "region": labelled(fields.get("region"), "dim.payslip_regions")}
    return out
