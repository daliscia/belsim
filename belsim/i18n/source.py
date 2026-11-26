"""Collects every English source text the model serves, by kind and id: what the catalogs
`fr.yaml` / `nl.yaml` are expected to translate. Each section is collected defensively, so a
module under construction only empties its own section."""
from __future__ import annotations

from typing import Any

from ..core import dims

# Kinds whose coverage is reported; `text` and `target` are free text keyed by the
# English sentence itself.
KINDS = ["line.label", "line.description", "group", "dim", "param", "unit", "measure.name",
         "measure.description", "tag", "household", "household_group", "preset", "validation",
         "target", "series", "text"]


def _try(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _lines() -> dict[str, dict]:
    from ..fiscal import expenditure, revenue
    return {"revenue": revenue.LINES, "expenditure": expenditure.LINES}


def collect(with_runs: bool = True) -> dict[str, dict[str, str]]:
    """{kind: {id: English text}}. `with_runs=False` skips what needs a simulation (series
    keys, distribution caveats)."""
    from ..api import units as U
    from . import labels as L

    out: dict[str, dict[str, str]] = {k: {} for k in KINDS}

    lines = _try(_lines, {"revenue": {}, "expenditure": {}})
    for side, table in lines.items():
        for lid, meta in table.items():
            out["line.label"][f"{side}.{lid}"] = meta.label
            if meta.description:
                out["line.description"][f"{side}.{lid}"] = meta.description
            if meta.group:
                out["group"][meta.group] = meta.group

    for dim, table in U.dim_labels().items():
        for code, label in table.items():
            out["dim"][f"{dim}.{code}"] = label
    for eid, info in L.entity_detail().items():
        out["dim"][f"entities.{eid}"] = info["label"]
    for table, dim in ((_try(lambda: __import__("belsim.api.distribution", fromlist=["x"]).KIND_LABELS, {}), "payslip_kinds"),
                       (_try(lambda: __import__("belsim.api.distribution", fromlist=["x"]).REGION_LABELS, {}), "payslip_regions")):
        for code, label in table.items():
            out["dim"][f"{dim}.{code}"] = label

    def params():
        from ..api import service as svc
        for spec in svc.load_params().specs.values():
            if spec.description:
                out["param"][spec.name] = spec.description
            if spec.unit:
                out["unit"][spec.unit] = spec.unit
    _try(params, None)

    def library():
        from ..api import service as svc
        for m in svc.measure_library()["measures"]:
            out["measure.name"][m["name"]] = m["name"]
            if m.get("description"):
                out["measure.description"][m["name"]] = m["description"]
            for t in m.get("tags") or []:
                out["tag"][t] = t
    _try(library, None)

    def households():
        from ..distribution import households as H
        for row in H.DEFINITIONS:
            out["household"][row[0]] = row[1]
            out["household_group"][row[2]] = row[2]
    _try(households, None)

    def presets():
        from ..distribution.calc import HOUSEHOLD_PRESETS
        for pid, spec in HOUSEHOLD_PRESETS.items():
            out["preset"][pid] = spec["label"]
    _try(presets, None)

    def validation():
        from .. import validation as V
        data = V.load()
        for e in list(data.get("baseline") or []) + list(data.get("measures") or []) + list(data.get("not_found") or []):
            if e.get("id") and e.get("description"):
                out["validation"][e["id"]] = e["description"]
            if e.get("unit"):
                out["unit"][e["unit"]] = e["unit"]
            for t in e.get("targets") or []:
                if t.get("label"):
                    out["target"][t["label"]] = t["label"]
                if t.get("ref_unit"):
                    out["unit"][t["ref_unit"]] = t["ref_unit"]
        conv = (data.get("meta") or {}).get("conventions")
        if conv:
            out["text"][str(conv).strip()] = str(conv).strip()
    _try(validation, None)

    if with_runs:
        def series():
            from ..api import service as svc
            seen: dict[str, str] = {}

            def rec(kind: str, id: Any, default: str) -> str:
                if kind == "series":
                    seen[id] = default
                return default

            lab = L.Labeller("en", rec, svc.line_meta(lines["revenue"]), svc.line_meta(lines["expenditure"]))
            keys = list(svc.baseline_series())
            lab.labels(keys)
            lab.regional_metrics(keys)
            out["series"].update(seen)
        _try(series, None)

        def distribution_text():
            from ..api import distribution as D
            res = D.distribution(D.DistributionIn(scenario={}, year=dims.BASE_YEAR + 6))
            for s in meta_texts(res.get("meta") or {}):
                out["text"][s] = s
            pov = res.get("poverty") or {}
            for rows in (pov.get("by_household_type"), pov.get("by_activity_status")):
                for row in rows or []:
                    if isinstance(row, dict) and row.get("label"):
                        out["text"][row["label"]] = row["label"]
        _try(distribution_text, None)
    return out


def meta_texts(meta: dict) -> list[str]:
    """The free-text sentences of a distribution `meta` block (method, caveats, notes)."""
    found: list[str] = []
    for key in ("method", "income_concept"):
        if isinstance(meta.get(key), str):
            found.append(meta[key])
    cav = meta.get("caveats")
    if isinstance(cav, str):
        found.append(cav)
    elif isinstance(cav, (list, tuple)):
        found.extend(c for c in cav if isinstance(c, str))
    causes = meta.get("causes")
    if isinstance(causes, dict):
        found.extend(v for v in causes.values() if isinstance(v, str))
    return found
