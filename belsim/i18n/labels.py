"""Human labels of series keys, composed from translated parts.

A key such as `fiscal.balance.walloon_region` or `regional.unemployment_rate.BRU` is not
translated as a whole: its metric ("Balance", "Unemployment rate") and its member (entity,
region, budget line, sector ...) are looked up separately, so new regions, entities and lines
get a label without a catalog entry of their own.
"""
from __future__ import annotations

from typing import Any, Callable

from ..core import dims
from . import translate

Tr = Callable[[str, Any, str], str]       # (kind, id, English default) -> text

_ACRONYMS = {"gdp": "GDP", "cpi": "CPI", "tfr": "Total fertility rate", "pct": "%", "vat": "VAT",
             "pit": "PIT", "cit": "CIT", "ssc": "SSC", "eu": "EU", "10y": "10-year", "rcc": "RCC",
             "gni": "GNI"}

# English labels where the humanised key reads badly. Everything else: `humanise(key tail)`.
SERIES_EN = {
    "pop.total": "Population",
    "pop.age_0_17": "Population aged 0-17",
    "pop.age_18_66": "Population aged 18-66",
    "pop.age_67_plus": "Population aged 67 and over",
    "fiscal.revenue_total": "Total revenue",
    "fiscal.expenditure_total": "Total expenditure",
    "fiscal.revenue": "Revenue",
    "fiscal.expenditure": "Expenditure",
    "fiscal.balance": "Balance",
    "fiscal.debt": "Debt",
    "fiscal.transfer.*": "Transfer from {from} to {to}",
    "regional.revenue.*": "Revenue: {line}",
    "regional.expenditure.*": "Expenditure: {line}",
    }


def humanise(code: str) -> str:
    words = [_ACRONYMS.get(w.lower(), w) for w in str(code).replace(".", " ").replace("_", " ").split()]
    text = " ".join(words).replace("% GDP", "(% of GDP)")
    return text[:1].upper() + text[1:]


def entity_detail() -> dict[str, dict]:
    """Federated entities below the general-government subsectors, as
    {entity id: {"label": English label, "parent": subsector id}}; {} while the regional model
    does not publish them. Accepts the shapes `ENTITY_DETAIL` may take: child -> label,
    child -> {label, parent, ...} or parent -> [children] / {child: label}."""
    try:
        from .. import regional
        raw = getattr(regional, "ENTITY_DETAIL", None)
    except Exception:
        raw = None
    out: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return out
    default_parent = "regions_communities"
    for key, val in raw.items():
        key = str(key)
        if key in dims.ENTITIES and isinstance(val, (list, tuple, dict)) and not (
                isinstance(val, dict) and ({"label", "name", "parent"} & set(val))):
            children = val.items() if isinstance(val, dict) else ((c, None) for c in val)
            for child, info in children:
                if isinstance(child, dict):
                    info, child = child, child.get("id")
                label = info.get("label") or info.get("name") if isinstance(info, dict) else info
                out[str(child)] = {"label": str(label or humanise(str(child))), "parent": key}
        elif isinstance(val, dict):
            parent = val.get("parent") or val.get("subsector") or val.get("entity") or default_parent
            out[key] = {"label": str(val.get("label") or val.get("name") or humanise(key)),
                        "parent": str(parent) if str(parent) in dims.ENTITIES else default_parent,
                        **{k: v for k, v in val.items() if k in ("esa", "regions", "kind", "note")}}
        else:
            out[key] = {"label": str(val) if isinstance(val, str) and val else humanise(key),
                        "parent": default_parent}
    return out


class Labeller:
    """Labels series keys in one language. `tr` defaults to the catalog lookup; `source.py`
    passes a recording function to list every id that needs a translation."""

    def __init__(self, lang: str = "en", tr: Tr | None = None, revenue_lines: dict | None = None,
                 expenditure_lines: dict | None = None):
        from ..api import units as U

        self.lang = lang
        self.tr: Tr = tr or (lambda kind, id, default: translate(kind, id, lang, default))
        self.lines = {"revenue": revenue_lines or {}, "expenditure": expenditure_lines or {}}
        self.detail = entity_detail()
        self._dim_en = U.dim_labels()
        self._dim_en["entities"] = {**self._dim_en["entities"],
                                    **{k: v["label"] for k, v in self.detail.items()}}
        self.entities = set(self._dim_en["entities"])
        self.regions = set(dims.REGIONS)

    def dim(self, dim: str, code: str) -> str:
        return self.tr(f"dim.{dim}", code, self._dim_en.get(dim, {}).get(code, humanise(code)))

    def line(self, side: str, lid: str) -> str | None:
        meta = self.lines[side].get(lid)
        if meta is None:
            return None
        en = meta.get("label") if isinstance(meta, dict) else getattr(meta, "label", None)
        return self.tr("line.label", f"{side}.{lid}", en or humanise(lid))

    def series(self, id: str, default: str | None = None) -> str:
        return self.tr("series", id, default or SERIES_EN.get(id) or self._metric_en(id) or humanise(id.split(".", 1)[-1]))

    @staticmethod
    def _metric_en(id: str) -> str | None:
        """English label of `regional.<metric>` from the regional module's METRICS table."""
        if not id.startswith("regional."):
            return None
        try:
            from ..regional import METRICS
            return str(METRICS[id[len("regional."):]][0])
        except Exception:
            return None

    def label(self, key: str) -> str:
        parts = key.split(".")
        sec, rest = parts[0], parts[1:]
        tail = ".".join(rest)
        if sec in self.lines and (lab := self.line(sec, tail)) is not None:
            return lab
        if sec == "status" and rest:
            return self.dim("statuses", tail)
        if sec in ("emp", "va") and len(rest) == 2 and rest[0] == "sector":
            return self.dim("sectors", rest[1])
        if sec == "emp" and len(rest) == 2 and rest[0] == "jobtype":
            return self.dim("jobtypes", rest[1])
        if sec == "pop" and len(rest) == 2 and rest[0] == "region":
            return self.dim("regions", rest[1])
        if sec == "pop" and len(rest) == 2 and rest[0] == "sex":
            return self.dim("sexes", rest[1])
        if sec == "macro" and len(rest) >= 2 and rest[0] == "cons":
            return self.dim("cons_cats", ".".join(rest[1:]))
        if sec == "fiscal" and len(rest) == 2 and rest[0] == "transfer" and "_to_" in rest[1]:
            for a in sorted(self.entities, key=len, reverse=True):
                if rest[1].startswith(a + "_to_") and rest[1][len(a) + 4:] in self.entities:
                    b = rest[1][len(a) + 4:]
                    return self.series("fiscal.transfer.*").replace("{from}", self.dim("entities", a)) \
                        .replace("{to}", self.dim("entities", b))
        if len(rest) >= 2 and rest[-1] in self.regions:
            return f"{self.label('.'.join(parts[:-1]))}, {self.dim('regions', rest[-1])}"
        if len(rest) >= 2 and rest[-1] in self.entities:
            return f"{self.label('.'.join(parts[:-1]))}, {self.dim('entities', rest[-1])}"
        if sec == "regional" and len(rest) >= 2 and rest[0] in self.lines:
            lab = self.line(rest[0], ".".join(rest[1:]))
            if lab is not None:
                return self.series(f"regional.{rest[0]}.*").replace("{line}", lab)
        return self.series(key)

    def labels(self, keys) -> dict[str, str]:
        out = {}
        for k in keys:
            try:
                out[k] = self.label(k)
            except Exception:
                out[k] = humanise(k.split(".", 1)[-1])
        return out

    def regional_metrics(self, keys) -> dict[str, str]:
        """{metric: label} for every `regional.<metric>.<REG>` key (label without the region)."""
        out = {}
        for k in keys:
            parts = k.split(".")
            if parts[0] == "regional" and len(parts) >= 3 and parts[-1] in self.regions:
                metric = ".".join(parts[1:-1])
                if metric not in out:
                    try:
                        out[metric] = self.label("regional." + metric)
                    except Exception:
                        out[metric] = humanise(metric)
        return out
