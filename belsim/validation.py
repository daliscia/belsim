"""Validation of the model against official reference figures.

    uv run python -m belsim.validation

Reads `data/reference/validation.yaml`, runs the baseline and every costed measure through
the engine and writes `data/reference/validation_report.md` (reference vs model, absolute and
relative gap). Nothing here asserts closeness: it only measures the gaps.

Format of the YAML
------------------
baseline: list of references on the no-policy-change projection
    id, description, unit, source {institution, title, date}, url, confidence, note
    values: {<year> | "<y0>-<y1>": number}
    model:                       (omit when the model has no equivalent: listed, not compared)
      num: [series keys]         summed; keys are those of YearState.series() (fnmatch
                                 wildcards allowed) plus pop.age_<a>_plus / pop.age_<a>_<b>
      den: [series keys]         optional denominator (summed)
      scale: 100                 multiplies the result (default 1)
      transform: level           level | growth_pct (year on year) | change_since (needs
                                 base_year). For period keys "<y0>-<y1>": avg_growth_pct |
                                 change | mean

measures: list of published costings
    id, description, source, url, confidence, note
    measure: {name, changes: [{param, value, start, op, end, phase_in_years, key}]}
    reference_measure: {changes} optional counterfactual. The effect is then run(measure)
                                 minus run(reference_measure) instead of minus the baseline:
                                 used for reforms already in the baseline (measure restates
                                 the reform, reference_measure removes it) and for costings
                                 made against another starting point than the model baseline
    (an entry without `measure` is documented but has no model equivalent)
    targets: list of
      label, ref_value, ref_unit, ref_year, note
      variant: static | behavioural | unspecified   what the reference figure is
      model_year: year read in the model (default ref_year)
      model: {num, den, scale, mode}   mode: diff (scenario - baseline; of the ratio when
                                       `den` is given) | pct (relative change of num, in %)
      normalise: gdp             express both sides in % of GDP of their own year (for
                                 references about past years or in euros of another year)
      price_year: 2024           the reference is in constant prices of that year; it is
                                 inflated to ref_year prices with the baseline CPI

not_found: list of {id, description, searched}   references looked for without success
"""
from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .core import engine
from .core.dims import BASE_YEAR, END_YEAR
from .core.params import Change, Measure, ParamSet, Scenario
from .core.state import Results, YearState

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILE = ROOT / "data" / "reference" / "validation.yaml"
REPORT_FILE = ROOT / "data" / "reference" / "validation_report.md"
GDP_HISTORY_FILE = ROOT / "data" / "base" / "macro_gdp_history.csv"

VARIANTS = ("static", "behavioural", "unspecified")
CONFIDENCE = ("high", "medium", "low")
STATIC_SWITCH = "beh.enabled"

_AGE_PLUS = re.compile(r"^pop\.age_(\d+)_plus$")
_AGE_BAND = re.compile(r"^pop\.age_(\d+)_(\d+)$")
_PERIOD = re.compile(r"^(\d{4})\s*-\s*(\d{4})$")


class ValidationError(Exception):
    """A reference cannot be evaluated (unknown series, year out of range, ...)."""


# --------------------------------------------------------------------------- loading

def load(path: Path | str | None = None) -> dict:
    data = yaml.safe_load(Path(path or REFERENCE_FILE).read_text()) or {}
    data.setdefault("baseline", [])
    data.setdefault("measures", [])
    data.setdefault("not_found", [])
    return data


def measure_of(entry: dict, key: str = "measure") -> Measure:
    """The `Measure` equivalent to a costed reference (or its counterfactual)."""
    d = dict(entry[key])
    d.setdefault("name", f"{entry['id']} ({key})")
    return Measure.from_dict(d)


def missing_params(entry: dict, known: set[str] | None = None) -> list[str]:
    """Parameters used by the measure of `entry` that the model does not have."""
    if known is None:
        known = set(ParamSet.load().specs)
    used = {c["param"] for key in ("measure", "reference_measure")
            for c in (entry.get(key) or {}).get("changes", [])}
    return sorted(used - known)


def gdp_history() -> dict[int, float]:
    """Nominal GDP (EUR mn) of past years, from the calibration tables."""
    out: dict[int, float] = {}
    if GDP_HISTORY_FILE.exists():
        with GDP_HISTORY_FILE.open() as f:
            for row in csv.DictReader(f):
                try:
                    out[int(row["year"])] = float(row["gdp_nominal"])
                except (KeyError, ValueError):
                    continue
    return out


# --------------------------------------------------------------------------- series

def series_keys(state: YearState, patterns: list[str]) -> list[str]:
    """Expand wildcards against the scalar series of `state`; unknown plain keys are kept."""
    flat = state.series()
    out: list[str] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            hits = sorted(fnmatch.filter(flat, pat))
            if not hits:
                raise ValidationError(f"no series matches '{pat}'")
            out.extend(hits)
        else:
            out.append(pat)
    return out


def series_value(state: YearState, key: str, flat: dict[str, float] | None = None) -> float:
    flat = state.series() if flat is None else flat
    if key in flat:
        return float(flat[key])
    m = _AGE_PLUS.match(key)
    if m:
        return float(state.pop[int(m.group(1)):].sum())
    m = _AGE_BAND.match(key)
    if m:
        return float(state.pop[int(m.group(1)):int(m.group(2)) + 1].sum())
    raise ValidationError(f"unknown series '{key}'")


def _state(res: Results, year: int) -> YearState:
    if not res.has(year):
        raise ValidationError(f"year {year} outside the simulated range")
    return res.at(year)


def quantity(res: Results, year: int, spec: dict) -> float:
    """sum(num) / sum(den) * scale for one year of one run."""
    state = _state(res, year)
    flat = state.series()
    num = sum(series_value(state, k, flat) for k in series_keys(state, _as_list(spec.get("num"))))
    den_keys = _as_list(spec.get("den"))
    if den_keys:
        den = sum(series_value(state, k, flat) for k in series_keys(state, den_keys))
        if den == 0:
            raise ValidationError("denominator is zero")
        num /= den
    return num * float(spec.get("scale", 1.0))


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def baseline_value(res: Results, when: Any, spec: dict) -> float:
    """Model equivalent of one baseline reference value (`when` = year or 'y0-y1')."""
    if not _as_list(spec.get("num")):
        raise ValidationError("model spec has no 'num'")
    transform = spec.get("transform", "level")
    period = _PERIOD.match(str(when))
    if period:
        y0, y1 = int(period.group(1)), int(period.group(2))
        if y1 <= y0:
            raise ValidationError(f"bad period '{when}'")
        if transform in ("avg_growth_pct", "growth_pct"):
            a, b = quantity(res, y0, spec), quantity(res, y1, spec)
            if a <= 0 or b <= 0:
                raise ValidationError("growth of a non-positive quantity")
            return ((b / a) ** (1.0 / (y1 - y0)) - 1.0) * 100.0
        if transform in ("change", "change_since"):
            return quantity(res, y1, spec) - quantity(res, y0, spec)
        if transform in ("mean", "level"):
            return sum(quantity(res, y, spec) for y in range(y0, y1 + 1)) / (y1 - y0 + 1)
        raise ValidationError(f"unknown transform '{transform}' for a period")
    year = int(when)
    if transform == "level":
        return quantity(res, year, spec)
    if transform == "growth_pct":
        a, b = quantity(res, year - 1, spec), quantity(res, year, spec)
        if a == 0:
            raise ValidationError("growth from zero")
        return (b / a - 1.0) * 100.0
    if transform == "change_since":
        if "base_year" not in spec:
            raise ValidationError("change_since needs base_year")
        return quantity(res, year, spec) - quantity(res, int(spec["base_year"]), spec)
    raise ValidationError(f"unknown transform '{transform}'")


def effect(base: Results, scen: Results, year: int, spec: dict) -> float:
    """Scenario minus baseline for one target metric."""
    if not _as_list(spec.get("num")):
        raise ValidationError("model spec has no 'num'")
    mode = spec.get("mode", "diff")
    if mode == "diff":
        return quantity(scen, year, spec) - quantity(base, year, spec)
    if mode == "pct":
        plain = {k: v for k, v in spec.items() if k != "scale"}
        b = quantity(base, year, plain)
        if b == 0:
            raise ValidationError("relative change from zero")
        return (quantity(scen, year, plain) / b - 1.0) * 100.0
    raise ValidationError(f"unknown mode '{mode}'")


# --------------------------------------------------------------------------- rows

@dataclass
class BaselineRow:
    id: str
    description: str
    when: str
    unit: str
    confidence: str
    ref: float
    model: float | None = None
    status: str = "ok"

    @property
    def gap(self) -> float | None:
        return None if self.model is None else self.model - self.ref

    @property
    def gap_pct(self) -> float | None:
        return _pct(self.gap, self.ref)


@dataclass
class MeasureRow:
    id: str
    label: str
    variant: str
    ref_year: int
    model_year: int
    unit: str
    confidence: str
    ref: float
    static: float | None = None
    behavioural: float | None = None
    status: str = "ok"

    @property
    def compared(self) -> float | None:
        """Model figure the reference is compared with (static only if it says so)."""
        return self.static if self.variant == "static" else self.behavioural

    @property
    def gap(self) -> float | None:
        return None if self.compared is None else self.compared - self.ref

    @property
    def gap_pct(self) -> float | None:
        return _pct(self.gap, self.ref)


@dataclass
class Report:
    baseline: list[BaselineRow] = field(default_factory=list)
    measures: list[MeasureRow] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    not_found: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


def _pct(gap: float | None, ref: float) -> float | None:
    if gap is None or ref == 0:
        return None
    return gap / abs(ref) * 100.0


# --------------------------------------------------------------------------- running

def _run(measures: list[Measure], base: Results, end: int, name: str) -> Results:
    return engine.run(Scenario(name=name, measures=measures), end=end, baseline=base)


def _target_years(target: dict) -> tuple[int, int]:
    ref_year = int(target["ref_year"])
    return ref_year, int(target.get("model_year", ref_year))


def _normalise_gdp(ref: float, ref_year: int, base: Results, history: dict[int, float]) -> float:
    if ref_year >= BASE_YEAR and base.has(ref_year):
        gdp = base.at(ref_year).macro["gdp_nominal"]
    elif ref_year in history:
        gdp = history[ref_year]
    else:
        raise ValidationError(f"no GDP for {ref_year} to normalise the reference")
    return ref / gdp * 100.0


def run_validation(path: Path | str | None = None, *, base: Results | None = None,
                   end: int = END_YEAR) -> Report:
    data = load(path)
    report = Report(not_found=list(data.get("not_found") or []))
    try:
        base = base or engine.run(None, end=end)
    except Exception as exc:  # model under construction elsewhere: report, do not crash
        report.problems.append(f"baseline run failed: {type(exc).__name__}: {exc}")
        return report
    last = base.years[-1]
    known = set(ParamSet.load().specs)
    history = gdp_history()

    for entry in data["baseline"]:
        report.sources.append(_source_row(entry))
        spec = entry.get("model")
        for when, ref in (entry.get("values") or {}).items():
            row = BaselineRow(id=entry["id"], description=entry.get("description", ""),
                              when=str(when), unit=entry.get("unit", ""),
                              confidence=entry.get("confidence", ""), ref=float(ref))
            if not spec:
                row.status = "no model equivalent"
            else:
                try:
                    row.model = baseline_value(base, when, spec)
                except (ValidationError, KeyError, ZeroDivisionError) as exc:
                    row.status = str(exc)
                    report.problems.append(f"{entry['id']} [{when}]: {exc}")
            report.baseline.append(row)

    for entry in data["measures"]:
        report.sources.append(_source_row(entry))
        targets = entry.get("targets") or []
        rows = []
        for t in targets:
            ref_year, model_year = _target_years(t)
            rows.append(MeasureRow(
                id=entry["id"], label=t.get("label", ""), variant=t.get("variant", "unspecified"),
                ref_year=ref_year, model_year=model_year, unit=t.get("ref_unit", ""),
                confidence=entry.get("confidence", ""), ref=float(t["ref_value"])))
        report.measures.extend(rows)

        def fail(msg: str) -> None:
            report.problems.append(f"{entry['id']}: {msg}")
            for r in rows:
                r.status = msg

        if not entry.get("measure"):
            for r in rows:
                r.status = "no model equivalent"
            continue
        missing = missing_params(entry, known)
        if missing:
            fail("missing parameter: " + ", ".join(missing))
            continue
        if not rows:
            continue
        run_end = min(last, max(r.model_year for r in rows))
        runs: dict[str, Results] = {}
        refs: dict[str, Results] = {}
        try:
            off = Measure(name="static", changes=[Change(STATIC_SWITCH, 0, BASE_YEAR + 1)])
            variants = {"behavioural": []}
            if STATIC_SWITCH in known:
                variants["static"] = [off]
            for variant, extra in variants.items():
                name = f"{entry['id']} ({variant})"
                runs[variant] = _run([measure_of(entry)] + extra, base, run_end, name)
                if entry.get("reference_measure"):
                    refs[variant] = _run([measure_of(entry, "reference_measure")] + extra, base,
                                         run_end, name + " reference")
                else:
                    refs[variant] = base
        except Exception as exc:
            fail(f"run failed: {type(exc).__name__}: {exc}")
            continue

        for t, r in zip(targets, rows):
            try:
                spec = t.get("model") or {}
                for variant, res in runs.items():
                    setattr(r, variant, effect(refs[variant], res, r.model_year, spec))
                if t.get("price_year") is not None:
                    r.ref *= (_state(base, r.ref_year).macro["cpi"]
                              / _state(base, int(t["price_year"])).macro["cpi"])
                if t.get("normalise") == "gdp":
                    r.ref = _normalise_gdp(r.ref, r.ref_year, base, history)
                    gdp = base.at(r.model_year).macro["gdp_nominal"]
                    for variant in runs:
                        setattr(r, variant, getattr(r, variant) / gdp * 100.0)
                    r.unit = "% of GDP"
                elif t.get("normalise"):
                    raise ValidationError(f"unknown normalise '{t['normalise']}'")
            except (ValidationError, KeyError, ZeroDivisionError) as exc:
                r.static = r.behavioural = None
                r.status = str(exc)
                report.problems.append(f"{entry['id']} [{r.label}]: {exc}")
    return report


def _source_row(entry: dict) -> dict:
    src = entry.get("source") or {}
    if isinstance(src, str):
        src = {"title": src}
    return {"id": entry.get("id", ""), "institution": src.get("institution", ""),
            "title": src.get("title", ""), "date": str(src.get("date", "")),
            "url": entry.get("url", ""), "confidence": entry.get("confidence", "")}


# --------------------------------------------------------------------------- report

def _num(v: float | None) -> str:
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _pct_txt(v: float | None) -> str:
    return "-" if v is None else f"{v:+.0f}%"


def _cell(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def render(report: Report) -> str:
    lines = [
        "# Validation report - model vs official references",
        "",
        "Generated by `uv run python -m belsim.validation` from `data/reference/validation.yaml`.",
        "Gap = model - reference; gap % is relative to the reference. For measures the gap uses",
        "the static run when the reference is a static (ex-ante) costing and the behavioural run",
        "otherwise; both runs are always shown. `status` other than ok means the row could not be",
        "evaluated.",
        "",
        "## Baseline projection",
        "",
    ]
    lines += _table(
        ["id", "description", "year", "unit", "reference", "model", "gap", "gap %", "conf.", "status"],
        [[r.id, r.description, r.when, r.unit, _num(r.ref), _num(r.model), _num(r.gap),
          _pct_txt(r.gap_pct), r.confidence, r.status] for r in report.baseline])
    lines += ["", "## Measure costings", ""]
    lines += _table(
        ["id", "target", "reference is", "ref. year", "model year", "unit", "reference",
         "model static", "model behavioural", "gap", "gap %", "conf.", "status"],
        [[r.id, r.label, r.variant, r.ref_year, r.model_year, r.unit, _num(r.ref), _num(r.static),
          _num(r.behavioural), _num(r.gap), _pct_txt(r.gap_pct), r.confidence, r.status]
         for r in report.measures])

    ranked = sorted((r for r in report.measures if r.gap_pct is not None),
                    key=lambda r: -abs(r.gap_pct))[:15]
    if ranked:
        lines += ["", "## Largest relative gaps (measures)", ""]
        lines += _table(["id", "target", "unit", "reference", "model", "gap %"],
                        [[r.id, r.label, r.unit, _num(r.ref), _num(r.compared), _pct_txt(r.gap_pct)]
                         for r in ranked])
    ranked_b = sorted((r for r in report.baseline if r.gap_pct is not None),
                      key=lambda r: -abs(r.gap_pct))[:15]
    if ranked_b:
        lines += ["", "## Largest relative gaps (baseline)", ""]
        lines += _table(["id", "year", "unit", "reference", "model", "gap", "gap %"],
                        [[r.id, r.when, r.unit, _num(r.ref), _num(r.model), _num(r.gap),
                          _pct_txt(r.gap_pct)] for r in ranked_b])

    lines += ["", "## Rows that could not be evaluated", ""]
    lines += [f"- {p}" for p in report.problems] or ["None."]
    if report.not_found:
        lines += ["", "## References looked for and not found", ""]
        lines += [f"- **{n.get('id', '')}** - {n.get('description', '')}"
                  + (f" ({n['searched']})" if n.get("searched") else "") for n in report.not_found]
    lines += ["", "## Sources", ""]
    lines += _table(["id", "institution", "document", "date", "confidence", "url"],
                    [[s["id"], s["institution"], s["title"], s["date"], s["confidence"], s["url"]]
                     for s in report.sources])
    return "\n".join(lines) + "\n"


def write_report(report: Report, path: Path | str | None = None) -> Path:
    path = Path(path or REPORT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report))
    return path


def main() -> None:
    report = run_validation()
    path = write_report(report)
    print(render(report))
    print(f"written to {path}")


if __name__ == "__main__":
    main()
