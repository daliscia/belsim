"""Versioned parameters, measures and scenarios.

A parameter file (params/*.yaml) is a mapping  name -> spec :

    pit.basic_allowance:
      description: Tax-free allowance (EUR/year)
      unit: EUR
      source: FPS Finance, income year 2024
      uprate: health_index        # optional: none (default) | cpi | health_index | wages
      values:
        2024: 10160
        2030: 15000               # legislated future values are extra dated entries

Value at year t = the entry with the latest year <= t (step function). With `uprate`, that
entry is multiplied by index(t)/index(max(entry_year, BASE_YEAR)) of the named index, i.e.
automatic indexation from the year the value was set. Values may be scalars, lists or dicts;
uprating applies to every number inside (so never uprate a parameter that mixes amounts and
rates - split it in two parameters).

A Measure is a list of dated Changes to parameters; a Scenario is a list of Measures.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from .dims import BASE_YEAR

PARAMS_DIR = Path(__file__).resolve().parents[2] / "params"
UPRATE_KINDS = {"none", "cpi", "health_index", "wages"}


@dataclass
class ParamSpec:
    name: str
    values: dict[int, Any]
    description: str = ""
    unit: str = ""
    source: str = ""
    uprate: str = "none"
    file: str = ""
    wage_link: bool = False   # also follows core.tax_threshold_wage_link (see engine)

    def raw_at(self, year: int) -> tuple[int, Any]:
        years = [y for y in self.values if y <= year]
        if not years:
            y0 = min(self.values)
            return y0, self.values[y0]
        y = max(years)
        return y, self.values[y]


@dataclass
class Change:
    """One dated modification of a parameter.

    op: 'set' (replace), 'add' (value + x), 'mul' (value * x). For list/dict parameters use
    `key` to address one element (list index or dict key; dotted for nesting, e.g. 'rates.2'),
    or op='set' with a full replacement value.
    phase_in_years: linear phase-in from the pre-change value over that many years
    (0 = immediate). end: last year the change applies (None = permanent).
    """
    param: str
    value: Any
    start: int
    op: str = "set"
    end: int | None = None
    phase_in_years: int = 0
    key: str | None = None


@dataclass
class Measure:
    name: str
    changes: list[Change]
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Measure":
        return Measure(
            name=d["name"],
            description=d.get("description", ""),
            tags=list(d.get("tags", [])),
            changes=[Change(**c) for c in d.get("changes", [])],
        )


@dataclass
class Scenario:
    name: str = "baseline"
    measures: list[Measure] = field(default_factory=list)
    description: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(
            name=d.get("name", "scenario"),
            description=d.get("description", ""),
            measures=[Measure.from_dict(m) for m in d.get("measures", [])],
        )


def _map_numbers(v: Any, f: Callable[[float], float]) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return f(v)
    if isinstance(v, list):
        return [_map_numbers(x, f) for x in v]
    if isinstance(v, dict):
        return {k: _map_numbers(x, f) for k, x in v.items()}
    return v


def _combine(a: Any, b: Any, f: Callable[[float, float], float]) -> Any:
    """Element-wise combine of two same-shaped values (b may be a scalar broadcast)."""
    if isinstance(a, bool):
        return b
    if isinstance(a, (int, float)):
        return f(a, b) if isinstance(b, (int, float)) and not isinstance(b, bool) else b
    if isinstance(a, list):
        if isinstance(b, list):
            return [_combine(x, y, f) for x, y in zip(a, b)]
        return [_combine(x, b, f) for x in a]
    if isinstance(a, dict):
        if isinstance(b, dict):
            return {k: (_combine(x, b[k], f) if k in b else x) for k, x in a.items()}
        return {k: _combine(x, b, f) for k, x in a.items()}
    return b


def _get_key(v: Any, key: str) -> Any:
    for part in key.split("."):
        v = v[int(part)] if isinstance(v, list) else v[_dict_key(v, part)]
    return v


def _dict_key(d: dict, part: str) -> Any:
    if part in d:
        return part
    for k in d:
        if str(k) == part:
            return k
    raise KeyError(part)


def _set_key(v: Any, key: str, new: Any) -> Any:
    v = copy.deepcopy(v)
    parts = key.split(".")
    cur = v
    for part in parts[:-1]:
        cur = cur[int(part)] if isinstance(cur, list) else cur[_dict_key(cur, part)]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = new
    else:
        cur[_dict_key(cur, last)] = new
    return v


_LOAD_CACHE: dict[str, tuple[tuple, dict]] = {}


class ParamView:
    """Parameters resolved for one year. `p["name"]` or `p.get("name", default)`."""

    def __init__(self, year: int, values: dict[str, Any]):
        self.year = year
        self._values = values

    def __getitem__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise KeyError(f"unknown parameter '{name}' (year {self.year})") from None

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def keys(self):
        return self._values.keys()


class ParamSet:
    """All parameter specs plus the scenario changes applied on top of them."""

    def __init__(self, specs: dict[str, ParamSpec], scenario: Scenario | None = None):
        self.specs = specs
        self.scenario = scenario or Scenario()
        self._changes: dict[str, list[Change]] = {}
        for m in self.scenario.measures:
            for c in m.changes:
                if c.param not in specs:
                    raise KeyError(f"measure '{m.name}' changes unknown parameter '{c.param}'")
                if c.op not in ("set", "add", "mul"):
                    raise ValueError(f"measure '{m.name}': bad op '{c.op}'")
                self._changes.setdefault(c.param, []).append(c)

    @classmethod
    def load(cls, directory: Path | str | None = None, scenario: Scenario | None = None) -> "ParamSet":
        directory = Path(directory) if directory else PARAMS_DIR
        signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                          for p in sorted(directory.glob("*.yaml")))
        cached = _LOAD_CACHE.get(str(directory))
        if cached is not None and cached[0] == signature:
            return cls(cached[1], scenario)
        specs = cls._read(directory)
        _LOAD_CACHE[str(directory)] = (signature, specs)
        return cls(specs, scenario)

    @staticmethod
    def _read(directory: Path) -> dict[str, ParamSpec]:
        specs: dict[str, ParamSpec] = {}
        for path in sorted(directory.glob("*.yaml")):
            data = yaml.safe_load(path.read_text()) or {}
            for name, spec in data.items():
                if name in specs:
                    raise ValueError(f"parameter '{name}' defined twice ({specs[name].file}, {path.name})")
                if not isinstance(spec, dict) or "values" not in spec:
                    raise ValueError(f"{path.name}: parameter '{name}' needs a 'values' mapping")
                uprate = spec.get("uprate", "none") or "none"
                if uprate not in UPRATE_KINDS:
                    raise ValueError(f"{path.name}: parameter '{name}' has unknown uprate '{uprate}'")
                specs[name] = ParamSpec(
                    name=name,
                    values={int(y): v for y, v in spec["values"].items()},
                    description=spec.get("description", ""),
                    unit=str(spec.get("unit", "")),
                    source=spec.get("source", ""),
                    uprate=uprate,
                    file=path.name,
                    wage_link=bool(spec.get("wage_link", False)),
                )
        return specs

    def with_scenario(self, scenario: Scenario | None) -> "ParamSet":
        return ParamSet(self.specs, scenario)

    def changed_params(self) -> set[str]:
        return set(self._changes)

    def _legislated(self, spec: ParamSpec, year: int, index: Callable[[str, int], float] | None) -> Any:
        y0, v = spec.raw_at(year)
        if spec.uprate != "none" and index is not None and year > y0:
            ref = max(y0, BASE_YEAR)
            if year > ref:
                ratio = _ratio(spec, index, year, ref)
                v = _map_numbers(v, lambda x: x * ratio)
        return copy.deepcopy(v) if isinstance(v, (list, dict)) else v

    def value(self, name: str, year: int, index: Callable[[str, int], float] | None = None) -> Any:
        spec = self.specs[name]
        if name not in self._changes and spec.uprate == "none":
            # constant legislation: hand out the stored value (callers must not mutate it)
            return spec.raw_at(year)[1]
        v = self._legislated(spec, year, index)
        for c in self._changes.get(name, []):
            if year < c.start or (c.end is not None and year > c.end):
                continue
            # A 'set' value on an indexed parameter is an amount in start-year euros:
            # keep indexing it afterwards, like legislated values.
            ratio = 1.0
            ref = max(c.start, BASE_YEAR)
            if c.op == "set" and spec.uprate != "none" and index is not None and year > ref:
                ratio = _ratio(spec, index, year, ref)
            if c.key is not None:
                old = _get_key(v, c.key)
                new = _apply_op(old, c)
                if ratio != 1.0:
                    new = _map_numbers(new, lambda x: x * ratio)
                new = _phase(old, new, c, year)
                v = _set_key(v, c.key, new)
            else:
                new = _apply_op(v, c)
                if ratio != 1.0:
                    new = _map_numbers(new, lambda x: x * ratio)
                v = _phase(v, new, c, year)
        return v

    def at(self, year: int, index: Callable[[str, int], float] | None = None) -> ParamView:
        """Resolve every parameter for `year`.

        `index(kind, year)` returns the level of 'cpi' | 'health_index' | 'wages' in that year
        (1.0 in BASE_YEAR); the engine supplies it from the simulated state. With index=None
        nothing is uprated (the "plain" view used by macro.prices to compute the indices).
        """
        return ParamView(year, {name: self.value(name, year, index) for name in self.specs})

    def describe(self) -> list[dict]:
        return [
            {
                "name": s.name, "description": s.description, "unit": s.unit, "source": s.source,
                "uprate": s.uprate, "wage_link": s.wage_link, "file": s.file, "values": s.values,
            }
            for s in self.specs.values()
        ]


def _ratio(spec: ParamSpec, index: Callable[[str, int], float], year: int, ref: int) -> float:
    r = index(spec.uprate, year) / index(spec.uprate, ref)
    if spec.wage_link:
        r *= index("wage_link", year) / index("wage_link", ref)
    return r


def _apply_op(old: Any, c: Change) -> Any:
    if c.op == "set":
        return copy.deepcopy(c.value)
    if c.op == "add":
        return _combine(old, c.value, lambda a, b: a + b)
    return _combine(old, c.value, lambda a, b: a * b)


def _phase(old: Any, new: Any, c: Change, year: int) -> Any:
    if c.phase_in_years <= 0:
        return new
    w = min(1.0, (year - c.start + 1) / (c.phase_in_years + 1))
    if w >= 1.0:
        return new
    try:
        return _combine(old, new, lambda a, b: a + (b - a) * w)
    except Exception:
        return new
