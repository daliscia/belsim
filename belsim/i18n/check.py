"""`uv run python -m belsim.i18n.check [--lang fr nl] [--no-runs] [--dump]`

Lists what the French and Dutch catalogs do not translate yet: model-provided text
(`belsim/i18n/<lang>.yaml`) and dashboard strings (`web/i18n/<lang>.json`). Exit code 1 when
anything is missing, so it can gate a release; untranslated text falls back to English at run
time, it is never an error.

`--dump` prints the English sources as YAML in the layout of a catalog, ready to translate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from . import LANGS, catalog, has_translation
from .source import KINDS, collect

WEB_I18N = Path(__file__).resolve().parents[2] / "web" / "i18n"
TRANSLATED = [lang for lang in LANGS if lang != "en"]


def _translated(kind: str, id: str, lang: str) -> bool:
    if kind == "dim":
        dim, _, code = id.partition(".")
        return has_translation(f"dim.{dim}", code, lang)
    return has_translation(kind, id, lang)


def coverage(lang: str, sources: dict[str, dict[str, str]] | None = None) -> dict[str, dict]:
    """{kind: {"total": n, "missing": [ids]}} for one language."""
    sources = sources if sources is not None else collect()
    out = {}
    for kind in KINDS:
        ids = list(sources.get(kind, {}))
        out[kind] = {"total": len(ids), "missing": [i for i in ids if not _translated(kind, i, lang)]}
    return out


def flatten(obj, prefix: str = "") -> dict[str, object]:
    """Leaves of a nested UI catalog as {dotted.key: value}; plural objects count as one leaf."""
    out: dict[str, object] = {}
    for k, v in (obj or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and not (set(v) <= {"zero", "one", "two", "few", "many", "other"}):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def ui_catalog(lang: str) -> dict[str, object]:
    path = WEB_I18N / f"{lang}.json"
    if not path.is_file():
        return {}
    return flatten(json.loads(path.read_text(encoding="utf-8")))


def ui_coverage() -> dict[str, dict]:
    """{lang: {"total": n, "missing": [keys absent], "extra": [keys not in English]}}."""
    en = ui_catalog("en")
    out = {}
    for lang in TRANSLATED:
        cat = ui_catalog(lang)
        out[lang] = {"total": len(en), "missing": sorted(k for k in en if k not in cat or cat[k] in ("", None)),
                     "extra": sorted(k for k in cat if k not in en)}
    return out


def dump(sources: dict[str, dict[str, str]]) -> str:
    """English sources in catalog layout."""
    doc: dict = {}
    for side_id, text in sources["line.label"].items():
        doc.setdefault("line", {})[side_id] = {"label": text}
    for side_id, text in sources["line.description"].items():
        doc.setdefault("line", {}).setdefault(side_id, {})["description"] = text
    doc["group"] = dict(sources["group"])
    for id, text in sources["dim"].items():
        dim, _, code = id.partition(".")
        doc.setdefault("dim", {}).setdefault(dim, {})[code] = text
    for kind in ("param", "unit"):
        doc[kind] = dict(sources[kind])
    for name in sources["measure.name"]:
        doc.setdefault("measure", {})[name] = {"name": name, "description": sources["measure.description"].get(name, "")}
    for kind in ("tag", "household", "household_group", "preset", "validation", "target", "series", "text"):
        doc[kind] = dict(sources[kind])
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=1000)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="translation coverage of belsim")
    ap.add_argument("--lang", nargs="*", default=TRANSLATED, choices=TRANSLATED)
    ap.add_argument("--no-runs", action="store_true", help="skip what needs a simulation (series labels, caveats)")
    ap.add_argument("--dump", action="store_true", help="print the English sources in catalog layout and exit")
    ap.add_argument("--max", type=int, default=400, help="ids listed per kind (default 400)")
    a = ap.parse_args(argv)
    sources = collect(with_runs=not a.no_runs)
    if a.dump:
        print(dump(sources))
        return 0
    missing_any = False
    for lang in a.lang:
        print(f"== {lang}: model text ({'belsim/i18n/' + lang + '.yaml'}{'' if catalog(lang) else ', FILE MISSING OR EMPTY'})")
        for kind, c in coverage(lang, sources).items():
            done = c["total"] - len(c["missing"])
            pct = 100.0 * done / c["total"] if c["total"] else 100.0
            print(f"  {kind:<20} {done:>4}/{c['total']:<4} {pct:5.1f}%")
            for i in c["missing"][: a.max]:
                missing_any = True
                print(f"      missing: {i!r}")
            if len(c["missing"]) > a.max:
                print(f"      ... and {len(c['missing']) - a.max} more")
    ui = ui_coverage()
    for lang in a.lang:
        c = ui[lang]
        print(f"== {lang}: dashboard strings (web/i18n/{lang}.json): {c['total'] - len(c['missing'])}/{c['total']}")
        for k in c["missing"][: a.max]:
            missing_any = True
            print(f"      missing: {k}")
        for k in c["extra"][: a.max]:
            print(f"      not in en.json (unused?): {k}")
    return 1 if missing_any else 0


if __name__ == "__main__":
    sys.exit(main())
