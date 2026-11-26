"""Translations of model-provided text (French, Dutch). English is the source, in the code and
YAML files where it always was; `fr.yaml` / `nl.yaml` here hold the translations by stable id.

    translate(kind, id, lang, default)      -> str, `default` when no translation exists

Kinds and their ids (see `source.py` for where each English text comes from):

    line.label / line.description   "<side>.<line id>"         e.g. "revenue.pit"
    group                           English group name of a budget line
    dim.<dim>                       member code                e.g. dim.regions / "FLA"
    param                           parameter name             (its description)
    unit                            English unit string        e.g. "EUR/year"
    measure.name / measure.description   English measure name  (library measures)
    tag                             tag of a library measure
    household                       household id               (distribution view)
    household_group                 English household group
    preset                          payslip household preset id
    validation                      reference id               (its description)
    target                          English label of a costing target (validation)
    series                          series key, a prefix without its region / entity suffix, or
                                    a pattern ending in ".*"   e.g. "macro.gdp_real"
    text                            the English sentence itself (caveats, method notes)

A missing catalog, section or entry is never an error: the English default is returned.
`uv run python -m belsim.i18n.check` lists what is not translated.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

LANGS = ("en", "fr", "nl")
DEFAULT_LANG = "en"
CATALOG_DIR = Path(__file__).resolve().parent

_lock = threading.Lock()
_cache: dict[str, tuple[tuple[int, int], dict]] = {}


def catalog_path(lang: str) -> Path:
    return CATALOG_DIR / f"{lang}.yaml"


def catalog(lang: str) -> dict:
    """The catalog of one language ({} for English, a missing or a malformed file). Reloaded
    when the file changes, so a translator sees edits without restarting the server."""
    if lang not in LANGS or lang == DEFAULT_LANG:
        return {}
    path = catalog_path(lang)
    try:
        st = path.stat()
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(lang)
        if hit and hit[0] == sig:
            return hit[1]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    with _lock:
        _cache[lang] = (sig, data)
    return data


def _lookup(kind: str, id: Any, lang: str) -> str | None:
    cat = catalog(lang)
    if not cat:
        return None
    root, _, field = kind.partition(".")
    section = cat.get(root)
    if root == "dim":                       # dim.<dim>: nested by dimension
        section = (section or {}).get(field) if isinstance(section, dict) else None
        field = ""
    if not isinstance(section, dict):
        return None
    entry = section.get(id)
    if entry is None and not isinstance(id, str):
        entry = section.get(str(id))
    if isinstance(entry, dict):
        # entries with several fields: line {label, description}, measure {name, description}
        entry = entry.get(field or "label")
    if isinstance(entry, str) and entry.strip():
        return entry.strip() if root != "text" else entry
    return None


def translate(kind: str, id: Any, lang: str | None, default: str | None = None) -> str:
    """Translation of `id` for `kind` in `lang`; `default` (or the id) when there is none."""
    fallback = default if default is not None else ("" if id is None else str(id))
    if not lang or lang == DEFAULT_LANG or id is None or id == "":
        return fallback
    try:
        hit = _lookup(kind, id, lang)
    except Exception:
        hit = None
    return hit if hit is not None else fallback


def has_translation(kind: str, id: Any, lang: str) -> bool:
    try:
        return _lookup(kind, id, lang) is not None
    except Exception:
        return False


def pick_lang(lang: str | None = None, accept_language: str | None = None) -> str:
    """Language of a request: the `lang` query value, else the first supported language of the
    Accept-Language header, else English."""
    if lang:
        code = lang.strip().lower().replace("_", "-").split("-")[0]
        if code in LANGS:
            return code
    if accept_language:
        ranked = []
        for i, part in enumerate(accept_language.split(",")):
            piece, _, q = part.strip().partition(";")
            try:
                weight = float(q.strip()[2:]) if q.strip().startswith("q=") else 1.0
            except ValueError:
                weight = 0.0
            ranked.append((-weight, i, piece.strip().lower().split("-")[0]))
        for _, _, code in sorted(ranked):
            if code in LANGS:
                return code
    return DEFAULT_LANG
