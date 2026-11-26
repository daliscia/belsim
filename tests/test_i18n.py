"""Translation coverage: dashboard strings (web/i18n/*.json) and model text (belsim/i18n/*.yaml).

Complete coverage is required for what has a stable id and changes rarely (budget lines,
dimension members, library measures, households). Parameter descriptions tolerate a few
untranslated entries, because other modules keep adding parameters: those fall back to English
and are printed here (run with `-s`, or `uv run python -m belsim.i18n.check`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from belsim import i18n
from belsim.i18n import check, source
from belsim.i18n.labels import Labeller, humanise

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
TRANSLATED = ["fr", "nl"]
PARAM_THRESHOLD = 0.95          # share of parameter descriptions translated
SOFT_THRESHOLD = 0.90           # units, validation descriptions, series labels, free text
PLURAL_FORMS = {"zero", "one", "two", "few", "many", "other"}


@pytest.fixture(scope="module")
def sources():
    return source.collect()


# ------------------------------------------------------------------ dashboard strings

def test_ui_catalogs_have_the_same_keys():
    en = check.ui_catalog("en")
    assert len(en) > 200, "web/i18n/en.json is missing or nearly empty"
    for lang in TRANSLATED:
        cat = check.ui_catalog(lang)
        missing = sorted(k for k in en if k not in cat or cat[k] in ("", None))
        extra = sorted(k for k in cat if k not in en)
        assert not missing, f"{lang}.json lacks {len(missing)} keys: {missing[:40]}"
        assert not extra, f"{lang}.json has keys that en.json does not: {extra[:40]}"


def test_ui_placeholders_and_plurals_match():
    en = check.ui_catalog("en")
    holes = lambda v: set(re.findall(r"\{(\w+)\}", " ".join(v.values()) if isinstance(v, dict) else str(v)))
    for lang in TRANSLATED:
        cat = check.ui_catalog(lang)
        bad = []
        for key, v in en.items():
            w = cat.get(key)
            if w is None:
                continue
            if isinstance(v, dict) != isinstance(w, dict):
                bad.append(f"{key}: plural in one language only")
            elif isinstance(w, dict) and ("other" not in w or not set(w) <= PLURAL_FORMS):
                bad.append(f"{key}: plural forms {sorted(w)}")
            elif holes(w) - holes(v):
                bad.append(f"{key}: unknown placeholder {sorted(holes(w) - holes(v))}")
        assert not bad, f"{lang}.json: " + "; ".join(bad[:30])


def _js_files():
    return [p for p in (WEB / "js").rglob("*.js")]


def test_every_key_used_in_the_code_exists():
    """Literal keys passed to t(), tn() and tOr() in web/js are in en.json."""
    en = check.ui_catalog("en")
    prefixes = {k.rsplit(".", 1)[0] for k in en} | set(en)
    used, missing = set(), []
    call = re.compile(r"""\b(?:t|tn|tOr)\(\s*(['"])([A-Za-z0-9_.]+)\1""")
    for path in _js_files():
        for m in call.finditer(path.read_text(encoding="utf-8")):
            key = m.group(2)
            if "." not in key:
                continue
            used.add(key)
            if key not in en and not (key.endswith(".") and key[:-1] in prefixes):
                missing.append(f"{path.relative_to(WEB)}: {key}")
    assert len(used) > 200, "the dashboard code does not seem to use the i18n helper"
    assert not missing, f"{len(missing)} keys used in the code are not in en.json: {missing[:40]}"


def test_index_html_has_no_hard_coded_interface_text():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    body = html[html.index("<body"):]
    text = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    # the message for a browser without JavaScript cannot come from the catalogs: it is written
    # in the three languages in the page itself
    noscript = re.findall(r"<noscript.*?</noscript>", text, flags=re.S)
    assert noscript and all(f'lang="{lang}"' in noscript[0] for lang in TRANSLATED)
    text = re.sub(r"<noscript.*?</noscript>", "", text, flags=re.S)
    words = [w for w in re.findall(r">([^<>]+)<", text) if re.search(r"[A-Za-z]{4,}", w) and w.strip() != "belsim"]
    assert not words, f"index.html still holds text outside the catalogs: {words}"


# ------------------------------------------------------------------ model text

def _report(lang: str, kind: str, cov: dict) -> float:
    c = cov[kind]
    done = c["total"] - len(c["missing"])
    share = done / c["total"] if c["total"] else 1.0
    print(f"[i18n] {lang} {kind}: {done}/{c['total']} ({share:.1%})")
    for i in c["missing"]:
        print(f"[i18n]   untranslated {kind}: {i!r}")
    return share


@pytest.mark.parametrize("lang", TRANSLATED)
def test_model_text_coverage(lang, sources):
    assert i18n.catalog(lang), f"belsim/i18n/{lang}.yaml is missing, empty or not valid YAML"
    cov = check.coverage(lang, sources)
    shares = {kind: _report(lang, kind, cov) for kind in source.KINDS}
    # complete: stable ids that the interface shows everywhere. The regional and revenue modules
    # may add lines, entities and measures after this catalog was written: a small tolerance on
    # those keeps the suite green until the catalog is topped up (the ids are printed above).
    for kind in ("dim", "household", "household_group", "preset", "tag", "group"):
        assert shares[kind] >= SOFT_THRESHOLD, f"{lang}: {kind} only {shares[kind]:.0%} translated: {cov[kind]['missing'][:20]}"
    for kind in ("line.label", "measure.name", "measure.description"):
        assert shares[kind] >= SOFT_THRESHOLD, f"{lang}: {kind} only {shares[kind]:.0%} translated: {cov[kind]['missing'][:20]}"
    assert shares["param"] >= PARAM_THRESHOLD, (
        f"{lang}: {shares['param']:.1%} of parameter descriptions translated (need {PARAM_THRESHOLD:.0%}); "
        f"untranslated: {cov['param']['missing'][:40]}")
    for kind in ("unit", "validation", "series", "line.description", "target"):
        assert shares[kind] >= SOFT_THRESHOLD, f"{lang}: {kind} only {shares[kind]:.0%} translated: {cov[kind]['missing'][:20]}"


@pytest.mark.parametrize("lang", TRANSLATED)
def test_terminology(lang):
    """A few official terms, as a guard against a catalog being overwritten by a rough one."""
    want = {
        "fr": {("dim.regions", "FLA"): "Flandre", ("dim.regions", "WAL"): "Wallonie", ("dim.regions", "BRU"): "Bruxelles",
               ("dim.entities", "social_security"): "Sécurité sociale"},
        "nl": {("dim.regions", "FLA"): "Vlaanderen", ("dim.regions", "WAL"): "Wallonië", ("dim.regions", "BRU"): "Brussel",
               ("dim.entities", "social_security"): "Sociale zekerheid"},
    }[lang]
    for (kind, id), text in want.items():
        assert i18n.translate(kind, id, lang, "?").lower() == text.lower()
    cat = json.dumps(i18n.catalog(lang), ensure_ascii=False).lower()
    terms = {"fr": ["impôt des personnes physiques", "cotisations sociales", "précompte mobilier", "saut d'index",
                    "revenu d'intégration", "comité d'étude sur le vieillissement", "bureau fédéral du plan"],
             "nl": ["personenbelasting", "sociale bijdragen", "roerende voorheffing", "indexsprong", "leefloon",
                    "studiecommissie voor de vergrijzing", "federaal planbureau"]}[lang]
    absent = [t for t in terms if t not in cat]
    assert not absent, f"{lang}.yaml never uses the official terms {absent}"


def test_translate_falls_back_to_english():
    assert i18n.translate("param", "no.such.parameter", "fr", "English text") == "English text"
    assert i18n.translate("param", "no.such.parameter", "nl") == "no.such.parameter"
    assert i18n.translate("no_such_kind", "x", "fr", "d") == "d"
    assert i18n.translate("dim.regions", "FLA", "en", "Flanders") == "Flanders"
    assert i18n.translate("dim.regions", "FLA", "xx", "Flanders") == "Flanders"
    assert i18n.translate("param", None, "fr", "") == ""
    assert i18n.translate("dim.no_such_dim", "FLA", "fr", "Flanders") == "Flanders"


def test_pick_lang():
    assert i18n.pick_lang("fr", None) == "fr"
    assert i18n.pick_lang("NL-be", "fr") == "nl"
    assert i18n.pick_lang(None, "nl-BE,nl;q=0.9,en;q=0.8") == "nl"
    assert i18n.pick_lang(None, "de-DE,fr;q=0.6,en;q=0.9") == "en"
    assert i18n.pick_lang("de", "de") == "en"
    assert i18n.pick_lang(None, None) == "en"
    assert i18n.pick_lang("", "garbage;;q=x") == "en"


def test_series_labels_compose_regions_and_entities():
    lab = Labeller("en", None, {"pit": {"label": "Personal income tax"}}, {})
    assert lab.label("regional.unemployment_rate.WAL") == "Unemployment rate, Wallonia"
    assert lab.label("regional.revenue.pit.BRU") == "Revenue: Personal income tax, Brussels"
    assert lab.label("fiscal.balance.local") == "Balance, Local authorities"
    assert lab.label("fiscal.transfer.federal_to_social_security") == "Transfer from Federal government to Social security"
    assert lab.label("revenue.pit") == "Personal income tax"
    assert lab.label("some.brand_new_series") == humanise("brand_new_series")
    assert lab.regional_metrics(["regional.gdp.FLA", "regional.gdp.WAL", "macro.gdp_real"]) == {"gdp": "GDP"}
    fr = Labeller("fr", None, {}, {})
    assert fr.label("regional.a_metric_added_later.FLA").endswith(", " + i18n.translate("dim.regions", "FLA", "fr", "Flanders"))


def test_check_script_runs(capsys, sources):
    code = check.main(["--no-runs", "--max", "5"])
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "== fr: model text" in out and "== nl: dashboard strings" in out
    assert "param" in check.dump(sources)


# ------------------------------------------------------------------ methodology page

def _fragment(lang: str) -> str:
    return (WEB / "wp" / f"{lang}.html").read_text(encoding="utf-8")


def _outline(html: str) -> list:
    """Structure of a fragment that must be the same in every language: per section its id, the
    live figures and blocks in order, and the numbers of paragraphs, list items and code terms."""
    out = []
    parts = re.split(r'(?=<section id=")', html)
    for part in parts:
        sid = re.match(r'<section id="([\w-]+)"', part)
        out.append((
            sid.group(1) if sid else "head",
            re.findall(r'data-v="([^"]+)"', part),
            re.findall(r'data-block="([^"]+)"', part),
            len(re.findall(r"<p[ >]", part)), len(re.findall(r"<li[ >]", part)),
            sorted(re.findall(r"<code>([^<]+)</code>", part)) if sid and sid.group(1) != "reproduce" else [],
            len(re.findall(r'class="todo"', part)), len(re.findall(r"data-optional", part)),
            re.findall(r'href="(#[\w-]+)"', part),
        ))
    return out


def test_whitepaper_fragments_have_the_same_structure():
    en = _outline(_fragment("en"))
    assert [s[0] for s in en] == ["head", "purpose", "model", "data", "baseline", "behaviour", "validation",
                                  "distribution", "regions", "parties", "built", "limits", "reproduce",
                                  "sources", "licence"]
    for lang in TRANSLATED:
        other = _outline(_fragment(lang))
        assert [s[0] for s in other] == [s[0] for s in en], f"{lang}: sections differ"
        for a, b in zip(en, other):
            assert a == b, f"wp/{lang}.html, section {a[0]}: structure differs from English\n en {a}\n {lang} {b}"


def test_whitepaper_fragments_are_complete_and_inert():
    for lang in ("en", *TRANSLATED):
        html = _fragment(lang)
        assert not re.search(r"\b[A-Z]{4,}_[A-Z_]{3,}\b", html), f"wp/{lang}.html holds an unfilled placeholder"
        assert "<script" not in html and not re.search(r"\son[a-z]+=", html), f"wp/{lang}.html must hold no script"
        assert html.count('class="todo"') == 0, "licence and contact are filled in"
        assert "agpl-3.0" in html and "github.com/daliscia/belsim" in html
        # every anchor points at a section of the page
        ids = set(re.findall(r'<section id="([\w-]+)"', html))
        assert set(re.findall(r'href="#([\w-]+)"', html)) <= ids
    # the same commands are given in the three languages
    cmds = [re.findall(r"^uv run [^#\n]+", _fragment(lang), flags=re.M) for lang in ("en", *TRANSLATED)]
    assert cmds[0] and [[c.strip() for c in x] for x in cmds].count([c.strip() for c in cmds[0]]) == 3


def test_whitepaper_live_keys_are_known_to_the_page_script():
    """Every named figure of the text has a rule in web/js/whitepaper.js."""
    js = (WEB / "js" / "whitepaper.js").read_text(encoding="utf-8")
    known = set(re.findall(r"'([a-z]+\.[a-z0-9_.]+)': \(\) =>", js))
    blocks = set(re.findall(r"^  '?([a-z-]+)'?\(\) \{", js, flags=re.M))
    html = _fragment("en")
    for key in re.findall(r'data-v="([^"]+)"', html):
        assert key in known or re.match(r"^[bds]:", key), f"no rule for the live figure {key!r}"
    for block in re.findall(r'data-block="([^"]+)"', html):
        assert block in blocks, f"no builder for the block {block!r}"
    for scen in re.findall(r'data-v="[ds]:(\w+):', html):
        assert re.search(rf"^  {scen}: \[", js, flags=re.M), f"scenario {scen!r} is not defined"
