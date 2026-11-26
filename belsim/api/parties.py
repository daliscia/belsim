"""Prebuilt party-programme scenarios (scenarios/parties/<party>.yaml).

Each file holds the measures one party submitted to the Federal Planning Bureau's costing of
the 2024 election programmes (DC2024), translated into belsim parameters by
`belsim.api.party_build`. This module serves them: list with coverage, runnable scenario,
per-measure model effects, comparison of all parties.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..core.dims import END_YEAR
from . import service as svc
from .party_build import STATUSES, coverage, reference_impulse

LANG_FIELDS = {"en": "name", "fr": "name_fr", "nl": "name_nl"}
COMMUNITIES = ["nl", "fr", "national"]          # display order of the groups
DETAIL_END = 2040                               # horizon of the per-measure effects

# Text shown with every party scenario. Wording is the same for all parties.
CAVEAT = {
    "en": "These are the measures each party submitted to the Federal Planning Bureau for the costing of the "
          "2024 election programmes (DC2024), as translated into the parameters of this model. They are not the "
          "parties' current positions, the selection is the one each party made in 2024, and the results shown "
          "here are those of this model, not of the Planning Bureau.",
    "fr": "Il s'agit des mesures que chaque parti a soumises au Bureau fédéral du Plan pour le chiffrage des "
          "programmes électoraux de 2024 (DC2024), telles que traduites dans les paramètres de ce modèle. Ce ne sont "
          "pas les positions actuelles des partis, la sélection est celle que chaque parti a faite en 2024, et les "
          "résultats présentés ici sont ceux de ce modèle, pas ceux du Bureau du Plan.",
    "nl": "Dit zijn de maatregelen die elke partij aan het Federaal Planbureau heeft voorgelegd voor de doorrekening "
          "van de verkiezingsprogramma's van 2024 (DC2024), zoals vertaald naar de parameters van dit model. Het zijn "
          "niet de huidige standpunten van de partijen, de selectie is die van elke partij in 2024, en de resultaten "
          "hier zijn die van dit model, niet van het Planbureau.",
}
METHOD = {
    "en": "Same method for every party. Year T of the programme (2025-2029) is year T+2 of the model (2027-2031). "
          "Where a fiche fixes a value, that value is used; otherwise the parameter is sized so that the model's "
          "static effect on the primary balance equals the Bureau's ex-ante impulse as a share of GDP. A measure "
          "carried by a broader budget line, or modelled in part, is 'partly modelled'. Measures enacted since 2024 "
          "are in the baseline and add nothing; a proposal that goes further adds the difference only. Measures "
          "without a suitable parameter are listed and change nothing. Coverage shares weigh each measure by its "
          "largest yearly ex-ante impulse in absolute value.",
    "fr": "Même méthode pour chaque parti. L'année T du programme (2025-2029) est l'année T+2 du modèle (2027-2031). "
          "Quand une fiche fixe une valeur, elle est reprise ; sinon le paramètre est calibré pour que l'effet statique "
          "du modèle sur le solde primaire égale l'impulsion ex ante du Bureau en part du PIB. Une mesure portée par "
          "une ligne budgétaire plus large, ou modélisée en partie, est « partiellement modélisée ». Les mesures "
          "adoptées depuis 2024 sont dans le scénario de référence et n'ajoutent rien ; une proposition qui va plus loin "
          "n'ajoute que la différence. Les mesures sans paramètre adéquat sont listées et ne changent rien. Les parts "
          "de couverture pondèrent chaque mesure par sa plus grande impulsion annuelle ex ante en valeur absolue.",
    "nl": "Dezelfde methode voor elke partij. Jaar T van het programma (2025-2029) is jaar T+2 van het model "
          "(2027-2031). Waar een fiche een waarde vastlegt, wordt die gebruikt; anders wordt de parameter zo geschaald "
          "dat het statische effect van het model op het primair saldo gelijk is aan de ex-ante-impuls van het Bureau "
          "als aandeel van het bbp. Een maatregel die door een bredere begrotingslijn wordt gedragen, of maar deels "
          "is gemodelleerd, is 'deels gemodelleerd'. Maatregelen die sinds 2024 zijn ingevoerd zitten in het basisscenario "
          "en voegen niets toe; een voorstel dat verder gaat voegt enkel het verschil toe. Maatregelen zonder "
          "geschikte parameter staan in de lijst en veranderen niets. De dekkingsaandelen wegen elke maatregel met "
          "zijn grootste jaarlijkse ex-ante-impuls in absolute waarde.",
}
SCENARIO_NAME = {"en": "{party} - 2024 programme (DC2024)", "fr": "{party} - programme 2024 (DC2024)",
                 "nl": "{party} - programma 2024 (DC2024)"}
SCENARIO_DESC = {
    "en": "Measures {party} submitted for the Planning Bureau's 2024 costing, as translated into this model "
          "({modelled} of {total} measures carry changes).",
    "fr": "Mesures soumises par {party} pour le chiffrage 2024 du Bureau du Plan, telles que traduites dans ce modèle "
          "({modelled} mesures sur {total} portent des changements).",
    "nl": "Maatregelen die {party} voorlegde voor de doorrekening 2024 van het Planbureau, zoals vertaald naar dit "
          "model ({modelled} van de {total} maatregelen bevatten wijzigingen).",
}


def parties_dir() -> Path:
    return svc.scenarios_dir() / "parties"


def party_files() -> list[Path]:
    d = parties_dir()
    return sorted(p for p in d.glob("*.yaml") if p.is_file()) if d.is_dir() else []


_cache = svc._LRU(2)


def _signature() -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in party_files())


def load_parties() -> list[dict]:
    """Every party file, as written (cached on the files' signature)."""
    sig = _signature()
    hit = _cache.get(sig)
    if hit is None:
        hit = []
        for p in party_files():
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("kind") == "party":
                doc["file"] = f"scenarios/parties/{p.name}"
                hit.append(doc)
        _cache.put(sig, hit)
    return hit


def get_party(pid: str) -> dict:
    for doc in load_parties():
        if doc["id"] == pid:
            return doc
    raise svc.ApiError(404, f"no party scenario '{pid}'")


def _name(m: dict, lang: str) -> str:
    return m.get(LANG_FIELDS.get(lang, "name")) or m["name"]


def scenario_of(doc: dict, lang: str = "en") -> dict:
    """The runnable scenario of a party: its measures that carry changes (names in `lang`)."""
    ms = [{"name": f"{m['source']['fiche']}: {_name(m, lang)}", "description": m.get("title") or "",
           "changes": [dict(c) for c in m["changes"]]}
          for m in doc["measures"] if m.get("changes")]
    return {"name": SCENARIO_NAME.get(lang, SCENARIO_NAME["en"]).format(party=doc["name"]),
            "description": SCENARIO_DESC.get(lang, SCENARIO_DESC["en"]).format(
                party=doc["name"], modelled=len(ms), total=len(doc["measures"])),
            "measures": ms}


def _measure_row(m: dict, lang: str) -> dict:
    row = {k: m.get(k) for k in ("fiche", "title", "category", "source", "official_impulse", "status", "method",
                                 "lever", "baseline", "check", "validation_id", "sizing_residual")}
    row["name"] = _name(m, lang)
    row["name_en"] = m["name"]
    row["increment"] = bool(m.get("increment"))
    row["reversal"] = bool(m.get("reversal"))
    row["reference_impulse"] = reference_impulse(m)
    row["changes"] = m.get("changes") or []
    return row


def unknown_params(doc: dict) -> list[str]:
    specs = svc.load_params().specs
    return sorted({c["param"] for m in doc["measures"] for c in m.get("changes") or [] if c["param"] not in specs})


def party_summary(doc: dict, lang: str = "en", with_measures: bool = True) -> dict:
    out = {
        "id": doc["id"], "kind": "party", "name": doc["name"], "full_name": doc.get("full_name") or doc["name"],
        "name_en": doc.get("name_en") or doc["name"], "community": doc.get("community") or "national",
        "file": doc.get("file"), "source": doc.get("source") or {}, "official": doc.get("official") or {},
        "build": doc.get("build") or {}, "coverage": coverage(doc["measures"]),
        "scenario": scenario_of(doc, lang),
    }
    try:
        out["unknown_params"] = unknown_params(doc)
    except svc.ApiError:
        out["unknown_params"] = []
    if with_measures:
        out["measures"] = [_measure_row(m, lang) for m in doc["measures"]]
    return out


def list_parties(lang: str = "en", with_measures: bool = True) -> dict:
    """Party scenarios grouped by language community, alphabetical inside a group."""
    docs = sorted(load_parties(), key=lambda d: (COMMUNITIES.index(d.get("community")) if d.get("community") in COMMUNITIES
                                                 else len(COMMUNITIES), d["name"].casefold()))
    return svc.clean({
        "kind": "party", "lang": lang, "caveat": CAVEAT.get(lang, CAVEAT["en"]), "method": METHOD.get(lang, METHOD["en"]),
        "communities": COMMUNITIES, "statuses": STATUSES,
        "source": {"institution": "Federal Planning Bureau", "title": "DC2024", "date": "2024-05",
                   "url": "https://www.dc2024.be/"},
        "scenarios": [party_summary(d, lang, with_measures) for d in docs],
    })


# ---------------------------------------------------------------- model effects

_fx_cache = svc._LRU(16)
FX_KEYS = ["fiscal.balance", "fiscal.primary_balance", "fiscal.balance_pct_gdp", "labour.employment", "macro.gdp_real"]


def _static(scen: dict) -> dict:
    off = svc._band_measure("static")
    return {**scen, "measures": scen["measures"] + ([off] if off else [])}


def party_effects(pid: str, end: int | None = None, lang: str = "en") -> dict:
    """Each measure of a party simulated on its own: static and behavioural change against the
    baseline (balance, primary balance, employment), and the whole package."""
    doc = get_party(pid)
    end = min(svc.clamp_end(end), DETAIL_END) if end else DETAIL_END
    key = (svc.params_signature(), _signature(), pid, end)
    hit = _fx_cache.get(key)
    if hit is None:
        base = svc.baseline_series(end)
        years = svc.baseline(end).years
        has_static = svc.BAND_SWITCH_PARAM in svc.load_params().specs

        def deltas(scen: dict) -> dict:
            out: dict[str, Any] = {}
            for variant, s in (("behavioural", scen), ("static", _static(scen) if has_static else None)):
                if s is None:
                    continue
                try:
                    _, series = svc.headline(s, end)
                    out[variant] = {k: v for k, v in svc.delta_of(base, series).items() if k in FX_KEYS}
                except svc.ApiError as e:
                    if e.status >= 500:
                        raise
                    out["error"] = e.message
            return out

        rows = []
        for m in doc["measures"]:
            row: dict[str, Any] = {"fiche": m["fiche"]}
            if m.get("changes"):
                row.update(deltas({"name": f"{pid} {m['fiche']}", "measures": [{"name": m["name"], "changes": m["changes"]}]}))
            rows.append(row)
        hit = svc.clean({"id": pid, "years": years, "keys": FX_KEYS, "gdp": base.get("macro.gdp_nominal"),
                         "measures": rows, "package": deltas(scenario_of(doc)), "signature": svc.params_signature()})
        _fx_cache.put(key, hit)
    return {**hit, "lang": lang}


def parties_compare(end: int = END_YEAR, lang: str = "en") -> dict:
    """Headline series of every party's package (as /api/run_batch), with the Bureau's
    published results and the coverage next to them."""
    docs = list_parties(lang, with_measures=False)["scenarios"]
    batch = svc.run_batch([d["scenario"] for d in docs], end)
    for d, run in zip(docs, batch["scenarios"]):
        run["party"] = d["id"]
        run["name"] = d["name"]
        run["community"] = d["community"]
        run["official"] = d["official"]
        run["coverage"] = d["coverage"]
    return {**batch, "kind": "party", "lang": lang, "caveat": CAVEAT.get(lang, CAVEAT["en"])}


_dist_cache = svc._LRU(4)


def parties_distribution(year: int, lang: str = "en") -> dict:
    """Change of household disposable income by income decile for every party's package
    (the model's distribution analysis, one call per party: slow the first time)."""
    from . import distribution as D

    key = (svc.params_signature(), _signature(), year)
    hit = _dist_cache.get(key)
    if hit is None:
        rows = []
        for doc in load_parties():
            row: dict[str, Any] = {"party": doc["id"], "name": doc["name"], "community": doc.get("community"),
                                   "official_deciles_pct": (doc.get("official") or {}).get("disposable_income_by_decile_pct")}
            try:
                res = D.distribution(D.DistributionIn(scenario=scenario_of(doc), year=year))
                row["deciles_pct"] = [d.get("delta_pct") for d in res.get("deciles") or []]
                row["deciles_eur"] = [d.get("delta_eur") for d in res.get("deciles") or []]
                row["summary"] = res.get("summary")
            except svc.ApiError as e:
                row["error"] = e.message
            except Exception as e:  # the distribution module is owned elsewhere and may be in flux
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
        hit = svc.clean({"year": year, "parties": rows})
        _dist_cache.put(key, hit)
    return {**hit, "lang": lang}
