"""Builds the party-programme scenarios from the Federal Planning Bureau's 2024 costing.

    uv run python -m belsim.api.party_build              # (re)write scenarios/parties/<party>.yaml
    uv run python -m belsim.api.party_build --validation # also refresh the generated block of
                                                         # data/reference/validation.yaml and the
                                                         # matching entries of belsim/i18n/*.yaml

Inputs (scenarios/parties/source/):
  dc2024.json    what was read in the DC2024 workbooks of dc2024.be: per party the list of
                 measures (fiche number, title, ex-ante budgetary impulse 2025-2029 in EUR mn of
                 2024) and the Bureau's results for the whole package
  mapping.yaml   how each fiche is translated into belsim parameters (hand-written)

Method, identical for every party:
  * programme year T (2025-2029) is model year T+2 (2027-2031): the first full year after
    today, with the phase-in of the programme;
  * a measure whose fiche pins a parameter value uses that value ("pinned"); otherwise the
    lever is sized so that the model's static effect on the primary balance equals the Bureau's
    ex-ante impulse, both as a share of GDP (the impulse is in prices and quantities of 2024,
    so it is divided by the GDP of 2024), in the year of the largest impulse; the other years
    follow the profile of the impulse ("sized");
  * a lever that is a whole budget line while the measure concerns a part of it, or another
    design, gives status partly_modelled; no parameter is invented;
  * measures enacted since 2024 that are in the model baseline get no changes; a proposal that
    goes further is added as the increment, one that the enacted law contradicts as a reversal.
Re-run after a change of parameters: sized values and the recorded checks move with the model.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..core import engine
from ..core.params import Change, Measure, ParamSet, Scenario

ROOT = Path(__file__).resolve().parents[2]
PARTY_DIR = ROOT / "scenarios" / "parties"
SOURCE_DIR = PARTY_DIR / "source"
SHIFT = 2                       # programme year + SHIFT = model year
PROGRAMME_YEARS = [2025, 2026, 2027, 2028, 2029]
REF_GDP_YEAR = 2024             # impulses are in prices and quantities of 2024
RUN_END = PROGRAMME_YEARS[-1] + SHIFT + 2
SITE = "https://www.dc2024.be"

# source key -> party metadata. `community`: language community of the party's lists.
PARTIES: dict[str, dict] = {
    "cdv": {"id": "cdv", "name": "cd&v", "full_name": "Christen-Democratisch en Vlaams",
            "name_en": "Christian Democratic and Flemish", "community": "nl", "file": "CDV"},
    "defi": {"id": "defi", "name": "DéFI", "full_name": "Démocrate Fédéraliste Indépendant",
             "name_en": "Democratic Federalist Independent", "community": "fr", "file": "DEFI"},
    "ecolo": {"id": "ecolo", "name": "Ecolo", "full_name": "Ecolo",
              "name_en": "Ecolo (French-speaking greens)", "community": "fr", "file": "ECOLO"},
    "groen": {"id": "groen", "name": "Groen", "full_name": "Groen",
              "name_en": "Groen (Flemish greens)", "community": "nl", "file": "GROEN"},
    "le": {"id": "les-engages", "name": "Les Engagés", "full_name": "Les Engagés",
           "name_en": "Les Engagés (French-speaking centrists)", "community": "fr", "file": "LE"},
    "mr": {"id": "mr", "name": "MR", "full_name": "Mouvement Réformateur",
           "name_en": "Reformist Movement", "community": "fr", "file": "MR"},
    "nva": {"id": "n-va", "name": "N-VA", "full_name": "Nieuw-Vlaamse Alliantie",
            "name_en": "New Flemish Alliance", "community": "nl", "file": "NVA"},
    "openvld": {"id": "open-vld", "name": "Open Vld", "full_name": "Open Vlaamse Liberalen en Democraten",
                "name_en": "Open Flemish Liberals and Democrats", "community": "nl", "file": "OPENVLD"},
    "ps": {"id": "ps", "name": "PS", "full_name": "Parti Socialiste",
           "name_en": "Socialist Party (French-speaking)", "community": "fr", "file": "PS"},
    "pvda-ptb": {"id": "pvda-ptb", "name": "PVDA-PTB",
                 "full_name": "Partij van de Arbeid van België - Parti du Travail de Belgique",
                 "name_en": "Workers' Party of Belgium", "community": "national", "file": "PVDAPTB"},
    "vb": {"id": "vb", "name": "Vlaams Belang", "full_name": "Vlaams Belang",
           "name_en": "Vlaams Belang (Flemish Interest)", "community": "nl", "file": "VB"},
    "vooruit": {"id": "vooruit", "name": "Vooruit", "full_name": "Vooruit",
                "name_en": "Vooruit (Flemish socialists)", "community": "nl", "file": "VOORUIT"},
}
# fiche label used by the Bureau ("VB - 101")
FICHE_LABEL = {"cdv": "cd&v", "defi": "DéFI", "ecolo": "Ecolo", "groen": "Groen", "le": "Les Engagés",
               "mr": "MR", "nva": "N-VA", "openvld": "Open Vld", "ps": "PS", "pvda-ptb": "PVDA-PTB",
               "vb": "VB", "vooruit": "Vooruit"}
# fiches already in data/reference/validation.yaml before this module existed
EXISTING_VALIDATION = {
    ("vb", 101): "dc2024_allowance_13000", ("vb", 103): "dc2024_pit_45_to_40",
    ("nva", 101): "dc2024_abolish_special_ssc", ("mr", 102): "dc2024_abolish_special_ssc",
    ("pvda-ptb", 102): "dc2024_vat_food_zero", ("mr", 201): "dc2024_unemployment_limit_direct_saving",
    ("pvda-ptb", 301): "dc2024_pension_age_65", ("openvld", 402): "dc2024_health_norm_1_5",
    ("ecolo", 103): "dc2024_wealth_tax",
}
ENTITIES = ["federal", "social_security", "flemish_community", "walloon_region", "french_community",
            "brussels_region", "local"]

# Reforms enacted since 2024 that the model baseline contains (codes used in mapping.yaml).
BASELINE_CODES = {"unemp_limit", "rcc", "welfare_freeze", "pension_bonus", "perequation", "defence",
                  "replacement_reductions"}
# Levers the model lacks (codes used in mapping.yaml), reported per measure.
LEVERS = {
    "new_tax", "wealth_tax", "capital_income", "rental_income", "cit_base", "alt_remuneration",
    "targeted_tax_credit", "pit_brackets", "pit_expenditure", "targeted_reductions", "benefit_structure",
    "health_envelope", "programme_spending", "asylum_migration", "asset_sales", "energy_policy",
    "institutional", "activation", "regulation", "labour_regulation", "excise_structure", "vat_item",
    "local_finance",
}


# --------------------------------------------------------------------------- levers

def _lever(alias: str) -> tuple[list[tuple[str, str | None, str, float]], str]:
    """(param, key, op, weight) of one lever alias and the status it implies: 'm' when the
    lever is the budget line or legal parameter the measure is about, 'p' when it is broader."""
    kind, _, rest = alias.partition(":")
    one = lambda param, key, op="mul", st="m": ([(param, key, op, 1.0)], st)
    if kind == "rev":
        return one("rev.line_factor", rest, st="p")
    if kind == "fn":
        return one("exp.function_factor", rest, st="p")
    if kind == "sub":
        return one("subsidy.factor", rest, st="p")
    if kind == "inv":
        return one("invest.public_pct_gdp", None, "add")
    if kind == "invreg":
        return one("invest.regional_extra_pct_gdp", rest, "add")
    if kind == "capt":
        return one("invest.capital_transfers_pct_gdp", None, "add", st="p")
    if kind == "educ":
        return one("educ.community_cost_factor", rest)
    if kind == "childcare":
        return one("family.childcare_factor", rest)
    if kind == "ltc":
        return one("health.ltc_regional_factor", rest)
    if kind == "fam":
        return one("family.allowance_per_child", rest)
    if kind == "sv":
        return one("subsidy.service_voucher_factor", rest)
    if kind == "gen":
        return one({"unemp": "unemp.generosity_factor", "disab": "disab.generosity_factor",
                    "assist": "assist.generosity_factor", "sick": "disab.sickness_factor"}[rest], None, st="p")
    if kind == "pens":
        return one("pension.level_factor", rest, st="p")
    if kind == "grapa":
        return one("pension.grapa_amount", None, st="p")
    if kind in ("entp", "entsp", "ents", "entt"):
        params = {"entp": ["purchases"], "entsp": ["staff_cost", "purchases"], "ents": ["subsidies"],
                  "entt": ["current_transfers"]}[kind]
        ents = ENTITIES if rest == "all" else [rest]
        return [(f"entity.{p}_factor", e, "mul", 1.0) for p in params for e in ents], ("p" if kind == "entt" else "m")
    if kind == "workbonus":   # maxima scaled by (1+s), wage limits by (1+s/2)
        return ([("ssc.workbonus_a_max_monthly", None, "mul", 1.0), ("ssc.workbonus_b_max_monthly", None, "mul", 1.0),
                 ("ssc.workbonus_zero_wage_monthly", None, "mul", 0.5), ("ssc.workbonus_a_full_wage_monthly", None, "mul", 0.5),
                 ("ssc.workbonus_b_full_wage_monthly", None, "mul", 0.5)], "m")
    if kind == "employer_low":
        return ([("ssc.structural_reduction_low_slope", None, "add", 1.0),
                 ("ssc.structural_reduction_low_threshold", None, "mul", 1.0)], "m")
    if kind == "employer_rate":
        return one("ssc.employer_rate", None, "add")
    if kind == "expense_deduction":
        return ([("pit.expense_deduction_max", None, "mul", 1.0), ("pit.expense_deduction_rate", None, "mul", 0.25)], "m")
    if kind == "company_car":
        return one("pit.company_car_benefit", None, st="p")
    if kind == "child_allowance":
        return ([("pit.child_allowance_supplement", None, "mul", 1.0),
                 ("pit.child_allowance_extra_per_child", None, "mul", 1.0)], "m")
    if kind == "thr50":
        return one("pit.bracket_thresholds", "2")
    if kind == "thr_all":
        return ([("pit.bracket_thresholds", k, "mul", 1.0) for k in ("0", "1", "2")], "p")
    if kind == "sec_tax":
        return one("cap.securities_account_tax_rate", None, st="p")
    if kind == "withholding":
        return one("cap.withholding_rate", None, "add", st="p")
    if kind == "energy_excise":
        return one("excise.energy_duty", None)
    if kind == "minima":
        return ([(p, None, "mul", 1.0) for p in ("assist.generosity_factor", "pension.min_pension", "pension.grapa_amount",
                                                   "disab.allowance_amount", "unemp.min_benefit", "disab.min_benefit")], "m")
    if kind == "copay":
        return one("health.copayment_share", None, "add")
    if kind == "disab_rate":
        return one("labour.disability_rate_factor", None, st="p")
    if kind == "proptax":
        return one("prop.property_tax_rate", rest)
    if kind == "inherit":
        return one("prop.inheritance_rate", rest)
    if kind == "regduty":
        return one("prop.registration_rate", rest)
    if kind == "health_norm":
        return one("health.growth_norm", None, "add")
    if kind == "vat":         # share of a tax base moved from one VAT rate to another
        cat, i, j = rest.split(":")
        return [("vat.mix", cat, f"shift:{i}:{j}", 1.0)], "m"
    if kind == "p":           # p:<param>|<key>|<op>
        param, key, op = rest.split("|")
        return one(param, key or None, op or "mul")
    raise ValueError(f"unknown lever '{alias}'")


class Builder:
    def __init__(self) -> None:
        self.ps = ParamSet.load()
        self.base = engine.run(None, end=RUN_END, params=self.ps)
        self.index = engine._baseline_index_fn(self.base)
        self.gdp = {y: self.base.at(y).macro["gdp_nominal"] for y in self.base.years}
        self.runs = 0

    # ---- model access
    def pval(self, name: str, year: int) -> Any:
        return self.ps.value(name, year, self.index)

    def uprate(self, name: str, year: int) -> float:
        kind = self.ps.specs[name].uprate
        return 1.0 if kind == "none" else self.index(kind, year)

    def effect(self, changes: list[dict], static: bool = True) -> dict[int, dict[str, float]]:
        """Scenario minus baseline per year: primary balance, balance (EUR mn), employment."""
        ms = [Measure("m", [Change(**c) for c in changes])]
        if static:
            ms.append(Measure("static", [Change("beh.enabled", 0, 2025)]))
        res = engine.run(Scenario("x", ms), end=RUN_END, baseline=self.base, params=self.ps)
        self.runs += 1
        out = {}
        for y in res.years:
            s, b = res.at(y), self.base.at(y)
            out[y] = {"primary": s.fiscal["primary_balance"] - b.fiscal["primary_balance"],
                      "balance": s.fiscal["balance"] - b.fiscal["balance"],
                      "employment": s.labour["employment"] - b.labour["employment"]}
        return out

    def scaled(self, impulse: float, model_year: int) -> float:
        """An impulse in prices and quantities of 2024 as the same share of GDP in `model_year`."""
        return impulse * self.gdp[model_year] / self.gdp[REF_GDP_YEAR]

    # ---- recipes
    def lever_changes(self, levers: list[tuple], s: float, start: int, end: int | None = None) -> list[dict]:
        out = []
        for param, key, op, w in levers:
            if op.startswith("shift:"):
                _, i, j = op.split(":")
                base = list(self.pval(param, start)[key])
                x = max(-base[int(j)], min(base[int(i)], s * w))
                base[int(i)] -= x
                base[int(j)] += x
                c = {"param": param, "key": key, "value": [round(v, 5) for v in base], "start": start}
            elif op == "mul":
                c = {"param": param, "value": round(max(1e-6, 1.0 + s * w), 6), "op": "mul", "start": start}
            else:
                c = {"param": param, "value": round(s * w, 6), "op": "add", "start": start}
            if key is not None and "key" not in c:
                c["key"] = key
            if end is not None:
                c["end"] = end
            out.append(c)
        return out

    def size(self, levers: list[tuple], fixed: list[dict], target: float, year: int) -> tuple[float, float]:
        """s such that the static primary-balance effect in `year` equals `target`."""
        f = lambda s: self.effect(fixed + self.lever_changes(levers, s, PROGRAMME_YEARS[0] + SHIFT))[year]["primary"] - target
        f0 = f(0.0)
        if abs(f0) < 0.05:
            return 0.0, f0
        s1, f1 = None, None
        for guess in (0.01, -0.01, 0.1, -0.1, 0.5):
            v = f(guess)
            if abs(v - f0) > 1e-3:
                s1, f1 = guess, v
                break
        if s1 is None:
            raise ValueError("lever has no effect on the primary balance")
        s0 = 0.0
        best = (s1, f1) if abs(f1) < abs(f0) else (s0, f0)
        tol = max(0.3, 0.002 * abs(target))
        for _ in range(40):
            if abs(best[1]) <= tol or f1 == f0:
                break
            s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
            if any(op == "mul" and 1.0 + s2 * w <= 0.0 for _, _, op, w in levers):
                s2 = (s1 + min(-0.999 / w for _, _, op, w in levers if op == "mul")) / 2
            f2 = f(s2)
            s0, f0, s1, f1 = s1, f1, s2, f2
            if abs(f2) < abs(best[1]):
                best = (s2, f2)
        return best

    def allowance(self, work: list[float], repl: list[float] | str | None) -> list[dict]:
        """Tax-free allowance of the proposal (EUR of 2024 per programme year) on top of the
        enacted 2026-2030 reform: nobody gets less than in the baseline. `repl` is the allowance
        of pensions and replacement incomes ('keep' = as in the baseline, None = as workers)."""
        out = []
        last = PROGRAMME_YEARS[-1] + SHIFT
        for i, T in enumerate(PROGRAMME_YEARS):
            y = T + SHIFT
            hi = self.index("health_index", y)
            b_allow, b_neut = self.pval("pit.basic_allowance", y), self.pval("pit.allowance_neutralised_replacement", y)
            b_eff = b_allow - b_neut
            n_allow = max(b_allow, work[i] * hi)
            if repl == "keep":
                n_eff = b_eff
            else:
                n_eff = max(b_eff, (work[i] if repl is None else repl[i]) * hi)
            n_neut = max(0.0, n_allow - n_eff)
            end = {} if y == last else {"end": y}
            if abs(n_allow - b_allow) > 0.5:
                out.append({"param": "pit.basic_allowance", "value": round(n_allow), "start": y, **end})
            if abs(n_neut - b_neut) > 0.5:
                out.append({"param": "pit.allowance_neutralised_replacement", "value": round(n_neut), "start": y, **end})
        return out

    def inline(self, changes: list[dict]) -> list[dict]:
        """Changes written in mapping.yaml; `eur24` is an amount in EUR of 2024, carried to the
        start year with the index of the parameter."""
        out = []
        for c in changes:
            c = dict(c)
            if "eur24" in c:
                k = self.uprate(c["param"], c["start"])
                v = c.pop("eur24")
                c["value"] = [round(x * k) for x in v] if isinstance(v, list) else round(v * k)
            out.append(c)
        return out

    # ---- one measure
    def build_measure(self, key: str, src: dict, spec: dict) -> dict:
        imp = {int(y): (None if not isinstance(v, (int, float)) else float(v)) for y, v in src["impulse"].items()}
        costed = src["fpb_status"] == "SIMUL"
        m: dict[str, Any] = {
            "fiche": src["id"], "name": spec["en"], "name_fr": spec["fr"], "name_nl": spec["nl"],
            "title": src["title"], "category": int(str(src["category"]).split(".")[0]),
            "source": {"fiche": f"{FICHE_LABEL[key]} - {src['id']}", "url": f"{SITE}/uploaded/downloads/{PARTIES[key]['file']}.pdf"},
            "official_impulse": {y: (None if v is None else round(v)) for y, v in imp.items()} if costed else None,
        }
        recipe = str(spec.get("r") or "")
        changes: list[dict] = []
        status, method = "not_modelled", "no_lever"
        check = None
        nonzero = [T for T in PROGRAMME_YEARS if imp.get(T)]
        if not costed:
            method = "not_costed"
        elif recipe.startswith("base:"):
            code = recipe.split(":", 1)[1]
            assert code in BASELINE_CODES, code
            status, method = "already_in_baseline", "baseline"
            m["baseline"] = code
        elif recipe in ("zero", "none"):
            method = "zero_impulse" if not nonzero else "no_lever"
        elif recipe == "pin" or recipe.startswith("allow:") or recipe.startswith("assets:"):
            status, method = "modelled", "pinned"
            if recipe.startswith("allow:"):
                parts = recipe.split(":")
                work = [float(x) for x in parts[1].split("/")]
                repl = None if len(parts) < 3 else ("keep" if parts[2] == "keep" else [float(x) for x in parts[2].split("/")])
                changes += self.allowance(work, repl)
            elif recipe.startswith("assets:"):
                _, amount, holdings = recipe.split(":")
                y = PROGRAMME_YEARS[-1] + SHIFT
                changes.append({"param": "debt.stock_flow_adjustment_pct_gdp", "op": "add", "start": y, "end": y,
                                "value": round(-float(amount) / self.gdp[REF_GDP_YEAR] * 100.0, 4)})
                changes += [{"param": "rev.dividends", "key": h, "value": 0, "start": y + 1} for h in holdings.split("+")]
                status = "partly_modelled"
            changes += self.inline(spec.get("ch") or [])
        elif recipe.startswith("sz:"):
            if not nonzero:
                raise ValueError(f"{key} {src['id']}: sized recipe on a zero impulse")
            levers, st = [], "m"
            for alias in recipe[3:].split("+"):
                ls, s1 = _lever(alias)
                levers += ls
                st = "p" if "p" in (st, s1) else "m"
            fixed = self.inline(spec.get("fix") or [])
            ref = max(nonzero, key=lambda T: (abs(imp[T]), T))
            year = ref + SHIFT
            target = self.scaled(imp[ref], year)
            s, resid = self.size(levers, fixed, target, year)
            status, method = ("modelled" if st == "m" else "partly_modelled"), "sized"

            def profile(scale: float) -> list[dict]:
                if spec.get("natural"):     # a growth rate: one change, the effect builds up by itself
                    return self.lever_changes(levers, scale, PROGRAMME_YEARS[0] + SHIFT)
                # the years of the programme follow the profile of the impulse; equal consecutive
                # values are merged; the last value stays (unless the measure is temporary)
                shares = [(T + SHIFT, (imp.get(T) or 0.0) / imp[ref]) for T in PROGRAMME_YEARS]
                groups: list[list] = []
                for y, sh in shares:
                    if groups and abs(groups[-1][2] - sh) < 1e-9 and groups[-1][1] == y - 1:
                        groups[-1][1] = y
                    else:
                        groups.append([y, y, sh])
                last = PROGRAMME_YEARS[-1] + SHIFT
                out: list[dict] = []
                for y0, y1, sh in groups:
                    if abs(sh) < 1e-12:
                        continue
                    end = y1 if (y1 < last or spec.get("temp")) else None
                    out += self.lever_changes(levers, scale * sh, y0, end)
                return out

            # levers with a memory (stocks of beneficiaries, new pensions) do not reach the target
            # when phased in: correct the scale on the phased-in profile itself
            got = resid + target
            for _ in range(4):
                got = self.effect(fixed + profile(s))[year]["primary"]
                base_part = self.effect(fixed)[year]["primary"] if fixed else 0.0
                if abs(got - target) <= max(0.3, 0.002 * abs(target)) or abs(got - base_part) < 1e-6:
                    break
                s2 = s * (target - base_part) / (got - base_part)
                if any(op == "mul" and 1.0 + s2 * w <= 0.0 for _, _, op, w in levers):
                    break
                s = s2
            changes += fixed + profile(s)
            if abs(got - target) > max(1.0, 0.02 * abs(target)):
                # the lever cannot carry the whole impulse (it saturates)
                m["sizing_residual"] = round(got - target, 1)
                status = "partly_modelled"
        else:
            raise ValueError(f"{key} {src['id']}: no recipe")
        if spec.get("st") and status in ("modelled", "partly_modelled"):
            status = {"m": "modelled", "p": "partly_modelled"}[spec["st"]]
        if spec.get("lever"):
            assert spec["lever"] in LEVERS, spec["lever"]
        if status in ("partly_modelled", "not_modelled") and method not in ("not_costed",) and spec.get("lever"):
            m["lever"] = spec["lever"]
        m["status"], m["method"] = status, method
        if spec.get("inc"):
            m["increment"] = True
        if spec.get("rev"):
            m["reversal"] = True
        if changes and not recipe.startswith("assets:"):
            eff = self.effect(changes)
            if nonzero:
                ref = max(nonzero, key=lambda T: (abs(imp[T]), T))
                year = ref + SHIFT
                m["check"] = {"year": year, "official_impulse_scaled": round(self.scaled(imp[ref], year), 1),
                              "model_static": round(eff[year]["primary"], 1)}
            else:
                year = PROGRAMME_YEARS[-1] + SHIFT
                m["check"] = {"year": year, "official_impulse_scaled": 0.0, "model_static": round(eff[year]["primary"], 1)}
        vid = EXISTING_VALIDATION.get((key, src["id"]))
        if vid:
            m["validation_id"] = vid
        elif changes and nonzero and m.get("check") and status in ("modelled", "partly_modelled"):
            m["validation_id"] = f"dc2024_{PARTIES[key]['id'].replace('-', '_')}_{src['id']}"
        m["changes"] = changes
        return m


# --------------------------------------------------------------------------- coverage

STATUSES = ["modelled", "partly_modelled", "not_modelled", "already_in_baseline"]


def reference_impulse(m: dict) -> float:
    """Size of a measure for the coverage shares: its largest yearly ex-ante impulse (absolute)."""
    imp = m.get("official_impulse") or {}
    vals = [abs(v) for v in imp.values() if isinstance(v, (int, float))]
    return max(vals) if vals else 0.0


def coverage(measures: list[dict]) -> dict:
    """Count of measures and share of the ex-ante impulse (absolute amounts) per status."""
    count = {s: 0 for s in STATUSES}
    amount = {s: 0.0 for s in STATUSES}
    for m in measures:
        count[m["status"]] += 1
        amount[m["status"]] += reference_impulse(m)
    total = sum(amount.values())
    return {
        "measures": len(measures),
        "costed": sum(1 for m in measures if m.get("official_impulse") is not None),
        "count": count,
        "impulse_abs": {s: round(v) for s, v in amount.items()},
        "impulse_share": {s: (round(v / total, 4) if total else 0.0) for s, v in amount.items()},
    }


# --------------------------------------------------------------------------- writing

HERMES = {"gdp_pct": "Bbp", "employment": "Werkgelegenheid", "balance_pct_gdp": "Vorderingensaldo van de overheid",
          "debt_pct_gdp": "Overheidsschuld", "real_disposable_income_pct": "Reëel beschikbaar inkomen van de particulieren",
          "consumer_prices_pct": "Consumptieprijsindex"}

HEADER = """# {name} - measures submitted for the Federal Planning Bureau's costing of the 2024 election
# programmes (DC2024), translated into belsim parameters.
# GENERATED by `uv run python -m belsim.api.party_build` from scenarios/parties/source/
# (dc2024.json = what the Bureau published, mapping.yaml = the translation). Edit those, not this.
# These are the measures the party submitted in 2024, not its current positions, and the
# model's results are not the Planning Bureau's. Programme year T is model year T+{shift}.
"""


def build_party(b: Builder, key: str, src: dict, mapping: dict, today: str) -> dict:
    meta = PARTIES[key]
    measures = [b.build_measure(key, m, mapping[m["id"]]) for m in src["measures"]]
    total = {T: round(sum((m["official_impulse"] or {}).get(T) or 0 for m in measures)) for T in PROGRAMME_YEARS}
    official = {"impulse": total}
    for field, label in HERMES.items():
        vals = src["hermes"].get(label)
        if vals:
            official[field] = {T: round(v, 2) for T, v in zip(PROGRAMME_YEARS, vals)}
    dec = [r for r in src.get("expedition_deciles") or [] if "deciel" in r[0]]
    if len(dec) == 10:
        official["disposable_income_by_decile_pct"] = [round(r[3], 2) for r in dec]
    return {
        "id": meta["id"], "kind": "party", "name": meta["name"], "full_name": meta["full_name"],
        "name_en": meta["name_en"], "community": meta["community"],
        "source": {
            "institution": "Federal Planning Bureau",
            "document": "DC2024 - Doorrekening van de verkiezingsprogramma's / Chiffrage des programmes électoraux 2024",
            "date": "2024-05", "url": f"{SITE}/measures/measures_nl.html",
            "fiches_url": f"{SITE}/uploaded/downloads/{meta['file']}.pdf",
            "results_url": f"{SITE}/results/results_nl.html",
            "workbook_url": f"{SITE}/uploaded/downloads/{meta['file']}_nl.xlsx",
            "retrieved": src.get("retrieved", today),
        },
        "official": official,
        "build": {"date": today, "programme_years": PROGRAMME_YEARS, "year_shift": SHIFT,
                  "impulse_basis": "EUR mn, prices and quantities of 2024; compared as a share of GDP"},
        "coverage": coverage(measures),
        "measures": measures,
    }


def _dump(doc: dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10000, default_flow_style=None)


def build_all(only: list[str] | None = None, verbose: bool = True) -> dict[str, dict]:
    data = json.loads((SOURCE_DIR / "dc2024.json").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((SOURCE_DIR / "mapping.yaml").read_text(encoding="utf-8"))
    b = Builder()
    today = _dt.date.today().isoformat()
    out = {}
    for key in PARTIES:
        if only and key not in only and PARTIES[key]["id"] not in only:
            continue
        doc = build_party(b, key, data[key], mapping[key], today)
        (PARTY_DIR / f"{doc['id']}.yaml").write_text(
            HEADER.format(name=doc["name"], shift=SHIFT) + _dump(doc), encoding="utf-8")
        out[key] = doc
        if verbose:
            c = doc["coverage"]
            print(f"{doc['id']:<12} {c['measures']:>3} measures  " + "  ".join(f"{s}={c['count'][s]}" for s in STATUSES)
                  + f"  impulse modelled {c['impulse_share']['modelled']:.0%} partly {c['impulse_share']['partly_modelled']:.0%}")
            for m in doc["measures"]:
                if m.get("sizing_residual") is not None:
                    print(f"    residual {m['source']['fiche']}: {m['sizing_residual']} (target {m['check']['official_impulse_scaled']})")
    if verbose:
        print(f"{b.runs} model runs")
    return out


# --------------------------------------------------------------------------- validation entries

VALIDATION_FILE = ROOT / "data" / "reference" / "validation.yaml"
BEGIN = "  # >>> DC2024 party programmes: GENERATED by belsim.api.party_build (do not edit by hand)\n"
END = "  # <<< DC2024 party programmes\n"
TARGET_LABEL = "primary balance (ex-ante budgetary impulse)"
TARGET_LABEL_I18N = {"fr": "solde primaire (impulsion budgétaire ex ante)", "nl": "primair saldo (budgettaire impuls ex ante)"}
DESC = {
    "en": "DC2024 {fiche} - {name}. Ex-ante budgetary impulse in prices and quantities of 2024, year {T} of the programme read in model year {y}",
    "fr": "DC2024 {fiche} - {name}. Impulsion budgétaire ex ante aux prix et quantités de 2024, année {T} du programme lue dans l'année {y} du modèle",
    "nl": "DC2024 {fiche} - {name}. Budgettaire impuls ex ante in prijzen en hoeveelheden van 2024, jaar {T} van het programma gelezen in modeljaar {y}",
}
NOTE = {
    "sized": "Parameter sized on this impulse, so the gap only measures the precision of the sizing.",
    "pinned": "Parameter value taken from the fiche.",
    "increment": "Partly enacted since 2024 and in the model baseline: the model effect is the increment only, the reference is the whole measure.",
    "partly": "Carried by a broader budget line or only part of the measure is modelled.",
}


def load_parties() -> list[dict]:
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(PARTY_DIR.glob("*.yaml"))]


def validation_entries() -> list[tuple[dict, dict[str, str]]]:
    """(entry, {lang: description}) for every modelled measure with an impulse that has no
    entry of its own in the hand-written part of the file."""
    out = []
    existing = set(EXISTING_VALIDATION.values())
    for doc in load_parties():
        for m in doc["measures"]:
            vid = m.get("validation_id")
            if not vid or vid in existing or not m.get("changes") or not m.get("check"):
                continue
            y = m["check"]["year"]
            T = y - SHIFT
            imp = m["official_impulse"][T]
            desc = {lang: DESC[lang].format(fiche=m["source"]["fiche"], T=T, y=y,
                                            name=m["name" if lang == "en" else f"name_{lang}"]) for lang in DESC}
            notes = [NOTE[m["method"]]]
            if m.get("increment"):
                notes.append(NOTE["increment"])
            if m["status"] == "partly_modelled":
                notes.append(NOTE["partly"])
            entry = {
                "id": vid, "description": desc["en"],
                "source": {"institution": "Federal Planning Bureau",
                           "title": f"DC2024 - fiche {m['source']['fiche']} ({doc['name']})", "date": "2024-05"},
                "url": m["source"]["url"], "confidence": "high",
                "measure": {"changes": m["changes"]},
                "targets": [{"label": TARGET_LABEL, "variant": "static", "ref_value": imp, "ref_unit": "EUR mn",
                             "ref_year": REF_GDP_YEAR, "model_year": y, "normalise": "gdp",
                             "model": {"num": ["fiscal.primary_balance"]}}],
                "note": " ".join(notes),
            }
            out.append((entry, desc))
    return out


def write_validation() -> int:
    entries = validation_entries()
    text = VALIDATION_FILE.read_text(encoding="utf-8")
    block = BEGIN
    for e, _ in entries:
        body = yaml.safe_dump([e], sort_keys=False, allow_unicode=True, width=10000, default_flow_style=None)
        block += "".join("  " + line + "\n" for line in body.splitlines())
    block += END
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        m = re.search(r"^not_found:", text, flags=re.M)
        if not m:
            raise RuntimeError("validation.yaml: no 'not_found:' section to insert before")
        text = text[:m.start()] + block + "\n" + text[m.start():]
    VALIDATION_FILE.write_text(text, encoding="utf-8")
    # translations of the generated descriptions and of the target label
    from ..i18n import catalog_path
    for lang in ("fr", "nl"):
        path = catalog_path(lang)
        cat = path.read_text(encoding="utf-8")
        q = lambda s: json.dumps(s, ensure_ascii=False)
        lines = "".join(f"  {e['id']}: {q(d[lang])}\n" for e, d in entries)
        vb, ve = "  # >>> generated by belsim.api.party_build\n", "  # <<< generated by belsim.api.party_build\n"
        if vb in cat:
            head, rest = cat.split(vb, 1)
            cat = head + vb + lines + ve + rest.split(ve, 1)[1]
        else:
            m = re.search(r"^target:\n", cat, flags=re.M)
            cat = cat[:m.start()] + vb + lines + ve + cat[m.start():]
            m = re.search(r"^target:\n", cat, flags=re.M)
            cat = cat[:m.end()] + f"  {q(TARGET_LABEL)}: {q(TARGET_LABEL_I18N[lang])}\n" + cat[m.end():]
        path.write_text(cat, encoding="utf-8")
    return len(entries)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parties", nargs="*", help="party ids (default: all)")
    ap.add_argument("--validation", action="store_true", help="refresh the generated validation entries")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild the party files")
    a = ap.parse_args(argv)
    if not a.no_build:
        build_all(a.parties or None)
    if a.validation:
        print(f"{write_validation()} validation entries written")


if __name__ == "__main__":
    main()
