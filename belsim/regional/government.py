"""Federated entities inside "regions & communities" (S1312) and the regional incidence of the
federal budget and social security.

Entities (`ENTITIES4`): Flemish Community (Community + Region), Walloon Region (incl. the
German-speaking Community), French Community, Brussels-Capital Region (incl. the community
commissions). Their accounts are those published by the NBB, i.e. NOT consolidated with each
other; the pseudo-entity `interregional_consolidation` (interregional units, transfers between
entities, rounding) is the difference with the aggregate of `fiscal.balance`, so that the five
always add up exactly to `fiscal["<item>.regions_communities"]`.

Split of the aggregate, every year:

    expenditure_E = sum over lines of  line x share of regions & communities x w(line, E)
                    + what the region / Community keyed levers add for E (`regional_delta`)
                    + transfers paid by the sub-sector x w + residual_E x GDP
    revenue_E     = regional taxes of the territory (revenue module's `revenue.<line>.<REG>` if
                    published, else 2024 shares moving with the tax-base proxies) + other own
                    revenue x w + Special Financing Act grants by mechanism + other transfers
                    received x w + residual_E x GDP

`w(line, E)` (sums to 1 over the four entities) follows the competence that pays the line
(`LINE_KEYS`): pupils by Community, children x regional allowance, care needs, population of
the Region or of the Community, regional GDP, last year's debt (interest) or, for mixed lines,
the 2024 size of the entity; wage-heavy lines are tilted by the regional public-employment
norm (`labour.public_employment_growth` acts on public employees by region of residence, whatever
their employer: the entities of that territory get the regions-and-communities part of the saving). The residuals (constant % of GDP) are fitted so that 2024 equals the published accounts.

Financing Act grants: the 2024 amounts by mechanism (`regional.sfa_grants_2024`) move with
their legal key - pupils (VAT allocation), PIT key of the territory, population under 18 and
over 80 (transferred competences), the solidarity rule (population share minus PIT share of the
region, if positive), the phase-out of the transition mechanism (`grants_by_mechanism`).
`fiscal.balance` builds the federal transfer from these entity-level rules, so the national
envelope is their sum: a region gaining tax base gains PIT allocation and loses solidarity.

Budget paths of the entities (`entity.*_factor`, see `fiscal/expenditure/model.py`): the adopted
budgets and multi-year plans of each federated entity change its spending by category relative
to the unchanged-policy lines. The expenditure module applies them to the lines (the total
changes) and reports the amount per entity (`entity_delta`), which is attributed here to the
entity that decided it, not spread with the line's weights.
"""
from __future__ import annotations

import numpy as np

from ..core.dims import ENTITIES, REGIONS

FED, SS, RC, LOC = ENTITIES
ENTITIES4 = ["flemish_community", "walloon_region", "french_community", "brussels_region"]
OTHER = "interregional_consolidation"

# id -> label, parent entity, kind (read by the API)
ENTITY_DETAIL = {
    "flemish_community": {"label": "Flemish Community (Community + Region)", "parent": RC, "kind": "entity",
                          "regions": ["FLA", "BRU"]},
    "walloon_region": {"label": "Walloon Region (incl. German-speaking Community)", "parent": RC,
                       "kind": "entity", "regions": ["WAL"]},
    "french_community": {"label": "French Community (Federation Wallonie-Bruxelles)", "parent": RC,
                         "kind": "entity", "regions": ["WAL", "BRU"]},
    "brussels_region": {"label": "Brussels-Capital Region (incl. community commissions)", "parent": RC,
                        "kind": "entity", "regions": ["BRU"]},
    OTHER: {"label": "Interregional units and consolidation between entities", "parent": RC,
            "kind": "consolidation", "regions": []},
}

EDUC_AGES = {"educ_primary": slice(3, 12), "educ_secondary": slice(12, 18), "educ_tertiary": slice(18, 25),
             "educ_other": slice(3, 25)}
# which competence pays the regions-and-communities part of an expenditure line (default: mixed)
LINE_KEYS = {
    **{k: "educ" for k in EDUC_AGES},
    "family_allowances": "family", "long_term_care": "care",
    "childcare_family_services": "community", "culture_recreation": "community",
    "health_admin_prevention": "community", "pension_civil_servants": "community", "research": "community",
    "public_investment": "gdp", "capital_transfers": "gdp", "economic_affairs": "gdp",
    "subsidy_wage": "region", "subsidy_service_vouchers": "region", "subsidy_transport": "region",
    "subsidy_energy_environment": "region", "subsidy_other": "region", "transport": "region",
    "environment": "region", "housing_community": "region", "social_housing": "region",
    "labour_market_benefits": "region", "health_care": "care", "disability_allowances": "region",
    "interest": "debt",
}
REGIONAL_TAX_LINES = ("registration_duties", "inheritance_gift_tax", "vehicle_taxes", "energy_green_levies",
                      "environmental_taxes", "gambling_lottery", "excise_energy", "property_tax")
DELTA_ENTITY = {"FLA": 0, "WAL": 1, "BRU": 3, "flemish_community": 0, "french_community": 2}
GRANT_KEYS = ("vat_pupils", "pit_key", "needs_population", "solidarity", "transition", "other")


def region_w(x) -> np.ndarray:
    """Region vector -> weights of the four entities for a regional competence."""
    return np.array([x[0], x[1], 0.0, x[2]], dtype=float)


def community_w(x, b: float) -> np.ndarray:
    """Region vector -> weights of the four entities for a Community competence."""
    return np.array([x[0] + b * x[2], 0.0, x[1] + (1.0 - b) * x[2], 0.0], dtype=float)


def territory(x, b: float) -> np.ndarray:
    """Value of a regional key on the territory served by each entity."""
    return np.array([x[0] + b * x[2], x[1], x[1] + (1.0 - b) * x[2], x[2]], dtype=float)


def entity_to_region(pop_r, b: float) -> np.ndarray:
    """[entity, region] share of each entity's accounts attributed to each region (population)."""
    m = np.zeros((4, 3))
    fl = np.array([pop_r[0], 0.0, b * pop_r[2]])
    fr = np.array([0.0, pop_r[1], (1.0 - b) * pop_r[2]])
    m[0], m[1], m[2], m[3] = fl / fl.sum(), [0.0, 1.0, 0.0], fr / fr.sum(), [0.0, 0.0, 1.0]
    return m


def _sh(a) -> np.ndarray:
    a = np.maximum(np.asarray(a, dtype=float), 0.0)
    tot = a.sum()
    return a / tot if tot > 0 else np.full(len(a), 1.0 / len(a))


def _entity_share(meta, entity: str, default: float) -> float:
    sh = getattr(meta, "entity_shares", None)
    if not sh:
        return default
    vals = {e: max(float(sh.get(e, 0.0)), 0.0) for e in ENTITIES}
    tot = sum(vals.values())
    return vals[entity] / tot if tot > 0 else default


def _weights_fn(pop_age_r, family, ltc_need, pop_r, gdp_r, debt_prev, mixed_exp, b: float):
    """w(key, line) -> weights [4] of the federated entities for the competence `key`."""
    def weights(key: str, line: str = "") -> np.ndarray:
        if key == "educ":
            return _sh(community_w(pop_age_r[EDUC_AGES[line]].sum(axis=0), b))
        if key == "family":
            return _sh(region_w(family))
        if key == "care":
            return _sh(region_w(ltc_need))
        if key == "community":
            return _sh(community_w(pop_r, b))
        if key == "region":
            return _sh(region_w(pop_r))
        if key == "gdp":
            return _sh(region_w(gdp_r))
        if key == "debt":
            return _sh(debt_prev)
        return mixed_exp
    return weights


def path_weights(state, prev, p):
    """line -> weights [4] of the federated entities in the regions-and-communities part of an
    expenditure line, from what is known when the expenditure module runs (population, last
    year's debts, regional GDP of `belsim.regional.economy` if already computed). Used for the
    spending paths by entity (`entity.*_factor`); same rules as the split made in `compute`.
    None when the regional parameters are not loaded (unit-test worlds)."""
    acc = p.get("regional.entity_accounts_2024")
    if not acc:
        return None
    pop = state.pop
    pop_r, pop_age_r = pop.sum(axis=(0, 1)), pop.sum(axis=1)
    b = float(p.get("regional.brussels_dutch_share", 0.2))
    amounts = p.get("family.allowance_per_child") or {}
    kids = pop[:18].sum(axis=(0, 1)) + float(p.get("family.student_share", 0.0) or 0.0) * pop[18:25].sum(axis=(0, 1))
    family = kids * np.array([float(amounts.get(r, 1.0)) for r in REGIONS])
    ages, costs = p.get("health.ltc_profile_ages"), p.get("health.ltc_profile_cost")
    if ages is not None and costs is not None:
        prof = np.zeros(pop.shape[0])
        bounds = list(ages) + [pop.shape[0]]
        for lo, hi, c in zip(bounds[:-1], bounds[1:], costs):
            prof[int(lo):int(hi)] = c
        ltc_need = prof @ pop_age_r
    else:
        ltc_need = pop[80:].sum(axis=(0, 1))
    gdp_r = (state.extra.get("regional_economy") or {}).get("gdp")
    if gdp_r is None:
        gdp_r = pop_r
    acc_exp = np.array([float(acc[e]["expenditure"]) for e in ENTITIES4])
    acc_debt = np.array([float(acc[e]["debt"]) for e in ENTITIES4])
    debt_prev = (np.array([prev.fiscal[f"debt.{e}"] for e in ENTITIES4])
                 if (prev is not None and f"debt.{ENTITIES4[0]}" in prev.fiscal) else acc_debt)
    w = _weights_fn(pop_age_r, family, ltc_need, pop_r, np.asarray(gdp_r, dtype=float), debt_prev, _sh(acc_exp), b)
    return lambda line: w(LINE_KEYS.get(line, "mixed"), line)


def revenue_region_shares(state, line: str, K: dict, key0: dict, own_taxes: dict | None) -> np.ndarray:
    """Regional split of a revenue line: the revenue module's keys if published, else proxies."""
    vals = [state.regional.get(f"revenue.{line}.{r}") for r in REGIONS]
    if all(v is not None for v in vals) and sum(vals) > 0 and min(vals) >= 0:
        return _sh(vals)
    if line.startswith("pit"):
        if line == "pit_regional" and own_taxes:
            return _sh(np.array([own_taxes["pit"][r] for r in REGIONS]) * K["pit"] / key0["pit"])
        return _sh(K["pit"])
    if line in REGIONAL_TAX_LINES and own_taxes:
        return _sh(np.array([own_taxes["other"][r] for r in REGIONS]) * K["disp"] / key0["disp"])
    if line.startswith("ssc") or line == "special_ssc":
        return _sh(K["ssc"])
    if line.startswith(("vat", "excise", "insurance", "gambling", "vehicle", "registration", "inheritance",
                        "property", "environmental", "energy")):
        return _sh(K["disp"])
    return _sh(K["gdp"])


def grant_keys(state, pit_r=None) -> dict | None:
    """Keys of the Financing Act mechanisms by region: pupils, population under 18 and over 80,
    population, federal personal income tax of the residents (the revenue module's
    `revenue.pit_federal.<REG>`, or `pit_r`). None when the PIT by region is not available."""
    pop = state.pop
    if pit_r is None:
        vals = [state.regional.get(f"revenue.pit_federal.{r}") for r in REGIONS]
        if not all(v is not None and v > 0 for v in vals):
            return None
        pit_r = np.array(vals, dtype=float)
    return {"pupils": pop[6:18].sum(axis=(0, 1)),
            "young_old": pop[:18].sum(axis=(0, 1)) + pop[80:].sum(axis=(0, 1)),
            "pop": pop.sum(axis=(0, 1)), "pit": np.asarray(pit_r, dtype=float)}


def grants_by_mechanism(p, K: dict, key0: dict, nominal: float, year: int, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Federal Financing Act transfers by entity [4]: (indexed mechanisms, transition amounts).

    Each mechanism moves with its legal key: the VAT allocation with pupils, the transferred
    competences with the population under 18 and over 80, the PIT allocations with the share
    of the territory in the federal income tax (an indexed envelope shared by the PIT key, not
    the yield itself), national solidarity with the gap between the population share and the
    PIT share of the region (zero once the gap closes), the rest with the general index
    `nominal` (consumer prices and part of real growth). The transition mechanism is constant
    in nominal terms and then phased out. Nothing is scaled to a national envelope: a region
    whose tax base grows faster than the others gains PIT allocation and loses solidarity."""
    g0 = p["regional.sfa_grants_2024"]
    base = {k: np.array([float(g0[e].get(k, 0.0)) for e in ENTITIES4]) for k in GRANT_KEYS}

    def idx(name):
        return territory(K[name], b) / np.maximum(territory(key0[name], b), 1e-9)

    pit_key = territory(_sh(K["pit"]), b) / np.maximum(territory(_sh(key0["pit"]), b), 1e-9)
    out = base["vat_pupils"] * idx("pupils") * nominal
    out = out + base["pit_key"] * pit_key * nominal
    out = out + base["needs_population"] * idx("young_old") * nominal
    out = out + base["other"] * nominal
    # solidarity: population share minus PIT share of the region, if positive (regions: WR, BR;
    # for the Flemish Community the gap of Flanders applies)
    gap = region_w(np.maximum(_sh(K["pop"]) - _sh(K["pit"]), 0.0))
    gap0 = region_w(np.maximum(_sh(key0["pop"]) - _sh(key0["pit"]), 0.0))
    out = out + base["solidarity"] * np.where(gap0 > 0, gap / np.where(gap0 > 0, gap0, 1.0), 0.0) * nominal
    ph = p["regional.sfa_transition_phase_out"]
    left = min(1.0, max(0.0, 1.0 - (year - float(ph["last_full_year"])) / float(ph["years"])))
    return out, base["transition"] * left


def grants(p, K: dict, key0: dict, total: float, nominal: float, year: int, b: float) -> np.ndarray:
    """Federal Financing Act transfers by entity [4], scaled (all but the transition amounts) to
    a given national `total`. Fallback used when `fiscal.balance` did not build the transfers
    from the mechanisms (no PIT by region, unit tests)."""
    out, trans = grants_by_mechanism(p, K, key0, nominal, year, b)
    rest = float(out.sum())
    return out * ((total - float(trans.sum())) / rest if rest else 1.0) + trans


def compute(state, prev, p, K: dict, px: dict | None, bx: dict | None) -> tuple[dict, dict, dict]:
    """Returns (fiscal keys, by-region arrays, carried state). `px`: last year's carried state;
    `bx`: the baseline's base-year carried state when running a scenario's base year."""
    from ..fiscal import expenditure as exp_mod
    try:
        from ..fiscal import revenue as rev_mod
        rev_lines = getattr(rev_mod, "LINES", None) or {}
    except Exception:
        rev_lines = {}
    exp_lines = getattr(exp_mod, "LINES", {}) or {}
    f, m = state.fiscal, state.macro
    gdp = m["gdp_nominal"]
    gdp_mp = float(m.get("gdp_nominal_market_prices", gdp) or gdp)     # denominator of the ratios
    b = float(p["regional.brussels_dutch_share"])
    src = px if px is not None else bx
    acc = p["regional.entity_accounts_2024"]
    acc0 = {k: np.array([float(acc[e][k]) for e in ENTITIES4]) for k in ("revenue", "expenditure", "balance", "debt")}
    key0 = src["key0"] if src is not None else {k: np.array(v, dtype=float) for k, v in K.items()}
    own_taxes = p.get("regional.own_taxes_2024")
    pop_r = K["pop"]
    mixed_exp, mixed_rev = _sh(acc0["expenditure"]), _sh(acc0["revenue"])
    debt_prev = (np.array([prev.fiscal[f"debt.{e}"] for e in ENTITIES4])
                 if (prev is not None and f"debt.{ENTITIES4[0]}" in prev.fiscal) else acc0["debt"])
    tilt_r = np.asarray((state.extra.get("labour") or {}).get("public_tilt", np.ones(3)), dtype=float)
    tilt_e = territory(tilt_r * pop_r, b) / territory(pop_r, b)
    tilted = bool(np.any(tilt_e != 1.0))      # only lines whose wage part follows the headcount
    defs = exp_mod.lines.all_defs() if tilted else {}
    pub_r = K["public_employees"]
    tilt_nat = float(pub_r.sum() / max(float((pub_r / np.maximum(tilt_r, 1e-9)).sum()), 1e-9))
    ex = state.extra.get("expenditure") or {}
    delta = ex.get("regional_delta") or {}
    path = ex.get("entity_delta") or {}        # spending paths by entity: line -> entity -> EUR mn
    pop_age_r = state.pop.sum(axis=1)

    weights = _weights_fn(pop_age_r, K["family"], K["ltc_need"], pop_r, K["gdp"], debt_prev, mixed_exp, b)

    # ---------------- expenditure ----------------
    exp_e = np.zeros(4)
    for lid, v in state.expenditure.items():
        meta = exp_lines.get(lid)
        sh = _entity_share(meta, RC, 0.0)
        d = delta.get(lid) or {}
        d_tot = float(sum(d.values()))
        direct_scale = sh if lid in EDUC_AGES else 1.0      # as in fiscal.balance
        pd = path.get(lid) or {}
        base = (v - d_tot - float(sum(pd.values()))) * sh
        if base:
            w = weights(LINE_KEYS.get(lid, "mixed"), lid)
            d1 = 0.0
            if tilted and getattr(defs.get(lid), "driver", "") in ("generic", "children_services"):
                row = exp_mod.data.lines_2024().get(lid)
                d1 = row["D1"] / row["value"] if row and row["value"] else 0.0
                # scaled so that entities whose territory is not concerned keep their amount:
                # the national wage bill moved by `tilt_nat`, the entities' exposure by `den`
                den = float((w * (tilt_e - 1.0)).sum())
                theta = float(np.clip((tilt_nat - 1.0) / den, 0.0, 1.5)) if abs(den) > 1e-12 else 0.0
                w = _sh(w * (1.0 + d1 * theta * (tilt_e - 1.0)))
            exp_e += base * w
        for k, amount in d.items():
            exp_e[DELTA_ENTITY[k]] += amount * direct_scale
        for i, e in enumerate(ENTITIES4):
            exp_e[i] += pd.get(e, 0.0)
    t_loc = f.get(f"transfer.{RC}_to_{LOC}", 0.0)
    from ..fiscal.expenditure import data as exp_data
    es = exp_data.scalars().get("regions_to_local_education_share_2024", 0.0)
    exp_e += t_loc * (es * _sh(community_w(pop_age_r[3:18].sum(axis=0), b)) + (1.0 - es) * _sh(region_w(pop_r)))
    exp_e += (f.get(f"transfer.{RC}_to_{FED}", 0.0) + f.get(f"transfer.{RC}_to_{SS}", 0.0)) * mixed_exp
    exp_e += (f[f"own_expenditure.{RC}"] + t_loc + f.get(f"transfer.{RC}_to_{FED}", 0.0)
              + f.get(f"transfer.{RC}_to_{SS}", 0.0) - float(exp_e.sum())) * mixed_exp

    # ---------------- revenue ----------------
    rev_e = np.zeros(4)
    own_tax_e = np.zeros(4)
    for lid, v in state.revenue.items():
        sh = _entity_share(rev_lines.get(lid), RC, 0.0)
        if not sh or not v:
            continue
        if lid.startswith("pit") or lid in REGIONAL_TAX_LINES:
            part = v * sh * region_w(revenue_region_shares(state, lid, K, key0, own_taxes))
            own_tax_e += part
            rev_e += part
        elif lid == "ssc_imputed":          # imputed contributions: civil-servant (teacher) pensions
            w = _sh(community_w(pop_r, b))
            if tilted:      # they follow the public wage bill: same tilt as the wage-heavy spending lines
                den = float((w * (tilt_e - 1.0)).sum())
                theta = float(np.clip((tilt_nat - 1.0) / den, 0.0, 1.5)) if abs(den) > 1e-12 else 0.0
                w = _sh(w * (1.0 + theta * (tilt_e - 1.0)))
            rev_e += v * sh * w
        else:
            rev_e += v * sh * mixed_rev
    rev_e += (f[f"own_revenue.{RC}"] - float(rev_e.sum())) * mixed_rev      # lines without shares, rounding
    bal_x = state.extra.get("balance") or {}
    if bal_x.get("grant_e") is not None:        # built from the mechanisms by fiscal.balance
        grant_e = np.asarray(bal_x["grant_e"], dtype=float)
    else:
        nominal = float(bal_x.get("sfa_idx", 1.0))
        grant_e = grants(p, K, key0, f.get(f"transfer.{FED}_to_{RC}", 0.0), nominal, state.year, b)
    rev_e += grant_e
    rev_e += (f.get(f"transfer.{SS}_to_{RC}", 0.0) + f.get(f"transfer.{LOC}_to_{RC}", 0.0)
              + f.get(f"entity_residual.{RC}", 0.0)) * mixed_rev

    # ---------------- calibration on the 2024 accounts ----------------
    if src is not None:
        resid = src["resid"]
    else:
        resid = {"revenue": (acc0["revenue"] - rev_e) / gdp, "expenditure": (acc0["expenditure"] - exp_e) / gdp}
    rev_e = rev_e + resid["revenue"] * gdp
    exp_e = exp_e + resid["expenditure"] * gdp
    bal_e = rev_e - exp_e

    # ---------------- debt ----------------
    agg = {k: f[f"{k}.{RC}"] for k in ("revenue", "expenditure", "balance", "debt")}
    other = {"revenue": agg["revenue"] - float(rev_e.sum()), "expenditure": agg["expenditure"] - float(exp_e.sum())}
    other["balance"] = other["revenue"] - other["expenditure"]
    if prev is None:
        base_bal = bx["bal_e"] if bx is not None else bal_e
        debt_e = acc0["debt"] - (bal_e - base_bal)
    else:
        # stock-flow adjustment of the sub-sector (as in fiscal.balance) in proportion to debt
        prev_all = np.append(debt_prev, prev.fiscal[f"debt.{RC}"] - float(debt_prev.sum()))
        sfa = agg["debt"] - prev.fiscal[f"debt.{RC}"] + agg["balance"]
        tot_prev = float(np.abs(prev_all).sum())
        debt_e = debt_prev - bal_e + sfa * np.abs(debt_prev) / (tot_prev if tot_prev else 1.0)
    other["debt"] = agg["debt"] - float(debt_e.sum())

    out: dict[str, float] = {}
    for i, e in enumerate(ENTITIES4):
        out[f"revenue.{e}"], out[f"expenditure.{e}"] = float(rev_e[i]), float(exp_e[i])
        out[f"balance.{e}"], out[f"debt.{e}"] = float(bal_e[i]), float(debt_e[i])
        out[f"balance_pct_gdp.{e}"] = float(bal_e[i] / gdp_mp * 100.0)
        out[f"own_taxes.{e}"] = float(own_tax_e[i])
        out[f"federal_grants.{e}"] = float(grant_e[i])
    for k, v in other.items():
        out[f"{k}.{OTHER}"] = float(v)
    out[f"balance_pct_gdp.{OTHER}"] = float(other["balance"] / gdp_mp * 100.0)

    # ---------------- by region ----------------
    m_er = entity_to_region(pop_r, b)
    cons = _sh([float(p["regional.consolidation_shares"][r]) for r in REGIONS])
    pop_sh = _sh(pop_r)
    reg = {
        "gov_expenditure": exp_e @ m_er + other["expenditure"] * cons,
        "gov_balance": bal_e @ m_er + other["balance"] * pop_sh,
        "gov_debt": debt_e @ m_er + other["debt"] * pop_sh,
        "gov_own_taxes": own_tax_e @ m_er,
        "transfers_from_federal": grant_e @ m_er,
    }
    reg["gov_revenue"] = reg["gov_expenditure"] + reg["gov_balance"]
    x = {"key0": key0, "resid": resid, "bal_e": bal_e, "grant_e": grant_e, "m_er": m_er}
    return out, reg, x
