"""Budget balance, debt, government entities and EU fiscal-rule indicators.

Fills `state.fiscal` from `state.revenue` and `state.expenditure`:

- totals, balance, primary balance, % of GDP;
- Maastricht debt: debt_t = debt_{t-1} - balance_t + stock-flow adjustment (2024 = actual);
- per entity (federal, social security, regions & communities, local): own revenue and final
  expenditure from the lines' `entity_shares`, then transfers between entities by parameterised
  rules calibrated on the 2024 flows (Eurostat gov_10a_main D7/D9/D4 "of which to subsector"),
  giving `revenue.<e>`, `expenditure.<e>` (both including transfers, i.e. non-consolidated as in
  the subsector accounts), `balance.<e>` and `debt.<e>`;
- `net_expenditure_growth` (fraction) and `structural_balance_pct_gdp`.

Transfer rules (flow_t = flow_2024 x index_t x transfer.level_factor + transfer.adjustment):
  federal -> social security   alternative financing (VAT and PIT shares, follow those revenues) +
                               indexed state grants + balancing grant covering
                               `transfer.ss_balancing_share` of the gap between the social-security
                               balance and its 2024 level in % of GDP
  federal -> regions & comm.   Special Financing Act grants, built entity by entity from the
                               legal mechanisms (`belsim.regional.government.grants_by_mechanism`:
                               pupils, PIT key, population under 18 and over 80, solidarity,
                               transition phase-out), each indexed on CPI x (1 + link x real GDP
                               growth). Without the regional data (unit tests): 2024 flow x that
                               index x (1 + weight x growth of the population under 18)
  federal -> local             reimbursement share of local social assistance + rest with GDP
  regions & comm. -> local     education part follows education spending, the rest nominal GDP
  all other flows              nominal GDP (elasticity `transfer.gdp_elasticity`)

The federated entities inside "regions & communities" (Flemish Community, Walloon Region, French
Community, Brussels-Capital Region) are split by `belsim.regional` (engine step 8b), summing
exactly to the aggregate computed here.

Entity accounts: the revenue attribution of the revenue module does not have to reproduce the
subsector accounts exactly; a residual per entity (summing to zero, fitted on the actual 2024
entity balances, then constant in % of GDP) is added to entity revenue: `entity_residual.<e>`.

`debt.<e>` are the subsector debts as published (consolidated within the subsector only); their
sum exceeds the Maastricht debt by the holdings between subsectors, reported (negative) as
`debt.intragov_consolidation` and kept constant.

`net_expenditure_growth` approximates the EU governance indicator: growth of primary expenditure
net of cyclical unemployment spending and EU-funded expenditure, minus the yearly increment of
discretionary revenue measures. The latter is approximated by the change in (scenario revenue -
baseline revenue), which also contains induced revenue effects; one-offs are not identified.
"""
from __future__ import annotations

import numpy as np

from ..core.dims import ENTITIES
from .expenditure import data

NS = "balance"
FED, SS, RC, LOC = ENTITIES
ADJ_KEYS = {(FED, SS): "fed_to_ss", (FED, RC): "fed_to_regions", (FED, LOC): "fed_to_local",
            (RC, LOC): "regions_to_local"}
EDUC_LINES = ("educ_primary", "educ_secondary", "educ_tertiary", "educ_other")


def _norm(sh: dict | None) -> dict[str, float] | None:
    if not sh:
        return None
    vals = {e: max(float(sh.get(e, 0.0)), 0.0) for e in ENTITIES}
    tot = sum(vals.values())
    return {e: v / tot for e, v in vals.items()} if tot > 0 else None


def _by_entity(amounts: dict[str, float], lines: dict, default: dict[str, float]) -> dict[str, float]:
    out = {e: 0.0 for e in ENTITIES}
    for k, v in amounts.items():
        meta = lines.get(k)
        sh = _norm(getattr(meta, "entity_shares", None)) or default
        for e in ENTITIES:
            out[e] += v * sh[e]
    return out


def _prefix_sum(d: dict[str, float], prefix: str) -> float:
    return float(sum(v for k, v in d.items() if k == prefix or k.startswith(prefix + "_")
                     or k.startswith(prefix + ".")))


def run(state, prev, p, ctx) -> None:
    from . import expenditure as exp_mod
    try:
        from . import revenue as rev_mod
        rev_lines = getattr(rev_mod, "LINES", None) or {}
    except Exception:  # revenue package unavailable: fall back to the default split
        rev_lines = {}

    sc, T0 = data.scalars(), data.transfers_2024()
    m = state.macro
    gdp = m["gdp_nominal"]
    gap = m.get("output_gap", m.get("demand_gap", 0.0))   # total gap: exogenous cycle + demand feedback
    cpi, hi = m.get("cpi", 1.0), m.get("health_index", 1.0)
    gdp_real = m.get("gdp_real") or gdp / cpi
    first = prev is None
    px = prev.extra.get(NS) if not first else None
    bs = getattr(ctx, "base_state", None) if ctx is not None else None
    bx = bs.extra.get(NS) if (first and bs is not None) else None
    ex = state.extra.get("expenditure") or {}

    rev = float(sum(state.revenue.values()))
    exp = float(sum(state.expenditure.values()))
    interest = float(state.expenditure.get("interest", 0.0))
    bal = rev - exp

    # ---------------- own revenue / final expenditure per entity ----------------
    default_rev = _norm(p["transfer.default_revenue_split"])
    default_exp = _norm({e: sc[f"own_expenditure_{e}_2024"] for e in ENTITIES})
    own_rev = _by_entity(state.revenue, rev_lines, default_rev)
    own_exp = _by_entity(state.expenditure, getattr(exp_mod, "LINES", {}) or {}, default_exp)
    # What the region / Community keyed levers add to a line is spending of the regions and
    # communities, not of the other levels in proportion to their 2024 share of the line.
    # (Education excepted: schools run by local authorities are funded through the transfer rule.)
    for lid, d in (ex.get("regional_delta") or {}).items():
        amount = float(sum(d.values()))
        if lid in EDUC_LINES or amount == 0.0:
            continue
        sh = _norm(getattr((getattr(exp_mod, "LINES", {}) or {}).get(lid), "entity_shares", None)) or default_exp
        for e in ENTITIES:
            own_exp[e] -= amount * sh[e]
        own_exp[RC] += amount

    # Spending paths by entity (`entity.*_factor`): what an entity's budget path adds to or takes
    # from a line is spending of that entity (the four federated entities: of the regions and
    # communities), not of every level in proportion to its 2024 share of the line.
    for lid, d in (ex.get("entity_delta") or {}).items():
        amount = float(sum(d.values()))
        if amount == 0.0:
            continue
        sh = _norm(getattr((getattr(exp_mod, "LINES", {}) or {}).get(lid), "entity_shares", None)) or default_exp
        for e in ENTITIES:
            own_exp[e] -= amount * sh[e]
        for k, v in d.items():
            own_exp[k if k in ENTITIES else RC] += v

    # ---------------- drivers of the transfer rules ----------------
    vat, pit = _prefix_sum(state.revenue, "vat"), _prefix_sum(state.revenue, "pit")
    educ = float(sum(state.expenditure.get(k, 0.0) for k in EDUC_LINES))
    sa_meta = getattr(exp_mod, "LINES", {}).get("social_assistance")
    sa_local = state.expenditure.get("social_assistance", 0.0) * (
        (_norm(getattr(sa_meta, "entity_shares", None)) or default_exp)[LOC])
    pop_u18 = float(state.pop[:18].sum())

    src = px if px is not None else bx
    if src is not None:
        refs = src["refs"]
    else:
        refs = {"gdp": gdp, "vat": vat, "pit": pit, "educ": educ, "sa_local": sa_local}
    if first:
        grant_idx = sfa_idx = sfa_core = 1.0
    else:
        pm = prev.macro
        g_real = gdp_real / (pm.get("gdp_real") or pm["gdp_nominal"] / pm.get("cpi", 1.0)) - 1.0
        grant_idx = px["grant_idx"] * hi / pm.get("health_index", 1.0) * (1 + p["transfer.ss_grant_real_growth"])
        g_u18 = pop_u18 / px["pop_u18"] - 1.0 if px["pop_u18"] > 0 else 0.0
        core = cpi / pm.get("cpi", 1.0) * (1 + p["transfer.sfa_real_growth_link"] * g_real)
        sfa_core = px.get("sfa_core", px["sfa_idx"]) * core          # prices and part of real growth
        sfa_idx = px["sfa_idx"] * core * (1 + p["transfer.sfa_demography_weight"] * g_u18)

    # Special Financing Act transfers built from the entity-level mechanisms (belsim.regional):
    # needs the PIT by region of the revenue module and the regional parameters; otherwise the
    # aggregate rule below (2024 flow x sfa_idx) applies.
    grant_e = None
    if p.get("regional.sfa_grants_2024") is not None:
        from ..regional import government as _gov
        keys = _gov.grant_keys(state)
        if keys is not None and ("sfa_keys0" in refs or src is None):
            if "sfa_keys0" not in refs:
                refs["sfa_keys0"] = {k: np.array(v, dtype=float) for k, v in keys.items()}
            b_dutch = float(p.get("regional.brussels_dutch_share", 0.2))
            g_idx, g_trans = _gov.grants_by_mechanism(p, keys, refs["sfa_keys0"], sfa_core, state.year, b_dutch)
            if "sfa_k0" not in refs:        # 2024: the ESA flow is slightly wider than the budget tables
                f0 = T0.get((FED, RC), 0.0)
                refs["sfa_k0"] = (f0 - float(g_trans.sum())) / float(g_idx.sum()) if float(g_idx.sum()) else 1.0
            grant_e = g_idx * refs["sfa_k0"] + g_trans

    i_gdp = (gdp / refs["gdp"]) ** p["transfer.gdp_elasticity"]
    i_vat = vat / refs["vat"] if refs["vat"] > 0 and vat > 0 else i_gdp
    i_pit = pit / refs["pit"] if refs["pit"] > 0 and pit > 0 else i_gdp
    i_educ = educ / refs["educ"] if refs["educ"] > 0 and educ > 0 else i_gdp
    a_v, a_p = p["transfer.ss_altfin_vat_share"], p["transfer.ss_altfin_pit_share"]
    es = sc["regions_to_local_education_share_2024"]
    k_sa = p["transfer.fed_local_assistance_reimbursement"]
    adj = p["transfer.adjustment"]
    lvl = p.get("transfer.level_factor") or {}

    T: dict[tuple[str, str], float] = {}
    for pair, f0 in T0.items():
        if pair == (FED, SS):
            v = f0 * (a_v * i_vat + a_p * i_pit + (1 - a_v - a_p) * grant_idx)
        elif pair == (FED, RC):
            v = float(grant_e.sum()) if grant_e is not None else f0 * sfa_idx
        elif pair == (FED, LOC):
            sa0 = min(k_sa * refs["sa_local"], f0)
            v = (sa0 / refs["sa_local"] * sa_local if refs["sa_local"] > 0 else 0.0) + (f0 - sa0) * i_gdp
        elif pair == (RC, LOC):
            v = f0 * (es * i_educ + (1 - es) * i_gdp)
        else:
            v = f0 * i_gdp
        if pair in ADJ_KEYS:
            v = v * float(lvl.get(ADJ_KEYS[pair], 1.0)) + float(adj.get(ADJ_KEYS[pair], 0.0))
        T[pair] = v

    def net(e: str) -> float:
        return sum(v for (a, b), v in T.items() if b == e) - sum(v for (a, b), v in T.items() if a == e)

    # ---------------- entity calibration residual and social-security balancing ----------------
    if "resid_pct" not in refs:
        model_bal = {e: own_rev[e] - own_exp[e] + net(e) for e in ENTITIES}
        gap_tot = sc["balance_2024"] - bal
        w = {e: (own_rev[e] / rev if rev else 0.25) for e in ENTITIES}
        refs["resid_pct"] = {e: (sc[f"balance_{e}_2024"] - model_bal[e] - gap_tot * w[e]) / gdp for e in ENTITIES}
        # the published subsector balances may not add up exactly to the S13 balance: spread the rest
        rest = sum(refs["resid_pct"].values())
        for e in ENTITIES:
            refs["resid_pct"][e] -= rest / len(ENTITIES)
        refs["ss_target_pct"] = (model_bal[SS] + refs["resid_pct"][SS] * gdp) / gdp
    resid = {e: refs["resid_pct"][e] * gdp for e in ENTITIES}
    pre_ss = own_rev[SS] - own_exp[SS] + net(SS) + resid[SS]
    balancing = p["transfer.ss_balancing_share"] * (refs["ss_target_pct"] * gdp - pre_ss)
    T[(FED, SS)] += balancing

    recv = {e: sum(v for (a, b), v in T.items() if b == e) for e in ENTITIES}
    paid = {e: sum(v for (a, b), v in T.items() if a == e) for e in ENTITIES}
    rev_e = {e: own_rev[e] + recv[e] + resid[e] for e in ENTITIES}
    exp_e = {e: own_exp[e] + paid[e] for e in ENTITIES}
    bal_e = {e: rev_e[e] - exp_e[e] for e in ENTITIES}

    # ---------------- debt ----------------
    if first:
        base_bal = bs.fiscal.get("balance", bal) if bx is not None else bal
        base_bal_e = bx["bal_e"] if bx is not None else bal_e
        debt = sc["debt_2024"] - (bal - base_bal)
        debt_e = {e: sc[f"debt_{e}_2024"] - (bal_e[e] - base_bal_e[e]) for e in ENTITIES}
        sfa = debt - (sc["debt_2023"] - bal)
    else:
        sfa = p["debt.stock_flow_adjustment_pct_gdp"] / 100.0 * gdp
        debt = prev.fiscal["debt"] - bal + sfa
        prev_e = {e: prev.fiscal[f"debt.{e}"] for e in ENTITIES}
        tot_prev = sum(prev_e.values())
        debt_e = {e: prev_e[e] - bal_e[e] + (sfa * prev_e[e] / tot_prev if tot_prev else 0.0) for e in ENTITIES}

    # average interest rate on the long-term debt at the end of the year (used for next year's interest)
    s_short = p["debt.short_term_share"]
    roll = 1.0 / max(p["debt.avg_maturity_years"], 1.0)
    i10 = m.get("interest_rate_10y", 0.03)
    if "rate_long_prev" in ex:
        keep = (1 - roll) * (1 - s_short) * ex["debt_prev"]
        long_now = (1 - s_short) * debt
        new = long_now - keep
        rate_long = (keep * ex["rate_long_prev"] + new * i10) / long_now if new > 0 and long_now > 0 \
            else ex["rate_long_prev"]
    else:
        rate_long = interest / debt if debt > 0 else i10

    # ---------------- EU fiscal-rule indicators ----------------
    primary = exp - interest
    unemp = ex.get("unemployment_full", state.expenditure.get("unemployment", 0.0))
    e_u = p["exp.cyclical_unemployment_semi_elasticity"]
    den = 1 - e_u * gap
    cyclical_unemp = unemp * (-e_u * gap) / den if den > 0.1 else 0.0
    eu_funded = p["exp.eu_funded_pct_gdp"] / 100.0 * gdp
    net_exp = primary - cyclical_unemp - eu_funded
    drm = rev - bs.fiscal["revenue_total"] if (bs is not None and "revenue_total" in bs.fiscal) else 0.0
    if first:
        net_prev = sc["te_2023"] - sc["interest_2023"] - sc["eu_funded_2023"]
        drm_prev = 0.0
    else:
        net_prev, drm_prev = px["net_expenditure"], px["drm"]
    net_growth = (net_exp - (drm - drm_prev)) / net_prev - 1.0 if net_prev else 0.0
    # ratios to GDP: GDP at market prices of the run (differs from `gdp_nominal` only when a
    # scenario changes indirect taxes on consumption)
    gdp_mp = float(m.get("gdp_nominal_market_prices", gdp) or gdp)
    bal_pct = bal / gdp_mp * 100.0

    f = {
        "revenue_total": rev, "expenditure_total": exp, "primary_expenditure": primary, "interest": interest,
        "balance": bal, "primary_balance": bal + interest, "debt": debt,
        "balance_pct_gdp": bal_pct, "primary_balance_pct_gdp": (bal + interest) / gdp_mp * 100.0,
        "debt_pct_gdp": debt / gdp_mp * 100.0, "revenue_pct_gdp": rev / gdp_mp * 100.0,
        "expenditure_pct_gdp": exp / gdp_mp * 100.0, "interest_pct_gdp": interest / gdp_mp * 100.0,
        "net_expenditure_growth": net_growth,
        "net_primary_expenditure": net_exp,
        "structural_balance_pct_gdp": bal_pct - p["exp.budget_semi_elasticity"] * gap * 100.0,
        "stock_flow_adjustment": sfa,
        "transfers_total": float(sum(T.values())),
        "debt.intragov_consolidation": debt - sum(debt_e.values()),
    }
    for e in ENTITIES:
        f[f"revenue.{e}"] = rev_e[e]
        f[f"expenditure.{e}"] = exp_e[e]
        f[f"balance.{e}"] = bal_e[e]
        f[f"debt.{e}"] = debt_e[e]
        f[f"balance_pct_gdp.{e}"] = bal_e[e] / gdp_mp * 100.0
        f[f"own_revenue.{e}"] = own_rev[e]
        f[f"own_expenditure.{e}"] = own_exp[e]
        f[f"entity_residual.{e}"] = resid[e]
    for (a, b), v in T.items():
        f[f"transfer.{a}_to_{b}"] = v
    f["transfer.ss_balancing_grant"] = balancing
    state.fiscal = f
    # disposable income of households from the flows of the year (wages, self-employed and
    # capital income, cash benefits, less direct taxes and contributions): known only once
    # expenditure and revenue are, hence published here
    try:
        from ..behaviour.demand import household_income_level
        y_hh = household_income_level(state, p)
    except Exception:                      # partial states in unit tests, sibling module broken
        y_hh = None
    if y_hh is not None:
        m["household_income_after_tax"] = y_hh
    state.extra[NS] = {
        "refs": refs, "grant_idx": grant_idx, "sfa_idx": sfa_idx, "sfa_core": sfa_core, "pop_u18": pop_u18,
        "grant_e": grant_e,
        "rate_long": rate_long, "net_expenditure": net_exp, "drm": drm, "bal_e": bal_e,
        "implicit_rate": interest / ex["debt_prev"] if ex.get("debt_prev") else 0.0,
    }
