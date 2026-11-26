"""Short-run demand (Keynesian) feedback of the scenario's fiscal impulse.

After the first pass of the year (computed with a zero demand gap), the scenario is compared
with the baseline. Three parts, each in fractions of baseline GDP:

    direct  = m_purch x dPurchases(volume) + competitiveness
    firms   = - sum of m_firms(kind) x (dRevenue from firms - their share of sigma x dGDP)
    income  = m_income x (real disposable income of households, scenario - baseline, net of
              the change in real GDP)
              + m_income x (dOther transfers - dOther revenue net of its share of sigma x dGDP)

* Purchases are taken in volume: scenario purchases are deflated with their price relative to
  the baseline (wage index for the pay share, health index for the rest; health care, whose
  fees are set before the year starts, with last year's health index), so that indexation
  after a tax or wage shock is not mistaken for more public demand. The price of goods,
  services and investment bought from firms is neither demand nor household income.
* Household income (`household_income`) is built from the model's own flows: wages of private
  employees, income of the self-employed, capital income received, cash social benefits, less
  the direct taxes and social contributions of households
  (`beh.demand.household_direct_tax_esa_prefixes`). Public pay enters at the baseline's
  head count and the scenario's pay per head: a change in public staff is a purchase, a change
  in pay scales is income. The
  income of the scenario is deflated with its consumer prices relative to the baseline's,
  tax effect included, so that whoever is not (fully) indexed loses purchasing power after a
  rise in indirect taxes: capital income, and wages once bargaining has taken back part of the
  indexation. Income that merely follows a supply-side change in real GDP (pass 1 has no
  demand gap) is not an impulse: the baseline income times the change in real GDP is taken
  out, like the revenue removed with `sigma x dGDP` for firms and the induced purchases
  (`beh.demand.purchases_output_elasticity`). Jobs lost or gained at an output that changes
  less than employment (labour-cost response) do change household income.
* Transfers that are not cash benefits of households (subsidies, capital transfers, transfers
  abroad) and revenue that is neither a direct tax of households nor paid by firms (taxes on
  production and imports, sales) keep the same multiplier on their nominal change. Indirect
  taxes that are in consumer prices are taken out of that revenue (macro: GDP at the
  scenario's market prices minus GDP): they act through the deflator of household income.
* Taxes paid by firms (employer contributions, corporate tax: `beh.demand.firm_tax_keywords`)
  have their own multiplier by kind (`beh.demand.mult_taxes_firms`): corporate tax is mostly
  absorbed by profits; employer contributions reach output through channels that are modelled
  (jobs, wages, prices and dividends in household income; competitiveness), so they have none.
* Households adjust their spending to an income change gradually:
      income_eff_t = income_eff_{t-1} + mu x (income_t - income_eff_{t-1}),
  mu = `beh.demand.income_first_year_share`; `m_income` (`beh.demand.mult_transfers`: the
  marginal propensity to consume net of the import leakage) is the multiplier once spending has
  adjusted.

The stance is `direct + firms + income_eff` and the output gap follows

    gap_t = rho x gap_{t-1} + (stance_t - stance_{t-1})

so the effect of a permanent measure fades geometrically (rho = `beh.demand.persistence`) as
output returns to its supply-determined level. Competitiveness: a change in labour cost per unit
of output relative to the baseline (wages and employer contributions) adds
`- dyn.wage_cost.net_export_output_elasticity x ln(RULC scenario / RULC baseline)` (real unit
labour cost including contributions, published by macro.prices): net exports react first, the
lasting effect comes from labour demand (responses.py). Everything is carried in
`state.extra["behaviour"]["demand"]`; the stance is always the pass-1 (ex-ante) one, so the
second pass of the engine cannot feed back on it.
"""
from __future__ import annotations

import math

from ..labour.dynparams import par as dyn_par
from ._params import enabled, par, scale
from .responses import _wage_cost_gap


def _line_meta():
    try:
        from ..fiscal import expenditure
        lines = getattr(expenditure, "LINES", None)
        return lines if isinstance(lines, dict) else {}
    except Exception:                      # sibling module temporarily broken
        return {}


def _revenue_meta():
    try:
        from ..fiscal import revenue
        lines = getattr(revenue, "LINES", None)
        return lines if isinstance(lines, dict) else {}
    except Exception:                      # sibling module temporarily broken
        return {}


def _text(name: str, meta) -> str:
    return " ".join(str(x) for x in (name, getattr(meta, "label", ""), getattr(meta, "group", ""))).lower()


def _classify(name: str, meta, p) -> str:
    """'interest' | 'purchases' | 'transfers' for one expenditure line."""
    esa = str(getattr(meta, "esa", "") or "").upper()
    text = _text(name, meta)
    if any(kw in text or kw.upper() == esa for kw in par(p, "beh.demand.interest_keywords")):
        return "interest"
    tokens = [tok for tok in esa.replace("+", " ").replace(",", " ").replace("/", " ").split() if tok]
    if any(tok.startswith(str(pre).upper()) for tok in tokens for pre in par(p, "beh.demand.purchase_esa_prefixes")):
        return "purchases"
    if any(kw in text for kw in par(p, "beh.demand.purchase_keywords")):
        return "purchases"
    return "transfers"


def _paid_by_firms(name: str, meta, p) -> bool:
    text = _text(name, meta)
    return any(kw in text for kw in par(p, "beh.demand.firm_tax_keywords"))


def _firm_multiplier(name: str, meta, p) -> float:
    """Multiplier of a revenue line paid by firms: the entry of `beh.demand.mult_taxes_firms`
    whose keyword matches the line (a plain number applies to every line)."""
    m = par(p, "beh.demand.mult_taxes_firms")
    if not isinstance(m, dict):
        return float(m)
    text = _text(name, meta)
    return next((float(v) for kw, v in m.items() if str(kw) in text), 0.0)


def _index_ratio(state, base, key: str) -> float:
    """Scenario / baseline value of a price or wage index (1 if not available)."""
    try:
        r = float(state.macro[key]) / float(base.macro[key])
    except (KeyError, TypeError, ZeroDivisionError, AttributeError):
        return 1.0
    return r if math.isfinite(r) and r > 0.0 else 1.0


def _household_transfer(meta, p) -> bool:
    """True for an expenditure line that is a cash benefit of households (ESA code)."""
    esa = str(getattr(meta, "esa", "") or "").upper()
    tokens = [tok for tok in esa.replace("+", " ").replace(",", " ").replace("/", " ").split() if tok]
    pre = [str(x).upper() for x in par(p, "beh.demand.household_transfer_esa_prefixes")]
    return any(tok == x or (tok.startswith(x) and not tok[len(x):len(x) + 1].isdigit())
               for tok in tokens for x in pre)


def _direct_tax(meta, p) -> bool:
    """True for a revenue line that is a direct tax or social contribution (ESA code)."""
    esa = str(getattr(meta, "esa", "") or "").upper()
    tokens = [tok for tok in esa.replace("+", " ").replace(",", " ").replace("/", " ").split() if tok]
    pre = [str(x).upper() for x in par(p, "beh.demand.household_direct_tax_esa_prefixes")]
    return any(tok.startswith(x) for tok in tokens for x in pre)


def household_income(state, benefits: float, taxes: float) -> float | None:
    """Disposable income of households from the model's own flows (EUR mn, current prices),
    without public pay: wages of private employees + income of the self-employed + capital
    income received + cash benefits - direct taxes and contributions of households. None if the
    labour and macro keys are not there (unit tests with partial states)."""
    try:
        lab, mac = state.labour, state.macro
        cap = (state.extra.get("macro") or {}).get("capital_income_received")
        if cap is None:
            cap = mac["household_capital_income"]
        # income of the self-employed before the behavioural factor on what they declare
        # (`adjust["self_employed_income_factor"]` moves the tax base, not what they earn)
        declared = float((state.adjust or {}).get("self_employed_income_factor", 1.0) or 1.0)
        y = float(lab["wage_bill_private"]) + float(lab["self_employed_income"]) / declared + float(cap)
    except (KeyError, TypeError, AttributeError, ZeroDivisionError):
        return None
    y += benefits - taxes
    return y if math.isfinite(y) else None


def household_income_level(state, p) -> float | None:
    """Disposable income of households of the year from the model's flows, with the public pay
    actually paid (EUR mn, current prices): the published series
    `macro.household_income_after_tax` (balance module). The demand block uses the same flows
    with public staff held at the baseline's."""
    if not (state.revenue and state.expenditure):
        return None
    meta, rmeta = _line_meta(), _revenue_meta()
    ben = sum(float(v) for k, v in state.expenditure.items()
              if _classify(k, meta.get(k), p) == "transfers" and _household_transfer(meta.get(k), p))
    tax = sum(float(v) for k, v in state.revenue.items()
              if not _paid_by_firms(k, rmeta.get(k), p) and _direct_tax(rmeta.get(k), p))
    y = household_income(state, ben, tax)
    if y is None:
        return None
    lab = state.labour
    return y + float(lab.get("wage_bill_public_actual", lab.get("wage_bill_public", 0.0)) or 0.0)


def _tax_in_prices(state) -> float:
    """Value of the scenario's indirect taxes that is in consumer prices relative to the
    baseline's legislation (macro: GDP at the scenario's market prices minus GDP)."""
    try:
        v = float(state.macro["gdp_nominal_market_prices"]) - float(state.macro["gdp_nominal"])
    except (KeyError, TypeError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _purchase_prices(state, base, p, lagged: tuple | None = None) -> tuple[float, float, float]:
    """Price of public purchases in the scenario relative to the baseline, as (pay factor,
    factor of the rest, factor of the lines indexed a year late): wage index for the pay share,
    health index for the rest; the lines of `beh.demand.lagged_price_lines` follow last year's
    health index for the share `health.indexation_lag_share` (`lagged` = last year's states)."""
    rw, rh = _index_ratio(state, base, "wage_index"), _index_ratio(state, base, "health_index")
    w = min(1.0, max(0.0, float(par(p, "beh.demand.purchases_pay_share"))))
    late = rh
    share = p.get("health.indexation_lag_share") if p is not None else None
    share = min(1.0, max(0.0, float(share))) if share is not None else 0.0
    if share != 0.0:
        rh_prev = _index_ratio(lagged[0], lagged[1], "health_index") if lagged else 1.0
        if rh_prev != rh:
            late = rh_prev ** share * rh ** (1.0 - share)
    return (rw ** w if rw != 1.0 else 1.0), (rh ** (1.0 - w) if rh != 1.0 else 1.0), late


def stance(state, base, p, lagged: tuple | None = None) -> tuple[dict, dict]:
    """Multiplier-weighted fiscal expansion vs baseline (fractions of baseline GDP), unscaled:
    ({"direct", "firms", "income"}, detail). `lagged` = (scenario, baseline) states of the
    previous year, for the purchases whose prices are indexed a year late."""
    zero = {"direct": 0.0, "firms": 0.0, "income": 0.0}
    gdp_b = float(base.macro.get("gdp_nominal", 0.0))
    if not (math.isfinite(gdp_b) and gdp_b > 0.0):
        return zero, {}
    d_gdp = float(state.macro.get("gdp_nominal", gdp_b)) - gdp_b
    cyc = float(par(p, "beh.demand.budget_semi_elasticity")) * d_gdp
    out = dict(zero)

    if state.revenue and state.expenditure and base.revenue and base.expenditure:
        # revenue: firms / households
        rmeta = _revenue_meta()
        d_firms = rev_firms_b = rev_hh_s = rev_hh_b = d_other = rev_other_b = 0.0
        firm_lines = []                         # (multiplier, change, baseline amount)
        for k in state.revenue.keys() | base.revenue.keys():
            a, b = float(state.revenue.get(k, 0.0)), float(base.revenue.get(k, 0.0))
            if _paid_by_firms(k, rmeta.get(k), p):
                rev_firms_b += b
                firm_lines.append((_firm_multiplier(k, rmeta.get(k), p), a - b, b))
                if a != b:
                    d_firms += a - b
            elif _direct_tax(rmeta.get(k), p):
                rev_hh_s += a
                rev_hh_b += b
            else:
                rev_other_b += b
                if a != b:
                    d_other += a - b
        d_hh = rev_hh_s - rev_hh_b
        rev_b = rev_firms_b + rev_hh_b + rev_other_b
        share_f = min(1.0, max(0.0, rev_firms_b / rev_b)) if rev_b > 0.0 else 0.0
        cyc_f = cyc * share_f
        cyc_o = cyc * (min(1.0, max(0.0, rev_other_b / rev_b)) if rev_b > 0.0 else 0.0)
        # other revenue: the indirect taxes that are in consumer prices act through the deflator
        d_other -= _tax_in_prices(state) - _tax_in_prices(base)

        # expenditure: purchases in volume; cash benefits of households; other transfers. The
        # price change of purchases relative to the baseline is no demand of its own (the pay
        # part is income of households, counted below with their other income)
        parts = {"purchases": 0.0, "transfers": 0.0, "interest": 0.0}
        indexation = ben_s = ben_b = 0.0
        f_pay, f_rest, f_late = _purchase_prices(state, base, p, lagged)
        priced = f_pay != 1.0 or f_rest != 1.0 or f_late != 1.0
        late_lines = set(par(p, "beh.demand.lagged_price_lines")) if priced else set()
        meta = _line_meta()
        for k in state.expenditure.keys() | base.expenditure.keys():
            a, b = float(state.expenditure.get(k, 0.0)), float(base.expenditure.get(k, 0.0))
            kind = _classify(k, meta.get(k), p)
            if kind == "purchases" and priced:
                if k in late_lines:
                    parts[kind] += a / f_late - b
                    indexation += a - a / f_late
                else:
                    parts[kind] += a / (f_pay * f_rest) - b
                    indexation += a - a / f_pay
            elif kind == "transfers" and _household_transfer(meta.get(k), p):
                ben_s += a
                ben_b += b
            elif a != b:
                parts[kind] += a - b
        # public purchases that merely follow a supply-side change in real GDP (pass 1 has no
        # demand gap) are not a fiscal impulse, like the revenue removed with sigma x dGDP
        induced = g_real = pay_b = 0.0
        try:
            g_s, g_b = float(state.macro["gdp_real"]), float(base.macro["gdp_real"])
            if g_s != g_b and g_s > 0.0 and g_b > 0.0:
                p_b = sum(float(v) for k, v in base.expenditure.items()
                          if _classify(k, meta.get(k), p) == "purchases")
                induced = float(par(p, "beh.demand.purchases_output_elasticity")) * p_b * math.log(g_s / g_b)
                g_real = g_s / g_b - 1.0
        except (KeyError, TypeError, ValueError):
            induced = g_real = pay_b = 0.0
        parts["purchases"] -= induced

        # households: real disposable income of the scenario minus the baseline's, net of the
        # income that follows the supply-side change in real GDP
        y_b = household_income(base, ben_b, rev_hh_b)
        y_s = household_income(state, ben_s, rev_hh_s)
        m_inc = float(par(p, "beh.demand.mult_transfers"))
        if y_b is not None and y_s is not None:
            price = _index_ratio(state, base, "cpi")
            # public pay: the baseline's staff at the pay per head of each run
            pay_b = float(base.labour.get("wage_bill_public_actual", base.labour.get("wage_bill_public", 0.0)) or 0.0)
            w_s, w_b = float(state.labour.get("avg_wage_public", 0.0) or 0.0), float(base.labour.get("avg_wage_public", 0.0) or 0.0)
            y_b += pay_b
            y_s += pay_b * (w_s / w_b) if (w_s != w_b and w_s > 0.0 and w_b > 0.0) else pay_b
            d_income = (y_s / price if price != 1.0 else y_s) - y_b
            if g_real != 0.0:
                d_income -= y_b * g_real
            detail_income = {"household_income_baseline": y_b, "d_household_income_real": d_income,
                             "consumer_price_ratio": price}
        else:
            # partial states: nominal flows (benefits + indexation - taxes net of the cyclical part)
            d_income = (ben_s - ben_b) + indexation - (d_hh - (cyc - cyc_f - cyc_o))
            detail_income = {}
        out["direct"] = float(par(p, "beh.demand.mult_purchases")) * parts["purchases"] / gdp_b
        out["firms"] = -sum(m * (d - (cyc * b / rev_b if rev_b > 0.0 else 0.0))
                            for m, d, b in firm_lines if m != 0.0) / gdp_b + 0.0
        out["income"] = m_inc * (d_income + parts["transfers"] - (d_other - cyc_o)) / gdp_b
        detail = {"d_revenue": d_firms + d_hh + d_other, "d_revenue_firms": d_firms, "d_revenue_other": d_other, "d_purchases": parts["purchases"],
                  "d_transfers": parts["transfers"] + ben_s - ben_b, "indexation_public_pay": indexation,
                  "cyclical": cyc, **detail_income}
    else:
        # fallback: one average multiplier on the change in the primary balance
        try:
            d_pb = float(state.fiscal["primary_balance"]) - float(base.fiscal["primary_balance"])
        except (KeyError, TypeError):
            return zero, {}
        out["direct"] = -float(par(p, "beh.demand.mult_default")) * (d_pb - cyc) / gdp_b
        detail = {"d_primary_balance": d_pb, "cyclical": cyc}
    # competitiveness: labour cost per unit of output including employer contributions (the
    # wage-only index if macro does not publish the other one)
    key = "labour_cost_competitiveness_index"
    if key not in state.macro or key not in base.macro:
        key = "real_unit_labour_cost"
    comp = -float(dyn_par(p, "dyn.wage_cost.net_export_output_elasticity")) * _wage_cost_gap(state, base, key)
    if comp != 0.0:
        out["direct"] += comp
        detail["competitiveness"] = comp
    if not all(math.isfinite(v) for v in out.values()):
        return zero, {}
    return out, detail


def demand_gap(state, prev, p, ctx) -> float:
    mem = state.extra.setdefault("behaviour", {})
    if getattr(ctx, "pass_no", 1) == 2 and "demand" in mem:
        return mem["demand"]["gap"]         # never re-measure after the feedback pass

    base = ctx.base_state
    if base is None or not enabled(p) or float(par(p, "beh.demand.enabled")) == 0.0:
        mem["demand"] = {"stance": 0.0, "income_stance": 0.0, "gap": 0.0}
        return 0.0

    prev_d = ((prev.extra.get("behaviour") or {}).get("demand") if prev is not None else None) or {}
    lagged = None
    try:
        if prev is not None and ctx.baseline.has(prev.year):
            lagged = (prev, ctx.baseline.at(prev.year))
    except Exception:                      # fake contexts in unit tests
        lagged = None
    parts, detail = stance(state, base, p, lagged)
    sc = scale(p)
    mu = min(1.0, max(0.0, float(par(p, "beh.demand.income_first_year_share"))))
    inc_prev = float(prev_d.get("income_stance", 0.0))
    inc = inc_prev + mu * (parts["income"] * sc - inc_prev)
    s = (parts["direct"] + parts["firms"]) * sc + inc
    cap = abs(float(par(p, "beh.demand.max_gap")))
    rho = min(1.0, max(0.0, float(par(p, "beh.demand.persistence"))))
    gap = rho * float(prev_d.get("gap", 0.0)) + (s - float(prev_d.get("stance", 0.0)))
    gap = min(cap, max(-cap, gap)) + 0.0    # + 0.0 turns -0.0 into 0.0
    mem["demand"] = {"stance": s, "income_stance": inc, "gap": gap, **detail}
    return gap
