"""Macro-economy: prices and indices (`prices`), then output, expenditure, income and
tax-base proxies (`output`). See docs/ARCHITECTURE.md for the contract.

Supply-side model: real value added of a sector = employment x productivity per worker;
productivity grows at the trend rate `macro.productivity_growth`. 2024 reproduces the national
accounts (Eurostat nama_10_gdp / nama_10_a64 / nasa_10_nf_tr): every aggregate is
`driver x calibration factor`, the factors being fitted in the base year and carried unchanged
in `state.extra["macro"]["calib"]`.

Indirect taxes and prices (scenario runs only). `prices` compares the indirect-tax rate of each
consumption category (`revenue.calc.consumption_tax_rates`) under the scenario and under the
baseline legislation and passes the difference into the CPI with the pass-through rates of
`beh.cons.pass_through`; categories outside the health index do not move the health index, so
they do not feed automatic indexation of wages and benefits. Convention for values: the GDP
deflator, nominal GDP and `cons.<cat>` stay valued at BASELINE indirect-tax rates (revenue
strips the baseline rates from `cons.<cat>`, `cons_reflects_tax_changes` is not set); the
volume response to the price change is `adjust["cons_factor"]` (behaviour), computed at
unchanged nominal income, and the extra nominal income from indexation reaches consumption
through wages and benefits. Nothing is counted twice. The convention is kept because every rule
of the budget that follows nominal GDP (spending norms in % of GDP, tax bases) would otherwise
read a VAT rise as more activity. What a published ratio needs is the denominator:
`macro["gdp_nominal_market_prices"]` adds the scenario's indirect taxes on household
consumption (price change of each category x its value) and equals `gdp_nominal` whenever the
indirect taxes are the baseline's.

Slack and prices (scenario runs only). A demand gap moves the price level a little each year it
lasts (`dyn.prices.demand_gap_sensitivity` x last year's gap, cumulated; `cpi_demand_effect`):
consumer prices, the health index (hence indexation) and the price of output alike.

Output prices lead consumer prices. After a change in wage costs caused by taxes (employer
contributions, indexation of a tax-induced price rise, bargained shifts) the price of domestic
output moves at once to the level consumer prices reach gradually (`dyn.wage_cost.deflator_lead`),
so profits carry less of the shock in the first year. Indexation skipped or added relative to
the baseline (an index jump) is the exception: output prices follow consumer prices
(`dyn.wage_cost.deflator_lead_indexation`), so margins widen first and corporate tax rises.

Real unit labour cost. `macro["real_unit_labour_cost"]` = wage index / (underlying consumer
prices x the productivity level wages are meant to follow). Its ratio to the baseline is the
wage-cost gap that behaviour turns into labour demand. `macro["labour_cost_competitiveness_index"]`
is the same index times the change in employer contributions per euro of wage relative to the
baseline's legislation: behaviour turns its ratio to the baseline into the short-run
competitiveness effect (a contribution cut improves competitiveness even though gross wages rise
a little).

Wage bargaining (scenario runs only). Gross wages react, with a year's delay, to changes in the
tax wedge on labour relative to the baseline (`dyn.wage_bargaining.*`): they fall by part of a
rise in employer contributions, rise by part of the loss in net wage after a rise in employee
contributions or income tax, and lose again part of the indexation that first compensated a
rise in indirect taxes in full. The resulting shift of the wage index (`macro["wage_bargaining_index"]`, 1 in the baseline)
enters the real unit labour cost, hence labour demand, prices and competitiveness. A change in
employer contributions is also booked in compensation of employees, so that profits (and the
corporate tax base) carry the part that is not passed on.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from ..core.dims import BASE_YEAR, CONS_CATS, J, SECTORS
from ..labour.base import baseline_extra
from ..labour.dynparams import par as _dyn
from ..regional import economy as _regional

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "base"


@lru_cache(maxsize=1)
def _data() -> MappingProxyType:
    """Base-year national accounts, read once (read-only)."""
    gdp = pd.read_csv(DATA_DIR / "macro_gdp.csv").set_index("key")["value"].to_dict()
    hist = pd.read_csv(DATA_DIR / "macro_gdp_history.csv").set_index("year")
    b1g = pd.read_csv(DATA_DIR / "macro_sector.csv").set_index("sector")["b1g_meur"].reindex(SECTORS).to_numpy(dtype=float)
    lab = pd.read_csv(DATA_DIR / "labour_sector.csv").set_index("sector").reindex(SECTORS)
    acc = pd.read_csv(DATA_DIR / "macro_accounts.csv").set_index(["sector", "na_item", "direct"])["value"].to_dict()
    cons = pd.read_csv(DATA_DIR / "macro_consumption.csv").set_index("category")["value_meur"].to_dict()
    for a in (b1g,):
        a.setflags(write=False)
    d1_ratio = (lab["d1_meur"] / lab["d11_meur"]).to_numpy(dtype=float)
    d1_ratio.setflags(write=False)
    hh = "S14_S15"
    return MappingProxyType({
        "gdp": gdp["B1GQ"], "b1g": b1g, "b1g_total": gdp["B1G"], "d21x31": gdp["D21X31"],
        "d2x3": gdp["D2X3"], "d1": gdp["D1"], "d11": gdp["D11"],
        "cons": gdp["P31_S14_S15"], "gov_cons": gdp["P3_S13"], "investment": gdp["P5G"],
        "investment_housing": gdp["GFCF_N111G"], "exports": gdp["P6"], "imports": gdp["P7"],
        "d1_ratio": d1_ratio,
        "mixed_income": acc[(hh, "B3G", "RECV")],
        "hh_interest": acc[(hh, "D41", "RECV")], "hh_dividends": acc[(hh, "D42", "RECV")],
        "hh_disposable": acc[(hh, "B6G", "RECV")], "hh_saving": acc[(hh, "B8G", "RECV")],
        "corp_nos": acc[("S11", "B2A3N", "RECV")] + acc[("S12", "B2A3N", "RECV")],
        "corp_gos": acc[("S11", "B2A3G", "RECV")] + acc[("S12", "B2A3G", "RECV")],
        "hh_fin_wealth": gdp["HH_ASS_F"],
        "cons_raw": MappingProxyType(cons),
        "real_growth_base": float(hist.loc[BASE_YEAR, "real_growth_pct"]) / 100.0,
    })


# ---------------------------------------------------------------------------------------------
_INDEX_KEYS = {"cpi": "cpi", "health_index": "health_index", "wages": "wage_index",
               "wage_link": "tax_threshold_drift"}


class _LazyView:
    """Parameters of one year resolved on demand (only the few a calculator asks for)."""

    def __init__(self, pset, year: int, index):
        self._pset, self.year, self._index, self._cache = pset, year, index, {}

    def __getitem__(self, name: str):
        if name not in self._cache:
            self._cache[name] = self._pset.value(name, self.year, self._index)
        return self._cache[name]

    def get(self, name: str, default=None):
        return self[name] if name in self._pset.specs else default

    def __contains__(self, name: str) -> bool:
        return name in self._pset.specs


def _tax_price_shift(state, prev, p, ctx, provisional: dict) -> tuple[float, float, dict] | None:
    """Level effect on (CPI, health index) of the scenario's indirect taxes relative to the
    baseline's, as fractions, and the price change of each consumption category that has one.
    None if it cannot be evaluated (then last year's is kept).

    Both legislations are evaluated with their own uprated amounts (scenario: indices of the run
    so far and `provisional` ones for this year; baseline: the finished baseline run), so that
    CPI-indexed excise duties are compared like with like.
    """
    base = ctx.baseline
    if not base.has(state.year):
        return None
    try:
        from ..fiscal.revenue import calc
        year, hist = state.year, ctx.history

        def index_s(kind: str, y: int) -> float:
            if y < BASE_YEAR:
                return 1.0
            return provisional[kind] if y == year else hist.at(y).macro[_INDEX_KEYS[kind]]

        def index_b(kind: str, y: int) -> float:
            return 1.0 if y < BASE_YEAR else base.at(y).macro[_INDEX_KEYS[kind]]

        rs = calc.consumption_tax_rates(_LazyView(ctx.params, year, index_s))
        rb = calc.consumption_tax_rates(_LazyView(ctx.params.with_scenario(None), year, index_b))
    except Exception:                      # sibling module temporarily broken: no price effect
        return None
    pas = p.get("beh.cons.pass_through") or {}
    scale = float(_dyn(p, "dyn.indirect_tax_price_passthrough"))
    excluded = set(_dyn(p, "dyn.health_index_excluded_categories"))
    w = _cpi_weights(float(p["macro.fuel_share_of_cp072"]),
                     float(_dyn(p, "dyn.cpi.imputed_rent_share_of_housing")))
    cpi = health = w_health = 0.0
    by_cat: dict[str, float] = {}
    for c in CONS_CATS:
        a, b = float(rs[c]), float(rb[c])
        dp = (1.0 + a) / (1.0 + b) - 1.0
        if abs(dp) < 1e-12:                # same legislation (up to rounding of uprated amounts)
            dp = 0.0
        dp *= scale * float(pas.get(c, 1.0))
        if dp != 0.0:
            by_cat[c] = dp
        cpi += w[c] * dp
        if c not in excluded:
            health += w[c] * dp
            w_health += w[c]
    return cpi, (health / w_health if w_health > 0 else 0.0), by_cat


_NO_WEDGE = MappingProxyType({"employer": 0.0, "net": 0.0, "employer_ssc_ratio": 1.0})


def _log_ratio(a: float, b: float) -> float:
    return float(np.log(a / b)) if a > 0 and b > 0 and a != b else 0.0


def _labour_tax_wedge(state, ctx) -> MappingProxyType | dict:
    """Tax wedge on private employees' wages under the scenario's legislation relative to the
    baseline's, at an unchanged gross wage:
      employer            ln ratio of the labour cost per euro of gross wage (wage-bill weighted)
      net                 mean over the wage deciles of the ln ratio of the net wage (employee
                          contributions, income tax): the loss of the typical worker
      employer_ssc_ratio  ratio of employer contributions

    Both legislations are evaluated in the baseline's environment: its wage deciles of the year
    and its price and wage indices for the uprating of amounts. Only legislation differs, so
    identical legislation gives exactly 0 and a change in wages or prices alone (fiscal drag) is
    not read as a tax change. Neutral if it cannot be evaluated.
    """
    base, year = ctx.baseline, state.year
    if not base.has(year):
        return _NO_WEDGE
    try:
        from ..fiscal.revenue import calc

        def index_b(kind: str, y: int) -> float:
            return 1.0 if y < BASE_YEAR else base.at(y).macro[_INDEX_KEYS[kind]]

        grid = np.asarray(base.at(year).wage_dist[J["emp_private"]], dtype=float)
        if not (np.all(np.isfinite(grid)) and grid.sum() > 0):
            return _NO_WEDGE
        rs = calc.tax_on_income(grid, _LazyView(ctx.params, year, index_b), "emp_private")
        rb = calc.tax_on_income(grid, _LazyView(ctx.params.with_scenario(None), year, index_b), "emp_private")
        tot = {n: (float(np.sum(rs[n])), float(np.sum(rb[n]))) for n in ("employer_cost", "ssc_employer")}
        net_s, net_b = np.asarray(rs["net"], dtype=float), np.asarray(rb["net"], dtype=float)
        if not (all(np.isfinite(v) for pair in tot.values() for v in pair)
                and np.all(np.isfinite(net_s)) and np.all(net_s > 0) and np.all(net_b > 0)):
            return _NO_WEDGE
        es, eb = tot["ssc_employer"]
        net = 0.0 if np.array_equal(net_s, net_b) else float(np.mean(np.log(net_s / net_b)))
        return {"employer": _log_ratio(*tot["employer_cost"]), "net": net,
                "employer_ssc_ratio": es / eb if (eb > 0 and es > 0 and es != eb) else 1.0}
    except Exception:                      # sibling module temporarily broken: no effect
        return _NO_WEDGE


def _regional_productivity_growth(prev, p) -> float:
    """National productivity growth added this year by regional productivity wedges
    (`regional.productivity_growth_wedge`), as `output` applies it: GDP-weighted. Exactly 0 when
    no region has a wedge."""
    wedge = _regional.by_region(p.get("regional.productivity_growth_wedge", 0.0) or 0.0)
    if not np.any(wedge != 0.0):
        return 0.0
    try:
        share = np.asarray((prev.extra.get(_regional.NS) or {})["gdp_share0"], dtype=float)
        idx = np.asarray((prev.extra.get("macro") or {}).get("region_prod_index", np.ones(len(wedge))), dtype=float)
        before = 1.0 + float((share * (idx - 1.0)).sum())
        after = 1.0 + float((share * (idx * (1.0 + wedge) - 1.0)).sum())
        g = after / before - 1.0
        return g if np.isfinite(g) else 0.0
    except Exception:
        return 0.0


def prices(state, prev, p, ctx=None) -> None:
    """Inflation, price and wage indices, interest rates. Receives un-uprated parameters."""
    m = state.macro
    infl = float(p["macro.inflation"])
    h_growth = infl + float(p["macro.health_index_wedge"])
    g_trend = float(p["macro.productivity_growth"])
    m["inflation"] = infl
    m["interest_rate_10y"] = float(p["macro.interest_rate_10y"])
    m["interest_rate_short"] = float(p["macro.interest_rate_short"])
    px = (prev.extra.get("macro_prices") if prev is not None else None) or {}
    if prev is None or state.year <= BASE_YEAR:
        m.update(cpi=1.0, health_index=1.0, wage_index=1.0, real_unit_labour_cost=1.0,
                 labour_cost_competitiveness_index=1.0, cpi_tax_effect=0.0, cpi_wage_cost_effect=0.0, cpi_demand_effect=0.0,
                 wage_bargaining_index=1.0)
        state.extra["macro_prices"] = {
            "health_growth": h_growth, "trend_productivity": g_trend, "productivity_factor": 1.0,
            "indexation": h_growth, "real_wage_growth": 0.0, "core_inflation": infl,
            "cpi_core": 1.0, "health_core": 1.0, "tax_shift": (0.0, 0.0), "warranted_wage": 1.0,
        }
        return

    # underlying prices, then the level effect of indirect-tax changes relative to the baseline
    cpi_core = px.get("cpi_core", prev.macro["cpi"]) * (1.0 + infl)
    health_core = px.get("health_core", prev.macro["health_index"]) * (1.0 + h_growth)
    shift_prev = px.get("tax_shift", (0.0, 0.0))
    shift = shift_prev
    tax_prices = px.get("tax_prices") or {}        # price change by category (read-only)
    dynamic = (ctx is not None and ctx.baseline is not None and bool(ctx.params.changed_params())
               and float(p.get("beh.enabled", 1.0) or 0.0) != 0.0)
    if dynamic:
        provisional = {
            "cpi": cpi_core * (1.0 + shift_prev[0]), "health_index": health_core * (1.0 + shift_prev[1]),
            "wages": prev.macro["wage_index"] * (1.0 + h_growth),
            "wage_link": prev.macro.get("tax_threshold_drift", 1.0),
        }
        res = _tax_price_shift(state, prev, p, ctx, provisional)
        if res is not None:
            shift, tax_prices = (res[0], res[1]), res[2]
    h_tax = 0.0                             # health-index growth due to indirect-tax changes
    if shift[0] != shift_prev[0] or shift[1] != shift_prev[1]:
        m["inflation"] = (1.0 + infl) * (1.0 + shift[0]) / (1.0 + shift_prev[0]) - 1.0
        h_tax = (1.0 + shift[1]) / (1.0 + shift_prev[1]) - 1.0
        h_growth = (1.0 + h_growth) * (1.0 + h_tax) - 1.0

    # wage costs feed prices: a gap in unit labour costs relative to the baseline is passed on
    # to consumer prices gradually (and, through indexation, back to wages)
    wshift_prev = float(px.get("wage_price_shift", 0.0))
    wshift = wshift_prev
    wtarget = float(px.get("wage_price_target", 0.0))
    # cumulative wage-level effect of indexation skipped or added relative to the baseline (log),
    # its price target and the price effect reached so far
    moderation_prev = float(px.get("wage_moderation", 0.0))
    moderation = moderation_prev
    wshift_mod_prev = float(px.get("wage_price_shift_moderation", 0.0))
    wshift_mod = wshift_mod_prev
    wtarget_mod = float(px.get("wage_price_target_moderation", 0.0))
    # slack feeds prices: last year's demand gap lowers (raises) the price level a little each
    # year it lasts (margins and wage drift), on top of the wage-cost channel
    dshift_prev = float(px.get("demand_price_shift", 0.0))
    dshift = dshift_prev
    wedge = _labour_tax_wedge(state, ctx) if dynamic else _NO_WEDGE
    lag = float(p["labour.indexation_lag_weight"])
    lag_tax = float(_dyn(p, "dyn.indexation.tax_shift_lag_weight"))
    h_tax_prev = float(px.get("health_tax_growth", 0.0))

    # wage bargaining: gross wages react to last year's tax wedge and demand gap (see docstring)
    barg_prev = float(px.get("bargaining", 0.0))
    barg = barg_prev
    barg_target = 0.0
    if dynamic:
        barg_target = (-float(_dyn(p, "dyn.wage_bargaining.worker_share_employer_contributions")) * wedge["employer"]
                       - float(_dyn(p, "dyn.wage_bargaining.compensation_employee_taxes")) * wedge["net"]
                       - float(_dyn(p, "dyn.wage_bargaining.worker_share_indirect_taxes")) * float(np.log1p(shift[1]))) + 0.0
        speed = float(_dyn(p, "dyn.wage_bargaining.speed"))
        barg = barg_prev + speed * (float(px.get("bargaining_target", 0.0)) - barg_prev)
    # wages also follow the cyclical position (the productivity lost or gained with the demand
    # gap): this part moves the warranted wage too, so it leaves the real unit labour cost alone
    act_prev = float(px.get("activity_wage", 0.0))
    act = act_prev
    if dynamic:
        act_target = float(_dyn(p, "dyn.wage_bargaining.activity_weight")) * float(prev.macro.get("demand_gap", 0.0))
        act = act_prev + float(_dyn(p, "dyn.wage_bargaining.speed")) * (act_target - act_prev)


    if dynamic and ctx.baseline.has(prev.year):
        bprev = ctx.baseline.at(prev.year)
        bpx = bprev.extra.get("macro_prices") or {}
        gap = np.log(prev.macro["wage_index"] / bprev.macro["wage_index"]) - np.log(
            px.get("warranted_wage", 1.0) / bpx.get("warranted_wage", 1.0))
        # this year's wage changes that are already known: bargained shift, indexation of the
        # tax-induced price shift, indexation skipped or added relative to the baseline
        f_idx = float(p["labour.indexation_factor"])
        f_idx_b = float(ctx.params.with_scenario(None).value("labour.indexation_factor", state.year))
        known = (barg - barg_prev) + f_idx * (lag_tax * h_tax_prev + (1.0 - lag_tax) * h_tax)
        if f_idx != f_idx_b:
            skipped = (f_idx - f_idx_b) * (lag * px.get("health_growth", h_growth) + (1.0 - lag) * h_growth)
            known += skipped
            moderation = moderation_prev + skipped
        gap = float(gap) + known + float(_dyn(p, "dyn.wage_cost.employer_contribution_weight")) * wedge["employer"]
        passthrough = float(_dyn(p, "dyn.wage_cost.price_passthrough"))
        target = passthrough * gap
        speed = float(_dyn(p, "dyn.wage_cost.price_adjustment_speed"))
        wshift = wshift_prev + speed * (target - wshift_prev)
        wtarget = target if target != 0.0 else 0.0
        # the part of the target and of the price effect that comes from indexation skipped or
        # added relative to the baseline (same linear adjustment, so the parts add up): output
        # prices do not run ahead of consumer prices for that part (see `output`)
        if moderation != 0.0 or wshift_mod_prev != 0.0:
            wtarget_mod = passthrough * moderation
            wshift_mod = wshift_mod_prev + speed * (wtarget_mod - wshift_mod_prev)
    if dynamic:
        gap_prev = float(prev.macro.get("demand_gap", 0.0))
        if gap_prev != 0.0:
            dshift = dshift_prev + float(_dyn(p, "dyn.prices.demand_gap_sensitivity")) * gap_prev
    pshift, pshift_prev = wshift + dshift, wshift_prev + dshift_prev
    if pshift != pshift_prev:
        step = float(np.exp(pshift - pshift_prev))
        m["inflation"] = (1.0 + m["inflation"]) * step - 1.0
        h_growth = (1.0 + h_growth) * step - 1.0
    wfac = float(np.exp(pshift)) if pshift != 0.0 else 1.0

    # automatic indexation: health index, partly with a one-year lag. A level shift of prices
    # from an indirect-tax change (in force from 1 January) is indexed faster than inflation
    # that builds up during the year: own lag weight for that part.
    if h_tax != 0.0 or h_tax_prev != 0.0:
        h_prev = float(px.get("health_growth", h_growth))
        other, other_prev = (1.0 + h_growth) / (1.0 + h_tax) - 1.0, (1.0 + h_prev) / (1.0 + h_tax_prev) - 1.0
        indexation = float(p["labour.indexation_factor"]) * (
            lag * other_prev + (1.0 - lag) * other + lag_tax * h_tax_prev + (1.0 - lag_tax) * h_tax)
    else:
        indexation = float(p["labour.indexation_factor"]) * (
            lag * px.get("health_growth", h_growth) + (1.0 - lag) * h_growth)

    # real wages track trend productivity (and, with a year's lag, policy-induced level shifts)
    share = float(p["macro.real_wage_productivity_share"])
    lam = float(p["macro.real_wage_productivity_lag_weight"])
    pf_prev = float((prev.adjust or {}).get("productivity_factor", 1.0))
    pf_prev2 = float(px.get("productivity_factor", 1.0))
    warranted = (1.0 - lam) * g_trend + lam * px.get("trend_productivity", g_trend) + (pf_prev / pf_prev2 - 1.0)
    g_regional = _regional_productivity_growth(prev, p)      # wages follow regional productivity gains
    if g_regional != 0.0:
        warranted += g_regional
    real = share * warranted
    cap = p["labour.wage_norm_real_cap"]
    if cap is not None:
        real = min(real, float(cap))

    m["cpi"] = cpi_core * (1.0 + shift[0]) * wfac
    m["health_index"] = health_core * (1.0 + shift[1]) * wfac
    m["wage_index"] = prev.macro["wage_index"] * (1.0 + indexation) * (1.0 + real)
    if barg != barg_prev or act != act_prev:
        m["wage_index"] *= float(np.exp(barg - barg_prev + act - act_prev))
    m["wage_bargaining_index"] = float(np.exp(barg + act)) if (barg != 0.0 or act != 0.0) else 1.0
    # real unit labour cost: wages over underlying prices and the productivity they should follow
    warranted_wage = px.get("warranted_wage", 1.0) * (1.0 + warranted)
    if act != act_prev:
        warranted_wage *= float(np.exp(act - act_prev))
    m["real_unit_labour_cost"] = m["wage_index"] / (cpi_core * wfac * warranted_wage)
    # the same with employer contributions: labour cost per unit of output relative to the
    # baseline's legislation (what competitiveness reacts to); equal to the line above when
    # employer contributions are the baseline's
    m["labour_cost_competitiveness_index"] = (
        m["real_unit_labour_cost"] * float(np.exp(wedge["employer"])) if wedge["employer"] != 0.0
        else m["real_unit_labour_cost"])
    m["cpi_tax_effect"] = shift[0]
    m["cpi_wage_cost_effect"] = (float(np.exp(wshift)) - 1.0) if wshift != 0.0 else 0.0
    m["cpi_demand_effect"] = (float(np.exp(dshift)) - 1.0) if dshift != 0.0 else 0.0
    state.extra["macro_prices"] = {
        "health_growth": h_growth, "trend_productivity": g_trend, "productivity_factor": pf_prev,
        "indexation": indexation, "real_wage_growth": real, "core_inflation": infl,
        "cpi_core": cpi_core, "health_core": health_core, "tax_shift": shift,
        "warranted_wage": warranted_wage, "wage_price_shift": wshift, "wage_price_target": wtarget,
        "wage_moderation": moderation, "wage_price_shift_moderation": wshift_mod,
        "wage_price_target_moderation": wtarget_mod,
        "demand_price_shift": dshift, "tax_prices": tax_prices,
        "health_tax_growth": h_tax, "bargaining": barg, "bargaining_target": barg_target, "activity_wage": act,
        "employer_ssc_ratio": float(wedge["employer_ssc_ratio"]),
        "employer_wedge": float(wedge["employer"]),
    }


# ---------------------------------------------------------------------------------------------
def _calibrate(state, p, d, lab_bills_s: np.ndarray) -> dict:
    """Calibration factors, fitted on the base-year state so that 2024 equals the accounts."""
    emp_s = state.emp.sum(axis=1)
    lab = state.labour
    wages = lab["wage_bill_private"] + lab["wage_bill_public"]
    return {
        "prod0": d["b1g"] / np.maximum(emp_s, 1e-9),            # EUR mn of 2024 per worker
        "tau": d["d21x31"] / d["b1g_total"],                     # net taxes on products / GVA
        "d11": d["d11"] / wages,
        "d1": d["d1"] / float((lab_bills_s * d["d1_ratio"]).sum()),
        "mixed": d["mixed_income"] / lab["self_employed_income"],
        "d2x3_share": d["d2x3"] / d["gdp"],
        "non_profit_share": (d["gdp"] - d["d1"] - d["d2x3"] - d["mixed_income"] - d["corp_nos"]) / d["gdp"],
        "savings0": float(p["macro.household_savings_rate"]),
        "primary_income0": d["d1"] + d["mixed_income"] + d["hh_interest"] + d["hh_dividends"],
        "pub_bill0": lab["wage_bill_public"],
        "hours0": lab["hours_per_worker"],
        "r10_0": max(float(state.macro.get("interest_rate_10y", 0.03)), 1e-4),
        "deaths0": float(state.demo.get("deaths", 0.0)) or None,
        "pop0": float(state.pop.sum()),
        "regional": _regional.calibrate(state, p),    # workplace employment, productivity, wages
    }


def output(state, prev, p, ctx=None) -> None:
    d = _data()
    m = state.macro
    lab = state.labour
    adjust = state.adjust or {}
    m.setdefault("demand_gap", 0.0)
    gap = float(m["demand_gap"]) + float(p["macro.cyclical_output_gap"])
    prev_x = (prev.extra.get("macro") if prev is not None else None) or baseline_extra(ctx, "macro")

    emp_s = state.emp.sum(axis=1)
    # labour input in efficiency units: jobs gained or lost relative to the baseline carry the
    # productivity of their origin (labour module, dyn.marginal_productivity); 1 in a baseline run
    q_prod = (state.extra.get("labour") or {}).get("q_prod")
    emp_eff_s = emp_s if q_prod is None else (state.emp * q_prod).sum(axis=1)
    pub_bill = float(lab.get("wage_bill_public_actual", lab["wage_bill_public"]))   # bill actually paid
    employees_bill_s = (state.wage[:, :2] * state.emp[:, :2]).sum(axis=1) / 1e6      # EUR mn
    calib = prev_x.get("calib") or _calibrate(state, p, d, employees_bill_s)
    first = "gdp_real" not in prev_x                 # no macro stocks from a previous year

    # --- supply: employment x productivity ---
    g = float(p["macro.productivity_growth"])
    wedge = p["macro.productivity_growth_sector_wedge"]
    if first:
        prod_index = np.ones(len(SECTORS))
        deflator = float(m.get("cpi", 1.0))
    else:
        prod_index = prev_x["prod_index"] * (1.0 + g + np.array([float(wedge.get(s, 0.0)) for s in SECTORS]))
        # underlying inflation: values stay at baseline indirect-tax rates (see module docstring)
        core_infl = (state.extra.get("macro_prices") or {}).get("core_inflation", m["inflation"])
        deflator = prev_x["deflator"] * (1.0 + core_infl + float(p["macro.gdp_deflator_wedge"]))
    hours = (lab["hours_per_worker"] / calib["hours0"]) ** float(p["macro.hours_output_elasticity"])
    pf = float(adjust.get("productivity_factor", 1.0))
    va_real = emp_eff_s * calib["prod0"] * prod_index * hours * pf * (1.0 + gap)
    # --- regions: employment by place of work, GDP weights, regional productivity wedge ---
    base_state = getattr(ctx, "base_state", None)
    reg = _regional.workplace(
        state, p, calib.get("regional") or _regional.calibrate(state, p),
        (base_state.extra.get(_regional.NS) if base_state is not None else None),
        float(emp_eff_s.sum()) / max(float(emp_s.sum()), 1e-9))
    reg_wedge = _regional.by_region(p.get("regional.productivity_growth_wedge", 0.0))
    reg_index = np.ones(len(reg_wedge)) if first else prev_x.get("region_prod_index", np.ones(len(reg_wedge)))
    if not first and np.any(reg_wedge != 0.0):
        reg_index = reg_index * (1.0 + reg_wedge)
    if np.any(reg_index != 1.0):      # national output rises by each region's weight x its wedge
        va_real = va_real * (1.0 + float((reg["gdp_share0"] * (reg_index - 1.0)).sum()))
    gdp_real = float(va_real.sum() * (1.0 + calib["tau"]))
    # wage costs move the price of domestic output (relative to the baseline) more than they
    # move consumer prices, part of which are imports; `deflator` itself stays the underlying one
    # output prices lead: they move at once to the level consumer prices reach gradually
    # (`dyn.wage_cost.deflator_lead` of the remaining distance); slack moves both alike
    mp = state.extra.get("macro_prices") or {}
    wshift = float(mp.get("wage_price_shift", 0.0))
    wtarget = float(mp.get("wage_price_target", 0.0))
    dshift = float(mp.get("demand_price_shift", 0.0))
    deflator_out = deflator
    if wshift != 0.0 or wtarget != 0.0 or dshift != 0.0:
        lead = min(1.0, max(0.0, float(_dyn(p, "dyn.wage_cost.deflator_lead"))))
        ahead = lead * (wtarget - wshift)
        # skipped (or extra) indexation: its own lead, taken out of the distance still to cover
        ahead_mod = float(mp.get("wage_price_target_moderation", 0.0)) - float(mp.get("wage_price_shift_moderation", 0.0))
        if ahead_mod != 0.0:
            lead_mod = min(1.0, max(0.0, float(_dyn(p, "dyn.wage_cost.deflator_lead_indexation"))))
            ahead += (lead_mod - lead) * ahead_mod
        deflator_out = deflator * float(np.exp(
            (wshift + ahead) * float(_dyn(p, "dyn.wage_cost.deflator_ratio")) + dshift))
    # government output is valued at cost: a change in pay relative to output prices, compared
    # with the baseline, changes its value (and nominal GDP), not corporate profits
    if base_state is not None:
        try:
            rel = ((float(m["wage_index"]) / float(base_state.macro["wage_index"]))
                   / (deflator_out / float(base_state.macro["gdp_deflator"])))
        except (KeyError, TypeError, ZeroDivisionError):
            rel = 1.0
        bills = float(lab["wage_bill_private"]) + pub_bill
        if rel != 1.0 and np.isfinite(rel) and rel > 0.0 and bills > 0.0:
            d1_public = float((employees_bill_s * d["d1_ratio"]).sum()) * calib["d1"] * pub_bill / bills
            deflator_out *= 1.0 + d1_public * (1.0 - 1.0 / rel) / (gdp_real * deflator_out)
    gdp = gdp_real * deflator_out
    state.va = va_real * deflator_out
    employment = float(emp_s.sum())
    gdp_per_worker = gdp_real / max(employment, 1.0)
    if first:
        growth = d["real_growth_base"] if state.year == BASE_YEAR else 0.0
        prod_growth = g
        gdp_prev_nominal = None
    else:
        growth = gdp_real / prev_x["gdp_real"] - 1.0
        prod_growth = gdp_per_worker / prev_x["gdp_per_worker"] - 1.0
        gdp_prev_nominal = prev_x["gdp_nominal"]

    # --- income side ---
    wages = (lab["wage_bill_private"] + pub_bill) * calib["d11"]
    d1 = float((employees_bill_s * d["d1_ratio"]).sum()) * calib["d1"]
    mixed = lab["self_employed_income"] * calib["mixed"]
    d2x3 = calib["d2x3_share"] * gdp
    # a change in employer contributions relative to the baseline is part of compensation of
    # employees (not of household income): profits carry what is not passed on to prices
    d1_wages = d1
    ssc_ratio = float((state.extra.get("macro_prices") or {}).get("employer_ssc_ratio", 1.0))
    if ssc_ratio != 1.0 and base_state is not None:
        ssc_b = float((base_state.revenue or {}).get("ssc_employer_private", 0.0))
        bill_b = float(base_state.labour.get("wage_bill_private", 0.0))
        if ssc_b > 0.0 and bill_b > 0.0:
            d1 = d1 + ssc_b * (ssc_ratio - 1.0) * float(lab["wage_bill_private"]) / bill_b
    gos = gdp - d1 - d2x3 - mixed

    # --- tax-base proxies ---
    profits_raw = max(gos - calib["non_profit_share"] * gdp, 0.02 * gdp)
    corp_profits = profits_raw * float(adjust.get("cit_base_factor", 1.0))
    wealth_prev = prev_x.get("wealth", d["hh_fin_wealth"])
    wealth = wealth_prev * (gdp / gdp_prev_nominal if gdp_prev_nominal else 1.0)
    rate_effect = (max(m["interest_rate_10y"], 1e-4) / calib["r10_0"]) ** float(p["macro.interest_income_rate_elasticity"])
    interest = d["hh_interest"] * wealth / d["hh_fin_wealth"] * rate_effect
    dividends = d["hh_dividends"] * profits_raw / d["corp_nos"]
    cap_income_raw = interest + dividends
    cap_income = cap_income_raw * float(adjust.get("capital_income_base_factor", 1.0))

    # --- households: disposable-income proxy, consumption, wealth ---
    primary = d1_wages + mixed + cap_income_raw
    disposable = d["hh_disposable"] * primary / calib["primary_income0"]
    s_rate = float(p["macro.household_savings_rate"])
    cons_total = d["cons"] * (disposable * (1.0 - s_rate)) / (d["hh_disposable"] * (1.0 - calib["savings0"]))
    wealth += (s_rate - calib["savings0"]) * disposable
    shares = _cons_shares(float(p["macro.fuel_share_of_cp072"]))
    cons_factor = adjust.get("cons_factor") or {}
    cons = {c: cons_total * shares[c] * float(cons_factor.get(c, 1.0)) for c in CONS_CATS}
    cons_total = float(sum(cons.values()))

    # --- other expenditure components; imports close the identity ---
    investment = d["investment"] / d["gdp"] * gdp * float(p["macro.investment_share_factor"])
    inv_housing = d["investment_housing"] / d["gdp"] * gdp * float(p["macro.investment_share_factor"])
    w_g = float(p["macro.gov_consumption_wage_weight"])
    gov_cons = d["gov_cons"] * (w_g * pub_bill / calib["pub_bill0"] + (1.0 - w_g) * gdp / d["gdp"])
    exports = d["exports"] / d["gdp"] * gdp
    imports = cons_total + investment + gov_cons + exports - gdp

    deaths = float(state.demo.get("deaths", 0.0)) if state.demo else 0.0
    pop_ratio = float(state.pop.sum()) / calib["pop0"] if calib["pop0"] else 1.0
    death_ratio = deaths / calib["deaths0"] if (deaths and calib["deaths0"]) else pop_ratio
    wealth_per_head = wealth / d["hh_fin_wealth"] / max(pop_ratio, 1e-9)
    inheritance = float(p["macro.inheritance_flow_2024"]) * death_ratio * wealth_per_head
    property_tx = (float(p["macro.property_transactions_2024"]) * gdp / d["gdp"]
                   * float(adjust.get("property_transactions_factor", 1.0)))

    m.update(
        gdp_nominal=gdp, gdp_real=gdp_real, gdp_real_growth=growth, gdp_deflator=deflator_out,
        consumption_households=cons_total, investment=investment, investment_housing=inv_housing,
        gov_consumption=gov_cons, exports=exports, imports=imports,
        compensation_employees=d1, wages_salaries=wages, gross_operating_surplus=gos,
        mixed_income=mixed, corporate_profits_taxable=corp_profits,
        household_capital_income=cap_income, household_financial_wealth=wealth,
        property_transactions=property_tx, inheritance_flow=inheritance,
        productivity_growth=prod_growth,
        # --- additional indicators (not in the contract) ---
        household_disposable_income=disposable, household_savings_rate=s_rate,
        taxes_less_subsidies_production=d2x3, net_exports=exports - imports,
        gdp_per_worker=gdp_per_worker * 1e6, gdp_per_capita=gdp / max(float(state.pop.sum()), 1.0) * 1e6,
        wage_share=d1 / gdp, corporate_profit_share=profits_raw / gdp,
        trend_productivity_growth=g, output_gap=gap,
    )
    for c in CONS_CATS:
        m[f"cons.{c}"] = cons[c]
    # GDP at market prices of the scenario: the value above plus what the scenario's indirect
    # taxes add to (or take from) the price of household consumption. Equal to `gdp_nominal`
    # whenever the indirect taxes are the baseline's (see the module docstring); its deflator is
    # this value over `gdp_real`.
    tax_prices = mp.get("tax_prices") or {}
    tax_value = float(sum(cons[c] * dp for c, dp in tax_prices.items())) if tax_prices else 0.0
    m["gdp_nominal_market_prices"] = gdp + tax_value if tax_value != 0.0 else gdp

    state.extra["macro"] = {
        "calib": calib, "prod_index": prod_index, "deflator": deflator, "wealth": wealth,
        "gdp_real": gdp_real, "gdp_nominal": gdp, "gdp_per_worker": gdp_per_worker,
        "region_prod_index": reg_index,
        # interest and dividends received by households, before the behavioural factor on the
        # tax base (behaviour: household income in the demand block)
        "capital_income_received": cap_income_raw,
    }
    state.extra[_regional.NS] = _regional.finish(reg, reg_index, gdp, gdp_real, d1)


@lru_cache(maxsize=16)
def _cpi_weights(fuel_share: float, imputed_rent_share: float) -> MappingProxyType:
    """Weights of the consumption categories in the consumer price index: household consumption
    without imputed rents (owner-occupiers' rents are not in the national index)."""
    w = dict(_cons_shares(fuel_share))
    w["housing"] *= 1.0 - min(max(imputed_rent_share, 0.0), 1.0)
    total = sum(w.values())
    return MappingProxyType({c: v / total for c, v in w.items()})


@lru_cache(maxsize=16)
def _cons_shares(fuel_share: float) -> MappingProxyType:
    raw = dict(_data()["cons_raw"])
    transport = raw.pop("transport_operation")
    raw["fuel"] = fuel_share * transport
    raw["other_services"] += (1.0 - fuel_share) * transport
    total = sum(raw.values())
    return MappingProxyType({c: raw[c] / total for c in CONS_CATS})
