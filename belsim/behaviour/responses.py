"""Supply-side behavioural responses: fills `state.adjust` (see docs/ARCHITECTURE.md).

Method. The module never looks at tax/benefit parameter names. It evaluates the contract's
pure micro-calculators under the scenario parameters `p` and the baseline parameters
`ctx.base_p` on the baseline wage deciles, and reacts to the log-difference of the outcomes
(net gain from work, net-of-marginal-tax rate, employer cost, consumer prices, tax rates).
Every long-run target x* is a log change (or years, for retirement) that is exactly 0 when
the two parameter sets give identical calculator outputs. Targets are scaled by `beh.scale`,
clipped (`beh.clip`) and phased in by partial adjustment

    x_t = x_{t-1} + lambda (x*_t - x_{t-1})

with x_{t-1} read from `prev.extra["behaviour"]["x"]` (no module state). The published
adjustments are exp(x) (factors) or x (additive), hence exactly neutral for a no-measure
scenario.

Internal channel states (all stored in `state.extra["behaviour"]["x"]`):
    ext [3 age groups, 2 sexes]  log change of the employment rate (extensive margin)
    intensive_emp                log change of employees' effective labour input
    self_emp                     log change of self-employed taxable income
    labour_demand                log change of private employment (employer cost at unchanged
                                 gross wage, deciles weighted with their labour cost, plus
                                 the real-unit-labour-cost gap: wage index over underlying
                                 prices and warranted productivity, scenario relative to
                                 baseline, published by macro.prices)
    labour_composition           log change of private employment from a change in employer
                                 cost concentrated on some wage deciles: net share of the
                                 extra jobs bought per euro where jobs are cheap
                                 (`beh.ld.composition_net_share`), with its own speed
    labour_decile [decile]       log change of private employment of each wage decile around
                                 the mean response, after a change in employer cost that
                                 differs across deciles (zero in sum; same speed as
                                 labour_composition). Published as
                                 `adjust["labour_demand_decile_shift"]`: labour places the
                                 jobs gained and lost in the wage distribution with it
    retirement                   years of later effective retirement (replacement ratio, and the
                                 pension earned per extra year worked: a saturating response,
                                 `beh.ret.accrual_max_years`)
    cons [category]              log change of the volume consumed
    cit_base, cit_productivity   log change of CIT base / productivity (investment)
    capital_income, property     log change of those bases

Wage bargaining. macro.prices shifts gross wages after a change in the tax wedge
(`macro["wage_bargaining_index"]`, scenario relative to baseline); the jobs follow through
labour cost (channel 3b). The labour-supply channels therefore see legislation only (the
bargained shift is taken out of the wage ratio k), and in the gain from working each worker
recoups the share `dyn.wage_bargaining.compensation_employee_taxes` of a loss in net wage, so
that the same job loss is not counted twice.

Public capital. A change in public investment relative to the baseline builds up a gap in the
public capital stock (`state.extra["behaviour"]["public_capital_gap"]`, in points of baseline
GDP, depreciating at `beh.pubcap.depreciation_rate`); output follows with
`beh.pubcap.output_per_capital` through `adjust["productivity_factor"]`.

Benefit levels. The benefit calculator gives the legislated formula; the general level of
benefits relative to the health index (index jumps on benefits, welfare adjustments) is applied
by the expenditure module outside the calculator. The same relative level (scenario over
baseline, `state.extra["behaviour"]["benefit_level"]`) scales the amounts of the benefit formula
here (ceilings, minima, flat rates; the part of a benefit that is a percentage of the wage
follows the wage), so that an index jump on wages and benefits alike leaves the gain from work
unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.dims import CONS_CATS, J, N_AGE, N_DECILE, N_SEX, S, SEXES
from ..labour.dynparams import par as dyn_par
from ._params import enabled, par, scale

AGE_GROUPS = ["young", "prime", "older"]
_EMPLOYED = [S["emp_private"], S["emp_public"], S["self_employed"]]
_BENEFIT_KIND = {"unemployed": "unemployment", "disabled": "disability", "retired": "pension"}
_Z = np.linspace(-1.0, 1.0, N_DECILE)     # position of a decile in the distribution


# --------------------------------------------------------------------------- helpers
def _calculators():
    """Sibling calculators, imported defensively (None if the module is broken)."""
    try:
        from ..fiscal.revenue import calc as rev
    except Exception:                      # sibling module temporarily broken
        rev = None
    try:
        from ..fiscal.expenditure import calc as exp
    except Exception:
        exp = None
    return rev, exp


def _call(errors: list, fn_owner, fn_name: str, *args):
    """Call an external calculator; any failure -> None (channel treated as neutral)."""
    fn = getattr(fn_owner, fn_name, None) if fn_owner is not None else None
    if fn is None:
        errors.append(f"{fn_name}: not available")
        return None
    try:
        return fn(*args)
    except Exception as e:                 # external code: never crash the simulation
        errors.append(f"{fn_name}: {type(e).__name__}: {e}")
        return None


def _array(v, shape) -> np.ndarray | None:
    """Calculator output as a finite float array of the expected shape, else None."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.shape != shape or not np.isfinite(a).all():
        return None
    return a


def _decile_param(p, name: str) -> np.ndarray:
    a = np.asarray(par(p, name), dtype=float)
    if a.shape != (N_DECILE,):
        raise ValueError(f"parameter {name} must be a list of {N_DECILE} values")
    return a


def _log_ratio(scen: np.ndarray, base: np.ndarray) -> np.ndarray:
    """ln(scen/base); exactly 0 where the inputs are equal. Inputs must be > 0."""
    return np.log(scen / base)


def _same_policy(p, base_p) -> bool:
    """True if scenario and baseline parameter values are identical (fast path)."""
    a, b = getattr(p, "_values", None), getattr(base_p, "_values", None)
    if a is None or b is None:
        return False
    try:
        return bool(a == b)
    except Exception:                      # e.g. array-valued parameters
        return False


# ------------------------------------------------------- calculator evaluation (one side)
def _income_side(rev, exp, p, grid_emp, grid_se, k: float, h: float, errors: list,
                 benefit_level: float = 1.0) -> dict:
    """Net incomes on the wage grid under parameters `p`.

    `k` is the ratio of the scenario to the baseline wage index: the scenario is evaluated at
    wages k x grid and the results divided by k, so that differences in indexation of nominal
    thresholds do not show up as policy changes (k is exactly 1 on the baseline side).
    Returns per-decile arrays, or None for a piece whose calculator failed:
      net_w, cost (employee net wage, employer cost), keep_emp, keep_se (net-of-marginal-tax
      rates), out[status] (net out-of-work income of a worker of that decile).
    `benefit_level`: general level of unemployment / disability benefits relative to the
    baseline (the pension formula follows the worker's own wages and is not rescaled). It
    applies to the amounts of the benefit formula; a benefit that is a percentage of the wage
    follows the wage.
    """
    n = N_DECILE
    res: dict = {"net_w": None, "cost": None, "keep_emp": None, "keep_se": None, "out": {}}

    g = grid_emp * k
    gg = np.concatenate([g, g * (1.0 + h)])
    r = _call(errors, rev, "tax_on_income", gg, p, "emp_private")
    if isinstance(r, dict):
        net = _array(r.get("net"), gg.shape)
        cost = _array(r.get("employer_cost"), gg.shape)
        if net is not None:
            res["net_w"] = net[:n] / k
            res["keep_emp"] = (net[n:] - net[:n]) / (g * h)
        if cost is not None:
            res["cost"] = cost[:n] / k

    gs = grid_se * k
    gg = np.concatenate([gs, gs * (1.0 + h)])
    r = _call(errors, rev, "tax_on_income", gg, p, "self_employed")
    if isinstance(r, dict):
        net = _array(r.get("net"), gg.shape)
        if net is not None:
            res["keep_se"] = (net[n:] - net[:n]) / (gs * h)

    # social assistance: flat and untaxed; follows the general benefit level
    if float((par(p, "beh.ext.outside_option_weights") or {}).get("assisted", 0.0) or 0.0) != 0.0:
        ben = _array(_call(errors, exp, "replacement_income", "assisted", g, p), g.shape)
        if ben is not None:
            res["out"]["assisted"] = ben * benefit_level / k
    for status, kind in _BENEFIT_KIND.items():
        # the general level scales the amounts of the formula (ceilings, minima, flat rates), not
        # the part that is a percentage of the worker's own wage: level x formula(wage / level)
        scaled = status != "retired" and benefit_level != 1.0
        ben = _array(_call(errors, exp, "replacement_income", status, g / benefit_level if scaled else g, p), g.shape)
        if ben is None:
            continue
        if scaled:
            ben = ben * benefit_level
        r = _call(errors, rev, "tax_on_income", ben, p, kind)
        if isinstance(r, dict):
            net = _array(r.get("net"), g.shape)
            if net is not None:
                res["out"][status] = net / k
    return res


def _group_weights(p) -> np.ndarray:
    """Decile weights of each (age group, sex): array [3, 2, N_DECILE], rows sum to 1."""
    tilt = par(p, "beh.ext.decile_tilt")
    w = np.empty((len(AGE_GROUPS), N_SEX, N_DECILE))
    for i, ag in enumerate(AGE_GROUPS):
        for s, sx in enumerate(SEXES):
            t = float(tilt.get(ag, 0.0)) + float(tilt.get(sx, 0.0))
            e = np.exp(-t * _Z)
            w[i, s] = e / e.sum()
    return w


# ------------------------------------------------------------------- long-run targets
def _zero_targets() -> dict:
    return {
        "ext": np.zeros((len(AGE_GROUPS), N_SEX)),
        "intensive_emp": 0.0,
        "self_emp": 0.0,
        "labour_demand": 0.0,
        "labour_composition": 0.0,
        "labour_decile": np.zeros(N_DECILE),
        "retirement": 0.0,
        "cons": np.zeros(len(CONS_CATS)),
        "cit_base": 0.0,
        "cit_productivity": 0.0,
        "capital_income": 0.0,
        "property": 0.0,
    }


def _income_targets(t: dict, sc: dict, ba: dict, p, grid_emp, grid_se) -> None:
    """Channels 1-4 from the two evaluations of the income calculators."""
    # 1. extensive margin: d ln(emp rate) = mult_age x mult_sex x sum_d w_d eps_d ln(G_s/G_b),
    #    G = net wage - weighted net out-of-work income (the financial gain from working).
    wts = par(p, "beh.ext.outside_option_weights")
    need = [st for st in ("unemployed", "disabled", "assisted") if float(wts.get(st, 0.0)) != 0.0]
    if sc["net_w"] is not None and ba["net_w"] is not None and all(
        st in sc["out"] and st in ba["out"] for st in need
    ):
        floor = float(par(p, "beh.ext.min_gain_share"))

        # bargaining: each worker recoups part of the loss in net wage in the gross wage
        comp = min(1.0, max(0.0, float(dyn_par(p, "dyn.wage_bargaining.compensation_employee_taxes"))))

        def gain(side):
            out = sum(float(wts[st]) * side["out"][st] for st in need) if need else 0.0
            net = side["net_w"]
            if side is sc and comp != 0.0:
                net = net + comp * (ba["net_w"] - net)
            return np.maximum(net - out, np.maximum(floor * net, 1.0))

        ln_g = _log_ratio(gain(sc), gain(ba))
        resp = _decile_param(p, "beh.ext.elasticity_by_decile") * ln_g
        by_group = _group_weights(p) @ resp                       # [3, 2]
        m_age = par(p, "beh.ext.age_multiplier")
        m_sex = par(p, "beh.ext.sex_multiplier")
        mult = np.array([[float(m_age[a]) * float(m_sex[s]) for s in SEXES] for a in AGE_GROUPS])
        t["ext"] = by_group * mult

    # 2. intensive margin: income-weighted eps_d x ln((1-mtr_s)/(1-mtr_b)).
    floor_keep = 0.05   # numerical floor on the net-of-marginal-tax rate (mtr capped at 95%)
    if sc["keep_emp"] is not None and ba["keep_emp"] is not None:
        ln_k = _log_ratio(np.maximum(sc["keep_emp"], floor_keep), np.maximum(ba["keep_emp"], floor_keep))
        # bargaining: the gross wage recoups part of the mean loss in net wage, and with it the
        # net return to an extra hour (same share as in the gain from working above)
        if sc["net_w"] is not None and ba["net_w"] is not None and ln_k.any():
            comp = min(1.0, max(0.0, float(dyn_par(p, "dyn.wage_bargaining.compensation_employee_taxes"))))
            if comp != 0.0 and (sc["net_w"] > 0).all() and (ba["net_w"] > 0).all():
                ln_k = ln_k - comp * float(_log_ratio(sc["net_w"], ba["net_w"]).mean())
        share = grid_emp / grid_emp.sum()
        t["intensive_emp"] = float(share @ (_decile_param(p, "beh.eti.employee_by_decile") * ln_k))
    if sc["keep_se"] is not None and ba["keep_se"] is not None:
        ln_k = _log_ratio(np.maximum(sc["keep_se"], floor_keep), np.maximum(ba["keep_se"], floor_keep))
        share = grid_se / grid_se.sum()
        t["self_emp"] = float(share @ (_decile_param(p, "beh.eti.self_employed_by_decile") * ln_k))

    # 3. labour demand, at unchanged gross wage, in two parts:
    #    - the cost of labour as a whole: eps_d x ln(cost_s/cost_b) weighted with the deciles'
    #      shares in the baseline labour cost (what a euro of contributions buys);
    #    - the composition of employment: where the cost falls more than on average, jobs are
    #      cheap in euros, so the same budget buys more of them. That headcount premium (the
    #      headcount-weighted response minus the cost-weighted one) is taken with the share
    #      `beh.ld.composition_net_share`, because part of those jobs replace better-paid
    #      workers, and it builds up at its own, slower speed (`beh.lambda`).
    if sc["cost"] is not None and ba["cost"] is not None and (ba["cost"] > 0).all() and (sc["cost"] > 0).all():
        resp = _decile_param(p, "beh.ld.elasticity_by_decile") * _log_ratio(sc["cost"], ba["cost"])
        if resp.any():
            whole = float((ba["cost"] / ba["cost"].sum()) @ resp)
            t["labour_demand"] = whole
            t["labour_composition"] = float(par(p, "beh.ld.composition_net_share")) * (float(resp.mean()) - whole)
            # where the jobs are: each decile responds to its own cost change. Around the mean
            # response that is a shift of employment between deciles (the jobs created where
            # cost fell most, of which the share 1 - composition_net_share replaces workers
            # across the distribution, is exactly the two totals above plus this shift)
            if float(np.ptp(resp)) > 1e-9:
                t["labour_decile"] = resp - resp.mean()

    # 4. retirement: years = -kappa x 100 x mean_d ln(NRR_s/NRR_b), NRR = net pension / net wage.
    if (
        sc["net_w"] is not None and ba["net_w"] is not None
        and "retired" in sc["out"] and "retired" in ba["out"]
    ):
        ok = (sc["net_w"] > 0) & (ba["net_w"] > 0) & (sc["out"]["retired"] > 0) & (ba["out"]["retired"] > 0)
        if ok.all():
            ln_r = _log_ratio(sc["out"]["retired"] / sc["net_w"], ba["out"]["retired"] / ba["net_w"])
            t["retirement"] = -float(par(p, "beh.ret.years_per_pct_replacement")) * 100.0 * float(ln_r.mean())


def _scalar_rate(v) -> float | None:
    """A rate returned by a calculator as float (dict -> mean of values), None if unusable."""
    if isinstance(v, dict):
        vals = [float(x) for x in v.values()]
        v = sum(vals) / len(vals) if vals else None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rate_pair(errors, owner, name, p, base_p, optional=False):
    if optional and (owner is None or not hasattr(owner, name)):
        return None
    a, b = _scalar_rate(_call(errors, owner, name, p)), _scalar_rate(_call(errors, owner, name, base_p))
    return None if a is None or b is None else (a, b)


def _rate_targets(t: dict, rev, exp, p, base_p, errors: list) -> None:
    """Channels 4 (bonus/malus), 5, 6, 7 from the rate calculators."""
    # 5. consumption: ln factor = (eps_own + eps_cross_border) x ln(1 + pass_through x dP),
    #    dP = (1+t_s)/(1+t_b) - 1.
    rs = _call(errors, rev, "consumption_tax_rates", p)
    rb = _call(errors, rev, "consumption_tax_rates", base_p)
    if isinstance(rs, dict) and isinstance(rb, dict):
        eps, cross, pas = (par(p, f"beh.cons.{n}") for n in ("elasticity", "cross_border", "pass_through"))
        for i, cat in enumerate(CONS_CATS):
            a, b = _scalar_rate(rs.get(cat)), _scalar_rate(rb.get(cat))
            if a is None or b is None or a <= -1.0 or b <= -1.0:
                continue
            dp = float(pas.get(cat, 1.0)) * ((1.0 + a) / (1.0 + b) - 1.0)
            if dp <= -1.0:
                continue
            t["cons"][i] = (float(eps.get(cat, 0.0)) + float(cross.get(cat, 0.0))) * math.log1p(dp)

    # 6. corporate: ln base = semi x (tau_s - tau_b); productivity likewise (slow).
    pair = _rate_pair(errors, rev, "cit_effective_rate", p, base_p)
    if pair is not None:
        d = pair[0] - pair[1]
        t["cit_base"] = float(par(p, "beh.cit.base_semi_elasticity")) * d
        t["cit_productivity"] = float(par(p, "beh.cit.productivity_semi_elasticity")) * d

    # 7. capital income: ln base = eps x ln((1-t_s)/(1-t_b)).
    pair = _rate_pair(errors, rev, "capital_income_tax_rate", p, base_p)
    if pair is not None and pair[0] < 1.0 and pair[1] < 1.0:
        t["capital_income"] = float(par(p, "beh.cap.net_of_tax_elasticity")) * math.log(
            (1.0 - pair[0]) / (1.0 - pair[1])
        )

    # 7b. property transactions (optional calculator, not in the contract).
    pair = _rate_pair(errors, rev, "property_transaction_tax_rate", p, base_p, optional=True)
    if pair is not None:
        t["property"] = float(par(p, "beh.prop.transactions_semi_elasticity")) * (pair[0] - pair[1])

    # 4b. retirement bonus/malus (optional calculator, not in the contract): extra pension per
    #     additional year worked, as a fraction of the pension.
    #     The response saturates in the level of the incentive, years = f(scenario) - f(baseline)
    #     with f(a) = max x tanh(slope x a / max): those who react to a penalty already do so
    #     when it is small, and nobody can postpone beyond the legal age. Being a difference of
    #     levels, the effect of one reform does not depend on which other one is the baseline.
    pair = _rate_pair(errors, exp, "pension_accrual_incentive", p, base_p, optional=True)
    if pair is not None and pair[0] != pair[1]:
        slope = float(par(p, "beh.ret.years_per_pp_accrual")) * 100.0
        cap = abs(float(par(p, "beh.ret.accrual_max_years")))
        if cap > 0.0:
            t["retirement"] += cap * (math.tanh(slope * pair[0] / cap) - math.tanh(slope * pair[1] / cap))


_CLIP_KEY = {
    "ext": "emp_rate", "intensive_emp": "productivity", "self_emp": "self_employed",
    "labour_demand": "labour_demand", "labour_composition": "labour_demand",
    "labour_decile": "labour_demand",
    "retirement": "retirement", "cons": "consumption",
    "cit_base": "cit_base", "cit_productivity": "productivity",
    "capital_income": "capital_income", "property": "property",
}
_LAMBDA_KEY = {
    "ext": "labour_supply", "intensive_emp": "intensive", "self_emp": "intensive",
    "labour_demand": "labour_demand", "labour_composition": "labour_composition",
    "labour_decile": "labour_composition",
    "retirement": "retirement", "cons": "consumption",
    "cit_base": "cit_base", "cit_productivity": "cit_productivity",
    "capital_income": "capital_income", "property": "property",
}


def _real_step(p, hg: float) -> float:
    """One year's change of the general benefit level relative to the health index, as the
    expenditure module applies it (neutral if its parameters are not in the view)."""
    f = p.get("exp.benefit_indexation_factor") if p is not None else None
    w = p.get("exp.welfare_adjustment") if p is not None else None
    f = 1.0 if f is None else float(f)
    return (1.0 + f * hg) / (1.0 + hg) * (1.0 + (0.0 if w is None else float(w)))


def benefit_level(state, prev, p, ctx) -> float:
    """General benefit level of the scenario relative to the baseline (cumulative)."""
    last = ((prev.extra.get("behaviour") or {}).get("benefit_level", 1.0)) if prev is not None else 1.0
    try:
        y, base = state.year, ctx.baseline
        hg_s = state.macro["health_index"] / prev.macro["health_index"] - 1.0
        hg_b = base.at(y).macro["health_index"] / base.at(y - 1).macro["health_index"] - 1.0
        step_s, step_b = _real_step(p, hg_s), _real_step(ctx.base_p, hg_b)
        if step_s == step_b:
            return float(last)
        out = float(last) * step_s / step_b
    except Exception:                      # fake states in unit tests, missing years
        return float(last)
    return out if math.isfinite(out) and out > 0.0 else float(last)


def _wage_cost_gap(state, base, key: str = "real_unit_labour_cost") -> float:
    """ln(real unit labour cost of the scenario / of the baseline); 0 if not published."""
    try:
        a, b = float(state.macro[key]), float(base.macro[key])
    except (KeyError, TypeError, AttributeError):
        return 0.0
    if a == b or not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
        return 0.0
    return math.log(a / b)


def _bargained_wage_index(state) -> float:
    """Bargained shift of gross wages relative to the baseline (macro.prices); 1 if not published."""
    try:
        v = float(state.macro.get("wage_bargaining_index", 1.0))
    except (TypeError, ValueError, AttributeError):
        return 1.0
    return v if math.isfinite(v) and v > 0.0 else 1.0


def long_run_targets(state, p, ctx, errors: list, benefit_level: float = 1.0) -> dict:
    """Scaled and clipped long-run targets x* of every channel for this year."""
    base_p, base = ctx.base_p, ctx.base_state
    t = _zero_targets()

    k = 1.0
    try:
        k = float(state.macro["wage_index"]) / float(base.macro["wage_index"])
    except (KeyError, TypeError, ZeroDivisionError):
        k = 1.0
    if not (math.isfinite(k) and k > 0.0):
        k = 1.0

    # the income calculators see legislation only: the bargained wage shift acts through labour
    # cost (3b), so it is taken out of the wage ratio used to make nominal amounts comparable
    shift = _bargained_wage_index(state)
    if shift != 1.0:
        k = k / shift

    if not (k == 1.0 and benefit_level == 1.0 and _same_policy(p, base_p)):
        rev, exp = _calculators()
        wd = np.asarray(base.wage_dist, dtype=float)
        if wd.shape == (len(J), N_DECILE) and np.isfinite(wd).all():
            grid_emp, grid_se = wd[J["emp_private"]], wd[J["self_employed"]]
            if (grid_emp > 0).all() and (grid_se > 0).all():
                h = float(par(p, "beh.marginal_step"))
                sc = _income_side(rev, exp, p, grid_emp, grid_se, k, h, errors, benefit_level)
                ba = _income_side(rev, exp, base_p, grid_emp, grid_se, 1.0, h, errors)
                _income_targets(t, sc, ba, p, grid_emp, grid_se)
            else:
                errors.append("baseline wage_dist not positive: income channels neutral")
        else:
            errors.append("baseline wage_dist unusable: income channels neutral")
        _rate_targets(t, rev, exp, p, base_p, errors)

    # 3b. labour demand: wage level relative to baseline, for unchanged prices and productivity
    #     (index jump, wage norm, indexation of wages on tax-induced price rises), with the
    #     mean labour-demand elasticity.
    gap = _wage_cost_gap(state, base)
    if gap != 0.0:
        t["labour_demand"] += (float(dyn_par(p, "dyn.wage_cost.labour_demand_weight"))
                               * float(_decile_param(p, "beh.ld.elasticity_by_decile").mean()) * gap)

    sc_ = scale(p)
    clip = par(p, "beh.clip")
    for name, v in t.items():
        c = abs(float(clip[_CLIP_KEY[name]]))
        v = np.clip(np.nan_to_num(np.asarray(v, dtype=float) * sc_, nan=0.0, posinf=c, neginf=-c), -c, c)
        t[name] = v if v.ndim else float(v)
    return t


# --------------------------------------------------------------------------- dynamics
def _advance(x_prev: dict, target: dict, p) -> dict:
    lam = par(p, "beh.lambda")
    x: dict = {}
    for name, tv in target.items():
        lm = min(1.0, max(0.0, float(lam[_LAMBDA_KEY[name]])))
        pv = x_prev.get(name)
        if isinstance(tv, np.ndarray):
            pv = np.asarray(pv, dtype=float) if pv is not None else None
            if pv is None or pv.shape != tv.shape or not np.isfinite(pv).all():
                pv = np.zeros_like(tv)
            x[name] = pv + lm * (tv - pv)
        else:
            pv = float(pv) if isinstance(pv, (int, float, np.floating)) and math.isfinite(pv) else 0.0
            x[name] = pv + lm * (tv - pv)
    return x


def _base_emp_rate(base) -> np.ndarray:
    """Baseline employment rate [age, sex] (all regions)."""
    pop = base.pop.sum(axis=2)
    emp = base.status[_EMPLOYED].sum(axis=(0, 3))
    rate = np.divide(emp, pop, out=np.zeros_like(pop, dtype=float), where=pop > 0)
    return np.clip(rate, 0.0, 1.0)


def _emp_rate_delta(x_ext: np.ndarray, base, p) -> np.ndarray:
    """Additive employment-rate change [age, sex] = baseline rate x (exp(x) - 1), bounded."""
    delta = np.zeros((N_AGE, N_SEX))
    if not x_ext.any():
        return delta
    bands = [int(b) for b in par(p, "beh.ext.age_bands")]
    rate = _base_emp_rate(base)
    for i in range(len(AGE_GROUPS)):
        a0, a1 = max(0, bands[i]), min(N_AGE, bands[i + 1])
        delta[a0:a1] = rate[a0:a1] * np.expm1(x_ext[i])[None, :]
    cap = abs(float(par(p, "beh.clip")["emp_rate_delta_abs"]))
    delta = np.clip(delta, -cap, cap)
    return np.clip(delta, -rate, 1.0 - rate)      # the rate stays within [0, 1]


def _investment_line(meta, p) -> bool:
    """True for an expenditure line that is public capital formation only (ESA code)."""
    esa = str(getattr(meta, "esa", "") or "").upper()
    tokens = [tok for tok in esa.replace("+", " ").replace(",", " ").replace("/", " ").split() if tok]
    pre = [str(x).upper() for x in par(p, "beh.pubcap.investment_esa_prefixes")]
    return bool(tokens) and all(any(tok.startswith(x) for x in pre) for tok in tokens)


def public_capital_gap(prev, prev_mem: dict, p, ctx) -> float:
    """Public capital stock of the scenario minus the baseline's at the start of the year, in
    points of baseline GDP: last year's gap after depreciation plus last year's difference in
    public investment. Exactly 0 while public investment equals the baseline's."""
    last = prev_mem.get("public_capital_gap", 0.0)
    last = float(last) if isinstance(last, (int, float, np.floating)) and math.isfinite(last) else 0.0
    d_inv = 0.0
    try:
        base_prev = ctx.baseline.at(prev.year)
        gdp = float(base_prev.macro["gdp_nominal"])
        diff = {k: float(prev.expenditure.get(k, 0.0)) - float(base_prev.expenditure.get(k, 0.0))
                for k in prev.expenditure.keys() | base_prev.expenditure.keys()}
        diff = {k: d for k, d in diff.items() if d != 0.0}
        if diff and gdp > 0.0:
            from .demand import _line_meta
            meta = _line_meta()
            d_inv = sum(d for k, d in diff.items() if _investment_line(meta.get(k), p)) / gdp
    except Exception:                      # fake states in unit tests, missing years
        d_inv = 0.0
    if last == 0.0 and d_inv == 0.0:
        return 0.0
    out = (1.0 - float(par(p, "beh.pubcap.depreciation_rate"))) * last + d_inv
    return out if math.isfinite(out) else 0.0


# ------------------------------------------------------------------------------- run
def run(state, prev, p, ctx) -> None:
    state.adjust = {}
    mem: dict = {"x": {}, "target": {}, "errors": []}
    state.extra["behaviour"] = mem
    base = ctx.base_state
    if ctx.base_p is None or base is None or not enabled(p):
        return                               # static costing: every key absent = neutral

    prev_mem = (prev.extra.get("behaviour") if prev is not None else None) or {}
    mem["benefit_level"] = benefit_level(state, prev, p, ctx)
    target = long_run_targets(state, p, ctx, mem["errors"], mem["benefit_level"])
    x = _advance(prev_mem.get("x") or {}, target, p)
    mem["x"], mem["target"] = x, target
    k_gap = public_capital_gap(prev, prev_mem, p, ctx) if prev is not None else 0.0
    mem["public_capital_gap"] = k_gap
    c = abs(float(par(p, "beh.clip")["productivity"]))
    pubcap = min(c, max(-c, float(par(p, "beh.pubcap.output_per_capital")) * k_gap * scale(p))) if k_gap != 0.0 else 0.0

    adj = state.adjust
    adj["emp_rate_delta"] = _emp_rate_delta(x["ext"], base, p)
    adj["labour_demand_factor"] = math.exp(x["labour_demand"] + x["labour_composition"])
    if np.any(x["labour_decile"] != 0.0):
        adj["labour_demand_decile_shift"] = x["labour_decile"] - x["labour_decile"].mean()
    adj["retirement_shift"] = x["retirement"]
    adj["cons_factor"] = {cat: math.exp(v) for cat, v in zip(CONS_CATS, x["cons"])}
    adj["cit_base_factor"] = math.exp(x["cit_base"])
    adj["capital_income_base_factor"] = math.exp(x["capital_income"])
    adj["self_employed_income_factor"] = math.exp(x["self_emp"])
    adj["property_transactions_factor"] = math.exp(x["property"])
    # hours and effort of private employees move output by their share of labour income: the
    # output of government is measured at its cost and the self-employed have their own channel
    intensive = x["intensive_emp"]
    if intensive != 0.0:
        try:
            bl = base.labour
            priv = float(bl["wage_bill_private"])
            total = priv + float(bl.get("wage_bill_public_actual", bl["wage_bill_public"])) + float(bl["self_employed_income"])
            if 0.0 < priv < total:
                intensive *= priv / total
        except (KeyError, TypeError, AttributeError):
            pass
    adj["productivity_factor"] = math.exp(intensive + x["cit_productivity"] + pubcap)
