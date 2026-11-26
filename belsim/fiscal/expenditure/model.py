"""Expenditure model: every line of Belgian general-government expenditure (ESA 2010, S13).

    line_t = model_t(state, params) x calibration_factor,   factor = actual_2024 / model_2024

`model_t` is driver based (beneficiaries x average benefit, pupils x cost, % of GDP paths, ...).
The calibration factors and all year-to-year stocks (pension averages, cost indices, ...) are
carried in `state.extra["expenditure"]` and read back from `prev.extra["expenditure"]`: the
function is idempotent and keeps no module state.

Regional levers. Spending that is a competence of the Regions or Communities has, next to the
national lever, a factor keyed by region `{FLA, WAL, BRU}` or by Community
`{flemish_community, french_community}` (neutral = 1; public investment: an add-on in % of
regional GDP, neutral = 0): `educ.community_cost_factor`, `educ.community_enrolment_factor`,
`family.childcare_factor`, `health.ltc_regional_factor`, `subsidy.service_voucher_factor`,
`subsidy.target_group_factor`, `invest.regional_extra_pct_gdp` (and `family.allowance_per_child`,
an amount per region). A line is multiplied by 1 + sum_k weight_k x (factor_k - 1), the weight
being the share of the line spent for region / Community k (population, pupils, care needs ...;
only the Regions' and Communities' part of the line where other levels also spend). The amounts
so added per key are reported in the carried state (`regional_delta`: line -> key -> EUR mn) for
`balance` (they are spending of the regions-and-communities entity) and `belsim.regional`
(which federated entity pays). Exactly neutral at the neutral values.

Spending paths by entity. `entity.staff_cost_factor`, `entity.purchases_factor`,
`entity.subsidies_factor`, `entity.current_transfers_factor`, `entity.capital_transfers_factor`
and `entity.investment_factor` are level factors (1 = unchanged policy) keyed by entity
`{federal, social_security, flemish_community, walloon_region, french_community,
brussels_region, local}`: the budget path of that entity for that ESA category (wages,
purchases, subsidies, other current transfers, capital transfers, investment) relative to the
drivers of the lines. A line gets, on top of its drivers, the unchanged-policy line of the
baseline run x the sum over entities and categories of share of the entity in the line x
share of the category in the line (both 2024) x (factor - 1), so that the other levers of the
line keep their full effect;
interest, defence and the EU contribution keep their own path parameters, health care and
social benefits their own drivers. The amounts are reported per line and entity (`entity_delta`) so that
`balance` and `belsim.regional` book them on the entity that decided them.

Base year in a scenario run: the factors and reference values are taken from the baseline's
2024 state (`ctx.base_state`), so a measure that already applies in 2024 is not calibrated away.
"""
from __future__ import annotations

import numpy as np

from ...core.dims import ENTITIES, J, REGIONS, S
from . import data
from .calc import JOBTYPE_SCHEME, PENSION_SCHEMES, replacement_income
from .lines import ITEMS, all_defs

NS = "expenditure"
_SCHEME_JOB = {v: J[k] for k, v in JOBTYPE_SCHEME.items()}      # scheme -> jobtype index
_SUBSIDY_KEY = {"subsidy_transport": "transport", "subsidy_energy_environment": "energy_environment",
                "subsidy_other": "other"}
_EDUC = {  # line -> (age slice, enrolment key, cost key)
    "educ_primary": (slice(3, 12), "primary", "primary"),
    "educ_secondary": (slice(12, 18), "secondary", "secondary"),
    "educ_tertiary": (slice(18, 25), "tertiary", "tertiary"),
    "educ_other": (slice(3, 25), None, "other"),
}


def _profile(ages, costs, n=101) -> np.ndarray:
    out = np.zeros(n)
    bounds = list(ages) + [n]
    for lo, hi, c in zip(bounds[:-1], bounds[1:], costs):
        out[int(lo):int(hi)] = c
    return out


def _shifted(profile: np.ndarray, years: float, from_age: int = 50) -> np.ndarray:
    """Age profile moved `years` to older ages from `from_age` on (linear between single ages)."""
    if years <= 0.0:
        return profile
    ages = np.arange(len(profile), dtype=float)
    out = profile.copy()
    out[from_age:] = np.interp(ages[from_age:] - years, ages, profile)
    return out


def _labour(state) -> dict:
    """Labour-market aggregates with fallbacks (other modules may provide only part of the keys)."""
    lab, st, wd = state.labour, state.status, state.wage_dist
    wi = state.macro.get("wage_index", 1.0)
    n_priv, n_pub, n_self = (float(st[S[k]].sum()) for k in ("emp_private", "emp_public", "self_employed"))

    def avg(key, j, default):
        v = lab.get(key)
        if v:
            return float(v)
        v = float(wd[j].mean())
        return v if v > 0 else default * wi

    w_priv = avg("avg_wage_private", J["emp_private"], 50_000.0)
    w_pub = avg("avg_wage_public", J["emp_public"], 55_000.0)
    w_self = avg("avg_income_self_employed", J["self_employed"], 40_000.0)
    dist = np.array(wd, dtype=float)
    for j, w in ((J["emp_private"], w_priv), (J["emp_public"], w_pub), (J["self_employed"], w_self)):
        if dist[j].sum() <= 0:
            dist[j] = w
    emp = float(lab.get("employment") or (n_priv + n_pub + n_self))
    return {
        "w_priv": w_priv, "w_pub": w_pub, "w_self": w_self, "dist": dist,
        "wb_priv": float(lab.get("wage_bill_private") or n_priv * w_priv / 1e6),
        "wb_pub": float(lab.get("wage_bill_public") or n_pub * w_pub / 1e6),
        "employment": emp,
        "vouchers": float(lab.get("service_voucher_workers") or emp),
        "statutory": float(lab.get("statutory_share") or 0.5),
        # optional keys of the labour module (it models the RCC run-off and the benefit time limit)
        "rcc": lab.get("rcc_beneficiaries"),
        "excluded": float(lab.get("unemp_excluded_to_inactive") or 0.0),
    }


def _worker_mix(state, statutory: float) -> dict[str, float]:
    """Pension-scheme mix of workers aged 55-64 (the next retirees)."""
    st = state.status
    priv, pub, self_ = (float(st[S[k]][55:65].sum()) for k in ("emp_private", "emp_public", "self_employed"))
    mix = {"employees": priv + pub * (1 - statutory), "self_employed": self_, "civil_servants": pub * statutory}
    tot = sum(mix.values())
    return {k: (v / tot if tot > 0 else 1 / 3) for k, v in mix.items()}


REG_ENTITY = "regions_communities"
COMMUNITIES = ("flemish_community", "french_community")
FEDERATED = ("flemish_community", "walloon_region", "french_community", "brussels_region")
# Spending paths by entity: category -> (parameter, ESA transactions of the category)
PATH_CATEGORIES = {
    "staff": ("entity.staff_cost_factor", ("D1",)),
    "purchases": ("entity.purchases_factor", ("P2", "D29_D5_D8")),
    "subsidies": ("entity.subsidies_factor", ("D3",)),
    "transfers": ("entity.current_transfers_factor", ("D7",)),
    "capital_transfers": ("entity.capital_transfers_factor", ("D9",)),
    "investment": ("entity.investment_factor", ("OP5ANP",)),
}
# lines that carry the paths: operating, education, subsidy, investment and capital-transfer lines
# (interest, defence, the EU contribution, health care and cash benefits keep their own drivers)
PATH_DRIVERS = ("generic", "children_services", "education", "investment", "capital_transfers", "subsidy",
                "subsidy_wage", "service_vouchers")


def path_lines() -> tuple[str, ...]:
    return tuple(lid for lid, d in all_defs().items() if d.driver in PATH_DRIVERS)


def _entity_paths(state, prev, p, L0) -> dict[str, dict[str, float]]:
    """line -> entity -> fraction of the unchanged-policy line added by the spending paths of
    the entities (`entity.*_factor`, neutral = 1): share of the entity in the line (2024 accounts;
    for the federated entities their weight in the regions-and-communities part, see
    `regional.government.path_weights`) x share of the ESA category in the line (2024) x
    (factor - 1). Empty when every factor is 1."""
    active = {}
    for cat, (name, items) in PATH_CATEGORIES.items():
        f = p.get(name)
        if f:
            d = {e: float(v) - 1.0 for e, v in f.items() if float(v) != 1.0}
            if d:
                active[cat] = (d, items)
    if not active:
        return {}
    from ...regional import government as _gov
    w4 = _gov.path_weights(state, prev, p)
    out: dict[str, dict[str, float]] = {}
    lines = path_lines()
    for lid, row in L0.items():
        if lid not in lines or not row["value"]:
            continue
        sh = data.entity_shares(lid)
        if not sh:
            continue
        w = None
        for cat, (d, items) in active.items():
            part = sum(row[it] for it in items) / row["value"]
            if part == 0.0:
                continue
            for e, dv in d.items():
                if e in FEDERATED:
                    if w4 is None or not sh.get(REG_ENTITY):
                        continue
                    if w is None:
                        w = w4(lid)
                    frac = sh[REG_ENTITY] * float(w[FEDERATED.index(e)]) * part * dv
                else:
                    frac = sh.get(e, 0.0) * part * dv
                if frac:
                    out.setdefault(lid, {})
                    out[lid][e] = out[lid].get(e, 0.0) + frac
    return out


def _community_weights(pop_region: np.ndarray, p) -> dict[str, float]:
    """Share of a population by region served by each Community (Brussels split parameter)."""
    b = float(p.get("regional.brussels_dutch_share", 0.2) or 0.0)
    tot = max(float(pop_region.sum()), 1e-9)
    fl = (float(pop_region[0]) + b * float(pop_region[2])) / tot
    return {"flemish_community": fl, "french_community": 1.0 - fl}


def _region_weights(a: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    tot = max(float(np.sum(a)), 1e-9)
    return {r: scale * float(a[i]) / tot for i, r in enumerate(REGIONS)}


class _RegionalLevers:
    """Applies region / Community keyed factors to lines and records what they add."""

    def __init__(self):
        self.frac: dict[str, dict[str, float]] = {}      # line -> key -> fraction of the unlevered line
        self.mult: dict[str, float] = {}
        self.direct: dict[str, dict[str, float]] = {}    # line -> key -> raw amount added

    def factor(self, line: str, weights: dict[str, float], *factors) -> float:
        m = 1.0
        for k, w in weights.items():
            f = 1.0
            for fac in factors:
                if fac:
                    f *= float(fac.get(k, 1.0))
            if f != 1.0:
                m += w * (f - 1.0)
                self.frac.setdefault(line, {})[k] = w * (f - 1.0)
        if line in self.frac:
            self.mult[line] = m
        return m

    def amounts(self, raw: dict, calib: dict) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for line, fr in self.frac.items():
            base = raw[line] / self.mult[line] if self.mult[line] else 0.0
            out[line] = {k: float(v * base * calib[line]) for k, v in fr.items()}
        for line, d in self.direct.items():
            out.setdefault(line, {}).update({k: float(v * calib[line]) for k, v in d.items()})
        return out


def _norm(d: dict, keys) -> dict:
    tot = sum(max(float(d[k]), 0.0) for k in keys)
    return {k: (max(float(d[k]), 0.0) / tot if tot > 0 else 1.0 / len(keys)) for k in keys}


def _new_pension(scheme, dist0, hi, career_real, w_idx, p, extra_years: float = 0.0) -> float:
    """Average formula pension of a new retiree: career wages = base-year decile wages of the job
    type x health index x lagged real wage index; ceilings/minima raised by the welfare index;
    `extra_years` = career years gained by retiring later than in 2024."""
    wages = dist0[_SCHEME_JOB[scheme]] * hi * career_real
    return float(w_idx * replacement_income("retired", wages / w_idx, p, scheme, extra_years).mean())


def run(state, prev, p, ctx) -> None:
    px = prev.extra.get(NS) if prev is not None else None
    bx = None
    if prev is None and ctx is not None:
        bs = getattr(ctx, "base_state", None)
        if bs is not None:
            bx = bs.extra.get(NS)
    # unchanged-policy lines of the baseline run in the same year: the spending paths of the
    # entities are amounts relative to those, so that other levers keep their full effect
    by = getattr(ctx, "base_state", None) if ctx is not None else None
    base_lines = (by.extra.get(NS) or {}).get("line_unchanged") if by is not None else None
    hi_lag = 1.0
    base_res = getattr(ctx, "baseline", None) if ctx is not None else None
    if by is not None and prev is not None and base_res is not None and base_res.has(state.year - 1):
        dev_now = state.macro["health_index"] / by.macro["health_index"]
        dev_prev = prev.macro["health_index"] / base_res.at(state.year - 1).macro["health_index"]
        hi_lag = dev_prev / dev_now
    # pensioners of the baseline run this year, when a behavioural retirement response is active
    base_n = None
    if by is not None and float((state.adjust or {}).get("retirement_shift", 0.0) or 0.0) != 0.0:
        base_n = (by.extra.get(NS) or {}).get("pension_n")
    values, x = compute(state, prev, px, p, bx, base_lines, hi_lag, base_n)
    state.expenditure = values
    state.extra[NS] = x


def compute(state, prev, px, p, bx=None, base_lines=None, hi_lag: float = 1.0,
            base_pension_n: float | None = None) -> tuple[dict[str, float], dict]:
    """Returns (line -> EUR mn, carried state). `px` = last year's carried state (None in the base
    year); `bx` = the baseline's base-year carried state when running a scenario's base year;
    `base_lines` = the baseline run's unchanged-policy lines of this year (None in a baseline run)."""
    L0 = data.lines_2024()
    if not L0:
        raise RuntimeError("data/base/expenditure_lines_2024.csv missing: run belsim.data.fetch_expenditure")
    sc = data.scalars()
    m, st, pop = state.macro, state.status, state.pop
    first = px is None
    hi, cpi, wi = m["health_index"], m["cpi"], m["wage_index"]
    # Spending rules tied to GDP (targets in % of GDP, purchases, subsidies, other transfers)
    # follow GDP net of the demand gap of a scenario: budgets are not cut within the year
    # because demand dips (purchases are fixed in volume in the HERMES variants). In the
    # baseline the demand gap is zero and the two are identical.
    dgap = float(m.get("demand_gap", 0.0) or 0.0)
    gdp_actual = m["gdp_nominal"]
    gdp = gdp_actual / (1.0 + dgap) if dgap else gdp_actual
    gdp_real = m.get("gdp_real") or gdp_actual / cpi
    if dgap:
        gdp_real = gdp_real / (1.0 + dgap)
    gap = m.get("output_gap", m.get("demand_gap", 0.0))   # total gap: exogenous cycle + demand feedback
    pop_age = pop.sum(axis=(1, 2))
    pop_as = pop.sum(axis=2)
    pop_tot = float(pop_age.sum())
    lab = _labour(state)
    i10, ish = m["interest_rate_10y"], m["interest_rate_short"]
    pop_r_age = pop.sum(axis=1)                             # [age, region]
    rc_share = {lid: data.entity_shares(lid).get(REG_ENTITY, 0.0) for lid in L0}
    levers = _RegionalLevers()

    # ---------- needs indicators ----------
    # Healthy ageing: a share of the years of life expectancy gained since the base year is
    # spent in good health, i.e. the age profiles of care needs move to older ages by that many
    # years (profile read at age - shift).
    src0 = px if px is not None else bx
    life = float((state.demo or {}).get("life_expectancy", 0.0) or 0.0)
    life0 = float(src0["refs"].get("life_exp", life)) if src0 is not None else life
    gain = max(life - life0, 0.0) if life0 > 0 else 0.0
    sh_h = gain * float(p.get("health.healthy_ageing_share", 0.0) or 0.0)
    sh_l = gain * float(p.get("health.ltc_healthy_ageing_share", 0.0) or 0.0)
    prof_m = _shifted(_profile(p["health.profile_ages"], p["health.profile_cost"]), sh_h)
    prof_f = _shifted(_profile(p["health.profile_ages"], p["health.profile_cost"])
                      * _profile(p["health.profile_ages"], p["health.profile_female_ratio"]), sh_h)
    ltc_prof = _shifted(_profile(p["health.ltc_profile_ages"], p["health.ltc_profile_cost"]), sh_l)
    need_health = float(pop_as[:, 0] @ prof_m + pop_as[:, 1] @ prof_f)
    need_ltc = float(pop_age @ ltc_prof)
    kids_0_11 = float(pop_age[:12].sum())

    # ---------- reference values (base year of the baseline) ----------
    src = px if px is not None else bx
    if src is not None:
        refs = src["refs"]
    else:
        refs = {
            "gdp": gdp, "wb_pub": lab["wb_pub"], "wb_priv": lab["wb_priv"], "w_pub": lab["w_pub"],
            "w_priv": lab["w_priv"], "w_self": lab["w_self"], "employment": lab["employment"],
            "vouchers": lab["vouchers"], "dist": lab["dist"].copy(), "need_ltc": need_ltc,
            "kids_0_11": kids_0_11, "copay": p["health.copayment_share"],
            "ecr": p["pubwage.employer_contribution_rate"],
            "worker_mix": _worker_mix(state, lab["statutory"]), "life_exp": life,
        }
        # new-retiree pension = k x formula; k fitted so that it equals stock average x ratio in 2024
        refs["pension_k"] = {
            s: p["pension.avg_pension_2024"][s] * p["pension.new_to_stock_ratio"][s]
            / max(_new_pension(s, refs["dist"], 1.0, 1.0, 1.0, p), 1.0)
            for s in PENSION_SCHEMES
        }
        refs["min_pension0"] = {s: float(p["pension.min_pension"][s]) / hi for s in PENSION_SCHEMES}
        refs["career_s0"] = {
            s: (1 + p["pension.past_real_wage_growth"]) ** (-p["pension.career_wage_lag_years"][s])
            for s in PENSION_SCHEMES
        }

    # ---------- carried indices ----------
    if first:
        w_idx, bidx, interm_vol, health_real, ltc_cost, educ_cost = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        career_s = dict(refs["career_s0"])
        hg = g_real = 0.0
    else:
        pm = prev.macro
        hg = hi / pm["health_index"] - 1.0
        gr_prev = pm.get("gdp_real") or pm["gdp_nominal"] / pm["cpi"]
        dgap_prev = float(pm.get("demand_gap", 0.0) or 0.0)
        if dgap_prev:
            gr_prev = gr_prev / (1.0 + dgap_prev)
        g_real = gdp_real / gr_prev - 1.0
        w_idx = px["welfare_index"] * (1 + p["exp.welfare_adjustment"])
        bidx = px["benefit_index"] * (1 + p["exp.benefit_indexation_factor"] * hg) / (1 + hg)
        interm_vol = px["interm_vol"] * (1 + p["exp.intermediate_gdp_elasticity"] * g_real
                                         + p["exp.intermediate_volume_growth_extra"])
        g_pc = (1 + g_real) / (pop_tot / px["pop_total"]) - 1.0
        g_need = ((need_health / px["need_health"]) * (1 + p["health.income_elasticity"] * g_pc
                                                         + p["health.excess_cost_growth"]) - 1.0)
        # the norm caps need-driven growth; the share `health.norm_room_spent` of the room left
        # when need grows less than the norm is spent as well (the norm as a budget)
        enf = p["health.norm_enforcement"]
        norm = p["health.growth_norm"]
        spent = float(p.get("health.norm_room_spent", 0.0) or 0.0)
        g_health = (1 - enf) * g_need + enf * (min(g_need, norm) + spent * max(norm - g_need, 0.0))
        health_real = px["health_real"] * (1 + g_health)
        ltc_cost = px["ltc_cost"] * (1 + p["health.ltc_real_cost_growth"])
        educ_cost = px["educ_cost"] * (1 + p["educ.real_cost_growth"])
        career_s = {}
        for s in PENSION_SCHEMES:
            a = 1.0 / (p["pension.career_wage_lag_years"][s] + 1.0)
            key = {"employees": "w_priv", "self_employed": "w_self", "civil_servants": "w_pub"}[s]
            real_wage = lab[key] / refs[key] / hi
            career_s[s] = (1 - a) * px["career_s"][s] + a * real_wage

    i_gdp = gdp / refs["gdp"]
    i_pubwage = lab["wb_pub"] / refs["wb_pub"]
    i_pubavg = lab["w_pub"] / refs["w_pub"]
    i_wbpriv = lab["wb_priv"] / refs["wb_priv"]
    i_wb_all = (lab["wb_priv"] + lab["wb_pub"]) / (refs["wb_priv"] + refs["wb_pub"])
    ecf = (1 + p["pubwage.employer_contribution_rate"]) / (1 + refs["ecr"])
    staff = p["pubwage.staffing_factor"]
    ffac = p["exp.function_factor"]
    other_el = p["exp.other_gdp_elasticity"]
    interm_idx = cpi * interm_vol

    # ---------- beneficiaries ----------
    n_unemp = float(st[S["unemployed"]].sum())
    n_dis = float(st[S["disabled"]].sum())
    n_inactive = float(st[S["inactive"]][18:65].sum())
    rcc_n = float(lab["rcc"] if lab["rcc"] is not None else p["unemp.rcc_beneficiaries"])
    n_ret = max(float(st[S["retired"]].sum()) - rcc_n, 0.0)

    raw: dict[str, float] = {}

    # ---------- pensions (stock-flow per scheme) ----------
    level = p["pension.level_factor"]
    # Stock-flow per scheme, in pension-equivalents. New retirees = change of the stock + exits.
    # Their scheme mix starts from the 2024 mix of new pensions and follows the job-type mix of
    # older workers with the lag of a career (rights were earned over decades: the falling share
    # of statutory staff reaches new civil-servant pensions slowly). A new pension = k x formula
    # on lagged career wages; retiring later than in 2024 (labour: `retirement_delay`) adds
    # career years. Pensions in payment: indexation + welfare adjustment (employees,
    # self-employed) or the capped perequation (civil servants).
    delay = float((state.extra.get("labour") or {}).get("retirement_delay", 0.0)) \
        * float(p.get("pension.career_years_per_delay_year", 0.0) or 0.0)
    wm = _worker_mix(state, lab["statutory"])
    pension_bonus: dict[str, float] = {}
    if first:
        mix = _norm(p["pension.scheme_mix_2024"], PENSION_SCHEMES)
        avg = {s: float(p["pension.avg_pension_2024"][s]) for s in PENSION_SCHEMES}
        n_s = {s: n_ret * mix[s] for s in PENSION_SCHEMES}
        wm_lag = dict(refs["worker_mix"])
        mix_drift = {s: 1.0 for s in PENSION_SCHEMES}
    else:
        xr = p["pension.exit_rate"]         # by scheme (a scalar applies to the three schemes)
        xr = {s: float(xr[s] if isinstance(xr, dict) else xr) for s in PENSION_SCHEMES}
        surv_s = {s: px["pension_n"] * px["pension_mix"][s] * (1 - xr[s]) for s in PENSION_SCHEMES}
        surv = sum(surv_s.values())
        new = max(n_ret - surv, 0.0)
        a_mix = 1.0 / (float(p.get("pension.scheme_mix_lag_years", 0.0) or 0.0) + 1.0)
        wm_prev = px.get("pension_wm_lag") or refs["worker_mix"]
        wm_lag = {s: (1 - a_mix) * wm_prev[s] + a_mix * wm[s] for s in PENSION_SCHEMES}
        mix0 = p.get("pension.new_pension_mix_2024") or px["pension_mix"]
        drift = p.get("pension.new_pension_mix_drift") or {}
        mix_drift = {s: px.get("pension_mix_drift", {}).get(s, 1.0) * (1.0 + float(drift.get(s, 0.0)))
                     for s in PENSION_SCHEMES}
        inflow = _norm({s: mix0[s] * mix_drift[s] * wm_lag[s] / max(refs["worker_mix"][s], 1e-9)
                        for s in PENSION_SCHEMES}, PENSION_SCHEMES)
        wa = p["pension.welfare_adjustment"]     # by scheme (a scalar applies to both schemes)
        idx_gen = {s: (1 + p["exp.benefit_indexation_factor"] * hg)
                   * (1 + float(wa.get(s, 0.0) if isinstance(wa, dict) else wa)) for s in PENSION_SCHEMES}
        real_pub = (lab["w_pub"] / px["w_pub"]) / (1 + hg) - 1.0
        pereq = p["pension.perequation_share"] * real_pub
        cap = p.get("pension.perequation_cap")
        if cap is not None:
            pereq = min(pereq, float(cap))
        idx_civ = (1 + p["exp.benefit_indexation_factor"] * hg) * (1 + pereq)
        scale = min(1.0, n_ret / surv) if surv > 0 else 0.0   # stock shrinking faster than exits
        avg, n_s = {}, {}
        for s in PENSION_SCHEMES:
            old_n = surv_s[s] * scale
            new_n = new * inflow[s]
            old_avg = px["pension_avg"][s] * (idx_civ if s == "civil_servants" else idx_gen[s])
            career_real = career_s[s] / refs["career_s0"][s]
            new_avg = refs["pension_k"][s] * _new_pension(s, refs["dist"], hi, career_real, w_idx, p, delay)
            n_s[s] = old_n + new_n
            avg[s] = (old_n * old_avg + new_n * new_avg) / n_s[s] if n_s[s] > 0 else old_avg
        mix = _norm(n_s, PENSION_SCHEMES) if n_ret > 0 else dict(px["pension_mix"])
        # pension bonus paid as a capital at retirement (law of 25 April 2024): average amount per
        # new retiree, EUR of 2024, indexed; not scaled by the calibration of the pension lines
        # People who retire later (or earlier) than in the baseline because of a financial
        # incentive have a pension below the average new pension (the malus bears on early
        # retirees with short effective careers): the stock-flow above values them at the
        # average, the difference is given back here.
        rel = float(p.get("pension.postponement_pension_ratio", 1.0) or 1.0)
        if base_pension_n is not None and rel != 1.0:
            gap_n = float(base_pension_n) - n_ret
            for s in PENSION_SCHEMES:
                new_avg_s = refs["pension_k"][s] * _new_pension(s, refs["dist"], hi, career_s[s] / refs["career_s0"][s],
                                                                w_idx, p, delay)
                pension_bonus[s] = (1.0 - rel) * gap_n * inflow[s] * new_avg_s * level[s] / 1e6
        cap_bonus = float(p.get("pension.capital_bonus", 0.0) or 0.0)
        if cap_bonus:
            for s in PENSION_SCHEMES:
                pension_bonus[s] = pension_bonus.get(s, 0.0) + new * inflow[s] * cap_bonus * hi * level[s] / 1e6
    # A change of the guaranteed minimum pension relative to its 2024 level (in real terms)
    # moves the pensions at or near the minimum, in payment and new alike: the share
    # `pension.min_pension_spending_share` of the spending of the scheme follows it in
    # proportion. (The formula of new pensions above only sees the floor on a full-career
    # average wage, which hardly binds.) Exactly 1 while the minimum is the indexed 2024 amount.
    min0 = refs.get("min_pension0") or {}
    min_share = p.get("pension.min_pension_spending_share") or {}
    for s in PENSION_SCHEMES:
        raw[f"pension_{s}"] = n_s[s] * avg[s] * level[s] / 1e6
        if min0.get(s) and min_share.get(s):
            q = round(float(p["pension.min_pension"][s]) / hi / min0[s], 9)
            if q != 1.0:
                raw[f"pension_{s}"] *= 1.0 + float(min_share[s]) * (q - 1.0)
    # income guarantee for the elderly: open from the legal pension age (in the year of a change,
    # the average of the old and the new age as for pensions)
    lx = state.extra.get("labour") or {}
    age_now = int(round(float(lx.get("legal_age", p.get("labour.legal_pension_age", 65) or 65))))
    lx_prev = (prev.extra.get("labour") or {}) if prev is not None else {}
    age_prev = int(round(float(lx_prev.get("legal_age", age_now))))
    w_new = float(p.get("labour.pension_age_first_year_share", 1.0) or 1.0) if age_prev != age_now else 1.0
    n_grapa_age = w_new * float(pop_age[age_now:].sum()) + (1.0 - w_new) * float(pop_age[age_prev:].sum())
    # the take-up parameter is a share of the population aged 65+ in 2024 (legal age 65)
    raw["pension_grapa"] = (p["pension.grapa_takeup"] * n_grapa_age * p["pension.grapa_amount"]
                            * w_idx * bidx * level["grapa"] / 1e6)

    # ---------- unemployment, RCC, invalidity, sickness, allowances ----------
    dist_priv = lab["dist"][J["emp_private"]]
    ben_u = w_idx * float(replacement_income("unemployed", p["unemp.wage_position"] * dist_priv / w_idx, p).mean())
    cov = p["unemp.coverage_rate"]
    unemp_full = n_unemp * cov * ben_u * bidx / 1e6
    unemp_temp = p["unemp.temporary_rate"] * lab["wb_priv"] * max(0.2, 1 - p["unemp.temporary_gap_sensitivity"] * gap)
    raw["unemployment"] = unemp_full + unemp_temp
    raw["rcc_swt"] = rcc_n * p["unemp.rcc_avg_benefit"] * bidx / 1e6
    ben_d = w_idx * float(replacement_income("disabled", p["disab.wage_position"] * dist_priv / w_idx, p).mean())
    raw["invalidity"] = n_dis * ben_d * bidx / 1e6
    raw["sickness_primary"] = L0["sickness_primary"]["value"] * i_wb_all * p["disab.sickness_factor"]
    raw["disability_allowances"] = (p["disab.allowance_recipients_share"] * float(pop_age[18:].sum())
                                    * p["disab.allowance_amount"] * w_idx * bidx / 1e6)
    raw["labour_market_benefits"] = (L0["labour_market_benefits"]["value"] * lab["employment"] / refs["employment"]
                                     * hi * bidx)

    # ---------- family, social assistance ----------
    amounts = p["family.allowance_per_child"]
    fam = 0.0
    for r, name in enumerate(REGIONS):
        kids = float(pop[:18, :, r].sum()) + p["family.student_share"] * float(pop[18:25, :, r].sum())
        fam += kids * amounts[name]
    raw["family_allowances"] = fam * bidx / 1e6
    # people excluded from unemployment benefits (by labour's time limit: counted as inactive; by the
    # coverage lever: still counted as unemployed) claim assistance at the spill-over rate
    excluded = min(lab["excluded"], n_inactive)
    sa_amount = p["assist.amount"] * p["assist.generosity_factor"] * w_idx * bidx / 1e6
    raw["social_assistance"] = p["assist.takeup_rate"] * (n_inactive - excluded) * sa_amount
    # the spill-over is valued at the actual benefit amount, NOT scaled by the calibration factor
    # (which also covers other aid recorded in this line)
    add = {"social_assistance": p["assist.unemp_spillover"] * (excluded + n_unemp * max(1 - cov, 0.0)) * sa_amount}
    add.update({f"pension_{s}": v for s, v in pension_bonus.items()})

    # ---------- health and long-term care ----------
    # the health budget is set a year ahead: a deviation of the health index of a scenario from
    # the baseline reaches it with a lag (`hi_lag` = last year's deviation over this year's, 1 in
    # a baseline run), for the share `health.indexation_lag_share`
    lag_share = float(p.get("health.indexation_lag_share", 0.0) or 0.0)
    hi_health = hi * (hi_lag ** lag_share) if (hi_lag != 1.0 and lag_share) else hi
    raw["health_care"] = (L0["health_care"]["value"] * health_real * hi_health
                          * (1 - p["health.copayment_share"]) / (1 - refs["copay"]))
    ws = p["health.ltc_wage_share"]
    raw["long_term_care"] = (L0["long_term_care"]["value"] * need_ltc / refs["need_ltc"] * ltc_cost
                             * wi ** ws * cpi ** (1 - ws))
    need_ltc_r = ltc_prof @ pop_r_age
    raw["long_term_care"] *= levers.factor(
        "long_term_care", _region_weights(need_ltc_r, rc_share["long_term_care"]),
        p.get("health.ltc_regional_factor"))

    # ---------- education ----------
    enrol, cpp = p["educ.enrolment"], p["educ.cost_per_pupil"]
    for line, (ages, ekey, ckey) in _EDUC.items():
        pupils = float(pop_age[ages].sum()) * (enrol[ekey] if ekey else 1.0)
        c = L0[line]
        d1 = c["D1"] / c["value"] if c["value"] else 0.0
        price = d1 * i_pubavg * ecf * staff["education"] + (1 - d1) * cpi
        raw[line] = pupils * cpp[ckey] * price * educ_cost / 1e6
        raw[line] *= levers.factor(line, _community_weights(pop_r_age[ages].sum(axis=0), p),
                                   p.get("educ.community_cost_factor"), p.get("educ.community_enrolment_factor"))

    # ---------- % of GDP lines ----------
    raw["defence"] = p["defence.target_pct_gdp"] / 100.0 * gdp
    raw["public_investment"] = p["invest.public_pct_gdp"] / 100.0 * gdp
    extra_inv = p.get("invest.regional_extra_pct_gdp")
    if extra_inv and any(float(v) != 0.0 for v in extra_inv.values()):
        # add-on of a region's governments, in % of the GDP of that region
        gdp_r = (state.extra.get("regional_economy") or {}).get("gdp")
        if gdp_r is None:
            gdp_r = gdp * pop_r_age.sum(axis=0) / pop_tot
        levers.direct["public_investment"] = {
            r: float(extra_inv.get(r, 0.0)) / 100.0 * float(gdp_r[i]) for i, r in enumerate(REGIONS)
            if float(extra_inv.get(r, 0.0)) != 0.0}
    raw["capital_transfers"] = p["invest.capital_transfers_pct_gdp"] / 100.0 * gdp
    raw["eu_contribution"] = p["exp.eu_contribution_pct_gni"] / 100.0 * gdp * sc["gni_2024"] / sc["gdp_2024"]

    # ---------- subsidies ----------
    sfac = p["subsidy.factor"]
    raw["subsidy_wage"] = L0["subsidy_wage"]["value"] * i_wbpriv * sfac["wage"]
    raw["subsidy_service_vouchers"] = (L0["subsidy_service_vouchers"]["value"] * lab["vouchers"] / refs["vouchers"]
                                       * wi * sfac["service_vouchers"])
    priv_r = st[S["emp_private"]].sum(axis=(0, 1))
    raw["subsidy_wage"] *= levers.factor("subsidy_wage", _region_weights(priv_r, rc_share["subsidy_wage"]),
                                         p.get("subsidy.target_group_factor"))
    sv_w = p.get("regional.service_voucher_shares")
    sv_w = _norm(sv_w, REGIONS) if sv_w else _region_weights(pop_r_age.sum(axis=0))
    raw["subsidy_service_vouchers"] *= levers.factor("subsidy_service_vouchers", sv_w,
                                                     p.get("subsidy.service_voucher_factor"))
    sel = p["subsidy.gdp_elasticity"]
    for line, key in _SUBSIDY_KEY.items():
        raw[line] = L0[line]["value"] * i_gdp ** sel * sfac[key]

    # ---------- functional lines: wages + purchases + other ----------
    for lid, d in all_defs().items():
        if lid in raw or lid == "interest":
            continue
        c = L0[lid]
        other = c["value"] - c["D1"] - c["P2"]
        fg = d.fgroup
        v = c["D1"] * i_pubwage * ecf * staff.get(fg, 1.0) + c["P2"] * interm_idx + other * i_gdp ** other_el
        if d.driver == "children_services":
            v *= kids_0_11 / refs["kids_0_11"]
            cw = _community_weights(pop_r_age[:12].sum(axis=0), p)
            v *= levers.factor(lid, {k: w * rc_share[lid] for k, w in cw.items()},
                               p.get("family.childcare_factor"))
        raw[lid] = v * ffac.get(fg, 1.0)

    # ---------- interest: debt stock with a maturity structure ----------
    s_short = p["debt.short_term_share"]
    roll = 1.0 / max(p["debt.avg_maturity_years"], 1.0)
    if first:
        debt_prev = sc["debt_2023"]
        deficit_est = -sc["balance_2024"]
    else:
        debt_prev = prev.fiscal["debt"]
        deficit_est = -prev.fiscal["balance"] * gdp_actual / prev.macro["gdp_nominal"]
    new_rate = (1 - s_short) * i10 + s_short * ish
    if "rate_long_prev0" in refs:
        rate_long_prev = refs["rate_long_prev0"] if first else prev.extra["balance"]["rate_long"]
    else:
        # baseline base year: back out the average rate on long-term debt from actual 2024 interest
        eff0 = ((L0["interest"]["value"] - s_short * debt_prev * ish - 0.5 * deficit_est * new_rate)
                / ((1 - s_short) * debt_prev))
        rate_long_prev = (eff0 - 0.5 * roll * i10) / (1 - 0.5 * roll)
        refs["rate_long_prev0"] = rate_long_prev
    eff = rate_long_prev * (1 - 0.5 * roll) + 0.5 * roll * i10     # refinancing spread over the year
    raw["interest"] = (1 - s_short) * debt_prev * eff + s_short * debt_prev * ish + 0.5 * deficit_est * new_rate

    # ---------- spending paths by entity ----------
    # amount = fraction x unchanged-policy line of the baseline run (this run's own line in a
    # baseline run): a lever on the line (function factor, % of GDP target ...) is not scaled
    path_frac = _entity_paths(state, prev, p, L0)
    line_unchanged = {lid: float(raw[lid]) for lid in L0}   # before the paths and the calibration
    path_base = {lid: float((base_lines or line_unchanged).get(lid, raw[lid])) for lid in path_frac}
    for lid, fr in path_frac.items():
        raw[lid] += path_base[lid] * sum(fr.values())

    # ---------- calibration ----------
    if src is not None:
        calib = src["calib"]
    else:
        # fitted on the unchanged-policy lines, so a path already active in the base year is kept
        calib = {lid: (L0[lid]["value"] / line_unchanged[lid] if abs(line_unchanged[lid]) > 1e-9 else 1.0)
                 for lid in L0}
    for lid, d in levers.direct.items():
        raw[lid] += sum(d.values())
    values = {lid: float(raw[lid] * calib[lid] + add.get(lid, 0.0)) for lid in L0}

    x = {
        "refs": refs, "calib": calib,
        "welfare_index": w_idx, "benefit_index": bidx, "interm_vol": interm_vol,
        "health_real": health_real, "ltc_cost": ltc_cost, "educ_cost": educ_cost, "career_s": career_s,
        "need_health": need_health, "pop_total": pop_tot, "w_pub": lab["w_pub"],
        "pension_n": n_ret, "pension_mix": mix, "pension_avg": avg, "pension_wm_lag": wm_lag,
        "pension_mix_drift": mix_drift,
        "rate_long_prev": rate_long_prev, "debt_prev": debt_prev,
        "unemployment_full": unemp_full * calib["unemployment"],
        "avg_benefit": {"unemployed": ben_u * bidx, "disabled": ben_d * bidx},
        # amounts added by the region / Community keyed levers: line -> key -> EUR mn (empty if neutral)
        "regional_delta": levers.amounts(raw, calib),
        # amounts added by the spending paths of the entities: line -> entity -> EUR mn (empty if neutral)
        "entity_delta": {lid: {e: float(v * path_base[lid] * calib[lid]) for e, v in fr.items()}
                         for lid, fr in path_frac.items()},
        "line_unchanged": line_unchanged,
    }
    return values, x


def components() -> dict[str, dict[str, float]]:
    """2024 ESA composition of each line (EUR mn): useful for the API / documentation."""
    return {lid: {i: row[i] for i in ITEMS} for lid, row in data.lines_2024().items()}


def entity_amounts(values: dict[str, float]) -> dict[str, float]:
    """Final expenditure per entity (before inter-governmental transfers) from 2024 entity shares."""
    out = {e: 0.0 for e in ENTITIES}
    for lid, v in values.items():
        sh = data.entity_shares(lid)
        if not sh:
            continue
        for e in ENTITIES:
            out[e] += v * sh[e]
    return out
