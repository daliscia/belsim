"""Labour market: socio-economic status of every population cell, employment by sector and
job type, wages.

`run(state, prev, p, ctx)` fills `state.status, emp, wage, wage_dist, labour` (see
docs/ARCHITECTURE.md). 2024 reproduces observed LFS / national-accounts structures (base.py); later
years apply, in this order, to the 2024 status shares of each [age, sex, region] cell:

  1. cohort effects on activity rates (younger cohorts keep their higher participation);
  2. disability and enrolment levers;
  3. retirement rules: the status profile over age is shifted by the change in the legal
     pension age (in its first year the average of the old and the new age; an age below
     the legal age is held back by the years the legal age has moved past it), the early-retirement
     age / career condition, a cumulated trend of the exit age (maturing reforms, cohorts)
     and the behavioural `retirement_shift`. Work and pension take-up shift by different
     amounts: the legal age passes through more to pensions than to work, the trend less;
     the difference waits in (or comes from) unemployment, invalidity and inactivity.
     `extra["labour"]["retirement_delay"]` (years by which pensions start later than in
     2024) lengthens the careers behind new pensions in the expenditure module;
  4. phase-out of the early-retirement scheme RCC/SWT;
  5. convergence of regional unemployment rates to their structural level (NAIRU);
  6. time limit on unemployment benefits (excluded people split between work, disability and
     inactivity / social assistance);
  7. public employment headcount norm, staffing levers of the expenditure side
     (`pubwage.staffing_factor`, `educ.enrolment`) and gradual re-absorption by the private
     sector of a public headcount change relative to the baseline (the staff not yet
     re-absorbed stays in the region of the cut);
  8. behavioural labour-supply response (`adjust["emp_rate_delta"]`);
  9. labour-demand factor and Okun response to the demand gap (behavioural `demand_gap` plus
     the exogenous cyclical position `macro.cyclical_output_gap`) on private employees.

Marginal workers. Steps 3-6, 8 and 9 record how many jobs they add; the difference of each
with the same step of the baseline run (`ctx.base_state`) is the employment a scenario gains or
loses by origin. Those jobs carry the relative wage `dyn.marginal_wage` and productivity
`dyn.marginal_productivity` of their origin instead of the average. They are placed in the wage
distribution of their job type (`_jobs_by_decile`): the jobs of an origin are spread over the
ten wage deciles with a profile whose mean wage is the origin's relative wage, and the jobs of
the labour-cost response go, on top, where employer cost changed most
(`adjust["labour_demand_decile_shift"]`, behaviour). The heads added to (or taken from) each
decile of private employees are published as `labour["scenario_jobs_private_by_decile"]`; the
wage of the job type, its deciles (`wage_dist`: the tenths of the workers after the change),
the wage bill and (in macro) value added follow from that allocation. No difference, no effect:
the allocation is zero and the factors exactly 1 in the baseline and in a scenario without
measures.

Regional levers. Status shares are by region, so levers that are a regional competence are
region-keyed dicts `{FLA, WAL, BRU}` (a scalar is accepted and applied to the three regions):
`labour.nairu`, `labour.activation_factor` (structural unemployment and the share of people
excluded by the time limit who find work), `labour.disability_rate_factor`,
`labour.public_employment_growth` (public employees residing in the region). With equal values
in the three regions the results are those of the national levers.

The function is idempotent: all year-to-year stocks are read from `prev.extra["labour"]`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..core.dims import BASE_YEAR, J, JOBTYPES, N_AGE, N_DECILE, N_JOBTYPE, N_SECTOR, REGIONS, S, SEC, SECTORS
from . import base as _base
from .dynparams import SOURCES, par as _dyn

AGE = np.arange(N_AGE)
I_PRIV, I_PUB, I_SELF = S["emp_private"], S["emp_public"], S["self_employed"]
I_UNE, I_DIS, I_RET, I_INA, I_STU = S["unemployed"], S["disabled"], S["retired"], S["inactive"], S["student"]
EMPLOYED = [I_PRIV, I_PUB, I_SELF]
ACTIVE = EMPLOYED + [I_UNE]
NON_ACTIVE = [S["child"], I_STU, I_DIS, I_RET, I_INA]
MARKET = [I_PRIV, I_SELF]              # employment that reacts to supply / demand shocks


def by_region(v) -> np.ndarray:
    """Region-keyed parameter `{FLA, WAL, BRU}` (or a scalar for all regions) -> array [region]."""
    if isinstance(v, dict):
        return np.array([float(v[r]) for r in REGIONS])
    return np.full(len(REGIONS), float(v))


def community_shares(pop_region: np.ndarray, dutch_share_brussels: float) -> dict[str, np.ndarray]:
    """Weights [region] of each Community in a population by region: the Flemish Community serves
    Flanders and `dutch_share_brussels` of Brussels, the French Community Wallonia (incl. the
    German-speaking Community, not modelled separately) and the rest of Brussels."""
    b = float(dutch_share_brussels)
    w = {"flemish_community": np.array([1.0, 0.0, b]), "french_community": np.array([0.0, 1.0, 1.0 - b])}
    tot = max(float(np.sum(pop_region)), 1e-9)
    return {c: m * np.asarray(pop_region, dtype=float) / tot for c, m in w.items()}


def _transfer(sh: np.ndarray, src: list[int], dst: list[int], amount: np.ndarray) -> np.ndarray:
    """Move `amount` (share of the cell's population, >= 0) from statuses `src` to `dst`.

    Taken proportionally to the current shares within `src` (capped at what is there) and
    distributed proportionally to the current shares within `dst` (all to dst[0] if empty).
    Returns the amount actually moved.
    """
    s = sh[src]
    tot = s.sum(axis=0)
    amt = np.minimum(np.maximum(amount, 0.0), tot)
    sh[src] = s - s * (amt / np.where(tot > 0, tot, 1.0))
    d = sh[dst]
    dtot = d.sum(axis=0)
    w = d / np.where(dtot > 0, dtot, 1.0)
    w[0] = np.where(dtot > 0, w[0], 1.0)
    sh[dst] = d + w * amt
    return amt


def _shift_ages(sh: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Status profile read `delta[age]` years earlier: new[:, a] = old[:, a - delta[a]]."""
    if not np.any(delta != 0.0):
        return sh
    src = np.clip(AGE - delta, 0.0, N_AGE - 1.0)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, N_AGE - 1)
    w = (src - lo)[None, :, None, None]
    return (1.0 - w) * sh[:, lo] + w * sh[:, hi]


def _retirement_shift(p, year: int, adjust: dict, trend: float | None = None,
                      legal: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Years by which the 2024 status-by-age profile is shifted at each age: (work, benefits).

    Below the calibration pension age (65) the two are equal: the change in the early-retirement
    rules plus `trend`, the cumulated rise in the effective exit age from maturing past reforms.
    From 65 up to the current legal age people are in the early-retirement window too: the legal
    change (x pass-through) plus the same trend. From the legal age on, only the legal change.
    The pass-through of the legal age differs for work (`labour.pension_age_passthrough`) and for
    drawing a pension (`labour.pension_age_benefit_passthrough`, higher): someone who neither
    works longer nor may retire yet waits on another benefit or without one.
    `trend=None` (no carried state): the dated per-year value times the years elapsed.
    `legal`: effective legal age of the year (annual average in the year of a change); default
    the parameter.
    """
    l0 = float(p["labour.calibration_legal_pension_age"])
    e0 = float(p["labour.calibration_early_retirement_age"])
    c0 = float(p["labour.calibration_early_retirement_career_years"])
    beh = float(adjust.get("retirement_shift", 0.0))
    if trend is None:
        trend_years = min(year, int(p["labour.exit_age_trend_end_year"])) - BASE_YEAR
        trend = float(p["labour.exit_age_trend_per_year"]) * max(trend_years, 0)
    legal = float(p["labour.legal_pension_age"]) if legal is None else float(legal)
    rules = ((float(p["labour.early_retirement_age"]) - e0) * float(p["labour.early_retirement_passthrough"])
             + (float(p["labour.early_retirement_career_years"]) - c0)
             * float(p["labour.early_retirement_career_effect"]))
    pt_work = p["labour.pension_age_passthrough"]
    pt_ben = p.get("labour.pension_age_benefit_passthrough", pt_work)
    # only part of the trend delays the pension itself: the rest is people working longer who
    # would otherwise be unemployed, in invalidity or inactive before their pension
    trend_ben = float(p.get("labour.exit_age_trend_pension_share", 1.0)) * trend
    out = []
    def cum(pt, years: float) -> float:
        """Shift for `years` of increase of the legal age: a scalar pass-through applies to every
        year, a list gives the pass-through of the first, second ... year of increase."""
        if not isinstance(pt, (list, tuple)):
            return float(pt) * years
        out_, k = 0.0, 0
        while years > 1e-12:
            step_ = min(years, 1.0)
            out_ += float(pt[min(k, len(pt) - 1)]) * step_
            years -= step_
            k += 1
        return out_

    late_share = float(p.get("labour.exit_age_trend_late_share", 0.0))
    for pt, tr in ((pt_work, trend), (pt_ben, trend_ben)):
        late = late_share * tr      # work: the whole trend; pensions: the part that delays the pension
        d_legal = cum(pt, legal - l0) + beh if legal >= l0 else (legal - l0) * float(pt[0] if isinstance(pt, (list, tuple)) else pt) + beh
        d_early = rules + tr + beh
        xs, ys = [e0 - 5.0, e0, l0 - 1.0], [0.0, d_early, d_early]
        if legal > l0:          # ages between the calibration age and the legal age
            xs += [l0, legal - 1e-6]
            ys += [d_legal + late, d_legal + late]
        xs.append(max(legal, l0))
        ys.append(d_legal)
        d = np.interp(AGE, xs, ys)
        # A person aged a < legal age is held back by the years the legal age has moved past
        # his age, not by the whole change: when the age goes from 66 to 67 nothing changes
        # for the 65-year-olds (they could not retire at 65 under 66 either).
        # (The window ages all read the steep stretch of the 2024 profile just below 65; a
        # behavioural shift moves that stretch once, so it is shared between the window ages.)
        n_win = max(legal - l0, 1.0)
        for a in range(int(l0), int(np.ceil(legal))):
            if a < legal:
                d[a] = cum(pt, min(a - l0 + 1.0, legal - l0)) + beh / n_win + late
        out.append(d)
    return out[0], out[1]


def _staffing_lever(p, base_p, pop=None) -> float:
    """Public headcount factor implied by the staffing levers of the expenditure side:
    `pubwage.staffing_factor` by function (absolute lever, neutral = 1) and, relative to the
    baseline, the enrolment rates of `educ.enrolment` (unchanged pupil-teacher ratio)."""
    w = _dyn(p, "dyn.public_employment.function_weights")
    staff = p.get("pubwage.staffing_factor") or {}
    f = 1.0 + sum(float(w.get(k, 0.0)) * (float(v) - 1.0) for k, v in staff.items())
    enrol_b = base_p.get("educ.enrolment") if base_p is not None else None
    enrol = p.get("educ.enrolment")
    if enrol and enrol_b:
        lw = _dyn(p, "dyn.public_employment.education_level_weights")
        d = sum(float(lw.get(k, 0.0)) * (float(enrol[k]) / float(enrol_b[k]) - 1.0)
                for k in enrol if enrol_b.get(k))
        f *= 1.0 + float(w.get("education", 0.0)) * d
    # enrolment lever by Community (absolute, neutral = 1): teachers follow pupils
    ef = p.get("educ.community_enrolment_factor")
    if ef and pop is not None and any(float(v) != 1.0 for v in ef.values()):
        pupils = pop[3:25].sum(axis=(0, 1))
        cs = community_shares(pupils, float(p.get("regional.brussels_dutch_share", 0.2)))
        d = sum(float(cs[c].sum()) * (float(v) - 1.0) for c, v in ef.items() if c in cs)
        f *= 1.0 + float(w.get("education", 0.0)) * d
    return max(f, 0.05)


def _status_shares(state, prev_x: dict, p, bs: _base.LabourBase, base_x: dict | None = None,
                   base_p=None, tg_ratio: float = 0.0) -> tuple[np.ndarray, dict]:
    """`base_x`: the baseline run's `extra["labour"]` of the same year (None in a baseline run)."""
    year, pop = state.year, state.pop
    adjust = state.adjust or {}
    info: dict = {}
    sh = bs.shares.copy()
    steps = {k: np.zeros(3) for k in SOURCES}       # jobs added by each step, by job type

    def jobs() -> np.ndarray:
        return (sh[EMPLOYED] * pop).sum(axis=(1, 2, 3))

    # 1. cohort effects on activity
    k = int(np.clip(year - BASE_YEAR, 0, bs.cohort_mult.shape[0] - 1))
    m = 1.0 + float(p["labour.cohort_effect_strength"]) * (bs.cohort_mult[k] - 1.0)
    act = sh[ACTIVE].sum(axis=0)
    new = np.minimum(act * m[:, :, None], np.maximum(act, _base.MAX_ACTIVITY))
    sh[ACTIVE] *= np.where(act > 0, new / np.where(act > 0, act, 1.0), 1.0)
    sh[NON_ACTIVE] *= np.where(act < 1, (1.0 - new) / np.where(act < 1, 1.0 - act, 1.0), 1.0)

    # 2. disability and enrolment levers
    f_dis = by_region(p["labour.disability_rate_factor"]) * float(p.get("labour.disability_trend_factor", 1.0))
    if np.any(f_dis != 1.0):
        d = sh[I_DIS] * (f_dis - 1.0)[None, None, :]
        up, down = np.maximum(d, 0.0), np.maximum(-d, 0.0)
        w_emp = float(p["labour.disability_offset_employment"])
        if np.any(up > 0.0):
            _transfer(sh, MARKET + [I_PUB], [I_DIS], up * w_emp)
            _transfer(sh, [I_INA, I_UNE], [I_DIS], up * (1.0 - w_emp))
        if np.any(down > 0.0):
            _transfer(sh, [I_DIS], MARKET, down * w_emp)
            _transfer(sh, [I_DIS], [I_INA], down * (1.0 - w_emp))
    f_stu = float(p["labour.student_rate_factor"])
    if f_stu != 1.0:
        d = sh[I_STU] * (f_stu - 1.0)
        d[:18] = 0.0
        if f_stu > 1.0:
            _transfer(sh, [I_INA, I_UNE] + MARKET, [I_STU], d)
        else:
            _transfer(sh, [I_STU], MARKET + [I_UNE, I_INA], -d)
    leave = int(p["labour.school_leaving_age"])
    if leave > 18:                      # longer compulsory schooling: everybody below is a student
        a = slice(18, min(leave, N_AGE))
        others = [i for i in range(sh.shape[0]) if i not in (I_STU, I_DIS, S["child"])]
        sh[I_STU, a] += sh[others, a].sum(axis=0)
        sh[others, a] = 0.0

    # 3. retirement rules: shift of the status-by-age profile. The trend of the effective exit
    #    age is a carried stock: last year's level plus this year's `exit_age_trend_per_year`.
    n0 = jobs()
    trend = prev_x.get("exit_trend")
    if year <= BASE_YEAR:
        trend = 0.0
    elif trend is not None and year <= int(p["labour.exit_age_trend_end_year"]):
        trend = float(trend) + float(p["labour.exit_age_trend_per_year"])
    info["exit_trend"] = trend
    # a new legal age applies to people reaching it from 1 February: in annual averages only
    # part of the step is effective in the first year
    legal_now = float(p["labour.legal_pension_age"])
    legal_prev = prev_x.get("legal_age")
    info["legal_age"] = legal_now
    l0_i = int(p["labour.calibration_legal_pension_age"])

    def retire(sh0: np.ndarray, legal: float) -> tuple[np.ndarray, float]:
        """Status shares under the legal age `legal` and the years by which pensions start
        later than under the 2024 profile (ages 63 to 66)."""
        d_work, d_ben = _retirement_shift(p, year, adjust, trend, legal)
        delay = float(d_ben[l0_i - 2:l0_i + 2].mean())
        if not np.any(d_ben != d_work):
            return _shift_ages(sh0, d_work), delay
        # pensions follow the benefit shift; whoever is retired under the work shift but not yet
        # entitled waits as unemployed / in invalidity / inactive (the mix of the age read)
        ben = _shift_ages(sh0, d_ben)
        out = _shift_ages(sh0, d_work)
        wait = [I_UNE, I_DIS, I_INA]
        gap_ret = out[I_RET] - ben[I_RET]
        mix = ben[wait] / np.maximum(ben[wait].sum(axis=0), 1e-12)
        mix[2] = np.where(ben[wait].sum(axis=0) > 0, mix[2], 1.0)
        moved = np.clip(gap_ret, -out[wait].sum(axis=0), out[I_RET])
        out[I_RET] = out[I_RET] - moved
        up = np.maximum(moved, 0.0)
        out[wait] = out[wait] + mix * up
        # benefit shift smaller than the work shift: the extra workers come from unemployment
        # and inactivity first, from invalidity only if those are exhausted
        down = np.maximum(-moved, 0.0)
        for grp in ([I_UNE, I_INA], [I_DIS]):
            g_tot = out[grp].sum(axis=0)
            take = np.minimum(down, g_tot)
            out[grp] = out[grp] - out[grp] * (take / np.where(g_tot > 0, g_tot, 1.0))
            down = down - take
        return out, delay

    if year > BASE_YEAR and legal_prev is not None and legal_now != float(legal_prev):
        # year of a change: annual average of the population under the old and the new age
        w1 = float(p.get("labour.pension_age_first_year_share", 1.0))
        sh_old, delay_old = retire(sh, float(legal_prev))
        sh_new, delay_new = retire(sh, legal_now)
        sh = (1.0 - w1) * sh_old + w1 * sh_new
        info["retirement_delay"] = (1.0 - w1) * delay_old + w1 * delay_new
    else:
        sh, info["retirement_delay"] = retire(sh, legal_now)
    if info["exit_trend"] is None:      # no carried state (unit tests): level implied by the parameters
        t_y = min(year, int(p["labour.exit_age_trend_end_year"])) - BASE_YEAR
        info["exit_trend"] = float(p["labour.exit_age_trend_per_year"]) * max(t_y, 0)

    # 4. RCC / SWT phase-out
    l0 = int(p["labour.calibration_legal_pension_age"])
    rcc_ages = slice(60, l0)
    rcc_ratio = prev_x.get("ref", {}).get("rcc_ratio")
    if rcc_ratio is None:
        retired_60 = float((bs.shares[I_RET, rcc_ages] * pop[rcc_ages]).sum())
        rcc_ratio = float(p["labour.rcc_beneficiaries_2024"]) / max(retired_60, 1.0)
    info["rcc_ratio"] = rcc_ratio
    f_rcc = float(p["labour.rcc_stock_factor"])
    rcc_share = np.zeros_like(sh[I_RET])
    rcc_share[rcc_ages] = np.minimum(bs.shares[I_RET, rcc_ages] * rcc_ratio, sh[I_RET, rcc_ages])
    if f_rcc < 1.0:
        gone = rcc_share * (1.0 - f_rcc)
        w = float(p["labour.rcc_exit_to_employment"])
        _transfer(sh, [I_RET], [I_PRIV], gone * w)
        _transfer(sh, [I_RET], [I_UNE], gone * (1.0 - w))
    info["rcc"] = float((rcc_share * min(f_rcc, 1.0) * pop).sum())
    steps["pension_age"] = jobs() - n0

    # 5. structural unemployment by region
    reg_axes = (0, 1)
    act = by_region(p.get("labour.activation_factor", 1.0))      # regional activation effort
    activated = bool(np.any(act != 1.0))
    u_cnt = (sh[I_UNE] * pop).sum(axis=reg_axes)
    lf_cnt = u_cnt + (sh[EMPLOYED].sum(axis=0) * pop).sum(axis=reg_axes)
    u_now = u_cnt / np.maximum(lf_cnt, 1.0)
    u_prev = prev_x.get("u_struct")
    n0 = jobs()
    if year <= BASE_YEAR or u_prev is None:
        u_struct = u_now
    else:
        nairu = by_region(p["labour.nairu"]) * float(p.get("labour.nairu_cyclical_factor", 1.0))
        if activated:
            nairu = nairu * np.maximum(act, 1e-6) ** (-float(p["labour.activation_nairu_elasticity"]))
        u_struct = np.asarray(u_prev) + float(p["labour.nairu_convergence_speed"]) * (nairu - np.asarray(u_prev))
        phi = np.where(u_now > 0, u_struct / np.where(u_now > 0, u_now, 1.0), 1.0)
        d = sh[I_UNE] * (phi - 1.0)[None, None, :]
        _transfer(sh, MARKET, [I_UNE], np.maximum(d, 0.0))
        _transfer(sh, [I_UNE], MARKET, np.maximum(-d, 0.0))
    info["u_struct"] = u_struct
    # Registered job-seekers outside the ILO definition (inactive in the model) find work in the
    # same proportion as the unemployed when the structural rate of the region falls below its
    # 2024 level: hidden stock = ratio x 2024 unemployed of the cell, share absorbed = 1 - u / u_2024.
    u0 = prev_x.get("u_struct0") if year > BASE_YEAR else None
    info["u_struct0"] = u_struct if u0 is None else u0
    hidden = float(p.get("labour.hidden_unemployment_ratio", 0.0) or 0.0)
    if hidden > 0.0 and u0 is not None:
        u0 = np.asarray(u0, dtype=float)
        absorbed = np.maximum(1.0 - np.asarray(u_struct) / np.where(u0 > 0, u0, 1.0), 0.0)
        if np.any(absorbed > 0.0):
            d = hidden * bs.shares[I_UNE] * absorbed[None, None, :]
            d[:18] = 0.0
            d[65:] = 0.0
            _transfer(sh, [I_INA], MARKET, d)
    steps["structural_unemployment"] = jobs() - n0

    # 6. time limit on unemployment benefits
    limit = float(p["labour.unemp_benefit_max_years"])
    excluded_inactive = np.zeros_like(sh[I_UNE])
    n0 = jobs()
    if limit > 0:
        surv = np.asarray(p["labour.unemp_duration_survival"], dtype=float)
        over = float(np.interp(limit, surv[:, 0], surv[:, 1]))
        x = over * float(p["labour.unemp_time_limit_enforcement"])
        d = sh[I_UNE] * x
        d[int(p["labour.unemp_time_limit_max_age"]):] = 0.0
        w_e = float(p["labour.unemp_exclusion_to_employment"])
        w_d = float(p["labour.unemp_exclusion_to_disability"])
        if activated:       # more of the excluded find work where activation is stronger
            w_e = np.clip(w_e * act, 0.0, 1.0 - w_d)[None, None, :]
        _transfer(sh, [I_UNE], [I_PRIV], d * w_e)
        _transfer(sh, [I_UNE], [I_DIS], d * w_d)
        excluded_inactive = _transfer(sh, [I_UNE], [I_INA], d * (1.0 - w_e - w_d))
        info["excluded_total"] = float((d * pop).sum())
    info["excluded_inactive_region"] = (excluded_inactive * pop).sum(axis=reg_axes)
    info["excluded_inactive"] = float(info["excluded_inactive_region"].sum())
    steps["unemp_exclusion"] = jobs() - n0

    # 7. public employment: headcount norm x change in the staffing levers of the expenditure side
    #    The norm is region-keyed (public employees residing in the region). `tilt` is the
    #    cumulated difference of each region's norm with the first region's; the national
    #    target is computed with the latter on a headcount stock free of the tilt, so that equal
    #    norms give exactly the national rule.
    pub_now = float((sh[I_PUB] * pop).sum())
    pub_prev, pop_prev = prev_x.get("public_headcount"), prev_x.get("pop_total")
    pub_prev = prev_x.get("public_headcount_untilted", pub_prev)
    staff = _staffing_lever(p, base_p, pop)
    info["staff_applied"] = 1.0                 # staffing-lever factor embodied in the headcount
    tilt = np.asarray(prev_x.get("public_tilt", np.ones(len(REGIONS))), dtype=float)
    target = None
    if year > BASE_YEAR and pub_prev is not None and pub_now > 0:
        pop_growth = float(pop.sum()) / pop_prev - 1.0 if pop_prev else 0.0
        g_pub = by_region(p["labour.public_employment_growth"])
        g_pop = float(p["labour.public_employment_pop_elasticity"]) * pop_growth
        g_base = float(p.get("labour.public_employment_outlook_growth", 0.0) or 0.0)
        target = pub_prev * (1.0 + float(g_pub[0]) + g_base + g_pop)
        target *= staff / float(prev_x.get("staff_applied", 1.0))
        info["staff_applied"] = staff
        if np.any(g_pub != g_pub[0]):
            tilt = tilt * (1.0 + g_pub + g_base + g_pop) / (1.0 + g_pub[0] + g_base + g_pop)
        if np.any(tilt != 1.0):
            d = sh[I_PUB] * (target / pub_now * tilt - 1.0)[None, None, :]
        else:
            d = sh[I_PUB] * (target / pub_now - 1.0)
        _transfer(sh, [I_PRIV], [I_PUB], np.maximum(d, 0.0))
        _transfer(sh, [I_PUB], [I_PRIV], np.maximum(-d, 0.0))
    info["public_headcount"] = float((sh[I_PUB] * pop).sum())
    info["public_tilt"] = tilt
    info["public_headcount_untilted"] = (target if (target is not None and np.any(tilt != 1.0))
                                         else info["public_headcount"])

    # 7b. the exchange with private employment above is one for one. Relative to the baseline,
    #     the private sector absorbs displaced public staff (or replaces those hired away)
    #     only gradually: the not-yet-absorbed stock is unemployed / inactive meanwhile, in the
    #     region where the public jobs were cut (or created).
    pub_r = (sh[I_PUB] * pop).sum(axis=reg_axes)
    info["public_headcount_region"] = pub_r
    info["public_diff"], info["public_unabsorbed"] = 0.0, 0.0
    info["public_diff_region"], info["public_unabsorbed_region"] = np.zeros(len(REGIONS)), np.zeros(len(REGIONS))
    if base_x is not None and base_x.get("public_headcount_region") is not None:
        diff = pub_r - np.asarray(base_x["public_headcount_region"], dtype=float)
        lam = min(1.0, max(0.0, float(_dyn(p, "dyn.public_employment.reabsorption_speed"))))
        prev_stock = np.asarray(prev_x.get("public_unabsorbed_region", np.zeros(len(REGIONS))), dtype=float)
        prev_diff = np.asarray(prev_x.get("public_diff_region", np.zeros(len(REGIONS))), dtype=float)
        stock = (1.0 - lam) * (prev_stock - (diff - prev_diff))
        info["public_diff_region"], info["public_unabsorbed_region"] = diff, stock
        info["public_diff"], info["public_unabsorbed"] = float(diff.sum()), float(stock.sum())
        priv_r = (sh[I_PRIV] * pop).sum(axis=reg_axes)
        if np.any(stock != 0.0):
            ratio = np.where(priv_r > 0, np.abs(stock) / np.where(priv_r > 0, priv_r, 1.0), 0.0)
            d = sh[I_PRIV] * ratio[None, None, :]
            w_u = float(p["labour.demand_shock_to_unemployment"])
            out_, in_ = (stock > 0.0)[None, None, :], (stock < 0.0)[None, None, :]
            if np.any(stock > 0.0):
                _transfer(sh, [I_PRIV], [I_UNE], np.where(out_, d, 0.0) * w_u)
                _transfer(sh, [I_PRIV], [I_INA], np.where(out_, d, 0.0) * (1.0 - w_u))
            if np.any(stock < 0.0):
                _transfer(sh, [I_UNE], [I_PRIV], np.where(in_, d, 0.0) * w_u)
                _transfer(sh, [I_INA], [I_PRIV], np.where(in_, d, 0.0) * (1.0 - w_u))

    # 8. behavioural labour-supply response (additive on the employment rate of the cell)
    n0 = jobs()
    delta = adjust.get("emp_rate_delta")
    if delta is not None:
        d = np.asarray(delta, dtype=float)[:, :, None] * np.ones_like(sh[0])
        _transfer(sh, [I_INA, I_UNE], MARKET, np.maximum(d, 0.0))
        _transfer(sh, MARKET, [I_INA, I_UNE], np.maximum(-d, 0.0))
    steps["labour_supply"] = jobs() - n0

    # 9. labour demand: labour-cost response and Okun's law on private employees
    n0 = jobs()
    gap = float(state.macro.get("demand_gap", 0.0)) + float(p["macro.cyclical_output_gap"])
    ldf = float(adjust.get("labour_demand_factor", 1.0))
    factor = ldf * (1.0 + float(p["labour.okun_coefficient"]) * gap)
    # Regional target-group reductions: a change of `subsidy.target_group_factor` relative to
    # the baseline changes the labour cost of private employers of the region by
    # (factor - baseline factor) x `tg_ratio` (regional target-group subsidies per euro of
    # private labour cost, last year); private employment of the region responds with the
    # elasticity and speed below. No difference with the baseline, no effect.
    tg_now = np.zeros(len(REGIONS))
    f_tg, f_tg0 = p.get("subsidy.target_group_factor"), (base_p.get("subsidy.target_group_factor") if base_p is not None else None)
    if f_tg and f_tg0 and tg_ratio > 0.0:
        diff = np.array([float(f_tg[r]) - float(f_tg0[r]) for r in REGIONS])
        tg_prev = np.asarray(prev_x.get("target_group_effect", np.zeros(len(REGIONS))), dtype=float)
        if np.any(diff != 0.0) or np.any(tg_prev != 0.0):
            target = float(p.get("labour.target_group_employment_elasticity", 0.0) or 0.0) * diff * tg_ratio
            lam = float(p.get("labour.target_group_adjustment_speed", 1.0) or 1.0)
            tg_now = tg_prev + lam * (target - tg_prev)
    info["target_group_effect"] = tg_now
    tg_jobs = np.zeros(3)
    if np.any(tg_now != 0.0):
        d = sh[I_PRIV] * tg_now[None, None, :]
        w_u = float(p["labour.demand_shock_to_unemployment"])
        up, down = np.maximum(d, 0.0), np.maximum(-d, 0.0)
        if np.any(up > 0.0):
            _transfer(sh, [I_UNE], [I_PRIV], up * w_u)
            _transfer(sh, [I_INA], [I_PRIV], up * (1.0 - w_u))
        if np.any(down > 0.0):
            _transfer(sh, [I_PRIV], [I_UNE], down * w_u)
            _transfer(sh, [I_PRIV], [I_INA], down * (1.0 - w_u))
        tg_jobs = jobs() - n0
        n0 = jobs()
    if factor != 1.0:
        d = sh[I_PRIV] * (factor - 1.0)
        w_u = float(p["labour.demand_shock_to_unemployment"])
        if factor > 1.0:
            _transfer(sh, [I_UNE], [I_PRIV], d * w_u)
            _transfer(sh, [I_INA], [I_PRIV], d * (1.0 - w_u))
        else:
            _transfer(sh, [I_PRIV], [I_UNE], -d * w_u)
            _transfer(sh, [I_PRIV], [I_INA], -d * (1.0 - w_u))
        moved = jobs() - n0
        steps["labour_demand"] = moved * ((ldf - 1.0) / (factor - 1.0))
        steps["cyclical"] = moved - steps["labour_demand"]
    steps["labour_demand"] = steps["labour_demand"] + tg_jobs

    sh = np.maximum(sh, 0.0)
    sh /= sh.sum(axis=0, keepdims=True)
    info["steps"] = steps
    return sh, info


@lru_cache(maxsize=256)
def _tilted_profile(rho: float, rel: tuple, shape: tuple) -> tuple:
    """Distribution over the wage deciles of jobs paid `rho` times the average: weights
    proportional to shape_d x exp(-theta x z_d), z from -1 (lowest decile) to +1, theta solved so
    that the mean of `rel` (decile wage over the average) under the weights is `rho` (the nearest
    attainable value if `rho` lies outside the deciles). Pure function of its arguments."""
    r, b = np.asarray(rel, dtype=float), np.asarray(shape, dtype=float)
    z = np.linspace(-1.0, 1.0, len(r))

    def weights(theta: float) -> np.ndarray:
        w = b * np.exp(-theta * z)
        return w / w.sum()

    if abs(float(weights(0.0) @ r) - rho) < 1e-12:
        return tuple(weights(0.0))
    lo, hi = -40.0, 40.0                       # the mean falls as theta rises
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(weights(mid) @ r) > rho else (lo, mid)
    return tuple(weights(0.5 * (lo + hi)))


def _recut(mass: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Mean of each tenth of a population made of groups of size `mass` at `level` (ascending)."""
    cum = np.cumsum(mass)
    total = cum[-1]
    edges = total * np.arange(len(mass) + 1) / len(mass)
    lo = np.concatenate([[0.0], cum[:-1]])
    overlap = np.clip(np.minimum(edges[1:, None], cum[None, :]) - np.maximum(edges[:-1, None], lo[None, :]), 0.0, None)
    return overlap @ level / (total / len(mass))


def _jobs_by_decile(steps: dict, base_steps: dict | None, totals: np.ndarray, base_totals,
                    rel: np.ndarray, shift, p) -> tuple[np.ndarray, np.ndarray]:
    """Heads a scenario adds to each wage decile of each job type relative to the baseline:
    (jobs [job type, decile], head-equivalents of wage lost [job type]).

    The deciles are those of the workers the scenario shares with the baseline (a tenth of the
    baseline head count each, at `rel` times the average wage). Jobs of an origin k (difference of
    step k with the baseline) follow `_tilted_profile` with the relative wage
    `dyn.marginal_wage[k]`, starting from the profile of entrants
    (`dyn.marginal_wage_decile_profile`) for the origins that bring people in from
    non-employment at low wages and from a flat one for the others; any other difference in the
    head count (population, public staff, disability ...) is spread evenly. `shift` [decile]
    (behaviour, private employees only) moves jobs between deciles after a change in employer
    cost that differs across them: log change of employment of each decile around the average
    response, zero in sum. Rows sum to the difference in the head count of the job type.
    """
    n_j, n_d = rel.shape
    jobs, lost = np.zeros((n_j, n_d)), np.zeros(n_j)
    if base_steps is None or base_totals is None:
        return jobs, lost
    rel_w = _dyn(p, "dyn.marginal_wage")
    entrants = tuple(float(x) for x in _dyn(p, "dyn.marginal_wage_decile_profile"))
    from_entrants = set(_dyn(p, "dyn.marginal_job_entrant_origins"))
    flat = (1.0,) * n_d
    base_totals = np.asarray(base_totals, dtype=float)
    diff = {k: steps[k] - np.asarray(base_steps.get(k, 0.0)) for k in SOURCES}
    for j in range(n_j):
        d_tot = float(totals[j] - base_totals[j])
        rest = d_tot
        row = np.zeros(n_d)
        r_j = tuple(float(x) for x in rel[j])
        for k in SOURCES:
            d = float(diff[k][j])
            if d == 0.0:
                continue
            w = _tilted_profile(float(rel_w.get(k, 1.0)), r_j, entrants if k in from_entrants else flat)
            row += d * np.asarray(w)
            rest -= d
        if rest != 0.0:
            row += rest / n_d
        if j == J["emp_private"] and shift is not None:
            sh_ = np.asarray(shift, dtype=float)
            if sh_.shape == (n_d,) and np.any(sh_ != 0.0):
                row += base_totals[j] / n_d * (sh_ - sh_.mean())
        if not np.any(row != 0.0):
            continue
        # a decile cannot lose more than (nearly) all its workers: the excess is taken elsewhere
        floor = -0.9 * base_totals[j] / n_d
        if row.min() < floor:
            excess = float(np.minimum(row - floor, 0.0).sum())
            row = np.maximum(row, floor)
            room = row - floor
            if room.sum() > 0.0:
                row += excess * room / room.sum()
        jobs[j] = row
        lost[j] = float((row * (1.0 - rel[j])).sum())
    return jobs, lost


def _marginal_factors(steps: dict, base_steps: dict | None, emp: np.ndarray, emp_up: np.ndarray,
                      up: float, p, lost_w: np.ndarray | None = None) -> tuple:
    """(wage factor, productivity factor) [sector, job type] and jobs vs baseline by origin.

    Effective labour of a job type = headcount - sum over origins of (1 - relative value) x
    (jobs of that origin in this run - in the baseline); for wages, `lost_w` (head-equivalents
    lost through the place of the jobs in the wage distribution, `_jobs_by_decile`) when given.
    The correction is made in the sectors where marginal jobs land (`emp_up` = sector employment
    with `up` more jobs of every type); factor = effective / headcount of the sector x job-type
    cell.
    """
    q_w, q_p = np.ones(emp.shape), np.ones(emp.shape)
    diff = dict.fromkeys(SOURCES, 0.0)
    if base_steps is None:
        return q_w, q_p, diff
    rel_w, rel_p = _dyn(p, "dyn.marginal_wage"), _dyn(p, "dyn.marginal_productivity")
    by_origin_w, lost_p = np.zeros(3), np.zeros(3)
    for k in SOURCES:
        d = steps[k] - np.asarray(base_steps.get(k, 0.0))
        diff[k] = float(d.sum())
        by_origin_w += (1.0 - float(rel_w.get(k, 1.0))) * d
        lost_p += (1.0 - float(rel_p.get(k, 1.0))) * d
    if lost_w is None:
        lost_w = by_origin_w
    landing = (emp_up - emp) / np.maximum(up * emp.sum(axis=0), 1e-9)     # columns sum to 1
    cell = np.maximum(emp, 1e-9)
    return (np.clip(1.0 - landing * lost_w / cell, 0.5, 1.5),
            np.clip(1.0 - landing * lost_p / cell, 0.5, 1.5), diff)


def _statutory_share(p, prev, prev_x: dict, t: int) -> float:
    """Share of statutory staff among public employees: last year's share plus this year's
    trend (a dated trend cumulates) plus any change of the level parameter."""
    level = float(p["labour.statutory_share"])
    last = (prev.labour or {}).get("statutory_share") if prev is not None else None
    if last is None or t <= 0:
        share = level + float(p["labour.statutory_share_trend"]) * max(t, 0)
    else:
        share = float(last) + float(p["labour.statutory_share_trend"]) + (
            level - float(prev_x.get("statutory_level", level)))
    return float(np.clip(share, 0.05, 0.9))


def _drift_weights(p, year: int) -> np.ndarray:
    """Sector share drift, decaying so that shares stabilise in the long run."""
    drift = p["labour.sector_share_drift"]
    half = float(p["labour.sector_drift_half_life"])
    kk = np.log(2.0) / half
    t_eff = (1.0 - np.exp(-kk * max(year - BASE_YEAR, 0))) / kk
    return np.exp(np.array([float(drift.get(s, 0.0)) for s in SECTORS]) * t_eff)


def _sector_employment(totals: np.ndarray, p, year: int, bs: _base.LabourBase, idx: dict,
                       priv_ref: float) -> np.ndarray:
    pub_sh = np.array([float(p["labour.public_share_by_sector"].get(s, 0.0)) for s in SECTORS])
    priv0, pub0, self0 = bs.employees * (1 - pub_sh), bs.employees * pub_sh, bs.self_emp
    w_drift = _drift_weights(p, year)
    emp = np.zeros((N_SECTOR, N_JOBTYPE))
    q = SEC["Q"]
    rel_care = idx["care"] / idx["pop"]
    rel_school = idx["school"] / idx["pop"]

    # private employees: health & care follows the care-weighted population, others share the rest
    t_priv = totals[J["emp_private"]]
    q_priv = priv_ref * priv0[q] / priv0.sum() * idx["care"] ** float(p["labour.care_employment_elasticity"])
    q_priv = min(q_priv, float(p["labour.care_employment_max_share"]) * t_priv)
    w = priv0 * w_drift
    w[q] = 0.0
    emp[:, J["emp_private"]] = (t_priv - q_priv) * w / w.sum()
    emp[q, J["emp_private"]] = q_priv

    # public employees: total set by the headcount norm; composition follows pupils / care needs
    w = pub0.copy()
    w[SEC["P"]] *= rel_school ** float(p["labour.education_employment_elasticity"])
    w[q] *= rel_care ** float(p["labour.care_employment_elasticity"])
    emp[:, J["emp_public"]] = totals[J["emp_public"]] * w / w.sum()

    w = self0 * w_drift
    emp[:, J["self_employed"]] = totals[J["self_employed"]] * w / w.sum()
    return emp


def run(state, prev, p, ctx=None) -> None:
    bs = _base.load(_base.calibration_key(p))
    year, pop = state.year, state.pop
    prev_x = (prev.extra.get("labour") if prev is not None else None) or _base.baseline_extra(ctx, "labour")
    adjust = state.adjust or {}
    base_state = getattr(ctx, "base_state", None)
    base_x = (base_state.extra.get("labour") or {}) if base_state is not None else None

    # regional target-group subsidies per euro of private labour cost, last year
    tg_ratio = 0.0
    if prev is not None and getattr(ctx, "base_p", None) is not None and prev.expenditure and prev.labour:
        from ..fiscal.expenditure import data as _exp_data
        rc = _exp_data.entity_shares("subsidy_wage").get("regions_communities", 0.0)
        pm = prev.macro
        cost = float(prev.labour.get("wage_bill_private", 0.0)) * (
            float(pm.get("compensation_employees", 1.0)) / max(float(pm.get("wages_salaries", 1.0)), 1e-9))
        if cost > 0:
            tg_ratio = float(prev.expenditure.get("subsidy_wage", 0.0)) * rc / cost
    sh, info = _status_shares(state, prev_x, p, bs, base_x, getattr(ctx, "base_p", None), tg_ratio)
    status = sh * pop[None]
    state.status = status
    totals = status[EMPLOYED].sum(axis=(1, 2, 3))

    # --- reference (2024) stocks used as index bases, carried forward unchanged ---
    care_w = np.interp(AGE, *np.asarray(p["labour.care_demand_age_weights"], dtype=float).T)
    age_pop = pop.sum(axis=(1, 2))
    now = {
        "care": float((care_w * age_pop).sum()), "school": float(age_pop[3:23].sum()),
        "pop": float(age_pop.sum()), "emp_private": float(totals[0]),
        "students": float(status[I_STU, 15:].sum()), "employment": float(totals.sum()),
    }
    ref = dict(prev_x.get("ref") or now)
    ref.setdefault("rcc_ratio", info["rcc_ratio"])
    idx = {k: now[k] / ref[k] if ref.get(k) else 1.0 for k in ("care", "school", "pop")}

    # --- employment by sector x job type ---
    emp = _sector_employment(totals, p, year, bs, idx, ref["emp_private"])
    state.emp = emp

    # --- wages ---
    t = year - BASE_YEAR
    wage_index = float(state.macro.get("wage_index", 1.0))
    raw = bs.d11 / bs.employees * 1e6                                   # EUR per employee, 2024
    calib = prev_x.get("calib")
    if calib is None:
        # fitted once: wage bill = national-accounts D11, self-employed income = mixed income B3G
        calib = {
            "employees": float(bs.d11.sum() * 1e6 / (raw * emp[:, :2].sum(axis=1)).sum()),
            "self": float(bs.misc["mixed_income_meur"] * 1e6 / (raw * emp[:, 2]).sum()),
        }
    # the wedge by job type cumulates year by year (a dated change of `labour.wage_drift` leaves
    # the level reached before it); without carried state: constant wedge since the base year
    drift = p["labour.wage_drift"]
    drift_prev = prev_x.get("wage_drift_index") if year > BASE_YEAR else None
    drift_idx = {}
    for jt in JOBTYPES:
        g = float(drift.get(jt, 0.0))
        if year <= BASE_YEAR:
            drift_idx[jt] = 1.0
        elif drift_prev is not None:
            drift_idx[jt] = float(drift_prev.get(jt, 1.0)) * (1.0 + g)
        else:
            drift_idx[jt] = (1.0 + g) ** t
    # Public pay scales: `labour.public_wage_link` = share of the growth of the (private) wage
    # index passed on to public pay per head, the rest following the health index (automatic
    # indexation only); the wedge `labour.wage_drift.emp_public` comes on top. The link acts on
    # the baseline path of the indices: any difference between the wage index of a scenario and
    # that of the baseline (index jump, wage moderation ...) is passed on in full.
    link = float(p.get("labour.public_wage_link", 1.0))
    pub_rel = float(prev_x.get("public_wage_rel", 1.0)) if year > BASE_YEAR else 1.0
    if year > BASE_YEAR and link != 1.0 and prev is not None:
        bm, bpm = state.macro, prev.macro
        base_res = getattr(ctx, "baseline", None)
        if base_state is not None and base_res is not None and base_res.has(year - 1):
            bm, bpm = base_state.macro, base_res.at(year - 1).macro
        g_w = float(bm["wage_index"]) / float(bpm["wage_index"])
        g_h = float(bm["health_index"]) / float(bpm["health_index"])
        pub_rel *= (g_h / g_w) ** (1.0 - link)
    wage = np.zeros((N_SECTOR, N_JOBTYPE))
    for j, jt in enumerate(JOBTYPES):
        k = calib["self"] if jt == "self_employed" else calib["employees"]
        wage[:, j] = raw * k * wage_index * drift_idx[jt]
    wage[:, J["emp_public"]] *= pub_rel
    wage[:, J["self_employed"]] *= float(adjust.get("self_employed_income_factor", 1.0))
    # The self-employed are paid by their customers, not by a wage agreement: relative to the
    # baseline their income follows prices net of indirect taxes and productivity, not the part
    # of the wage index that is specific to employees (indexation of a tax-induced price rise,
    # skipped indexation, bargained shifts), i.e. the real-unit-labour-cost gap of macro.prices
    if base_state is not None:
        try:
            rulc_s, rulc_b = float(state.macro["real_unit_labour_cost"]), float(base_state.macro["real_unit_labour_cost"])
        except (KeyError, TypeError):
            rulc_s = rulc_b = 1.0
        if rulc_s != rulc_b and rulc_s > 0.0 and rulc_b > 0.0:
            wage[:, J["self_employed"]] *= rulc_b / rulc_s
    # jobs gained or lost relative to the baseline carry the wage of their origin
    base_steps = base_x.get("steps") if base_x is not None else None
    up = 0.01
    emp_up = (_sector_employment(totals * (1.0 + up), p, year, bs, idx, ref["emp_private"])
              if base_steps is not None else emp)
    # place of those jobs in the wage distribution of each job type (zero without a difference)
    disp = p["labour.wage_dispersion_factor"]
    rel_raw = [bs.decile_rel ** float(disp.get(jt, 1.0)) for jt in JOBTYPES]
    rel = np.array([r / r.mean() for r in rel_raw])                     # decile wage over the average
    base_totals = None
    if base_steps is not None and base_state.labour:
        bl = base_state.labour
        base_totals = [bl.get("employees_private"), bl.get("employees_public"), bl.get("self_employed")]
        if any(v is None for v in base_totals):
            base_totals = None
    jobs_dec, lost_w = _jobs_by_decile(info["steps"], base_steps, totals, base_totals, rel,
                                       adjust.get("labour_demand_decile_shift"), p)
    q_wage, q_prod, emp_diff = _marginal_factors(info["steps"], base_steps, emp, emp_up, up, p,
                                                 lost_w if base_totals is not None else None)
    avg_std = (wage * emp).sum(axis=0) / np.maximum(emp.sum(axis=0), 1e-9)   # before composition
    wage = wage * q_wage
    state.wage = wage

    bills = (wage * emp).sum(axis=0)                                    # EUR
    avg = bills / np.maximum(emp.sum(axis=0), 1e-9)
    dist = np.zeros((N_JOBTYPE, N_DECILE))
    q_avg = avg / np.where(avg_std > 0, avg_std, 1.0)                   # exactly 1 without marginal jobs
    decile_std = np.zeros((N_JOBTYPE, N_DECILE))                        # wage of the baseline's deciles
    for j, jt in enumerate(JOBTYPES):
        d0 = avg_std[j] * rel_raw[j] / rel_raw[j].mean()
        if not np.any(jobs_dec[j] != 0.0):
            dist[j] = d0 * q_avg[j]                                     # q_avg is 1 here (or all jobs average)
            decile_std[j] = dist[j]
            continue
        # tenths of the workers after the change: the baseline's tenths plus the jobs added to
        # each; the mean is the job-type average (rescaled if a sector factor hit its bound)
        cut = _recut(base_totals[j] / N_DECILE + jobs_dec[j], d0)
        dist[j] = cut * (avg[j] / cut.mean())
        decile_std[j] = d0 * (avg[j] / cut.mean())
    state.wage_dist = dist

    # --- indicators ---
    employed = status[EMPLOYED].sum(axis=0)
    unemployed = status[I_UNE]
    n_emp, n_une = float(employed.sum()), float(unemployed.sum())
    pop_2064 = float(pop[20:65].sum())
    priv_ratio = totals[0] / ref["emp_private"]
    lab = {
        "employment": n_emp,
        # jobless job-seekers incl. those who lost their benefit through the time limit and are
        # neither working nor in invalidity (ILO-like); benefit recipients = status "unemployed"
        "unemployment_rate": (n_une + info["excluded_inactive"]) / max(n_emp + n_une + info["excluded_inactive"], 1.0),
        "employment_rate_20_64": float(employed[20:65].sum()) / max(pop_2064, 1.0),
        "activity_rate_20_64": float(employed[20:65].sum() + unemployed[20:65].sum()) / max(pop_2064, 1.0),
        "hours_per_worker": bs.hours_per_worker * (1.0 + float(p["labour.hours_growth"])) ** t,
        "blue_collar_share": float(np.clip(p["labour.blue_collar_share"]
                                           + p["labour.blue_collar_share_trend"] * t, 0.15, 0.6)),
        "statutory_share": _statutory_share(p, prev, prev_x, t),
        "flexi_jobs": float(p["labour.flexi_jobs"]) * priv_ratio,
        "student_jobs": float(p["labour.student_jobs"]) * now["students"] / ref["students"],
        "service_voucher_workers": float(p["labour.service_voucher_workers"]) * priv_ratio,
        "cross_border_out": float(p["labour.cross_border_out"]) * now["employment"] / ref["employment"],
        "cross_border_in": float(p["labour.cross_border_in"]) * now["employment"] / ref["employment"],
        "avg_wage_private": float(avg[0]),
        "avg_wage_public": float(avg[1]),
        "avg_income_self_employed": float(avg[2]),
        "wage_bill_private": float(bills[0] / 1e6),
        # Expenditure multiplies its wage lines by this bill AND by `pubwage.staffing_factor`
        # (per function); education follows pupils x enrolment. So that the staffing levers are
        # counted once, the contract key is the bill before the staffing levers; the bill
        # actually paid is `wage_bill_public_actual` (identical when the levers are neutral).
        "wage_bill_public": float(bills[1] / 1e6) / info["staff_applied"],
        "wage_bill_public_actual": float(bills[1] / 1e6),
        "public_staffing_lever_factor": info["staff_applied"],
        "self_employed_income": float(bills[2] / 1e6),
        # --- additional indicators (not in the contract) ---
        "employees_private": float(totals[0]),
        "employees_public": float(totals[1]),
        "self_employed": float(totals[2]),
        "unemployed": n_une,
        "part_time_share": float(p["labour.part_time_share"]),
        "unemp_excluded_to_inactive": info["excluded_inactive"],
        "unemp_excluded_total": info.get("excluded_total", 0.0),
        "unemployment_benefit_rate": n_une / max(n_emp + n_une, 1.0),
        "rcc_beneficiaries": info["rcc"],
        "employment_rate_55_64": float(employed[55:65].sum()) / max(float(pop[55:65].sum()), 1.0),
        "employment_65_plus": float(employed[65:].sum()),
        # composition of the employment difference with the baseline (all 0 / 1 in a baseline run)
        "marginal_wage_factor_private": float(q_avg[0]),
        "marginal_productivity_factor": float((q_prod * emp).sum() / max(float(emp.sum()), 1.0)),
        "public_staff_unabsorbed": info["public_unabsorbed"],
        # private-sector jobs added relative to the baseline by wage decile of the baseline's
        # private employees (negative: lost); zeros in a baseline run. Not a scalar series.
        "scenario_jobs_private_by_decile": [float(x) for x in jobs_dec[J["emp_private"]]],
    }
    for ri, r in enumerate(REGIONS):
        lab[f"public_staff_unabsorbed.{r}"] = float(info["public_unabsorbed_region"][ri])
    for k in SOURCES:
        lab[f"employment_vs_baseline.{k}"] = emp_diff[k]
    for ri, r in enumerate(REGIONS):
        e_r = float(employed[:, :, ri].sum())
        u_r = float(unemployed[:, :, ri].sum()) + float(info["excluded_inactive_region"][ri])
        lab[f"unemployment_rate.{r}"] = u_r / max(e_r + u_r, 1.0)
        lab[f"employment_rate_20_64.{r}"] = float(employed[20:65, :, ri].sum()) / max(float(pop[20:65, :, ri].sum()), 1.0)
    state.labour = lab

    state.extra["labour"] = {
        "u_struct": info["u_struct"], "u_struct0": info["u_struct0"], "public_headcount": info["public_headcount"],
        "pop_total": now["pop"], "ref": ref, "calib": calib,
        "steps": info["steps"], "q_prod": q_prod,
        # jobs added by decile [job type, decile] and the wage of those deciles (EUR): heads
        # `baseline head count / 10 + jobs` at these wages give the wage bill of the job type
        "jobs_by_decile": jobs_dec, "decile_wage_standard": decile_std,
        "staff_applied": info["staff_applied"], "public_diff": info["public_diff"],
        "public_unabsorbed": info["public_unabsorbed"],
        "public_headcount_region": info["public_headcount_region"],
        "public_diff_region": info["public_diff_region"],
        "public_unabsorbed_region": info["public_unabsorbed_region"],
        "public_tilt": info["public_tilt"], "public_headcount_untilted": info["public_headcount_untilted"],
        "excluded_inactive_region": info["excluded_inactive_region"],
        "exit_trend": info["exit_trend"], "retirement_delay": info["retirement_delay"],
        "legal_age": info["legal_age"], "statutory_level": float(p["labour.statutory_share"]),
        "wage_drift_index": drift_idx, "public_wage_rel": pub_rel,
        "target_group_effect": info["target_group_effect"],
    }
