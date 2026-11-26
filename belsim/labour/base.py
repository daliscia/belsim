"""Base-year (2024) calibration tables of the labour module, built once from data/base/.

Everything here is read-only after construction (cached with lru_cache; arrays are flagged
non-writeable). Sources of each table are documented in belsim/data/fetch_labour.py.

How the 2024 status shares [status, age, sex, region] are built
    1. Employment and unemployment per head, national, 5-year age x sex: Eurostat LFS
       (lfsa_pganws), spread to single ages with a shape-preserving spline that keeps every
       5-year mean.
    2. Regional differentiation: ratio regional / national rate by broad age band x sex
       (lfst_r_lfe2emprt, lfst_r_lfp2actrt), interpolated over age.
    3. Employees vs self-employed by age x sex: LFS (lfsa_egaps), rescaled so that the national
       share of self-employed equals the national-accounts one (nama_10_pe, national concept),
       because the tax/social-security notion of self-employed (incl. company directors) is the
       national-accounts one. Public share of employees: uniform over cells (no data by age).
    4. Non-employed, non-unemployed remainder split into student (edat_lfse_18), disabled
       (age profile and total are parameters), retired (EU-SILC ilc_lvhl02 totals for 18-64 and
       65+, age profile below 65 is a parameter) and other inactive (residual).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtri

from ..core.dims import BASE_YEAR, N_AGE, N_DECILE, N_REGION, N_SECTOR, N_SEX, N_STATUS, REGION_NUTS1, REGIONS, S, SECTORS, SEXES

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "base"
AGE = np.arange(N_AGE)
G5_LO = np.arange(15, 75, 5)           # 12 five-year groups 15-19 ... 70-74 (+ 75 and over)
N_COHORT_YEARS = 60                    # cohort multipliers tabulated for 2024 .. 2024+59
MAX_ACTIVITY = 0.97                    # numerical ceiling on any activity rate


@dataclass(frozen=True)
class LabourBase:
    shares: np.ndarray        # [status, age, sex, region] 2024 status shares of population
    cohort_mult: np.ndarray   # [year - 2024, age, sex] activity-rate multiplier from cohort effects
    employees: np.ndarray     # [sector] national-accounts employees 2024 (persons, domestic)
    self_emp: np.ndarray      # [sector] national-accounts self-employed 2024 (persons)
    d11: np.ndarray           # [sector] wages and salaries 2024 (EUR mn)
    d1: np.ndarray            # [sector] compensation of employees 2024 (EUR mn)
    hours_per_worker: float   # national accounts hours worked per person employed, 2024
    decile_rel: np.ndarray    # [decile] employee earnings by tenth relative to the mean (SES 2022)
    misc: dict                # scalar data points (labour_misc.csv)


def _freeze(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _spread5(vals12: np.ndarray, v75: float, lo_anchor: float | None) -> np.ndarray:
    """Rates for 5-year groups 15-19..70-74 and 75+ -> single ages 0..100 (0 below 15).

    Shape-preserving cubic through the group mid-points, then each 5-year group is rescaled so
    that its (unweighted) mean equals the published value.
    """
    xs = list(G5_LO + 2.5) + [77.5, 87.5, 100.5]
    ys = list(vals12) + [min(2.0 * v75, vals12[-1]), 0.3 * v75, 0.0]
    if lo_anchor is not None:
        xs, ys = [14.5] + xs, [lo_anchor] + ys
    f = PchipInterpolator(xs, ys, extrapolate=True)
    out = np.zeros(N_AGE)
    out[15:] = np.clip(f(AGE[15:] + 0.5), 0.0, 1.0)
    for g, lo in enumerate(G5_LO):
        m = out[lo:lo + 5].mean()
        if m > 0:
            out[lo:lo + 5] *= vals12[g] / m
    return np.clip(out, 0.0, 1.0)


def _bands_to_ages(bands: list[tuple[int, int]], vals: list[float], end_zero_age: int) -> np.ndarray:
    """Band values -> single ages: linear between band mid-points, band means preserved."""
    xs = [(lo + hi + 1) / 2 for lo, hi in bands] + [end_zero_age + 0.5]
    ys = list(vals) + [0.0]
    out = np.interp(AGE + 0.5, xs, ys, left=vals[0], right=0.0)
    out[: bands[0][0]] = 0.0
    for (lo, hi), v in zip(bands, vals):
        m = out[lo:hi + 1].mean()
        if m > 0:
            out[lo:hi + 1] *= v / m
    return np.clip(out, 0.0, 1.0)


def _knots(profile: list, ages: np.ndarray = AGE) -> np.ndarray:
    """[[age, value], ...] -> values at single ages (linear, 0 outside the knots)."""
    xs = [float(k[0]) for k in profile]
    ys = [float(k[1]) for k in profile]
    return np.interp(ages, xs, ys, left=0.0, right=0.0)


def _decile_profile(d1: float, med: float, d9: float, mean: float) -> np.ndarray:
    """Mean earnings of each tenth of employees relative to the overall mean.

    Quantile function: log-linear in the normal score, with separate slopes below and above
    the median fitted on D1 and D9; the top tenth is then adjusted so that the published mean
    is reproduced.
    """
    z9 = float(ndtri(0.9))
    s_lo = (np.log(med) - np.log(d1)) / z9
    s_hi = (np.log(d9) - np.log(med)) / z9
    pgrid = (np.arange(10_000) + 0.5) / 10_000
    z = ndtri(pgrid)
    q = med * np.exp(np.where(z < 0, s_lo, s_hi) * z)
    dec = q.reshape(N_DECILE, -1).mean(axis=1)
    dec[-1] += (mean - dec.mean()) * N_DECILE
    return dec / dec.mean()


def _cohort_multipliers(status: pd.DataFrame, start_age: int, last_age: int,
                        r_lo: float, r_hi: float) -> np.ndarray:
    """Activity-rate multipliers [year-2024, age, sex] relative to 2024 from cohort effects.

    R(g) = change in the activity rate of 5-year group g between 2018-19 and 2023-24. A cohort
    that was more active at age g than its predecessors is assumed to remain so as it ages:
    m(g, 2024+k) = m(g-1, 2024+k-5) * R(g-1), phased in linearly over the 5 years it takes the
    cohort to move up one group. Only groups from `start_age` to `last_age` are affected.
    """
    out = np.ones((N_COHORT_YEARS, N_AGE, N_SEX))
    status = status[status["age_lo"] < 75]
    for si, sex in enumerate(SEXES):
        t = status[status.sex == sex].assign(act=lambda d: (d.EMP + d.UNE) / d.POP)
        t = t.pivot_table(index="age_lo", columns="year", values="act")
        ratio = (t[[BASE_YEAR - 1, BASE_YEAR]].mean(axis=1) / t[[BASE_YEAR - 6, BASE_YEAR - 5]].mean(axis=1))
        ratio = ratio.clip(r_lo, r_hi).reindex(G5_LO).to_numpy()
        m = np.ones((N_COHORT_YEARS, len(G5_LO)))
        for k in range(1, N_COHORT_YEARS):
            for g, lo in enumerate(G5_LO):
                if lo < start_age or lo > last_age or g == 0:
                    continue
                if k <= 5:
                    m[k, g] = 1.0 + (k / 5.0) * (ratio[g - 1] - 1.0)
                else:
                    m[k, g] = ratio[g - 1] * m[k - 5, g - 1]
        mids = G5_LO + 2.5
        for k in range(N_COHORT_YEARS):
            out[k, :, si] = np.interp(AGE + 0.5, mids, m[k], left=1.0, right=1.0)
    return out


def baseline_extra(ctx, key: str) -> dict:
    """Calibration factors / reference stocks of the baseline run's base year, so that a measure
    starting in the base year is not absorbed by the calibration."""
    b = getattr(ctx, "baseline", None)
    if b is not None and b.has(BASE_YEAR):
        x = b.at(BASE_YEAR).extra.get(key) or {}
        return {k: x[k] for k in ("calib", "ref") if k in x}
    return {}


def calibration_key(p) -> str:
    """Hashable key of the parameters that shape the 2024 calibration (not policy levers)."""
    names = [
        "labour.public_share_by_sector", "labour.disabled_2024", "labour.disability_age_profile",
        "labour.disability_female_ratio", "labour.early_retired_age_profile",
        "labour.calibration_legal_pension_age", "labour.cohort_start_age", "labour.cohort_last_age",
        "labour.cohort_ratio_bounds",
    ]
    return json.dumps({n: p[n] for n in names}, sort_keys=True)


@lru_cache(maxsize=8)
def load(key: str) -> LabourBase:
    c = json.loads(key)
    status_all = pd.read_csv(DATA_DIR / "labour_lfs_status.csv")
    status = status_all[status_all.year == BASE_YEAR]
    prof = pd.read_csv(DATA_DIR / "labour_lfs_profstatus.csv")
    stud = pd.read_csv(DATA_DIR / "labour_students.csv")
    reg = pd.read_csv(DATA_DIR / "labour_lfs_region.csv")
    sector = pd.read_csv(DATA_DIR / "labour_sector.csv").set_index("sector").reindex(SECTORS)
    earn = pd.read_csv(DATA_DIR / "labour_earnings_dist.csv")
    misc = pd.read_csv(DATA_DIR / "labour_misc.csv").set_index("key")["value"].to_dict()
    acc = pd.read_csv(DATA_DIR / "macro_accounts.csv")
    misc["mixed_income_meur"] = float(acc[(acc.sector == "S14_S15") & (acc.na_item == "B3G")
                                          & (acc.direct == "RECV")]["value"].iloc[0])

    legal_age = int(c["labour.calibration_legal_pension_age"])

    # --- sector tables (national accounts) ---
    employees = sector["employees_ths"].to_numpy() * 1e3
    self_emp = sector["self_ths"].to_numpy() * 1e3
    pub_sh = np.array([float(c["labour.public_share_by_sector"].get(s, 0.0)) for s in SECTORS])
    public_share_employees = float((employees * pub_sh).sum() / employees.sum())
    self_share_na = misc["na_SELF_NC_ths"] / misc["na_EMP_NC_ths"]

    # --- national single-age rates by sex ---
    emp = np.zeros((N_AGE, N_SEX))
    une = np.zeros((N_AGE, N_SEX))
    selfsh = np.zeros((N_AGE, N_SEX))
    pop5 = np.zeros((N_AGE, N_SEX))            # LFS population spread uniformly inside each group
    emp_w = np.zeros((N_AGE, N_SEX))
    for si, sex in enumerate(SEXES):
        d = status[status.sex == sex].set_index("age_lo")
        e12 = (d.loc[G5_LO, "EMP"] / d.loc[G5_LO, "POP"]).to_numpy()
        u12 = (d.loc[G5_LO, "UNE"] / d.loc[G5_LO, "POP"]).to_numpy()
        e75 = float(d.loc[75, "EMP"] / d.loc[75, "POP"]) if not np.isnan(d.loc[75, "EMP"]) else 0.0
        emp[:, si] = _spread5(e12, e75, lo_anchor=0.0)
        une[:, si] = _spread5(u12, 0.0, lo_anchor=0.0)
        une[legal_age + 5:, si] = 0.0
        for lo in G5_LO:
            pop5[lo:lo + 5, si] = d.loc[lo, "POP"] * 1e3 / 5
        q = prof[prof.sex == sex].set_index("age_lo")
        s12 = ((q.loc[G5_LO, "SELF"] + q.loc[G5_LO, "CFAM"]) / q.loc[G5_LO, "EMP"]).to_numpy()
        s75 = float((q.loc[75, "SELF"] + q.loc[75, "CFAM"]) / q.loc[75, "EMP"])
        selfsh[:, si] = np.interp(AGE + 0.5, list(G5_LO + 2.5) + [80.0], list(s12) + [s75])
        emp_w[:, si] = emp[:, si] * pop5[:, si]
    # rescale the self-employed share to the national-accounts concept
    k_self = self_share_na * emp_w.sum() / (selfsh * emp_w).sum()
    selfsh = np.clip(selfsh * k_self, 0.0, 0.95)

    # --- student, disabled, early-retired shares of population by age x sex (national) ---
    student = np.zeros((N_AGE, N_SEX))
    disabled = np.zeros((N_AGE, N_SEX))
    early_ret = np.zeros((N_AGE, N_SEX))
    ret_share_65 = np.zeros(N_SEX)
    dis_shape = _knots(c["labour.disability_age_profile"])
    dis_shape[legal_age:] = 0.0
    dis_sex = np.array([1.0, float(c["labour.disability_female_ratio"])])
    dis_total = float(c["labour.disabled_2024"])
    dis_scale = dis_total / (dis_shape[:, None] * dis_sex[None, :] * pop5).sum()
    ret_shape = _knots(c["labour.early_retired_age_profile"])
    ret_shape[legal_age:] = 0.0
    for si, sex in enumerate(SEXES):
        d = stud[stud.sex == sex]
        bands = list(zip(d.age_lo, d.age_hi))
        student[:, si] = _bands_to_ages(bands, list(d.student_share), end_zero_age=39)
        disabled[:, si] = dis_shape * dis_sex[si] * dis_scale
        pop_18_64 = pop5[18:65, si].sum()
        target = misc[f"silc_RET_Y18-64_{sex}"] * pop_18_64
        early_ret[:, si] = ret_shape * target / max((ret_shape * pop5[:, si]).sum(), 1e-9)
        r, o = misc[f"silc_RET_Y_GE65_{sex}"], misc[f"silc_INAC_OTH_Y_GE65_{sex}"]
        ret_share_65[si] = r / (r + o)

    # --- regional ratios by broad age band ---
    bands = ["Y15-24", "Y25-34", "Y35-44", "Y45-54", "Y55-64", "Y_GE65"]
    mids = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    shares = np.zeros((N_STATUS, N_AGE, N_SEX, N_REGION))
    for ri, rname in enumerate(REGIONS):
        for si, sex in enumerate(SEXES):
            nat = reg[(reg.geo == "BE") & (reg.sex == sex)].set_index("age")
            loc = reg[(reg.geo == REGION_NUTS1[rname]) & (reg.sex == sex)].set_index("age")
            r_e = [loc.loc[b, "emp_rate"] / nat.loc[b, "emp_rate"] for b in bands]
            u_nat = [max(nat.loc[b, "act_rate"] - nat.loc[b, "emp_rate"], 1e-4) for b in bands]
            u_loc = [max(loc.loc[b, "act_rate"] - loc.loc[b, "emp_rate"], 0.0) for b in bands]
            r_u = [ul / un for ul, un in zip(u_loc, u_nat)]
            r_u[-1] = 1.0                                     # 65+: too few observations
            e = np.clip(emp[:, si] * np.interp(AGE + 0.5, mids, r_e), 0.0, 0.95)
            u = une[:, si] * np.interp(AGE + 0.5, mids, r_u)
            u = np.minimum(u, MAX_ACTIVITY - e)
            rest = 1.0 - e - u
            st = np.minimum(student[:, si], rest)
            st[15:18] = rest[15:18]                           # compulsory schooling until 18
            rest = rest - st
            dis = np.minimum(disabled[:, si], 0.9 * rest)
            dis[:18] = 0.0
            rest = rest - dis
            ret = np.minimum(early_ret[:, si], 0.95 * rest)
            ret[legal_age:] = rest[legal_age:] * ret_share_65[si]
            inact = rest - ret
            se = e * selfsh[:, si]
            pub = (e - se) * public_share_employees
            pub[:18] = 0.0
            shares[S["emp_private"], :, si, ri] = e - se - pub
            shares[S["emp_public"], :, si, ri] = pub
            shares[S["self_employed"], :, si, ri] = se
            shares[S["unemployed"], :, si, ri] = u
            shares[S["student"], :, si, ri] = st
            shares[S["disabled"], :, si, ri] = dis
            shares[S["retired"], :, si, ri] = ret
            shares[S["inactive"], :, si, ri] = inact
    shares[:, :15] = 0.0
    shares[S["child"], :15] = 1.0
    shares = np.clip(shares, 0.0, 1.0)
    shares /= shares.sum(axis=0, keepdims=True)

    lo_b, hi_b = c["labour.cohort_ratio_bounds"]
    cohort = _cohort_multipliers(status_all, int(c["labour.cohort_start_age"]),
                                 int(c["labour.cohort_last_age"]), float(lo_b), float(hi_b))

    e_row = earn[(earn.nace_r2 == "B-S_X_O") & (earn.worktime == "TOTAL")].iloc[0]
    decile_rel = _decile_profile(e_row.D1_E_EUR, e_row.MED_E_EUR, e_row.D9_E_EUR, e_row.MEAN_E_EUR)

    hours = float(sector["emp_hours_ths"].sum() / sector["emp_ths"].sum())
    assert employees.shape == (N_SECTOR,)
    return LabourBase(
        shares=_freeze(shares), cohort_mult=_freeze(cohort), employees=_freeze(employees),
        self_emp=_freeze(self_emp), d11=_freeze(sector["d11_meur"].to_numpy(dtype=float)),
        d1=_freeze(sector["d1_meur"].to_numpy(dtype=float)), hours_per_worker=hours,
        decile_rel=_freeze(decile_rel), misc=misc,
    )
