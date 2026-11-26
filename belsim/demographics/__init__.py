"""Cohort-component population projection for Belgium by single year of age, sex and region.

Conventions
-----------
- `state.pop[age, sex, region]` is the population at 1 January of `state.year` (2024 = observed).
- `state.demo` flows (births, deaths, migration) are those occurring *during* `state.year`, so
  `pop[t+1] = pop[t] + births[t] - deaths[t] + net_migration[t]` holds exactly.
- The population of next 1 January is computed together with the flows and carried in
  `state.extra["demographics"]` (no module-level state between years).

One year, starting from the population P at 1 January:
1. deaths: each cohort is exposed to the death rates of the two ages it spans during the year;
   rates = observed 2024 regional rates x cumulated EUROPOP2023 improvement;
2. births: regional age-specific fertility x women 15-49, national TFR path from parameters;
3. international migration: emigration = age/sex/region rates x population; immigration =
   (net migration parameter + emigration at baseline rates), spread by an age-sex profile and
   regional shares;
4. inter-regional migration: origin-destination rates by age and sex x origin population.
The migration factors `demo.immigration_factor` (immigrants settling in the region),
`demo.emigration_factor` (emigration rates of the region's residents) and
`demo.internal_migration_factor` (moves out of the region to the two others) are region-keyed
levers `{FLA, WAL, BRU}`; a scalar is accepted and applies to the three regions.
Migrants are added at the end of the year (no deaths or births for them in the year of the move).

All tables come from data/base/demographics_*.csv (see belsim/data/fetch_demographics.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.dims import BASE_YEAR, N_AGE, N_REGION, N_SEX, REGIONS, SEXES

DATA = Path(__file__).resolve().parents[2] / "data" / "base"
FERT_AGES = slice(15, 50)          # mothers' ages 15..49 (younger/older folded into the ends)
MAX_OUT_RATE = 0.9                 # cap on the share of a cell that can leave in one year
KEY = "demographics"


# ----------------------------------------------------------------------------- tables
@dataclass(frozen=True)
class Tables:
    pop0: np.ndarray            # [age, sex, region] population 1 Jan 2024
    mort0: np.ndarray           # [age, sex, region] death rates 2024 (before calibration)
    asfr_reg: np.ndarray        # [35, region] observed fertility rates 2024
    asfr_nat: np.ndarray        # [35] observed national fertility rates 2024
    tfr_obs: float              # observed national TFR 2024
    years: np.ndarray           # years covered by the EUROPOP assumption tables
    ep_tfr: np.ndarray          # [year] EUROPOP national TFR
    ep_asfr_ratio: np.ndarray   # [year, 35] EUROPOP ASFR relative to 2024
    ep_log_asmr: np.ndarray     # [year, age, sex] log EUROPOP death rates relative to 2024
    imm_profile: np.ndarray     # [age, sex] immigration shares (age at end of year), sums to 1
    emi_rate: np.ndarray        # [age, sex, region] emigration rates (age at end of year)
    imm_region: np.ndarray      # [region] regional shares of immigration
    internal: np.ndarray        # [age, sex, origin, destination] inter-regional rates
    births_obs: np.ndarray      # [region] observed births 2024
    deaths_obs: np.ndarray      # [region] observed deaths 2024
    europop: pd.DataFrame       # EUROPOP2023 baseline summary by year (validation target)

    def yi(self, year: int) -> int:
        """Index of `year` in the assumption tables (held at the last available year)."""
        return int(min(max(year - int(self.years[0]), 0), len(self.years) - 1))


def _cube(df: pd.DataFrame, col: str) -> np.ndarray:
    out = np.zeros((N_AGE, N_SEX, N_REGION))
    a = df["age"].to_numpy()
    s = df["sex"].map({x: i for i, x in enumerate(SEXES)}).to_numpy()
    r = df["region"].map({x: i for i, x in enumerate(REGIONS)}).to_numpy()
    out[a, s, r] = df[col].to_numpy(float)
    return out


@lru_cache(maxsize=1)
def tables() -> Tables:
    """Load every calibration table once (read-only afterwards)."""
    pop0 = _cube(pd.read_csv(DATA / "demographics_population_2024.csv"), "persons")
    mort0 = _cube(pd.read_csv(DATA / "demographics_mortality_2024.csv"), "death_rate")

    fert = pd.read_csv(DATA / "demographics_fertility_2024.csv").set_index("age").reindex(range(15, 50)).fillna(0.0)
    asfr_reg = fert[REGIONS].to_numpy(float)
    asfr_nat = fert["BE"].to_numpy(float)

    summ = pd.read_csv(DATA / "demographics_europop_summary.csv").set_index("year")
    summ = summ.loc[BASE_YEAR:]
    years = summ.index.to_numpy()
    asfr = pd.read_csv(DATA / "demographics_europop_asfr.csv").pivot(index="year", columns="age", values="asfr")
    asfr = asfr.reindex(index=years, columns=range(15, 50)).to_numpy(float)
    ratio = np.divide(asfr, asfr[0], out=np.ones_like(asfr), where=asfr[0] > 0)

    asmr = pd.read_csv(DATA / "demographics_europop_asmr.csv")
    arr = np.zeros((len(years), N_AGE, N_SEX))
    asmr = asmr[asmr["year"].isin(years)]
    arr[asmr["year"].to_numpy() - years[0], asmr["age"].to_numpy(),
        asmr["sex"].map({"M": 0, "F": 1}).to_numpy()] = asmr["rate"].to_numpy(float)
    arr = np.maximum(arr, 1e-7)
    log_asmr = np.log(arr / arr[0])

    mig = pd.read_csv(DATA / "demographics_migration_age.csv")
    si = mig["sex"].map({"M": 0, "F": 1}).to_numpy()
    imm_profile = np.zeros((N_AGE, N_SEX))
    emi_age = np.zeros((N_AGE, N_SEX))
    imm_profile[mig["age"].to_numpy(), si] = mig["immigration_share"].to_numpy(float)
    emi_age[mig["age"].to_numpy(), si] = mig["emigration_rate"].to_numpy(float)
    reg = pd.read_csv(DATA / "demographics_migration_region.csv").set_index("region").reindex(REGIONS)
    emi_rate = emi_age[:, :, None] * reg["emigration_factor"].to_numpy(float)

    od = pd.read_csv(DATA / "demographics_internal_migration.csv")
    internal = np.zeros((N_AGE, N_SEX, N_REGION, N_REGION))
    ri = {x: i for i, x in enumerate(REGIONS)}
    internal[od["age"].to_numpy(), od["sex"].map({"M": 0, "F": 1}).to_numpy(),
             od["origin"].map(ri).to_numpy(), od["destination"].map(ri).to_numpy()] = od["rate"].to_numpy(float)

    obs = pd.read_csv(DATA / "demographics_observed.csv")
    o24 = obs[obs["year"] == BASE_YEAR].set_index("region").reindex(REGIONS)

    t = Tables(
        pop0=pop0, mort0=mort0, asfr_reg=asfr_reg, asfr_nat=asfr_nat, tfr_obs=float(asfr_nat.sum()),
        years=years, ep_tfr=summ["tfr"].to_numpy(float), ep_asfr_ratio=ratio, ep_log_asmr=log_asmr,
        imm_profile=imm_profile / imm_profile.sum(), emi_rate=emi_rate,
        imm_region=reg["immigration_share"].to_numpy(float), internal=internal,
        births_obs=o24["births"].to_numpy(float), deaths_obs=o24["deaths"].to_numpy(float),
        europop=summ,
    )
    for v in t.__dict__.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return t


def reference() -> pd.DataFrame:
    """EUROPOP2023 baseline for Belgium by year (population, 67+, births, deaths, migration, TFR)."""
    return tables().europop.copy()


# ----------------------------------------------------------------------------- building blocks
def _by_region(v) -> np.ndarray:
    """Region-keyed parameter `{FLA, WAL, BRU}` (or a scalar for all regions) -> array [region]."""
    if isinstance(v, dict):
        return np.array([float(v[r]) for r in REGIONS])
    return np.full(N_REGION, float(v))


def _hazard(m: np.ndarray, infant_share: float) -> np.ndarray:
    """Death hazard over one calendar year for the cohort aged `a` at 1 January."""
    h = np.empty_like(m)
    h[:-1] = 0.5 * (m[:-1] + m[1:])
    h[0] = (1.0 - infant_share) * m[0] + 0.5 * m[1]
    h[-1] = m[-1]
    return h


@lru_cache(maxsize=8)
def _calibration(infant_share: float) -> tuple[np.ndarray, np.ndarray]:
    """Regional factors making the 2024 births and deaths equal the observed regional totals.

    They absorb the difference between the model's exposure (population at 1 January) and the
    true person-years; fitted once on 2024 and held constant.
    """
    t = tables()
    births0 = (t.asfr_reg * t.pop0[FERT_AGES, 1, :]).sum(axis=0)
    cal_b = t.births_obs / births0
    cal_m = np.ones(N_REGION)
    for _ in range(12):
        m = t.mort0 * cal_m
        d = (t.pop0 * -np.expm1(-_hazard(m, infant_share))).sum(axis=(0, 1))
        d += t.births_obs * -np.expm1(-infant_share * m[0].mean(axis=0))
        cal_m = cal_m * t.deaths_obs / d
    cal_b.setflags(write=False)
    cal_m.setflags(write=False)
    return cal_b, cal_m


def life_expectancy(m: np.ndarray) -> np.ndarray:
    """Period life expectancy at birth from death rates `m[age, ...]` (last age open-ended)."""
    q = -np.expm1(-m)
    q[-1] = 1.0
    l = np.concatenate([np.ones_like(m[:1]), np.cumprod(1.0 - q[:-1], axis=0)], axis=0)
    d = l * q
    a = np.full(m.shape[0], 0.5)
    a[0] = 0.1
    L = l - d * (1.0 - a.reshape((-1,) + (1,) * (m.ndim - 1)))
    L[-1] = l[-1] / np.maximum(m[-1], 1e-9)
    return L.sum(axis=0)


def tfr_path(year: int, p) -> float:
    """National total fertility rate implied by the parameters for `year`."""
    t = tables()
    ref = t.ep_tfr[t.yi(year)]
    target_year = int(p["demo.tfr_target_year"])
    conv_year = float(p["demo.tfr_convergence_year"])
    w = np.clip((year - BASE_YEAR) / max(target_year - BASE_YEAR, 1), 0.0, 1.0)
    scale = 1.0 + (float(p["demo.tfr_target"]) / t.ep_tfr[t.yi(target_year)] - 1.0) * w
    left = np.clip(1.0 - (year - BASE_YEAR) / max(conv_year - BASE_YEAR, 1e-9), 0.0, 1.0)
    gap = (t.tfr_obs / t.ep_tfr[0] - 1.0) * left
    return float(ref * scale * (1.0 + gap))


def _log_mort(year: int, prev_log: np.ndarray | None, p) -> np.ndarray:
    """Cumulated log change of death rates since 2024, [age, sex]."""
    t = tables()
    if year <= BASE_YEAR:
        return np.zeros((N_AGE, N_SEX))
    mult = float(p["demo.mortality_improvement"])
    by_sex = p.get("demo.mortality_improvement_by_sex")
    if by_sex:                # [sex] multipliers on top of the general lever (men, women)
        mult = mult * np.array([float(by_sex["M"]), float(by_sex["F"])])[None, :]
    if prev_log is None:      # no history available: apply this year's multiplier to the whole path
        return mult * t.ep_log_asmr[t.yi(year)]
    return prev_log + mult * (t.ep_log_asmr[t.yi(year)] - t.ep_log_asmr[t.yi(year - 1)])


def step(pop: np.ndarray, year: int, p, log_mort: np.ndarray) -> dict:
    """Flows during `year` and the population at 1 January `year + 1`.

    Returns a dict with `pop_next` and the flow arrays; pure function of its arguments.
    """
    t = tables()
    f_inf = float(p["demo.infant_death_share_birth_year"])
    cal_b, cal_m = _calibration(f_inf)

    # --- mortality
    m = t.mort0 * np.exp(log_mort)[:, :, None] * cal_m
    deaths_c = pop * -np.expm1(-_hazard(m, f_inf))
    surv = pop - deaths_c
    s = np.empty_like(pop)                 # survivors by age at the end of the year
    s[1:] = surv[:-1]
    s[-1] += surv[-1]

    # --- fertility
    yi = t.yi(year)
    shape = t.ep_asfr_ratio[yi]
    k = tfr_path(year, p) / float((t.asfr_nat * shape).sum())
    ff = p["demo.fertility_factor"]
    ffac = np.array([float(ff[r]) for r in REGIONS]) if isinstance(ff, dict) else np.full(N_REGION, float(ff))
    asfr = t.asfr_reg * shape[:, None] * (k * ffac)
    women = pop[FERT_AGES, 1, :]
    births_r = (asfr * women).sum(axis=0) * cal_b
    male = float(p["demo.sex_ratio_at_birth"])
    male = male / (1.0 + male)
    births = np.stack([births_r * male, births_r * (1.0 - male)], axis=0)      # [sex, region]
    deaths_b = births * -np.expm1(-f_inf * m[0])
    s[0] = births - deaths_b

    # --- international migration (the three factors are region-keyed levers; a scalar applies
    #     to every region: immigrants settling in / emigrants leaving / movers leaving the region)
    e_fac = _by_region(p["demo.emigration_factor"])
    i_fac = _by_region(p["demo.immigration_factor"])
    # age pattern of emigration relative to the rates observed in 2023-2024 (knots [age, factor],
    # linear in between; part of the reference path, so it also sets the immigration that goes
    # with a given net migration)
    rate0 = t.emi_rate
    knots = p.get("demo.emigration_rate_age_profile")
    if knots:
        kn = np.asarray(knots, dtype=float)
        a_fac = np.interp(np.arange(N_AGE), kn[:, 0], kn[:, 1])
        if np.any(a_fac != 1.0):
            rate0 = rate0 * a_fac[:, None, None]
    emi_base = np.minimum(rate0, MAX_OUT_RATE) * s
    emi_rate = np.minimum(rate0 * e_fac, MAX_OUT_RATE)
    emi = emi_rate * s
    imm_total = max(0.0, float(p["demo.net_migration"]) + float(emi_base.sum()))
    imm = imm_total * t.imm_profile[:, :, None] * (t.imm_region * i_fac)

    # --- inter-regional migration (origin-destination rates on the survivors of the origin)
    od = t.internal * _by_region(p["demo.internal_migration_factor"])[:, None]
    out_rate = od.sum(axis=3)
    room = np.maximum(MAX_OUT_RATE - emi_rate, 0.0)
    squeeze = np.where(out_rate > room, room / np.maximum(out_rate, 1e-12), 1.0)
    flows = od * (s * squeeze)[:, :, :, None]
    internal_net = flows.sum(axis=2) - flows.sum(axis=3)

    pop_next = s - emi + imm + internal_net
    return {
        "pop_next": pop_next, "deaths_cohort": deaths_c, "deaths_newborn": deaths_b,
        "births": births, "immigration": imm, "emigration": emi, "internal_net": internal_net,
        "asfr": asfr, "mort": m,
    }


# ----------------------------------------------------------------------------- engine entry
def run(state, prev, p, ctx) -> None:
    t = tables()
    year = state.year
    if prev is None:
        pop = t.pop0.copy()
        log_mort = _log_mort(year, None, p)
    else:
        carry = prev.extra.get(KEY) if isinstance(prev.extra, dict) else None
        if carry and "pop_next" in carry:
            pop = carry["pop_next"].copy()
            log_mort = _log_mort(year, carry["log_mort"], p)
        else:
            # `prev` was built without this module (unit tests of other modules): redo last year's
            # step from prev.pop with the current parameters.
            lm_prev = _log_mort(prev.year, None, p)
            pop = step(prev.pop, prev.year, p, lm_prev)["pop_next"]
            log_mort = _log_mort(year, lm_prev, p)

    out = step(pop, year, p, log_mort)

    births = float(out["births"].sum())
    deaths = float(out["deaths_cohort"].sum() + out["deaths_newborn"].sum())
    imm = float(out["immigration"].sum())
    emi = float(out["emigration"].sum())

    women = pop[FERT_AGES, 1, :]
    w_nat = women.sum(axis=1)
    tfr = float(((out["asfr"] * women).sum(axis=1) / np.maximum(w_nat, 1e-9)).sum())
    w = np.maximum(pop, 1e-9)
    m_nat = (out["mort"] * w).sum(axis=2) / w.sum(axis=2)
    e0 = life_expectancy(m_nat)
    e0_both = life_expectancy((out["mort"] * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2)))
    e0_reg = life_expectancy((out["mort"] * w).sum(axis=1) / w.sum(axis=1))
    tfr_reg = out["asfr"].sum(axis=0)
    net_int = out["internal_net"].sum(axis=(0, 1))

    state.pop = pop
    state.demo = {
        "births": births,
        "deaths": deaths,
        "net_migration": imm - emi,
        "immigration": imm,
        "emigration": emi,
        "tfr": tfr,
        "life_expectancy_M": float(e0[0]),
        "life_expectancy_F": float(e0[1]),
        "life_expectancy": float(e0_both),
        "old_age_dependency": float(pop[67:].sum() / pop[18:67].sum()),
        "natural_change": births - deaths,
        "population": float(pop.sum()),
    }
    for i, r in enumerate(REGIONS):
        state.demo[f"tfr.{r}"] = float(tfr_reg[i])
        state.demo[f"life_expectancy.{r}"] = float(e0_reg[i])
        state.demo[f"net_internal_migration.{r}"] = float(net_int[i])
    state.extra[KEY] = {
        "pop_next": out["pop_next"],          # population at 1 January of year + 1
        "log_mort": log_mort,                 # cumulated log change of death rates used this year
        "births": out["births"],              # [sex, region]
        "deaths": out["deaths_cohort"],       # [age at 1 Jan, sex, region] (newborn deaths apart)
        "deaths_newborn": out["deaths_newborn"],
        "immigration": out["immigration"],    # [age at 31 Dec, sex, region]
        "emigration": out["emigration"],
        "internal_net": out["internal_net"],
    }


def load_params():
    """ParamSet holding only params/demographics.yaml (enough to run this module alone)."""
    import yaml
    from ..core.params import PARAMS_DIR, ParamSet, ParamSpec

    data = yaml.safe_load((PARAMS_DIR / "demographics.yaml").read_text())
    return ParamSet({
        name: ParamSpec(name=name, values={int(y): v for y, v in spec["values"].items()},
                        description=spec.get("description", ""), unit=str(spec.get("unit", "")),
                        source=spec.get("source", ""), file="demographics.yaml")
        for name, spec in data.items()
    })


def project(pset=None, end: int | None = None, scenario=None):
    """Run the demographic module alone (no economy). Returns a core `Results`."""
    from ..core.dims import END_YEAR
    from ..core.engine import Context
    from ..core.params import ParamSet
    from ..core.state import Results, YearState

    pset = (pset or load_params()).with_scenario(scenario)
    res = Results(name=pset.scenario.name)
    prev = None
    for year in range(BASE_YEAR, (end or END_YEAR) + 1):
        st = YearState(year=year)
        ctx = Context(year=year, params=pset, history=res, baseline=None, base_p=None)
        run(st, prev, pset.at(year, None), ctx)
        res.add(st)
        prev = st
    return res
