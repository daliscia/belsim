"""Download and clean the demographic calibration tables.

    uv run python -m belsim.data.fetch_demographics [--refresh]

Sources (all official, no key needed):
- Eurostat dissemination API (JSON-stat): demo_r_d2jan, demo_r_magec, demo_r_frate2, demo_r_mlife,
  demo_mlifetable, demo_gind, demo_r_gind3, demo_fasec, migr_imm8, migr_emi2 and the EUROPOP2023
  projection (proj_23np, proj_23ndbi, proj_23naasfr, proj_23naasmr).
- Statbel "Population change" workbook (pop1992-mov_en.xlsx): international and internal migration
  totals by region.

Raw downloads are cached in data/raw/demographics/, cleaned tables written to
data/base/demographics_*.csv. The model reads only the cleaned tables.

Derived (not directly observed) quantities, all computed here from the downloads:
- net migration by age, sex and region = cohort residual of two consecutive 1-January populations
  and deaths (average of the flows of 2023 and 2024);
- "implied" immigration by age = national residual net migration + observed emigration (so that it
  includes the statistical adjustment that Eurostat adds to net migration);
- inter-regional migration by age = regional residual minus the region's share of international
  migration, expressed as origin-destination rates.
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "demographics"
BASE = ROOT / "data" / "base"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
STATBEL_MOV = ("https://statbel.fgov.be/sites/default/files/files/documents/bevolking/"
               "5.2%20Loop%20van%20de%20bevolking/pop1992-mov_en.xlsx")
HEADERS = {"User-Agent": "python-requests"}

NUTS = {"BE2": "FLA", "BE3": "WAL", "BE1": "BRU"}
REGIONS = ["FLA", "WAL", "BRU"]
SEXES = ["M", "F"]
N_AGE = 101
FLOW_YEARS = [2023, 2024]          # years whose flows calibrate the migration schedules
PROJ_YEARS = list(range(2022, 2072))
STATBEL_REGION = {"02000": "FLA", "03000": "WAL", "04000": "BRU"}
MORT_GROUPS = [(0, 39), (40, 64), (65, 79), (80, 100)]   # age bands of the regional mortality ratios


# ----------------------------------------------------------------------------- download helpers
def _get(url: str, params=None, tries: int = 6) -> requests.Response:
    """GET with a few retries (both servers occasionally drop the TLS handshake)."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=180)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 + 3 * i)
    raise RuntimeError("unreachable")


def eurostat(name: str, dataset: str, refresh: bool = False, **filters) -> pd.DataFrame:
    """Fetch one Eurostat extract (cached as JSON) and return it as a long DataFrame."""
    path = RAW / f"{name}.json"
    if refresh or not path.exists():
        params: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN")]
        for k, v in filters.items():
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                params.append((k, str(x)))
        r = _get(API + dataset, params=params)
        path.write_text(r.text)
        print(f"  downloaded {dataset} -> {path.name}")
    j = json.loads(path.read_text())
    dims, sizes = j["id"], j["size"]
    cats = []
    for d in dims:
        idx = j["dimension"][d]["category"]["index"]
        cats.append(sorted(idx, key=idx.get))
    rows = []
    for k, v in j["value"].items():
        k = int(k)
        key = []
        for sz, c in zip(reversed(sizes), reversed(cats)):
            key.append(c[k % sz])
            k //= sz
        rows.append(list(reversed(key)) + [v])
    df = pd.DataFrame(rows, columns=dims + ["value"])
    df["time"] = df["time"].astype(int)
    return df.drop(columns=[c for c in ("freq", "unit") if c in df.columns])


def age_index(code: str) -> int | None:
    """Eurostat age code -> model age index 0..100 (None for totals / unknown)."""
    if code in ("Y_LT1",):
        return 0
    if code in ("Y_OPEN", "Y_GE100"):
        return 100
    m = re.fullmatch(r"Y(\d+)", code)
    if m:
        return min(int(m.group(1)), 100)
    return None


def by_age(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep single-year ages, map to 0..100 (summing anything above 100), spread unknown ages."""
    d = df.copy()
    d["a"] = d["age"].map(age_index)
    unk = d[d["age"] == "UNK"].groupby(keys)["value"].sum()
    d = d.dropna(subset=["a"])
    d["a"] = d["a"].astype(int)
    out = d.groupby(keys + ["a"])["value"].sum().unstack("a").reindex(columns=range(N_AGE)).fillna(0.0)
    if len(unk):
        tot = out.sum(axis=1)
        add = unk.reindex(out.index).fillna(0.0)
        out = out.mul(1.0 + add / tot.where(tot > 0, 1.0), axis=0)
    return out


def read_xlsx(path: Path) -> dict[str, list[dict[str, str]]]:
    """Minimal .xlsx reader (no openpyxl dependency): sheet name -> list of {column: text}."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    z = zipfile.ZipFile(path)
    strings: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            strings.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out: dict[str, list[dict[str, str]]] = {}
    for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", ns):
        target = rels[s.get("{%s}id" % ns["r"])].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).iter("{%s}row" % ns["m"]):
            d = {}
            for c in row.findall("m:c", ns):
                v = c.find("m:v", ns)
                if v is None:
                    continue
                col = re.match(r"[A-Z]+", c.get("r")).group()
                d[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(d)
        out[s.get("name")] = rows
    return out


def statbel_regions(refresh: bool) -> pd.DataFrame:
    """International and internal migration totals by region and year (Statbel)."""
    path = RAW / "statbel_pop_mov.xlsx"
    if refresh or not path.exists():
        r = _get(STATBEL_MOV)
        path.write_bytes(r.content)
        print(f"  downloaded Statbel population change -> {path.name}")
    wb = read_xlsx(path)
    rows = []
    for year in FLOW_YEARS:
        sheet = wb[str(year)]
        assert str(year) in sheet[0].get("A", ""), f"unexpected Statbel sheet layout for {year}"
        for d in sheet:
            reg = STATBEL_REGION.get(d.get("A", "").strip().zfill(5))
            if reg is None:
                continue
            f = lambda c: float(d[c])  # noqa: E731
            rows.append({
                "year": year, "region": reg, "population": f("C"),
                "internal_in": f("G"), "internal_out": f("H"),
                # immigration + change of register (in) + re-registrations
                "immigration": f("J") + f("K") + f("L"),
                # emigration + change of register (out) + automatic deletions
                "emigration": f("M") + f("N") + f("O"),
            })
    return pd.DataFrame(rows)


def smooth(x: np.ndarray, half: int) -> np.ndarray:
    """Centred moving average along axis 0 for ages >= 1 (age 0 kept as is)."""
    out = x.copy()
    n = x.shape[0]
    for a in range(1, n):
        lo, hi = max(1, a - half), min(n, a + half + 1)
        out[a] = x[lo:hi].mean(axis=0)
    return out


# ----------------------------------------------------------------------------- main
def main(refresh: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    geo3 = list(NUTS)
    params = yaml.safe_load((ROOT / "params" / "demographics.yaml").read_text())
    infant_share = float(params["demo.infant_death_share_birth_year"]["values"][2024])

    # ---- population at 1 January by age, sex, region (2023-2025) -------------------------
    pop_raw = eurostat("demo_r_d2jan", "demo_r_d2jan", refresh, geo=geo3, sex=SEXES,
                       time=[2023, 2024, 2025])
    pop_w = by_age(pop_raw, ["time", "geo", "sex"])

    def cube(wide: pd.DataFrame, year: int) -> np.ndarray:
        """[age, sex, region] array for one year."""
        return np.stack([np.stack([wide.loc[(year, g, s)].to_numpy(float) for g in ("BE2", "BE3", "BE1")],
                                  axis=-1) for s in SEXES], axis=1)

    pop = {y: cube(pop_w, y) for y in (2023, 2024, 2025)}
    idx = pd.MultiIndex.from_product([range(N_AGE), SEXES, REGIONS], names=["age", "sex", "region"])
    pd.DataFrame({"persons": pop[2024].ravel()}, index=idx).reset_index().to_csv(
        BASE / "demographics_population_2024.csv", index=False)

    # ---- observed balances (national and regional) ---------------------------------------
    gind = eurostat("demo_gind", "demo_gind", refresh, geo="BE", sinceTimePeriod=2022,
                    indic_de=["JAN", "LBIRTH", "DEATH", "CNMIGRAT"])
    gind3 = eurostat("demo_r_gind3", "demo_r_gind3", refresh, geo=geo3, sinceTimePeriod=2022,
                     indic_de=["JAN", "LBIRTH", "DEATH", "CNMIGRAT"])
    names = {"JAN": "population_1jan", "LBIRTH": "births", "DEATH": "deaths",
             "CNMIGRAT": "net_migration_incl_adjustment"}
    obs = pd.concat([gind.assign(region="BE"), gind3.assign(region=gind3["geo"].map(NUTS))])
    obs = obs.assign(indicator=obs["indic_de"].map(names)).pivot_table(
        index=["time", "region"], columns="indicator", values="value").reset_index().rename(
        columns={"time": "year"})
    imm_raw = eurostat("migr_imm8", "migr_imm8", refresh, geo="BE", agedef="REACH", sex=SEXES,
                       time=FLOW_YEARS)
    # Belgium reports emigration by single year of age for both sexes together only; the split by
    # sex comes from the five-year age groups of migr_emi1ctz.
    emi_raw = eurostat("migr_emi2", "migr_emi2", refresh, geo="BE", agedef="REACH", sex="T",
                       time=FLOW_YEARS)
    emi_grp = eurostat("migr_emi1ctz", "migr_emi1ctz", refresh, geo="BE", agedef="REACH",
                       citizen="TOTAL", sex=SEXES, time=FLOW_YEARS)
    imm_w = by_age(imm_raw, ["time", "sex"])
    emi_w = by_age(emi_raw, ["time", "sex"])
    imm = {y: np.stack([imm_w.loc[(y, s)].to_numpy(float) for s in SEXES], axis=1) for y in FLOW_YEARS}
    emi = {}
    for y in FLOW_YEARS:
        g = emi_grp[emi_grp["time"] == y].pivot_table(index="age", columns="sex", values="value")
        male = np.zeros(N_AGE)
        for a0 in range(0, 105, 5):
            code = "Y_LT5" if a0 == 0 else "Y_GE100" if a0 == 100 else f"Y{a0}-{a0 + 4}"
            tot = g.loc[code, "M"] + g.loc[code, "F"]
            male[a0:a0 + 5] = g.loc[code, "M"] / tot if tot > 0 else 0.5
        both = emi_w.loc[(y, "T")].to_numpy(float)
        emi[y] = np.stack([both * male, both * (1 - male)], axis=1)
    obs["immigration"] = np.nan
    obs["emigration"] = np.nan
    for y in FLOW_YEARS:
        m = (obs["year"] == y) & (obs["region"] == "BE")
        obs.loc[m, "immigration"] = imm[y].sum()
        obs.loc[m, "emigration"] = emi[y].sum()
    fasec = eurostat("demo_fasec", "demo_fasec", refresh, geo="BE", age="TOTAL", sex=SEXES,
                     sinceTimePeriod=2019)
    boys = fasec.pivot_table(index="time", columns="sex", values="value")
    obs = obs.merge((boys["M"] / boys["F"]).rename("sex_ratio_at_birth").reset_index().rename(
        columns={"time": "year"}).assign(region="BE"), on=["year", "region"], how="left")
    obs.to_csv(BASE / "demographics_observed.csv", index=False)
    male_share = float(boys["M"].sum() / (boys["M"].sum() + boys["F"].sum()))

    def obs_val(year: int, region: str, col: str) -> float:
        return float(obs[(obs["year"] == year) & (obs["region"] == region)][col].iloc[0])

    # ---- fertility: observed ASFR 2024 by region + national ------------------------------
    fr = eurostat("demo_r_frate2", "demo_r_frate2", refresh, geo=["BE"] + geo3, time=2024)
    fr = fr[fr["age"] != "TOTAL"].copy()
    fr["a"] = fr["age"].map(lambda c: 15 if c == "Y10-14" else 49 if c == "Y_GE50" else int(c[1:]))
    fert = fr.groupby(["a", "geo"])["value"].sum().unstack("geo").rename(
        columns={**NUTS, "BE": "BE"})[REGIONS + ["BE"]]
    fert.index.name = "age"
    fert.reset_index().to_csv(BASE / "demographics_fertility_2024.csv", index=False)

    # ---- EUROPOP2023 baseline: population, balances, assumptions --------------------------
    ep_pop = eurostat("proj_23np", "proj_23np", refresh, geo="BE", projection="BSL", sex=SEXES,
                      time=PROJ_YEARS)
    ep_w = by_age(ep_pop, ["time", "sex"])
    ep_long = ep_w.stack().rename("persons").reset_index().rename(columns={"time": "year", "a": "age"})
    ep_long.to_csv(BASE / "demographics_europop_population.csv", index=False)

    ep_bal = eurostat("proj_23ndbi", "proj_23ndbi", refresh, geo="BE", projection="BSL",
                      indic_de=["JAN", "LBIRTH", "DEATH", "NMIGRAT"], time=PROJ_YEARS)
    ep_bal = ep_bal.pivot_table(index="time", columns="indic_de", values="value").rename(
        columns={"JAN": "population_1jan", "LBIRTH": "births", "DEATH": "deaths",
                 "NMIGRAT": "net_migration"})
    ep_asfr = eurostat("proj_23naasfr", "proj_23naasfr", refresh, geo="BE", projection="BSL",
                       time=PROJ_YEARS)
    ep_bal["tfr"] = ep_asfr[ep_asfr["age"] == "TOTAL"].set_index("time")["value"]
    tot = ep_w.groupby(level="time").sum()
    ep_bal["pop_0_17"] = tot.loc[:, 0:17].sum(axis=1)
    ep_bal["pop_18_66"] = tot.loc[:, 18:66].sum(axis=1)
    ep_bal["pop_67_plus"] = tot.loc[:, 67:].sum(axis=1)
    ep_bal.index.name = "year"
    ep_bal.reset_index().to_csv(BASE / "demographics_europop_summary.csv", index=False)

    a = ep_asfr[ep_asfr["age"] != "TOTAL"].copy()
    a["age"] = a["age"].map(lambda c: 15 if c == "Y_LT15" else 49 if c == "Y_GE50" else int(c[1:]))
    a = a.groupby(["time", "age"])["value"].sum().rename("asfr").reset_index().rename(
        columns={"time": "year"})
    a.to_csv(BASE / "demographics_europop_asfr.csv", index=False)

    ep_m = eurostat("proj_23naasmr", "proj_23naasmr", refresh, geo="BE", projection="BSL",
                    sex=SEXES, time=PROJ_YEARS)
    ep_mw = by_age(ep_m, ["time", "sex"])   # NB rates: age codes map one to one, nothing is summed
    ep_mw.stack().rename("rate").reset_index().rename(columns={"time": "year", "a": "age"}).to_csv(
        BASE / "demographics_europop_asmr.csv", index=False)
    ep_m2024 = np.stack([ep_mw.loc[(2024, s)].to_numpy(float) for s in SEXES], axis=1)

    # ---- mortality: national death rates x regional ratio ---------------------------------
    lt_nat = eurostat("demo_mlifetable", "demo_mlifetable", refresh, geo="BE", indic_de="DEATHRATE",
                      sex=SEXES, time=[2022, 2023, 2024])
    lt_reg = eurostat("demo_r_mlife", "demo_r_mlife", refresh, geo=geo3, indic_de="DEATHRATE",
                      sex=SEXES, time=[2022, 2023, 2024])

    def rates(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
        d = df[df["time"].isin(years)].copy()
        d["a"] = d["age"].map(lambda c: 0 if c == "Y_LT1" else 95 if c == "Y_GE95" else int(c[1:]))
        keys = [k for k in ("geo", "sex") if d[k].nunique() > 1 or k == "sex"]
        return d.groupby(keys + ["a"])["value"].mean().unstack("a")

    nat2 = rates(lt_nat, [2023, 2024])       # level: mean of the two latest years
    nat3 = rates(lt_nat, [2022, 2023, 2024])  # for regional ratios
    reg3 = rates(lt_reg, [2022, 2023, 2024])
    mort = np.zeros((N_AGE, 2, 3))
    for si, s in enumerate(SEXES):
        m = np.zeros(N_AGE)
        m[:95] = nat2.loc[s].to_numpy(float)[:95]
        # 95..100+: life tables stop at 95+; extend with the age gradient of the EUROPOP rates
        m[95:] = m[94] * ep_m2024[95:, si] / ep_m2024[94, si]
        for ri, g in enumerate(("BE2", "BE3", "BE1")):
            w = pop[2024][:95, si, ri]
            r_reg = reg3.loc[(g, s)].to_numpy(float)[:95]
            r_nat = nat3.loc[s].to_numpy(float)[:95]
            ratio = np.ones(N_AGE)
            for lo, hi in MORT_GROUPS:
                sl = slice(lo, min(hi, 94) + 1)
                ratio[lo:hi + 1] = (r_reg[sl] * w[sl]).sum() / (r_nat[sl] * w[sl]).sum()
            mort[:, si, ri] = m * ratio
    pd.DataFrame({"death_rate": mort.ravel()}, index=idx).reset_index().to_csv(
        BASE / "demographics_mortality_2024.csv", index=False)

    # ---- migration: cohort residuals -> immigration profile, emigration rates, internal ---
    deaths_raw = eurostat("demo_r_magec", "demo_r_magec", refresh, geo=geo3, sex=SEXES, time=FLOW_YEARS)
    deaths_w = by_age(deaths_raw, ["time", "geo", "sex"])
    stb = statbel_regions(refresh)
    stb.to_csv(BASE / "demographics_statbel_regions.csv", index=False)
    sh = stb.groupby("region")[["immigration", "emigration"]].sum()
    imm_share = (sh["immigration"] / sh["immigration"].sum()).reindex(REGIONS).to_numpy()
    emi_share = (sh["emigration"] / sh["emigration"].sum()).reindex(REGIONS).to_numpy()

    resid, start, implied_imm = {}, {}, {}
    for y in FLOW_YEARS:
        d = cube(deaths_w, y)
        births = np.array([obs_val(y, r, "births") for r in REGIONS])
        b = np.stack([births * male_share, births * (1 - male_share)], axis=0)       # [sex, region]
        st = np.zeros((N_AGE, 2, 3))      # cohort size at the start of the year, by age at year end
        st[0] = b
        st[1:] = pop[y][:-1]
        st[100] += pop[y][100]
        dc = np.zeros((N_AGE, 2, 3))      # deaths of that cohort during the year
        dc[0] = infant_share * d[0]
        dc[1] = (1 - infant_share) * d[0] + 0.5 * d[1]
        dc[2:100] = 0.5 * d[1:99] + 0.5 * d[2:100]
        dc[100] = 0.5 * d[99] + d[100]
        resid[y] = pop[y + 1] - st + dc
        start[y] = st
        implied_imm[y] = resid[y].sum(axis=2) + emi[y]
    st_mean = sum(start.values()) / len(FLOW_YEARS)
    st_nat = st_mean.sum(axis=2)

    # emigration rate by age reached during the year and sex (per person of the cohort)
    emi_rate = sum(emi[y] / start[y].sum(axis=2) for y in FLOW_YEARS) / len(FLOW_YEARS)
    emi_rate = smooth(emi_rate, 1)
    # regional factor so that regional emigration totals follow the Statbel shares
    raw_r = (emi_rate[:, :, None] * st_mean).sum(axis=(0, 1))
    emi_factor = emi_share * raw_r.sum() / raw_r
    # immigration age-sex profile: implied (residual + emigration) below 80, observed above
    imp = sum(implied_imm.values()) / len(FLOW_YEARS)
    obs_imm = sum(imm.values()) / len(FLOW_YEARS)
    imp[80:] = obs_imm[80:]
    imp = np.clip(smooth(imp, 1), 0.0, None)
    imm_profile = imp / imp.sum()

    # inter-regional migration: regional residual minus regional international migration
    internal = np.zeros((N_AGE, 2, 3))
    for y in FLOW_YEARS:
        imm_y = np.clip(implied_imm[y], 0.0, None)
        intl = imm_y[:, :, None] * imm_share - emi_rate[:, :, None] * emi_factor * start[y]
        x = resid[y] - intl
        x -= x.sum(axis=2, keepdims=True) * start[y] / start[y].sum(axis=2, keepdims=True)
        internal += x / len(FLOW_YEARS)
    internal = smooth(internal, 2)
    internal[91:] = 0.0                      # residuals above 90 are dominated by death-timing noise
    internal -= internal.sum(axis=2, keepdims=True) * st_mean / st_mean.sum(axis=2, keepdims=True)
    rows = []
    od = np.zeros((N_AGE, 2, 3, 3))
    for ai in range(N_AGE):
        for si in range(2):
            n = internal[ai, si]
            pos = np.clip(n, 0.0, None)
            if pos.sum() <= 0:
                continue
            for o in range(3):
                if n[o] < 0:
                    od[ai, si, o] = -n[o] * pos / pos.sum() / st_mean[ai, si, o]
    for ai in range(N_AGE):
        for si, s in enumerate(SEXES):
            for o, ro in enumerate(REGIONS):
                for dd, rd in enumerate(REGIONS):
                    if o != dd:
                        rows.append((ai, s, ro, rd, od[ai, si, o, dd]))
    pd.DataFrame(rows, columns=["age", "sex", "origin", "destination", "rate"]).to_csv(
        BASE / "demographics_internal_migration.csv", index=False)

    mig = pd.DataFrame({
        "age": np.repeat(np.arange(N_AGE), 2), "sex": SEXES * N_AGE,
        "immigration_share": imm_profile.ravel(), "emigration_rate": emi_rate.ravel(),
        "observed_immigration": obs_imm.ravel(),
        "observed_emigration": (sum(emi.values()) / len(FLOW_YEARS)).ravel(),
    })
    mig.to_csv(BASE / "demographics_migration_age.csv", index=False)
    pd.DataFrame({"region": REGIONS, "immigration_share": imm_share, "emigration_share": emi_share,
                  "emigration_factor": emi_factor}).to_csv(
        BASE / "demographics_migration_region.csv", index=False)

    # ---- report ---------------------------------------------------------------------------
    print(f"population 1 Jan 2024: {pop[2024].sum():,.0f}")
    print("TFR 2024:", fert.sum().round(4).to_dict())
    for y in FLOW_YEARS:
        print(f"{y}: residual net migration {resid[y].sum():,.0f} "
              f"(Eurostat incl. adjustment {obs_val(y, 'BE', 'net_migration_incl_adjustment'):,.0f}; "
              f"imm-emi {imm[y].sum() - emi[y].sum():,.0f}); by region "
              f"{dict(zip(REGIONS, resid[y].sum(axis=(0, 1)).round()))}")
    print("regional immigration shares:", dict(zip(REGIONS, imm_share.round(3))),
          "emigration factors:", dict(zip(REGIONS, emi_factor.round(3))))
    print("internal net migration (model schedule, persons/yr):",
          dict(zip(REGIONS, internal.sum(axis=(0, 1)).round())),
          "| Statbel:", (stb.groupby("region")["internal_in"].sum()
                         - stb.groupby("region")["internal_out"].sum()).div(len(FLOW_YEARS)).round().to_dict())
    print("EUROPOP net migration path (for params/demographics.yaml):")
    print("  " + ", ".join(f"{y}: {int(v)}" for y, v in ep_bal["net_migration"].loc[2026:2070].items()))
    print("EUROPOP TFR 2024/2070:", ep_bal["tfr"].loc[2024], ep_bal["tfr"].loc[2070],
          "| sex ratio at birth (M/F, 2019+):", round(male_share / (1 - male_share), 4))


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
