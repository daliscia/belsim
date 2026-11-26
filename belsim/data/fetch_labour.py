"""Download and clean the labour-market calibration tables (Eurostat, open API, no key).

    uv run python -m belsim.data.fetch_labour

Raw JSON-stat responses go to data/raw/labour/, cleaned tables to data/base/labour_*.csv.
The model only ever reads data/base/.

Tables written
    labour_lfs_status.csv      LFS population / employed / unemployed / inactive, 5-year age x sex,
                               2014-2024 (lfsa_pganws)
    labour_lfs_profstatus.csv  LFS employed by professional status, 5-year age x sex, 2024 (lfsa_egaps)
    labour_students.csv        Not employed and in education, % of population, ages 15-34 x sex, 2024
                               (edat_lfse_18)
    labour_lfs_region.csv      Employment and activity rates by broad age x sex x NUTS1, 2024
                               (lfst_r_lfe2emprt, lfst_r_lfp2actrt)
    labour_sector.csv          National-accounts employment (persons, hours), D1, D11 for the 16 model
                               sectors, 2024 (nama_10_a64_e, nama_10_a64)
    labour_earnings_dist.csv   Structure of Earnings Survey 2022: D1, median, D9, mean annual
                               earnings (earn_ses_annual)
    labour_misc.csv            Scalars: part-time share, weekly hours, NA employment national vs
                               domestic concept, outgoing cross-border workers, S13 compensation,
                               retired share of 18-64 / 65+ (EU-SILC ilc_lvhl02), pension beneficiaries
                               by type (ESSPROS spr_pns_ben), unemployment by duration (lfsa_ugad).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "base"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
HEADERS = {"User-Agent": "python-requests"}

YEAR = 2024
AGE5 = ["Y15-19", "Y20-24", "Y25-29", "Y30-34", "Y35-39", "Y40-44", "Y45-49", "Y50-54",
        "Y55-59", "Y60-64", "Y65-69", "Y70-74", "Y_GE75"]
NUTS1 = ["BE", "BE1", "BE2", "BE3"]

# Eurostat NACE letters -> the 16 model sectors of belsim.core.dims.SECTORS
NACE_TO_SECTOR = {
    "A": "A", "B": "BDE", "D": "BDE", "E": "BDE", "C": "C", "F": "F", "G": "G", "H": "H",
    "I": "I", "J": "J", "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q",
    "R": "RSTU", "S": "RSTU", "T": "RSTU", "U": "RSTU",
}


def eurostat(dataset: str, raw_dir: Path, tag: str = "", **filters) -> pd.DataFrame:
    """Fetch a JSON-stat slice and return it as a long DataFrame (one column per dimension + value).

    Filter values may be scalars or lists. The raw response is stored in `raw_dir`.
    """
    params: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN")]
    for k, v in filters.items():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            params.append((k, str(x)))
    r = requests.get(API + dataset, params=params, headers=HEADERS, timeout=120)
    r.raise_for_status()
    js = r.json()
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{dataset}{('_' + tag) if tag else ''}.json").write_text(json.dumps(js))
    dims = js["id"]
    sizes = js["size"]
    cats = []
    for d in dims:
        idx = js["dimension"][d]["category"]["index"]
        cats.append([c for c, _ in sorted(idx.items(), key=lambda kv: kv[1])])
    values = js.get("value", {})
    rows = []
    for flat, combo in enumerate(itertools.product(*cats)):
        v = values.get(str(flat))
        if v is not None:
            rows.append(combo + (float(v),))
    df = pd.DataFrame(rows, columns=dims + ["value"])
    assert len(sizes) == len(dims)
    return df.drop(columns=[c for c in ("freq",) if c in df.columns])


def latest(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Keep, for each key combination, the most recent year available."""
    df = df.copy()
    df["time"] = df["time"].astype(int)
    idx = df.groupby(keys)["time"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _band(code: str) -> tuple[int, int]:
    """'Y25-49' -> (25, 49); 'Y_GE75' -> (75, 100)."""
    if code.startswith("Y_GE"):
        return int(code[4:]), 100
    lo, hi = code[1:].split("-")
    return int(lo), int(hi)


def fetch_status(raw: Path) -> pd.DataFrame:
    df = eurostat("lfsa_pganws", raw, geo="BE", citizen="TOTAL", sex=["M", "F"], age=AGE5,
                  wstatus=["POP", "EMP", "UNE", "INAC"], sinceTimePeriod=2014)
    out = df.pivot_table(index=["time", "sex", "age"], columns="wstatus", values="value").reset_index()
    out = out.rename(columns={"time": "year"})
    out["UNE"] = out["UNE"].fillna(0.0)  # suppressed small cells at high ages
    out["INAC"] = out["INAC"].fillna(out["POP"] - out["EMP"] - out["UNE"])
    out["age_lo"] = out["age"].map(lambda a: _band(a)[0])
    out = out.sort_values(["year", "sex", "age_lo"])
    out.to_csv(BASE_DIR / "labour_lfs_status.csv", index=False)
    return out


def fetch_profstatus(raw: Path) -> pd.DataFrame:
    df = eurostat("lfsa_egaps", raw, geo="BE", sex=["M", "F"], age=AGE5,
                  wstatus=["EMP", "SAL", "SELF", "CFAM"], time=YEAR)
    out = df.pivot_table(index=["sex", "age"], columns="wstatus", values="value").reset_index()
    for c in ("SAL", "SELF", "CFAM"):
        if c not in out:
            out[c] = 0.0
    out[["SELF", "CFAM"]] = out[["SELF", "CFAM"]].fillna(0.0)
    out["SAL"] = out["SAL"].fillna(out["EMP"] - out["SELF"] - out["CFAM"])
    out["age_lo"] = out["age"].map(lambda a: _band(a)[0])
    out = out.sort_values(["sex", "age_lo"])
    out.to_csv(BASE_DIR / "labour_lfs_profstatus.csv", index=False)
    return out


def fetch_students(raw: Path) -> pd.DataFrame:
    """Share of the population that is in education and not employed (LFS), ages 15-34."""
    df = eurostat("edat_lfse_18", raw, geo="BE", sex=["M", "F"], wstatus="NEMP", training="FE_NFE",
                  age=["Y15-17", "Y15-19", "Y20-24", "Y25-29", "Y30-34"], time=YEAR)
    rows = []
    for sex in ("M", "F"):
        v = df[df.sex == sex].set_index("age")["value"] / 100.0
        bands = {
            (15, 17): v["Y15-17"],
            (18, 19): (5 * v["Y15-19"] - 3 * v["Y15-17"]) / 2,   # equal weight per single age
            (20, 24): v["Y20-24"], (25, 29): v["Y25-29"], (30, 34): v["Y30-34"],
        }
        for (lo, hi), x in bands.items():
            rows.append({"sex": sex, "age_lo": lo, "age_hi": hi, "student_share": float(x)})
    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / "labour_students.csv", index=False)
    return out


def fetch_region(raw: Path) -> pd.DataFrame:
    ages = ["Y15-24", "Y25-34", "Y35-44", "Y45-54", "Y55-64", "Y_GE65", "Y20-64", "Y15-64"]
    e = eurostat("lfst_r_lfe2emprt", raw, geo=NUTS1, sex=["M", "F", "T"], age=ages, time=YEAR)
    a = eurostat("lfst_r_lfp2actrt", raw, geo=NUTS1, sex=["M", "F", "T"], age=ages, time=YEAR)
    e = e.rename(columns={"value": "emp_rate"})[["geo", "sex", "age", "emp_rate"]]
    a = a.rename(columns={"value": "act_rate"})[["geo", "sex", "age", "act_rate"]]
    out = e.merge(a, on=["geo", "sex", "age"], how="outer")
    out[["emp_rate", "act_rate"]] /= 100.0
    out.to_csv(BASE_DIR / "labour_lfs_region.csv", index=False)
    return out


def fetch_sector(raw: Path) -> pd.DataFrame:
    nace = list(NACE_TO_SECTOR) + ["TOTAL"]
    emp = eurostat("nama_10_a64_e", raw, geo="BE", nace_r2=nace, unit=["THS_PER", "THS_HW"],
                   na_item=["EMP_DC", "SAL_DC", "SELF_DC"], sinceTimePeriod=YEAR - 2)
    inc = eurostat("nama_10_a64", raw, tag="income", geo="BE", nace_r2=nace, unit="CP_MEUR",
                   na_item=["D1", "D11"], sinceTimePeriod=YEAR - 2)
    emp["item"] = emp["na_item"] + "_" + emp["unit"]
    inc["item"] = inc["na_item"]
    both = pd.concat([emp[["nace_r2", "item", "time", "value"]], inc[["nace_r2", "item", "time", "value"]]])
    both["time"] = both["time"].astype(int)
    both = both[both["time"] <= YEAR]
    lat = latest(both, ["nace_r2", "item"])
    wide = lat.pivot_table(index="nace_r2", columns="item", values="value").fillna(0.0)
    years = lat.groupby("item")["time"].min().to_dict()
    total = wide.loc["TOTAL"]
    wide = wide.drop(index="TOTAL")
    wide["sector"] = wide.index.map(NACE_TO_SECTOR)
    out = wide.groupby("sector").sum(numeric_only=True)
    # items published for an earlier year than YEAR are scaled to the YEAR total
    for item in out.columns:
        s = out[item].sum()
        if s > 0 and total[item] > 0:
            out[item] *= total[item] / s
    out = out.rename(columns={
        "EMP_DC_THS_PER": "emp_ths", "SAL_DC_THS_PER": "employees_ths", "SELF_DC_THS_PER": "self_ths",
        "EMP_DC_THS_HW": "emp_hours_ths", "SAL_DC_THS_HW": "employees_hours_ths",
        "SELF_DC_THS_HW": "self_hours_ths", "D1": "d1_meur", "D11": "d11_meur"})
    out["data_year_min"] = min(years.values())
    out.reset_index().to_csv(BASE_DIR / "labour_sector.csv", index=False)
    return out


def fetch_earnings(raw: Path) -> pd.DataFrame:
    df = eurostat("earn_ses_annual", raw, geo="BE", time=2022, isco08="TOTAL", age="TOTAL", sex="T",
                  nace_r2=["B-S_X_O", "B-N", "P-S"], worktime=["TOTAL", "FT"],
                  indic_se=["MEAN_E_EUR", "MED_E_EUR", "D1_E_EUR", "D9_E_EUR"])
    out = df.pivot_table(index=["nace_r2", "worktime"], columns="indic_se", values="value").reset_index()
    out["year"] = 2022
    out.to_csv(BASE_DIR / "labour_earnings_dist.csv", index=False)
    return out


def fetch_misc(raw: Path) -> pd.DataFrame:
    rows: list[dict] = []

    def add(key: str, value: float, source: str) -> None:
        rows.append({"key": key, "value": value, "source": source})

    pt = eurostat("lfsa_eppga", raw, geo="BE", sex=["T", "M", "F"], age="Y20-64", time=YEAR)
    for _, r in pt.iterrows():
        add(f"part_time_share_{r.sex}", r.value / 100.0, "Eurostat lfsa_eppga 2024, age 20-64")
    hrs = eurostat("lfsa_ewhun2", raw, geo="BE", nace_r2="TOTAL", sex="T", age="Y20-64",
                   wstatus=["EMP", "SAL", "SELF"], worktime=["TOTAL", "FT", "PT"], time=YEAR)
    for _, r in hrs.iterrows():
        add(f"weekly_hours_{r.wstatus}_{r.worktime}", r.value, "Eurostat lfsa_ewhun2 2024, age 20-64")
    pe = eurostat("nama_10_pe", raw, geo="BE", unit="THS_PER", time=YEAR)
    for _, r in pe.iterrows():
        add(f"na_{r.na_item}_ths", r.value, "Eurostat nama_10_pe 2024")
    try:
        cb = eurostat("lfst_r_lfe2ecomm", raw, geo="BE", sex="T", age="Y15-64", time=YEAR)
        for _, r in cb.iterrows():
            add(f"commuting_{r.c_work}_ths", r.value, "Eurostat lfst_r_lfe2ecomm 2024, age 15-64")
    except Exception as exc:  # optional table
        print("  lfst_r_lfe2ecomm not available:", exc)
    gov = eurostat("gov_10a_main", raw, tag="d1", geo="BE", unit="MIO_EUR", sector="S13",
                   na_item=["D1PAY", "P3"], time=YEAR)
    for _, r in gov.iterrows():
        add(f"s13_{r.na_item}_meur", r.value, "Eurostat gov_10a_main 2024, S13")
    silc = eurostat("ilc_lvhl02", raw, geo="BE", rskpovth="TOTAL", sex=["M", "F"],
                    age=["Y18-64", "Y_GE65"], wstatus=["RET", "INAC_OTH", "EMP", "UNE"], time=YEAR)
    for _, r in silc.iterrows():
        add(f"silc_{r.wstatus}_{r.age}_{r.sex}", r.value / 100.0,
            "Eurostat ilc_lvhl02 2024, most frequent activity status, share of population")
    pens = eurostat("spr_pns_ben", raw, geo="BE", spdepm="TOTAL", sex=["T", "M", "F"],
                    spdepb=["OLD", "AOLD", "DIS", "ERLM", "ERRC", "SRV", "TOTAL"], sinceTimePeriod=YEAR - 2)
    pens["time"] = pens["time"].astype(int)
    pens = latest(pens[pens["time"] <= YEAR], ["spdepb", "sex"])
    for _, r in pens.iterrows():
        add(f"pension_beneficiaries_{r.spdepb}_{r.sex}", r.value, f"Eurostat spr_pns_ben {r.time} (ESSPROS)")
    dur = eurostat("lfsa_ugad", raw, geo="BE", sex="T", age="Y15-74", time=YEAR)
    for _, r in dur.iterrows():
        add(f"unemployed_duration_{r.duration}_ths", r.value, "Eurostat lfsa_ugad 2024, age 15-74")
    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / "labour_misc.csv", index=False)
    return out


def main() -> None:
    raw = ROOT / "data" / "raw" / "labour"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    status = fetch_status(raw)
    print("status", status.shape)
    print("profstatus", fetch_profstatus(raw).shape)
    print("students", fetch_students(raw).shape)
    print("region", fetch_region(raw).shape)
    print("sector", fetch_sector(raw).shape)
    print("earnings", fetch_earnings(raw).shape)
    print("misc", fetch_misc(raw).shape)


if __name__ == "__main__":
    main()
