"""Download and clean the macro-economic calibration tables (Eurostat, open API, no key).

    uv run python -m belsim.data.fetch_macro

Raw responses in data/raw/macro/, cleaned tables in data/base/macro_*.csv.

Tables written
    macro_gdp.csv          2024 GDP and main expenditure / income components, current prices
                           (nama_10_gdp), GFCF by asset (nama_10_an6), household financial balance
                           sheet (nasa_10_f_bs). Columns: key, value (EUR mn), source.
    macro_gdp_history.csv  Real GDP growth, nominal GDP, deflator growth 2015-latest (nama_10_gdp).
    macro_sector.csv       Gross value added, current prices, 16 model sectors, 2024 (nama_10_a64).
    macro_accounts.csv     Institutional-sector accounts 2024: households (S14_S15), non-financial
                           (S11) and financial (S12) corporations, total economy (nasa_10_nf_tr).
    macro_consumption.csv  Household consumption 2024 by the 10 model categories (nama_10_cp18,
                           COICOP 2018) with the COICOP codes mapped to each.
    macro_prices.csv       HICP inflation, 10-year government bond yield (Belgium), 3-month
                           money-market rate (euro area), 2019-latest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fetch_labour import BASE_DIR, NACE_TO_SECTOR, ROOT, eurostat, latest

YEAR = 2024

# COICOP 2018 (3-digit) -> model consumption categories. CP072 (operation of personal transport
# equipment) contains motor fuels (CP0722), which Eurostat does not publish separately at this
# level: the split is the parameter macro.fuel_share_of_cp072 applied by the model.
COICOP_MAP = {
    "CP011": "food", "CP012": "food",
    "CP021": "alcohol", "CP023": "tobacco", "CP024": "other_goods",
    "CP031": "other_goods", "CP032": "other_goods",
    "CP041": "housing", "CP042": "housing", "CP043": "renovation", "CP044": "other_services",
    "CP045": "energy_home",
    "CP051": "other_goods", "CP052": "other_goods", "CP053": "other_goods", "CP054": "other_goods",
    "CP055": "other_goods", "CP056": "other_services",
    "CP061": "other_goods", "CP062": "other_services", "CP063": "other_services",
    "CP064": "other_services",
    "CP071": "other_goods", "CP072": "transport_operation", "CP073": "other_services",
    "CP074": "other_services",
    "CP081": "other_goods", "CP082": "other_goods", "CP083": "other_services",
    "CP091": "other_goods", "CP092": "other_goods", "CP093": "other_goods", "CP094": "other_services",
    "CP095": "other_goods", "CP096": "other_services", "CP097": "other_goods", "CP098": "other_services",
    "CP101": "other_services", "CP102": "other_services", "CP103": "other_services",
    "CP104": "other_services", "CP105": "other_services",
    "CP111": "restaurants", "CP112": "restaurants",
    "CP121": "other_services", "CP122": "other_services",
    "CP131": "other_goods", "CP132": "other_goods", "CP133": "other_services", "CP139": "other_services",
}

ACCOUNT_ITEMS = ["B1G", "B2A3G", "B2A3N", "B2G", "B3G", "B4G", "B5G", "B6G", "B8G", "D1", "D11", "D4",
                 "D41", "D42", "D43", "D44", "D45", "D5", "D61", "D62", "D7", "D9", "D91", "P31", "P3",
                 "P51G", "P51C"]


def fetch_gdp(raw: Path) -> pd.DataFrame:
    rows: list[dict] = []
    g = eurostat("nama_10_gdp", raw, geo="BE", unit="CP_MEUR", time=YEAR)
    for _, r in g.iterrows():
        rows.append({"key": r.na_item, "value": r.value, "source": "Eurostat nama_10_gdp 2024, CP_MEUR"})
    a = eurostat("nama_10_an6", raw, geo="BE", unit="CP_MEUR", time=YEAR)
    for _, r in a.iterrows():
        rows.append({"key": f"GFCF_{r.asset10}", "value": r.value,
                     "source": "Eurostat nama_10_an6 2024, CP_MEUR"})
    f = eurostat("nasa_10_f_bs", raw, geo="BE", unit="MIO_EUR", sector="S14_S15", co_nco="NCO",
                 na_item=["F", "F2", "F3", "F4", "F5", "F51", "F52", "F6"], time=YEAR)
    for _, r in f.iterrows():
        rows.append({"key": f"HH_{r.finpos}_{r.na_item}", "value": r.value,
                     "source": "Eurostat nasa_10_f_bs 2024, S14_S15, non-consolidated, MIO_EUR"})
    out = pd.DataFrame(rows)
    out.to_csv(BASE_DIR / "macro_gdp.csv", index=False)

    h = eurostat("nama_10_gdp", raw, tag="history", geo="BE", na_item="B1GQ",
                 unit=["CLV_PCH_PRE", "CP_MEUR", "PD_PCH_PRE_EUR"], sinceTimePeriod=2015)
    hist = h.pivot_table(index="time", columns="unit", values="value").reset_index()
    hist = hist.rename(columns={"time": "year", "CLV_PCH_PRE": "real_growth_pct", "CP_MEUR": "gdp_nominal",
                                "PD_PCH_PRE_EUR": "deflator_growth_pct"})
    hist.to_csv(BASE_DIR / "macro_gdp_history.csv", index=False)
    return out


def fetch_sector(raw: Path) -> pd.DataFrame:
    nace = list(NACE_TO_SECTOR) + ["TOTAL"]
    v = eurostat("nama_10_a64", raw, tag="b1g", geo="BE", nace_r2=nace, unit="CP_MEUR", na_item="B1G",
                 sinceTimePeriod=YEAR - 2)
    v["time"] = v["time"].astype(int)
    v = latest(v[v["time"] <= YEAR], ["nace_r2"])
    total = float(v.loc[v.nace_r2 == "TOTAL", "value"].iloc[0])
    v = v[v.nace_r2 != "TOTAL"].copy()
    v["sector"] = v["nace_r2"].map(NACE_TO_SECTOR)
    out = v.groupby("sector")["value"].sum().rename("b1g_meur").reset_index()
    out["b1g_meur"] *= total / out["b1g_meur"].sum()
    out["data_year_min"] = int(v["time"].min())
    out.to_csv(BASE_DIR / "macro_sector.csv", index=False)
    return out


def fetch_accounts(raw: Path) -> pd.DataFrame:
    d = eurostat("nasa_10_nf_tr", raw, geo="BE", unit="CP_MEUR", sector=["S14_S15", "S11", "S12", "S1"],
                 na_item=ACCOUNT_ITEMS, time=YEAR)
    out = d[["sector", "na_item", "direct", "value"]].sort_values(["sector", "na_item", "direct"])
    out.to_csv(BASE_DIR / "macro_accounts.csv", index=False)
    return out


def fetch_consumption(raw: Path) -> pd.DataFrame:
    d = eurostat("nama_10_cp18", raw, geo="BE", unit="CP_MEUR", time=YEAR)
    d = d[d.coicop18.isin(COICOP_MAP)].copy()
    d["category"] = d["coicop18"].map(COICOP_MAP)
    out = d.groupby("category").agg(value_meur=("value", "sum"),
                                    coicop=("coicop18", lambda s: " ".join(sorted(s)))).reset_index()
    out.to_csv(BASE_DIR / "macro_consumption.csv", index=False)
    return out


def fetch_prices(raw: Path) -> pd.DataFrame:
    hicp = eurostat("prc_hicp_aind", raw, geo="BE", coicop="CP00", unit="RCH_A_AVG", sinceTimePeriod=2019)
    lt = eurostat("irt_lt_mcby_a", raw, geo="BE", sinceTimePeriod=2019)
    st = eurostat("irt_st_a", raw, geo="EA", int_rt="IRT_M3", sinceTimePeriod=2019)
    out = pd.DataFrame({
        "hicp_inflation": hicp.set_index("time")["value"] / 100.0,
        "interest_rate_10y": lt.set_index("time")["value"] / 100.0,
        "interest_rate_3m": st.set_index("time")["value"] / 100.0,
    })
    out.index.name = "year"
    out.reset_index().to_csv(BASE_DIR / "macro_prices.csv", index=False)
    return out


def main() -> None:
    raw = ROOT / "data" / "raw" / "macro"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print("gdp", fetch_gdp(raw).shape)
    print("sector", fetch_sector(raw).shape)
    print("accounts", fetch_accounts(raw).shape)
    print("consumption", fetch_consumption(raw).shape)
    print("prices", fetch_prices(raw).shape)


if __name__ == "__main__":
    main()
