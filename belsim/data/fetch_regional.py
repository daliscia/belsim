"""Download and clean the regional (NUTS1) calibration tables (Eurostat, open API, no key).

    uv run python -m belsim.data.fetch_regional

Raw responses in data/raw/regional/, cleaned table in data/base/regional_accounts.csv
(long format: indicator, region, year, value, unit, source). Regions: FLA (BE2), WAL (BE3),
BRU (BE1) and BE (national total of the same table, for shares).

Indicators
    gdp                        GDP at current market prices, by place of work (nama_10r_2gdp), EUR mn
    gdp_real_growth            Real growth of GDP, % (nama_10r_2gvagr, B1GQ, PCH_PRE)
    gva_<A10 code>             Gross value added by NACE A10 group (nama_10r_3gva), EUR mn (if served)
    employment_workplace       Persons employed, domestic concept = place of work (nama_10r_3empers), thousand
    employees_workplace        Employees, place of work (nama_10r_3empers), thousand
    compensation_workplace     Compensation of employees D1, place of work (nama_10r_2coe), EUR mn
    population_avg             Average population used in the regional accounts (nama_10r_3popgdp), thousand
    hh_primary_income          Households: balance of primary income, net (B5N), place of residence, EUR mn
    hh_disposable_income       Households: disposable income, net (B6N), place of residence, EUR mn
    hh_compensation_received   Households: compensation of employees received (D1), residence, EUR mn
    hh_property_income_received  Households: property income received (D4), EUR mn
    hh_social_benefits_received  Households: social benefits other than in kind received (D62), EUR mn
    hh_income_taxes_paid       Households: current taxes on income and wealth paid (D5), EUR mn
    hh_social_contributions_paid  Households: net social contributions paid (D61), EUR mn
    lfs_employed_15_64         LFS employed residents 15-64 (lfst_r_lfe2emp), thousand
    lfs_employment_rate_20_64  LFS employment rate 20-64 (lfst_r_lfe2emprt), %
    lfs_unemployment_rate      LFS unemployment rate 15-74 (lfst_r_lfu3rt), %
    lfs_work_other_region      LFS employed residents working in another NUTS2 region of the country
                               (lfst_r_lfe2ecomm OUTR). NB: NUTS2 concept - only for Brussels (one
                               NUTS2 region) is this the commuting out of the NUTS1 region.
    lfs_work_abroad            LFS employed residents working in a foreign country (lfst_r_lfe2ecomm FOR), thousand

Not available from the Eurostat API (see params/regional.yaml for what is read from documents or
assumed): the commuting matrix between the three regions, the accounts of the individual
federated entities, the Special Financing Act transfers by entity.
"""
from __future__ import annotations

import pandas as pd

from .fetch_labour import BASE_DIR, ROOT, eurostat

RAW = ROOT / "data" / "raw" / "regional"
GEO = {"BE": "BE", "BE1": "BRU", "BE2": "FLA", "BE3": "WAL"}
SINCE = 2021

HH_ITEMS = {
    ("BAL", "B5N"): "hh_primary_income", ("BAL", "B6N"): "hh_disposable_income",
    ("RECV", "D1"): "hh_compensation_received", ("RECV", "D4"): "hh_property_income_received",
    ("RECV", "D62"): "hh_social_benefits_received", ("PAID", "D5"): "hh_income_taxes_paid",
    ("PAID", "D61"): "hh_social_contributions_paid",
}


def _get(dataset: str, **filters) -> pd.DataFrame:
    d = eurostat(dataset, RAW, geo=list(GEO), sinceTimePeriod=SINCE, **filters)
    d["region"] = d["geo"].map(GEO)
    d["year"] = d["time"].astype(int)
    return d


def _rows(d: pd.DataFrame, indicator: str, unit: str, source: str) -> pd.DataFrame:
    return pd.DataFrame({"indicator": indicator, "region": d["region"], "year": d["year"],
                         "value": d["value"], "unit": unit, "source": source})


def main() -> None:
    out: list[pd.DataFrame] = []

    d = _get("nama_10r_2gdp", unit="MIO_EUR")
    out.append(_rows(d, "gdp", "EUR mn", "Eurostat nama_10r_2gdp, MIO_EUR"))

    d = _get("nama_10r_2gvagr", na_item="B1GQ", unit="PCH_PRE")
    out.append(_rows(d, "gdp_real_growth", "%", "Eurostat nama_10r_2gvagr, B1GQ, PCH_PRE"))

    try:
        d = _get("nama_10r_3gva", unit="CP_MEUR")
        for code, g in d.groupby("nace_r2"):
            out.append(_rows(g, f"gva_{code}", "EUR mn", "Eurostat nama_10r_3gva, CP_MEUR"))
    except Exception as e:  # table not served for these filters: GDP alone is enough for the model
        print(f"nama_10r_3gva not fetched ({e!r:.120})")

    d = _get("nama_10r_3empers", wstatus=["EMP", "SAL"], nace_r2="TOTAL")
    out.append(_rows(d[d.wstatus == "EMP"], "employment_workplace", "thousand persons",
                     "Eurostat nama_10r_3empers, EMP, TOTAL"))
    out.append(_rows(d[d.wstatus == "SAL"], "employees_workplace", "thousand persons",
                     "Eurostat nama_10r_3empers, SAL, TOTAL"))

    d = _get("nama_10r_2coe", currency="MIO_EUR", nace_r2="TOTAL")
    out.append(_rows(d, "compensation_workplace", "EUR mn", "Eurostat nama_10r_2coe, MIO_EUR, TOTAL"))

    d = _get("nama_10r_3popgdp")
    out.append(_rows(d, "population_avg", "thousand persons", "Eurostat nama_10r_3popgdp"))

    d = _get("nama_10r_2hhinc", unit="MIO_EUR", na_item=sorted({k[1] for k in HH_ITEMS}))
    for (direct, item), name in HH_ITEMS.items():
        g = d[(d.direct == direct) & (d.na_item == item)]
        out.append(_rows(g, name, "EUR mn", f"Eurostat nama_10r_2hhinc, {item} {direct}, MIO_EUR"))

    d = _get("lfst_r_lfe2emp", sex="T", age="Y15-64")
    out.append(_rows(d, "lfs_employed_15_64", "thousand persons", "Eurostat lfst_r_lfe2emp, T, Y15-64"))
    d = _get("lfst_r_lfe2emprt", sex="T", age="Y20-64")
    out.append(_rows(d, "lfs_employment_rate_20_64", "%", "Eurostat lfst_r_lfe2emprt, T, Y20-64"))
    d = _get("lfst_r_lfu3rt", sex="T", age="Y15-74", isced11="TOTAL")
    out.append(_rows(d, "lfs_unemployment_rate", "%", "Eurostat lfst_r_lfu3rt, T, Y15-74"))
    d = _get("lfst_r_lfe2ecomm", sex="T", age="Y15-64")
    out.append(_rows(d[d.c_work == "OUTR"], "lfs_work_other_region", "thousand persons",
                     "Eurostat lfst_r_lfe2ecomm, OUTR (other NUTS2 region), T, Y15-64"))
    out.append(_rows(d[d.c_work == "FOR"], "lfs_work_abroad", "thousand persons",
                     "Eurostat lfst_r_lfe2ecomm, FOR, T, Y15-64"))

    table = pd.concat(out, ignore_index=True).sort_values(["indicator", "year", "region"])
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BASE_DIR / "regional_accounts.csv", index=False)
    print(f"regional_accounts.csv: {len(table)} rows, indicators: {sorted(table.indicator.unique())}")


if __name__ == "__main__":
    main()
