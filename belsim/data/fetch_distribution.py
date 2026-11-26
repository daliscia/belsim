"""Download and clean the tables behind the distributional analysis (Eurostat, open API, no key).

    uv run python -m belsim.data.fetch_distribution

Raw JSON-stat responses go to data/raw/distribution/, cleaned tables to
data/base/distribution_*.csv. The model only ever reads data/base/.

Tables written
    distribution_households.csv     Private households (thousand) by composition, number of
                                    children and working status of the adults, Belgium, 2024
                                    (EU-LFS household statistics: lfst_hhnhtych, lfst_hhnhwhtc)
    distribution_hbs_structure.csv  Structure of household consumption by income quintile, mapped
                                    to the model's consumption categories (Household Budget Survey,
                                    hbs_str_t223, latest wave = 2020); shares of total consumption,
                                    imputed rents shown separately
    distribution_propensity.csv     Aggregate propensity to consume (consumption as % of disposable
                                    income) by income quintile, all available waves (Eurostat
                                    experimental statistics on income, consumption and wealth,
                                    icw_sr_10)
    distribution_reference.csv      Observed EU-SILC indicators used only as a yardstick for the
                                    stylised-household approximation: Gini (ilc_di12), S80/S20
                                    (ilc_di11), at-risk-of-poverty rate and threshold (ilc_li02,
                                    ilc_li01), decile cut-offs and shares of equivalised disposable
                                    income (ilc_di01); cross-tabulations of the at-risk-of-poverty
                                    rate by household type (ilc_li03), most frequent activity
                                    status (ilc_li04), age (ilc_li02), work intensity of the
                                    household (ilc_li06) and tenure (ilc_li08); in-work poverty
                                    (ilc_iw01, ilc_iw02); median equivalised income by household
                                    type (ilc_di04) and activity status (ilc_di05); distribution
                                    of the population by household type (ilc_lvps02) and people
                                    in households with very low work intensity (ilc_lvhl11n; by
                                    household type: ilc_lvhl13n). The last two also give the
                                    weights of the household set: persons by household type and
                                    the households without work among them.
                                    Row names: `<measure>.<breakdown>.<Eurostat code>`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fetch_labour import eurostat, latest

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "base"
RAW_DIR = ROOT / "data" / "raw" / "distribution"
YEAR = 2024
# EU-SILC asks for the income of the calendar year before the survey: the survey of YEAR + 1
# measures the incomes of YEAR, the base year of the model
SILC_YEAR = YEAR + 1

# COICOP (HBS classification, 3/4-digit) -> model consumption categories (belsim.core.dims.CONS_CATS).
# Same mapping as belsim.data.fetch_macro (CP072 "operation of personal transport equipment" is
# split into motor fuels and the rest at run time with macro.fuel_share_of_cp072); imputed rents
# (CP042) are kept apart because they are not a cash expense. Tobacco is CP022 in the HBS
# classification (COICOP 1999).
HBS_MAP = {
    "CP011": "food", "CP012": "food",
    "CP021": "alcohol", "CP022": "tobacco", "CP023": "other_goods",
    "CP031": "other_goods", "CP032": "other_goods",
    "CP041": "housing", "CP042": "imputed_rent", "CP043": "renovation", "CP045": "energy_home",
    "CP051": "other_goods", "CP052": "other_goods", "CP053": "other_goods", "CP054": "other_goods",
    "CP055": "other_goods", "CP061": "other_goods", "CP071": "other_goods",
    "CP072": "transport_operation",
    "CP082": "other_goods", "CP091": "other_goods", "CP092": "other_goods", "CP093": "other_goods",
    "CP095": "other_goods", "CP111": "restaurants", "CP112": "restaurants", "CP123": "other_goods",
}
QUINTILES = ["QU1", "QU2", "QU3", "QU4", "QU5"]


def households() -> pd.DataFrame:
    comp = eurostat("lfst_hhnhtych", RAW_DIR, geo="BE", time=YEAR, agechild="TOTAL")
    comp = comp.assign(hhwkstat="TOTAL", dataset="lfst_hhnhtych")
    work = eurostat("lfst_hhnhwhtc", RAW_DIR, geo="BE", time=YEAR).assign(dataset="lfst_hhnhwhtc")
    work = work[work["hhwkstat"] != "TOTAL"]
    df = pd.concat([comp, work], ignore_index=True)
    df = df[df["unit"] == "THS_HH"]
    out = df[["hhwkstat", "n_child", "phhcomp", "value", "time", "dataset"]].rename(
        columns={"value": "households_ths", "time": "year"})
    return out.sort_values(["dataset", "hhwkstat", "phhcomp", "n_child"]).reset_index(drop=True)


def hbs_structure() -> pd.DataFrame:
    df = eurostat("hbs_str_t223", RAW_DIR, geo="BE")
    df = df[df["quant_inc"].isin(QUINTILES)]
    df = df[df["time"].astype(int) == df["time"].astype(int).max()]
    year = int(df["time"].iloc[0])
    rows = []
    for q in QUINTILES:
        d = df[df["quant_inc"] == q].set_index("coicop")["value"]
        total = float(d[[c for c in d.index if len(c) == 4]].sum())      # 2-digit divisions
        shares: dict[str, float] = {}
        for code, cat in HBS_MAP.items():
            shares[cat] = shares.get(cat, 0.0) + float(d.get(code, 0.0))
        shares["other_services"] = total - sum(shares.values())           # everything else
        for cat, v in sorted(shares.items()):
            rows.append({"quintile": int(q[2]), "category": cat, "share": round(v / total, 5), "year": year})
    return pd.DataFrame(rows)


def propensity() -> pd.DataFrame:
    df = eurostat("icw_sr_10", RAW_DIR, geo="BE")
    df = df[df["quant_inc"].isin(QUINTILES + ["TOTAL"]) & (df["unit"] == "PC_DI")]
    out = pd.DataFrame({"year": df["time"].astype(int),
                        "quintile": [0 if q == "TOTAL" else int(q[2]) for q in df["quant_inc"]],
                        "propensity": (df["value"] / 100.0).round(4)})
    return out.sort_values(["year", "quintile"]).reset_index(drop=True)


def reference() -> pd.DataFrame:
    rows = []

    def add(name, dataset, **filters):
        try:
            d = latest(eurostat(dataset, RAW_DIR, tag=name, geo="BE", time=SILC_YEAR, **filters), ["geo"])
            rows.append({"indicator": name, "value": float(d["value"].iloc[0]), "year": int(d["time"].iloc[0]),
                         "dataset": dataset, "income_year": int(d["time"].iloc[0]) - 1})
        except Exception as e:                                             # table renamed / no data
            print(f"  {name}: not available ({type(e).__name__}: {e})")

    add("gini", "ilc_di12", age="TOTAL")
    add("s80s20", "ilc_di11", age="TOTAL", sex="T")
    add("poverty_rate", "ilc_li02", statinfo="MED_EI", rskpovth="B_60", age="TOTAL", sex="T", unit="PC")
    add("poverty_threshold_single_eur", "ilc_li01", statinfo="MED_EI", rskpovth="B_60", hhcomp="A1", unit="EUR")
    add("median_equivalised_income_eur", "ilc_di03", statinfo="MED_EI", age="TOTAL", sex="T", unit="EUR")
    di = eurostat("ilc_di01", RAW_DIR, geo="BE", time=SILC_YEAR)
    for _, r in di.iterrows():
        q = r["quant_inc"]
        if not q.startswith("D"):
            continue
        if r["statinfo"] == "TC" and r["unit"] == "EUR":
            rows.append({"indicator": f"decile_cutoff_{q[1:]}", "value": r["value"], "year": SILC_YEAR, "dataset": "ilc_di01",
                         "income_year": YEAR})
        if r["statinfo"] == "SHARE" and r["unit"] == "EUR":
            rows.append({"indicator": f"decile_share_{q[1:]}", "value": r["value"], "year": SILC_YEAR, "dataset": "ilc_di01",
                         "income_year": YEAR})
    rows += cross_tabulations()
    return pd.DataFrame(rows)


# EU-SILC cross-tabulations: (row prefix, dataset, fixed filters, dimension that varies)
CROSS_TABS = [
    ("poverty_rate.household", "ilc_li03", dict(statinfo="MED_EI", rskpovth="B_60", unit="PC"), "hhcomp"),
    ("poverty_persons_ths.household", "ilc_li03", dict(statinfo="MED_EI", rskpovth="B_60", unit="THS_PER"), "hhcomp"),
    ("poverty_rate.activity", "ilc_li04", dict(statinfo="MED_EI", rskpovth="B_60", unit="PC", sex="T", age="Y_GE18"), "wstatus"),
    ("poverty_persons_ths.activity", "ilc_li04", dict(statinfo="MED_EI", rskpovth="B_60", unit="THS_PER", sex="T", age="Y_GE18"), "wstatus"),
    ("poverty_rate.age", "ilc_li02", dict(statinfo="MED_EI", rskpovth="B_60", unit="PC", sex="T",
                                          age=["Y_LT18", "Y18-64", "Y_GE65"]), "age"),
    ("poverty_rate_line40.age", "ilc_li02", dict(statinfo="MED_EI", rskpovth="B_40", unit="PC", sex="T",
                                                 age=["TOTAL", "Y_LT18", "Y18-64", "Y_GE65"]), "age"),
    ("poverty_rate_line50.age", "ilc_li02", dict(statinfo="MED_EI", rskpovth="B_50", unit="PC", sex="T",
                                                 age=["TOTAL", "Y_LT18", "Y18-64", "Y_GE65"]), "age"),
    ("poverty_rate_line70.age", "ilc_li02", dict(statinfo="MED_EI", rskpovth="B_70", unit="PC", sex="T",
                                                 age=["TOTAL", "Y_LT18", "Y18-64", "Y_GE65"]), "age"),
    ("poverty_rate.work_intensity", "ilc_li06", dict(statinfo="MED_EI", rskpovth="B_60", unit="PC", sex="T", age="Y_LT65",
                                                     hhcomp="TOTAL"), "workint"),
    ("poverty_rate.work_intensity_children", "ilc_li06", dict(statinfo="MED_EI", rskpovth="B_60", unit="PC", sex="T",
                                                              age="Y_LT65", hhcomp="DCH"), "workint"),
    ("poverty_rate.tenure", "ilc_li08", dict(statinfo="MED_EI", rskpovth="B_60", unit="PC", sex="T", age="TOTAL"), "tenure"),
    ("inwork_poverty_rate.activity", "ilc_iw01", dict(sex="T", age="Y_GE18", unit="PC"), "wstatus"),
    ("inwork_poverty_rate.household", "ilc_iw02", dict(unit="PC"), "hhcomp"),
    ("median_equivalised_income_eur.household", "ilc_di04", dict(statinfo="MED_EI", unit="EUR"), "hhcomp"),
    ("median_equivalised_income_eur.activity", "ilc_di05", dict(statinfo="MED_EI", unit="EUR", sex="T", age="Y_GE18"), "wstatus"),
    ("population_share.household", "ilc_lvps02", dict(rskpovth="TOTAL", unit="PC"), "hhcomp"),
    ("low_work_intensity_ths.age", "ilc_lvhl11n", dict(unit="THS", sex="T", age=["Y_LT18", "Y18-64"]), "age"),
    # people under 65 living in households with very low work intensity, by household type: the
    # households without work of the set (belsim.distribution.households.base_weights)
    ("low_work_intensity_ths.household", "ilc_lvhl13n", dict(unit="THS_PER", quant_inc="TOTAL"), "hhcomp"),
    ("low_work_intensity_rate.household", "ilc_lvhl13n", dict(unit="PC", quant_inc="TOTAL"), "hhcomp"),
]


def cross_tabulations() -> list[dict]:
    """Published EU-SILC cross-tabulations for Belgium (survey of SILC_YEAR, incomes of YEAR)."""
    rows = []
    for prefix, dataset, filters, dim in CROSS_TABS:
        try:
            d = eurostat(dataset, RAW_DIR, tag=prefix.replace(".", "_"), geo="BE", time=SILC_YEAR, **filters)
        except Exception as e:                                             # table renamed / no data
            print(f"  {prefix}: not available ({type(e).__name__}: {e})")
            continue
        for _, r in d.sort_values(dim).iterrows():
            rows.append({"indicator": f"{prefix}.{r[dim]}", "value": float(r["value"]), "year": int(r["time"]),
                         "dataset": dataset, "income_year": int(r["time"]) - 1})
    return rows


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("households", households), ("hbs_structure", hbs_structure), ("propensity", propensity),
                     ("reference", reference)):
        df = fn()
        path = BASE_DIR / f"distribution_{name}.csv"
        df.to_csv(path, index=False)
        print(f"{path.relative_to(ROOT)}: {len(df)} rows")


if __name__ == "__main__":
    main()
