"""Download the 2024 revenue accounts of Belgian general government and build the
calibration tables of the revenue module.

    uv run python -m belsim.data.fetch_revenue

Sources (all official, open, no key):
- Eurostat `gov_10a_taxag`  : taxes and social contributions by detailed ESA code and subsector.
- Eurostat `gov_10a_main`   : total revenue and non-tax items (sales, D4, D7, D9) by subsector,
                              incl. the intra-government flows needed to consolidate.
- NBB.Stat `DF_NFGOVTSC_DISS`: the Belgian *national tax list* (every tax and contribution by
                              ESA code and subsector). The Eurostat API only carries the ESA
                              aggregates of this list; the national detail is published by the
                              NBB (same figures, it is the list Belgium transmits to Eurostat).

Outputs:
- data/raw/revenue/*.json|csv           raw downloads
- data/base/revenue_taxag.csv           Eurostat ESA tax categories x subsector (EUR mn)
- data/base/revenue_main.csv            Eurostat main revenue aggregates x subsector (EUR mn)
- data/base/revenue_ntl.csv             national tax list items x subsector (EUR mn), with the
                                        model line each item is mapped to
- data/base/revenue_lines.csv           one row per model line: 2024 amount (EUR mn, S13
                                        consolidated), ESA code, entity shares
- data/base/revenue_outturn.csv         the same lines for the observed year after the base
                                        year (2025 outturn, same sources and mapping): the
                                        baseline reproduces it, see `belsim.fiscal.revenue`

The script fails loudly if the lines do not add up to total general-government revenue (TR).
"""
from __future__ import annotations

import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

YEAR = 2024
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "revenue"
BASE = ROOT / "data" / "base"
UA = {"User-Agent": "python-requests"}
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
NBB = "https://nsidisseminate-stat.nbb.be/rest/data/BE2,DF_NFGOVTSC_DISS,1.0/all"

SUBSECTORS = {"S1311": "federal", "S1314": "social_security", "S1312": "regions_communities", "S1313": "local"}
ENTITIES = ["federal", "social_security", "regions_communities", "local"]

# ---------------------------------------------------------------------------------------------
# National tax list item -> model line. A value may be a dict subsector -> line when the same
# item is split by receiving entity (personal income tax: federal / regional / municipal).
# ---------------------------------------------------------------------------------------------
PIT = {"S1311": "pit_federal", "S1312": "pit_regional", "S1313": "pit_local", "S1314": "pit_federal"}
NTL_MAP: dict[str, object] = {
    # --- social contributions (D61) ---
    "D6111C_X13": "ssc_employer_private", "D6111C_S13": "ssc_employer_public",
    "D612": "ssc_imputed", "D614": "ssc_imputed", "D61SC": "ssc_imputed",
    "D613CE": "ssc_employee", "D613VE": "ssc_employee",
    "D613CS": "ssc_self_employed",
    "D613CN": "ssc_replacement_income", "D613VN": "ssc_replacement_income",
    "D51A_6": "special_ssc", "D51A_7": "special_ssc",
    # --- personal income tax (D51A, D51E) ---
    "D51A_2": PIT, "D51A_3": PIT, "D51A_4": PIT, "D51A_5": PIT, "D51A_8": PIT, "D51A_9": PIT,
    "D51E_2": PIT, "D51E_3": PIT,
    "D51A_1": "withholding_tax",
    # --- corporate income tax (D51B) ---
    "D51B_1": "cit", "D51B_2": "cit", "D51B_3": "cit", "D51B_4": "cit", "D51B_5": "cit",
    # --- VAT ---
    "D211P": "vat",
    # --- excise and similar ---
    "D214A_1": "excise_fuel", "D2122C_1": "excise_fuel", "D214A_28": "excise_fuel", "D214A_29": "excise_fuel",
    "D214A_3": "excise_tobacco", "D2122C_3": "excise_tobacco",
    "D214A_4": "excise_alcohol", "D214A_7": "excise_alcohol", "D214A_8": "excise_alcohol",
    "D214A_12": "excise_alcohol", "D2122C_4": "excise_alcohol", "D2122C_6": "excise_alcohol",
    "D2122C_7": "excise_alcohol", "D2122C_8": "excise_alcohol", "D214A_5": "excise_alcohol",
    "D214A_6": "excise_alcohol", "D2122C_5": "excise_alcohol", "D2122C_12": "excise_alcohol",
    "D214A_13": "excise_energy", "D214A_14": "excise_energy", "D214A_15": "excise_energy",
    "D214A_16": "excise_energy",
    "D214A_31": "energy_green_levies", "D214A_32": "energy_green_levies", "D214A_33": "energy_green_levies",
    "D214A_34": "energy_green_levies", "D214A_35": "energy_green_levies", "D214A_17": "energy_green_levies",
    "D214A_9": "excise_beverages_packaging", "D2122C_9": "excise_beverages_packaging",
    "D2122C_11": "excise_beverages_packaging", "D214A_11": "excise_beverages_packaging",
    "D214A_23": "excise_beverages_packaging", "D29H_16": "excise_beverages_packaging",
    # --- transactions, insurance, gambling ---
    "D214C_1": "registration_duties", "D214C_2": "registration_duties",
    "D214C_4": "stock_exchange_tax",
    "D214G_1": "insurance_premium_tax", "D214G_2": "insurance_premium_tax", "D214G_3": "insurance_premium_tax",
    "D214G_4": "insurance_premium_tax", "D214G_5": "insurance_premium_tax", "D214G_6": "insurance_premium_tax",
    "D214F_1": "gambling_lottery", "D29B_2": "gambling_lottery", "D214J_1": "gambling_lottery",
    "D29H_3": "gambling_lottery",
    "D214L_2": "pharma_levies", "D214L_3": "pharma_levies",
    # --- property, wealth, vehicles, environment, sector levies ---
    "D29A_1": "property_tax", "D29A_2": "property_tax", "D29A_5": "property_tax",
    "D91A_1": "inheritance_gift_tax", "D91A_2": "inheritance_gift_tax",
    "D59G": "securities_account_tax",
    "D214D_2": "vehicle_taxes", "D29B_1": "vehicle_taxes", "D59D_1": "vehicle_taxes",
    "D29F_1": "environmental_taxes", "D29F_2": "environmental_taxes", "D29F_3": "environmental_taxes",
    "D29F_4": "environmental_taxes",
    "D29E_2": "bank_levies", "D29E_3": "bank_levies", "D29H_12": "bank_levies", "D29H_13": "bank_levies",
    "D29H_8": "nuclear_levy",
    "D29H_15": "other_taxes_production", "D29H_1": "other_taxes_production", "D29H_4": "other_taxes_production",
    "D29H_5": "other_taxes_production", "D214C_3": "other_taxes_production", "D214C_6": "other_taxes_production",
    "D214H_1": "other_taxes_production", "D214L_1": "other_taxes_production",
    "D59A_1": "other_taxes_income_wealth", "D59A_2": "other_taxes_income_wealth",
    "D59A_5": "other_taxes_income_wealth", "D59F_1": "other_taxes_income_wealth",
    "D91B_1": "other_taxes_income_wealth", "D91B_4": "other_taxes_income_wealth",
    "D91B_5": "other_taxes_income_wealth",
}
# Unmapped leaf items fall back on a line by ESA prefix (so a new item in a later vintage is
# not lost); the totals check below still guarantees nothing is double counted.
FALLBACK_PREFIX = [("D2", "other_taxes_production"), ("D5", "other_taxes_income_wealth"),
                   ("D91", "other_taxes_income_wealth"), ("D61", "ssc_imputed")]

LINE_ESA = {
    "ssc_employer_private": "D611", "ssc_employer_public": "D611", "ssc_imputed": "D612",
    "ssc_employee": "D613", "ssc_self_employed": "D613", "ssc_replacement_income": "D613",
    "special_ssc": "D51", "pit_federal": "D51", "pit_regional": "D51", "pit_local": "D51",
    "withholding_tax": "D51", "cit": "D51", "vat": "D211", "excise_fuel": "D214", "excise_tobacco": "D214",
    "excise_alcohol": "D214", "excise_energy": "D214", "energy_green_levies": "D214",
    "excise_beverages_packaging": "D214", "registration_duties": "D214", "stock_exchange_tax": "D214",
    "insurance_premium_tax": "D214", "gambling_lottery": "D214", "pharma_levies": "D214",
    "property_tax": "D29", "inheritance_gift_tax": "D91", "securities_account_tax": "D59",
    "capital_gains_tax": "D51", "vehicle_taxes": "D59", "environmental_taxes": "D29", "bank_levies": "D29",
    "nuclear_levy": "D29", "other_taxes_production": "D29", "other_taxes_income_wealth": "D59",
    "sales": "P11+P12+P131", "interest_received": "D41", "dividends_received": "D42",
    "rent_other_property_income": "D44+D45", "eu_transfers_current": "D74", "other_current_transfers": "D7+D39",
    "eu_transfers_capital": "D92", "other_capital_transfers": "D92+D99", "uncollectible_taxes": "D995",
}


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers={**UA, **kw.pop("headers", {})}, timeout=180, **kw)
    r.raise_for_status()
    return r


def _jsonstat(d: dict) -> dict[tuple, float]:
    """JSON-stat -> {(sector, na_item, time): value}."""
    ids, size = d["id"], d["size"]
    cats = [sorted(d["dimension"][k]["category"]["index"], key=d["dimension"][k]["category"]["index"].get)
            for k in ids]
    pos = {k: i for i, k in enumerate(ids)}
    out = {}
    for key, v in d["value"].items():
        i, idx = int(key), []
        for s in reversed(size):
            idx.append(i % s)
            i //= s
        idx = idx[::-1]
        out[(cats[pos["sector"]][idx[pos["sector"]]], cats[pos["na_item"]][idx[pos["na_item"]]],
             cats[pos["time"]][idx[pos["time"]]])] = float(v)
    return out


def fetch_eurostat(dataset: str) -> tuple[dict, dict]:
    url = f"{EUROSTAT}/{dataset}?format=JSON&lang=EN&geo=BE&unit=MIO_EUR&time={YEAR}&time={YEAR - 1}"
    r = _get(url)
    (RAW / f"{dataset}.json").write_bytes(r.content)
    d = r.json()
    return _jsonstat(d), d["dimension"]["na_item"]["category"]["label"]


def fetch_ntl() -> list[dict]:
    r = _get(f"{NBB}?startPeriod={YEAR - 1}&endPeriod={YEAR}",
             headers={"Accept": "application/vnd.sdmx.data+csv;labels=both"})
    (RAW / "nbb_nfgovtsc.csv").write_bytes(r.content)
    return list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))


def _col(row: dict, name: str) -> str:
    for k, v in row.items():
        if k.split(":")[0].strip() == name:
            return v
    raise KeyError(name)


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"wrote {shown} ({len(rows)} rows)")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    yr = str(YEAR)

    main_v, main_lab = fetch_eurostat("gov_10a_main")
    tax_v, tax_lab = fetch_eurostat("gov_10a_taxag")
    if ("S13", "TR", yr) not in main_v:
        raise SystemExit(f"gov_10a_main has no {YEAR} total revenue for Belgium yet; "
                         "calibration needs the base-year accounts")
    secs = ["S13"] + list(SUBSECTORS)

    def mv(item: str, sec: str = "S13") -> float:
        return main_v.get((sec, item, yr), 0.0)

    # --- raw tables kept for reference / other modules ---
    _write(BASE / "revenue_main.csv", ["na_item", "label", "year"] + secs,
           [[it, main_lab[it], YEAR] + [main_v.get((s, it, yr), "") for s in secs]
            for it in main_lab if any((s, it, yr) in main_v for s in secs)])
    _write(BASE / "revenue_taxag.csv", ["na_item", "label", "year"] + secs,
           [[it, tax_lab[it], YEAR] + [tax_v.get((s, it, yr), "") for s in secs]
            for it in tax_lab if any((s, it, yr) in tax_v for s in secs)])

    # --- national tax list ---
    ntl: dict[str, dict[str, float]] = defaultdict(dict)
    labels: dict[str, str] = {}
    for row in fetch_ntl():
        if _col(row, "TIME_PERIOD") != yr or not _col(row, "FREQ").startswith("A"):
            continue
        code, _, label = _col(row, "ITEM").partition(":")
        sec = _col(row, "SECTOR").split(":")[0].strip()
        try:
            val = float(_col(row, "OBS_VALUE"))
        except ValueError:
            continue
        if val != val:  # NaN
            continue
        ntl[code.strip()][sec] = val
        labels[code.strip()] = label.strip()

    amounts: dict[str, dict[str, float]] = defaultdict(lambda: dict.fromkeys(ENTITIES, 0.0))
    ntl_rows = []
    codes = set(ntl)
    aggregates = {"1NF", "1NF_N", "TTR", "PTC", "PTC_E", "D611_3", "D613E", "D613N", "D613S", "D6111C",
                  "D995", "D995A", "D995B", "D995C", "D995E", "D995F", "D995G", "D211", "D613", "D611"}
    for code in sorted(codes):
        if code in aggregates:
            continue
        target = NTL_MAP.get(code)
        if target is None:
            # an aggregate if any other item extends it ("D214A" -> "D214A_1", "D21" -> "D211")
            if any(c != code and c.startswith(code) for c in codes):
                continue
            vals = [abs(ntl[code].get(s, 0.0)) for s in SUBSECTORS]
            if max(vals, default=0.0) < 0.05:
                continue
            target = next(line for pre, line in FALLBACK_PREFIX if code.startswith(pre))
            print(f"  note: unmapped item {code} ({labels[code]}) -> {target}")
        for sec, ent in SUBSECTORS.items():
            v = ntl[code].get(sec, 0.0)
            line = target[sec] if isinstance(target, dict) else target
            amounts[line][ent] += v
        ntl_rows.append([code, labels[code], YEAR, target if isinstance(target, str) else "pit_*",
                         ntl[code].get("S1300", 0.0)] + [ntl[code].get(s, 0.0) for s in SUBSECTORS])
    _write(BASE / "revenue_ntl.csv", ["item", "label", "year", "line", "S13"] + list(SUBSECTORS), ntl_rows)

    tax_total = sum(sum(a.values()) for a in amounts.values())
    official_tax = mv("D2REC") + mv("D5REC") + mv("D91REC") + mv("D61REC")
    if abs(tax_total - official_tax) > 2.0:
        raise SystemExit(f"national tax list mapping sums to {tax_total:.1f}, "
                         f"Eurostat D2+D5+D91+D61 = {official_tax:.1f}")
    # cross-check against the Eurostat ESA categories of gov_10a_taxag
    for item, lines in {"D211": ["vat"], "D51B": ["cit"], "D91A": ["inheritance_gift_tax"]}.items():
        a, b = tax_v.get(("S13", item, yr), 0.0), sum(sum(amounts[ln].values()) for ln in lines)
        if abs(a - b) > 1.0:
            raise SystemExit(f"gov_10a_taxag {item} = {a}, mapped lines = {b}")

    amounts["capital_gains_tax"] = {"federal": 0.0, "social_security": 0.0, "regions_communities": 0.0, "local": 0.0}

    # --- non-tax revenue from gov_10a_main, consolidated with the intra-government flows ---
    def by_sub(fn) -> dict[str, float]:
        return {ent: fn(sec) for sec, ent in SUBSECTORS.items()}

    def paid_to(prefix: str, sec: str) -> float:
        """Flows `prefix` paid by the other subsectors to `sec` (to be consolidated out)."""
        return sum(mv(f"{prefix}_{sec}", s) for s in SUBSECTORS if s != sec)

    amounts["sales"] = by_sub(lambda s: mv("P11_P12_P131", s))
    amounts["interest_received"] = by_sub(lambda s: mv("D41REC", s) - paid_to("D4PAY", s))
    amounts["dividends_received"] = by_sub(lambda s: mv("D42REC", s) + mv("D43REC", s))
    amounts["rent_other_property_income"] = by_sub(lambda s: mv("D44REC", s) + mv("D45REC", s))
    amounts["eu_transfers_current"] = by_sub(lambda s: mv("D7REC_S212", s))
    amounts["other_current_transfers"] = by_sub(
        lambda s: mv("D7REC", s) - mv("D73REC", s) - mv("D7REC_S212", s) + mv("D39REC", s))
    amounts["eu_transfers_capital"] = by_sub(lambda s: mv("D9REC_S212", s))
    amounts["other_capital_transfers"] = by_sub(
        lambda s: mv("D92_D99REC", s) + mv("D995REC", s) - mv("D9REC_S212", s) - paid_to("D9PAY", s))
    amounts["uncollectible_taxes"] = by_sub(lambda s: -mv("D995REC", s))

    total = sum(sum(a.values()) for a in amounts.values())
    tr = mv("TR")
    print(f"sum of lines {total:,.1f}  vs  total revenue S13 (TR) {tr:,.1f}")
    if abs(total - tr) > 3.0:
        raise SystemExit("lines do not add up to total general-government revenue")
    # rounding residual (tenths of a million) goes to other current transfers, pro rata
    resid = tr - total
    oc = amounts["other_current_transfers"]
    s_oc = sum(oc.values())
    for ent in oc:
        oc[ent] += resid * oc[ent] / s_oc

    rows = []
    for line in LINE_ESA:
        a = amounts[line]
        tot = sum(a.values())
        pos = {e: max(v, 0.0) for e, v in a.items()} if tot >= 0 else {e: max(-v, 0.0) for e, v in a.items()}
        sp = sum(pos.values())
        shares = {e: (pos[e] / sp if sp > 0 else (1.0 if e == "federal" else 0.0)) for e in ENTITIES}
        rows.append([line, LINE_ESA[line], YEAR, round(tot, 3)] + [round(shares[e], 6) for e in ENTITIES])
    missing = set(amounts) - set(LINE_ESA)
    if missing:
        raise SystemExit(f"lines without ESA code: {missing}")
    _write(BASE / "revenue_lines.csv", ["line", "esa", "year", "amount"] + ENTITIES, rows)
    _write(BASE / "revenue_totals.csv", ["item", "year", "amount"],
           [["total_revenue", YEAR, tr], ["taxes_and_contributions", YEAR, official_tax],
            ["gov_intermediate_consumption", YEAR, mv("P2")], ["gov_gfcf", YEAR, mv("P51G")],
            ["gov_final_consumption", YEAR, mv("P3")], ["gov_wages", YEAR, mv("D11PAY")]])
    return 0


def outturn(year: int = YEAR + 1) -> int:
    """Observed outturn of a later year by model line (same sources, same mapping as the base
    year) -> data/base/revenue_outturn.csv. Raw downloads go to data/raw/revenue/outturn_<year>/.
    Does nothing if the accounts of that year are not published yet."""
    global YEAR, RAW, BASE
    import tempfile
    keep = (YEAR, RAW, BASE)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            YEAR, RAW, BASE = year, keep[1] / f"outturn_{year}", Path(tmp)
            try:
                main()
            except SystemExit as e:
                if e.code not in (0, None):
                    print(f"no {year} outturn: {e}")
                    return 0
            with (BASE / "revenue_lines.csv").open() as fh:
                rows = [[r["line"], year, r["amount"]] for r in csv.DictReader(fh)]
    finally:
        YEAR, RAW, BASE = keep
    _write(BASE / "revenue_outturn.csv", ["line", "year", "amount"], rows)
    return 0


if __name__ == "__main__":
    sys.exit(main() or outturn())
