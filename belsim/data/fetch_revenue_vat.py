"""Build the VAT rate structure behind `vat.mix` (params/revenue.yaml): household consumption
by COICOP class with the legal VAT rate of each class.

    uv run python -m belsim.data.fetch_revenue_vat

Sources:
- Eurostat `nama_10_cp18` (open API): final consumption expenditure of households by COICOP 2018
  class, Belgium, 2024, current prices (domestic concept, VAT included).
- The legal rates: Royal Decree no. 20 (rates of 2024: 6 % food, water, medicines, books,
  passenger transport, household gas and electricity, hotels, takeaway meals, cultural and sport
  admissions; 12 % restaurant meals; 21 % otherwise; exemptions of art. 44 VAT Code: rents,
  medical care, education, insurance, finance, social services). `RATES` gives, for every class,
  the share of its value before VAT at 21 %, 12 %, 6 % and at zero / exempt. A class that mixes
  rates is split by judgement (the split is the assumption, the rate schedule is the law); the
  splits of restaurants, hotels and leisure are set on official costings (see `CHECKS`).
- Federal inventory of tax expenditures (annex to the 2026 budget, Chamber DOC 56 1279/002,
  table "BTW - TVA", year 2023): revenue forgone by the 6 % and 12 % rates, from which the
  taxable amounts declared at each reduced rate follow (`INVENTORY_2023`). They are the yardstick
  for the reduced-rate share of the non-household bases (housing investment, government
  purchases, inputs of exempt sectors), which Eurostat does not give.

Outputs:
- data/raw/revenue/nama_10_cp18.json
- data/base/revenue_vat_structure.csv   one row per COICOP class: model category, value, shares
                                        by rate; then one row per model category (level
                                        "category") with the shares to copy into `vat.mix`.
The script prints the comparison with the inventory and with the European Commission's rate gap.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import requests

YEAR = 2024
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "revenue"
OUT = ROOT / "data" / "base" / "revenue_vat_structure.csv"
UA = {"User-Agent": "python-requests"}
URL = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nama_10_cp18"
       f"?format=JSON&geo=BE&unit=CP_MEUR&time={YEAR}")
RATE = (0.21, 0.12, 0.06, 0.0)
FUEL_SHARE_CP072 = 0.40          # macro.fuel_share_of_cp072 (params/macro.yaml)

# class -> (model category, [share at 21 %, 12 %, 6 %, zero or exempt], note)
RATES: dict[str, tuple[str, list[float], str]] = {
    "CP011": ("food", [0.01, 0.01, 0.98, 0.0], "food 6 %; margarine 12 %; lobster, caviar 21 %"),
    "CP012": ("food", [0.0, 0.0, 1.0, 0.0], "water, soft drinks, juices, coffee, tea 6 %"),
    "CP021": ("alcohol", [1.0, 0.0, 0.0, 0.0], ""),
    "CP023": ("tobacco", [1.0, 0.0, 0.0, 0.0], ""),
    "CP024": ("other_goods", [0.0, 0.0, 0.0, 1.0], "illegal: no VAT"),
    "CP031": ("other_goods", [0.98, 0.0, 0.02, 0.0], "repair 6 %"),
    "CP032": ("other_goods", [0.97, 0.0, 0.03, 0.0], "repair 6 %"),
    "CP041": ("housing", [0.0, 0.0, 0.0, 1.0], "rents exempt"),
    "CP042": ("housing", [0.0, 0.0, 0.0, 1.0], "imputed rents"),
    "CP043": ("renovation", [0.40, 0.0, 0.60, 0.0], "work on dwellings older than 10 years 6 %, materials 21 %"),
    "CP044": ("other_services", [0.15, 0.0, 0.55, 0.30], "water 6 %; refuse collection and co-ownership charges outside VAT"),
    "CP045": ("energy_home", [0.16, 0.01, 0.83, 0.0], "electricity, natural gas, heat, firewood 6 %; heating oil 21 %; coal 12 %"),
    "CP051": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP052": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP053": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP054": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP055": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP056": ("other_services", [0.50, 0.0, 0.0, 0.50], "service vouchers and domestic staff outside VAT"),
    "CP061": ("other_goods", [0.25, 0.0, 0.75, 0.0], "medicines and most medical devices 6 %"),
    "CP062": ("other_services", [0.03, 0.0, 0.0, 0.97], "medical care exempt"),
    "CP063": ("other_services", [0.03, 0.0, 0.0, 0.97], "hospital care exempt"),
    "CP064": ("other_services", [0.03, 0.0, 0.0, 0.97], ""),
    "CP071": ("other_goods", [0.85, 0.0, 0.0, 0.15], "second-hand sales between households and margin scheme"),
    "CP072": ("other_services", [0.90, 0.0, 0.02, 0.08], "non-fuel part: repairs, parts, parking; bicycle repair 6 %"),
    "CP073": ("other_services", [0.0, 0.0, 0.70, 0.30], "domestic passenger transport 6 %, international zero"),
    "CP074": ("other_services", [0.70, 0.0, 0.0, 0.30], "universal postal service exempt"),
    "CP081": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP082": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP083": ("other_services", [1.0, 0.0, 0.0, 0.0], ""),
    "CP091": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP092": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP093": ("other_goods", [0.50, 0.0, 0.50, 0.0], "plants, flowers, pet food 6 %"),
    "CP094": ("other_services", [0.15, 0.0, 0.45, 0.40], "sport and leisure admissions 6 %; non-profit clubs and gambling exempt"),
    "CP095": ("other_goods", [0.70, 0.0, 0.30, 0.0], ""),
    "CP096": ("other_services", [0.10, 0.0, 0.60, 0.30], "cinema, theatre, concerts, museums 6 %; recognised institutions exempt"),
    "CP097": ("other_goods", [0.30, 0.0, 0.45, 0.25], "books 6 %; newspapers and weeklies zero; stationery 21 %"),
    "CP098": ("other_services", [0.10, 0.0, 0.0, 0.90], "margin scheme of travel agents"),
    "CP101": ("other_services", [0.0, 0.0, 0.0, 1.0], "education exempt"),
    "CP102": ("other_services", [0.0, 0.0, 0.0, 1.0], ""),
    "CP103": ("other_services", [0.0, 0.0, 0.0, 1.0], ""),
    "CP104": ("other_services", [0.0, 0.0, 0.0, 1.0], ""),
    "CP105": ("other_services", [0.0, 0.0, 0.0, 1.0], ""),
    "CP111": ("restaurants", [0.25, 0.33, 0.37, 0.05], "meals on the premises 12 %, drinks 21 %, takeaway 6 %, school and care canteens exempt"),
    "CP112": ("restaurants", [0.05, 0.0, 0.95, 0.0], "accommodation 6 % (12 % from March 2026)"),
    "CP121": ("other_services", [0.0, 0.0, 0.0, 1.0], "insurance exempt (premium tax)"),
    "CP122": ("other_services", [0.05, 0.0, 0.0, 0.95], "financial services exempt"),
    "CP131": ("other_goods", [0.90, 0.0, 0.10, 0.0], "hygiene products 6 %"),
    "CP132": ("other_goods", [1.0, 0.0, 0.0, 0.0], ""),
    "CP133": ("other_services", [0.0, 0.0, 0.0, 1.0], "social services exempt"),
    "CP139": ("other_services", [0.60, 0.0, 0.10, 0.30], "funeral services 6 %; administrative fees outside VAT"),
}

# Federal inventory of tax expenditures, table BTW - TVA, year 2023, EUR mn of revenue forgone
INVENTORY_2023 = {
    ("6", "construction"): 3261.04, ("6", "necessities and other"): 10589.01, ("6", "horeca"): 1199.27,
    ("6", "cars of disabled persons"): 5.11,
    ("12", "construction"): 39.89, ("12", "horeca"): 508.02, ("12", "other"): 15.69,
    ("0", "newspapers and weeklies"): 157.39,
}
# official costings used to split mixed classes: (description, EUR mn, rate change) -> taxable amount
CHECKS = [
    ("takeaway meals 6 -> 12 % (2026 budget, withdrawn)", 362.0, 0.06),
    ("hotels and campsites 6 -> 12 % (in force March 2026)", 158.0, 0.06),
    ("cultural, sport and leisure admissions 6 -> 12 % (2026 budget, withdrawn)", 253.0, 0.06),
    ("soft drinks in horeca 21 -> 12 % (2026 budget, withdrawn)", 140.0, 0.09),
    ("VAT on food to zero (Planning Bureau, DC2024)", 2093.0, 0.06),
]
# European Commission, VAT gap report 2025, Belgium 2023
EC_2023 = {"vat_revenue": 37413.0, "compliance_gap": 5230.0, "rate_gap_share": 0.131, "policy_gap_share": None}


def download() -> dict[str, float]:
    """COICOP class -> household consumption of the year, EUR mn."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "nama_10_cp18.json"
    if not path.exists():
        r = requests.get(URL, headers=UA, timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
    d = json.loads(path.read_text())
    ids, size = d["id"], d["size"]
    stride, strides = 1, {}
    for name, n in zip(reversed(ids), reversed(size)):
        strides[name] = stride
        stride *= n
    key = next(i for i in ids if i.startswith("coicop"))
    values = d["value"]
    out = {}
    for code, k in d["dimension"][key]["category"]["index"].items():
        v = values.get(str(k * strides[key])) if isinstance(values, dict) else values[k * strides[key]]
        if v is not None:
            out[code] = float(v)
    return out


def structure(cons: dict[str, float]) -> tuple[list[list], dict[str, list[float]], dict[str, float]]:
    """(rows per class, category -> shares of the value before VAT, category -> value before VAT)."""
    rows, base = [], {}
    for code, (cat, shares, note) in RATES.items():
        value = cons.get(code, 0.0)
        if code == "CP072":
            fuel = value * FUEL_SHARE_CP072
            net = fuel / (1.0 + RATE[0])
            rows.append(["class", code, "fuel", round(fuel, 1), round(net, 1), 1.0, 0.0, 0.0, 0.0, "motor fuel (share of CP072)"])
            base.setdefault("fuel", [0.0] * 4)[0] += net
            value -= fuel
        net = value / (1.0 + sum(s * r for s, r in zip(shares, RATE)))
        rows.append(["class", code, cat, round(value, 1), round(net, 1), *shares, note])
        acc = base.setdefault(cat, [0.0] * 4)
        for i, s in enumerate(shares):
            acc[i] += net * s
    mix = {cat: [round(x / sum(acc), 3) for x in acc] for cat, acc in base.items()}
    for cat, m in mix.items():                     # shares add up to one after rounding
        m[m.index(max(m))] = round(m[m.index(max(m))] + 1.0 - sum(m), 3)
    return rows, mix, {cat: sum(acc) for cat, acc in base.items()}


def main() -> int:
    cons = download()
    missing = [c for c in cons if len(c) == 5 and c.startswith("CP") and c not in RATES]
    if missing:
        print(f"COICOP classes without a rate: {missing}", file=sys.stderr)
        return 1
    rows, mix, net = structure(cons)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "coicop", "category", "value_meur", "value_before_vat_meur", "share_21", "share_12",
                    "share_6", "share_zero_exempt", "note"])
        w.writerows(rows)
        for cat in sorted(mix):
            w.writerow(["category", "", cat, "", round(net[cat], 1), *mix[cat], "copy into vat.mix"])
    print(f"written {OUT}")
    print("\nvat.mix rows of household consumption (share of the value before VAT at 21 / 12 / 6 % / zero):")
    for cat in sorted(mix):
        print(f"  {cat:15s} {mix[cat]}   base {net[cat]:9,.0f} mn")
    b6 = sum(net[c] * mix[c][2] for c in mix)
    b12 = sum(net[c] * mix[c][1] for c in mix)
    b21 = sum(net[c] * mix[c][0] for c in mix)
    print(f"\nhouseholds {YEAR}: taxable at 21 % {b21:,.0f} mn, at 12 % {b12:,.0f} mn, at 6 % {b6:,.0f} mn; "
          f"VAT {0.21 * b21 + 0.12 * b12 + 0.06 * b6:,.0f} mn")
    print("federal inventory of tax expenditures 2023 -> amounts declared at the reduced rates (all buyers):")
    for (rate, what), loss in INVENTORY_2023.items():
        print(f"  {rate:>2s} %  {what:28s} forgone {loss:9,.1f} mn -> taxable amount {loss / (0.21 - float(rate) / 100):9,.0f} mn")
    print("official costings -> taxable amounts:")
    for what, amount, step in CHECKS:
        print(f"  {what:75s} {amount:7,.0f} mn -> {amount / step:7,.0f} mn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
