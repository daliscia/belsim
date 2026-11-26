"""Download the Statbel fiscal income statistics (personal income tax returns by place of
residence) and build the table the revenue module uses for the tax base of the personal income
tax, the population of tax units and the regional attribution.

    uv run python -m belsim.data.fetch_revenue_pit

Source (official, open): Statbel, "Revenus fiscaux", income year 2024 (returns filed in 2025)
https://statbel.fgov.be/fr/themes/menages/revenus-fiscaux - workbooks
  C    total net taxable income, state tax (federal + regional) and municipal taxes by region
  B.2  components of net income (wages, self-employed, pensions, unemployment, sickness) by region
  B.3  joint returns by number of professional incomes
  B.5  returns by number of dependants, individual and joint returns
  B.1  upper limits and total income of the deciles of returns and of the top ten percentiles
and table C of income year 2023 (state tax and municipal taxes of the previous income year: the
municipal surcharges received in a year are those assessed on earlier incomes).

Outputs:
- data/raw/revenue/statbel_fisc2024/*.xlsx   raw downloads
- data/base/revenue_pit_statbel.csv          long table: table, region (BE, FLA, WAL, BRU), key, value
                                             (amounts in EUR mn, counts in numbers)
- data/base/revenue_pit_distribution.csv     distribution of net taxable income taxed at the
                                             progressive rates: region, type of return
                                             (individual, joint_one_income, joint_two_incomes),
                                             income (EUR per return), returns (number). Each row is a
                                             group of returns with the same income; see
                                             `distribution` for how they are built from the tables.
"""
from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import requests

YEAR = 2024
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "revenue" / f"statbel_fisc{YEAR}"
OUT = ROOT / "data" / "base" / "revenue_pit_statbel.csv"
UA = {"User-Agent": "python-requests"}
URL = ("https://statbel.fgov.be/sites/default/files/files/documents/Huishoudens/"
       "10.9%20Fiscale%20inkomens/fisc{year}_{table}_FR.xlsx")
TABLES = ("C", "B_1", "B_2", "B_3", "B_5")
PREVIOUS = YEAR - 1                       # table C of the previous income year (municipal taxes)
DIST = ROOT / "data" / "base" / "revenue_pit_distribution.csv"
POINTS = 8                                # groups of returns per decile and type of return
SHEETS = {"BE": "Belgique", "FLA": "Région flamande", "WAL": "Région wallonne",
          "BRU": "Région de Bruxelles-Capitale"}
C_ROWS = {"BE": "Belgique", "FLA": "Région flamande", "WAL": "Région wallonne", "BRU": "Région de Bruxelles-Capita"}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path: Path) -> dict[str, list[list[str]]]:
    """sheet name -> rows of cell strings (standard library only)."""
    z = zipfile.ZipFile(path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS)]
    out = {}
    for i, name in enumerate(names, 1):
        rows = []
        for r in ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml")).iter("{%s}row" % NS["m"]):
            row: dict[int, str] = {}
            for c in r.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is not None:
                    row[_col(c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else v.text
            rows.append([row.get(k, "") for k in range(max(row) + 1)] if row else [])
        out[name] = rows
    return out


def download() -> dict[str, Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {}
    for t in TABLES:
        path = RAW / f"fisc{YEAR}_{t}_FR.xlsx"
        if not path.exists():
            r = requests.get(URL.format(year=YEAR, table=t), headers=UA, timeout=180)
            r.raise_for_status()
            path.write_bytes(r.content)
        paths[t] = path
    path = RAW / f"fisc{PREVIOUS}_C_FR.xlsx"
    if not path.exists():
        r = requests.get(URL.format(year=PREVIOUS, table="C"), headers=UA, timeout=180)
        r.raise_for_status()
        path.write_bytes(r.content)
    paths["C_previous"] = path
    return paths


def _f(x) -> float:
    return float(str(x).strip() or 0.0)


def build(paths: dict[str, Path]) -> list[list]:
    rows: list[list] = []

    def add(table, region, key, value, scale=1.0):
        rows.append([table, region, key, round(_f(value) * scale, 3)])

    # C: totals by region (sheet 'Total': label, code, zero returns, returns, 6 classes, inhabitants,
    # net taxable income, ..., state tax, municipal taxes, total tax)
    total = read_xlsx(paths["C"])["Total"]
    head = total[6]
    idx = {k: next(i for i, h in enumerate(head) if h.startswith(k)) for k in
           ("Nombre d'habitants", "Revenu total net imposable", "Impôt d'Etat", "Taxes communales", "Impôt total")}
    for reg, label in C_ROWS.items():
        r = next(x for x in total[7:] if x and x[0].startswith(label))
        add("C", reg, "returns_zero_income", r[2])
        add("C", reg, "returns", r[3])
        add("C", reg, "inhabitants", r[idx["Nombre d'habitants"]])
        add("C", reg, "net_taxable_income", r[idx["Revenu total net imposable"]], 1e-6)
        add("C", reg, "state_tax", r[idx["Impôt d'Etat"]], 1e-6)
        add("C", reg, "municipal_tax", r[idx["Taxes communales"]], 1e-6)
        add("C", reg, "total_tax", r[idx["Impôt total"]], 1e-6)

    # C of the previous income year: state tax and municipal taxes
    total = read_xlsx(paths["C_previous"])["Total"]
    head = total[6]
    idx = {k: next(i for i, h in enumerate(head) if h.startswith(k)) for k in ("Impôt d'Etat", "Taxes communales")}
    for reg, label in C_ROWS.items():
        r = next(x for x in total[7:] if x and x[0].startswith(label))
        add("C_previous", reg, "state_tax", r[idx["Impôt d'Etat"]], 1e-6)
        add("C_previous", reg, "municipal_tax", r[idx["Taxes communales"]], 1e-6)

    # C.2 / C.3: individual and joint returns (number, income, state tax)
    for kind, sheet in (("individual", "par décl. indiv."), ("joint", "par décl. comm.")):
        data = read_xlsx(paths["C"])[sheet]
        head = data[6]
        idx = {k: next(i for i, h in enumerate(head) if h.startswith(k)) for k in
               ("Total", "Revenu total net imposable", "Impôt d'Etat")}
        for reg, label in C_ROWS.items():
            r = next(x for x in data[7:] if x and x[0].startswith(label))
            add("C", reg, f"{kind}.returns", r[idx["Total"]])
            add("C", reg, f"{kind}.net_taxable_income", r[idx["Revenu total net imposable"]], 1e-6)
            add("C", reg, f"{kind}.state_tax", r[idx["Impôt d'Etat"]], 1e-6)

    # B.2: components of net income (row 'Total': total amount, then shares)
    b2 = read_xlsx(paths["B_2"])
    comp = ("self_employed", "wages", "pensions", "unemployment", "sickness")
    for reg, sheet in SHEETS.items():
        r = next(x for x in b2[sheet] if x and x[0] == "Total")
        amount = _f(r[2])
        add("B2", reg, "net_income", amount, 1e-6)
        for k, share in zip(comp, r[3:8]):
            add("B2", reg, k, amount * _f(share), 1e-6)

    # B.3: returns by type (row 'Total': pairs number, amount)
    b3 = read_xlsx(paths["B_3"])
    kinds = ("individual_zero_m", "individual_zero_f", "individual_m", "individual_f",
             "joint_no_income", "joint_one_income", "joint_two_incomes")
    for reg, sheet in SHEETS.items():
        r = next(x for x in b3[sheet] if x and x[0] == "Total")
        for i, k in enumerate(kinds):
            add("B3", reg, f"{k}.returns", r[1 + 2 * i])
            add("B3", reg, f"{k}.income", r[2 + 2 * i], 1e-6)

    # B.5: returns by number of dependants (10 decile rows per type: pairs number, amount)
    b5 = read_xlsx(paths["B_5"])
    for reg, sheet in SHEETS.items():
        data = b5[sheet]
        start = {"individual": next(i for i, x in enumerate(data) if x and x[0].startswith("Déclaration individu")),
                 "joint": next(i for i, x in enumerate(data) if x and x[0].startswith("Déclaration commune"))}
        for kind, i0 in start.items():
            block = data[i0:i0 + 10]
            for n in range(6):
                add("B5", reg, f"{kind}.dependants_{n}", sum(_f(x[2 + 2 * n]) if len(x) > 2 + 2 * n else 0.0 for x in block))
    return rows


# ---------------------------------------------------------------------------------------------
# distribution of taxable income
# ---------------------------------------------------------------------------------------------
def _spread(lo: float, hi: float | None, mean: float, k: int) -> np.ndarray:
    """k equally likely incomes between `lo` and `hi` with the given mean: a density linear in
    income (uniform tilted towards the side of the mean); without upper limit a Pareto tail
    above `lo` (exact means of k equal slices)."""
    u = (np.arange(k) + 0.5) / k
    if hi is None:
        if mean <= lo * 1.02:
            return np.full(k, mean)
        a = mean / (mean - lo)                                   # Pareto index from mean / threshold
        e = np.arange(k + 1) / k
        tail = (1.0 - e) ** (1.0 - 1.0 / a)
        return lo * a / (a - 1.0) * (tail[:-1] - tail[1:]) * k
    mean = float(np.clip(mean, lo + (hi - lo) * 0.05, hi - (hi - lo) * 0.05))
    t = float(np.clip((mean - 0.5 * (lo + hi)) / ((hi - lo) / 6.0), -1.0, 1.0))
    tri = lo + (hi - lo) * np.sqrt(u) if t > 0 else hi - (hi - lo) * np.sqrt(1.0 - u)
    x = (1.0 - abs(t)) * (lo + (hi - lo) * u) + abs(t) * tri
    return x * mean / x.mean()


def _tilt(means: np.ndarray, target: float) -> np.ndarray:
    """Weights on the ten top percentiles (equal for all returns) tilted exponentially so that
    their mean income equals `target` (the mean of one type of return in the top decile)."""
    z = (means - means.mean()) / means.mean()
    lo, hi = -60.0, 60.0
    for _ in range(80):
        lam = 0.5 * (lo + hi)
        w = np.exp(lam * z)
        w /= w.sum()
        if float(w @ means) < target:
            lo = lam
        else:
            hi = lam
    return w


def distribution(paths: dict[str, Path]) -> list[list]:
    """Groups of returns (region, type, income, number) reproducing, for each region:
    - the number and income of individual returns, joint returns with one and with two
      professional incomes in each decile of returns (table B.3; deciles of all returns ranked
      by total net taxable income, limits from table B.1);
    - inside the top decile, the limits and incomes of the ten top percentiles (table B.1), a
      type of return being spread over them so as to reproduce its mean income in the decile;
      the top percentile is a Pareto tail;
    - inside the other deciles, incomes spread between the limits around the mean.
    Incomes are then reduced to the part taxed at the progressive rates: the ratio of globally
    taxable net income (table B.2) to total net taxable income (table B.1) of the decile or
    percentile - separately taxed income (arrears, termination pay, pension capital) is very
    concentrated at the top."""
    b1, b2, b3 = read_xlsx(paths["B_1"]), read_xlsx(paths["B_2"]), read_xlsx(paths["B_3"])
    out: list[list] = []
    for reg, sheet in SHEETS.items():
        r1 = {(r[0], r[1]): r for r in b1[sheet] if len(r) > 3}
        r2 = {(r[0], r[1]): r for r in b2[sheet] if len(r) > 13}
        r3 = {r[0]: [_f(x) for x in r[1:]] for r in b3[sheet] if r and r[0][:2].isdigit()}
        dec = [f"{d:02d}" for d in range(1, 11)]
        pct = [f"{k:03d}" for k in range(91, 101)]
        upper = [_f(r1[(d, "")][2]) for d in dec[:-1]]
        lows, highs = [0.0] + upper, upper + [None]
        p_upper = [_f(r1[("", k)][2]) for k in pct[:-1]]
        p_lows, p_highs = [upper[-1]] + p_upper, p_upper + [None]
        total_n = sum(sum(r3[d][0::2]) for d in dec)
        p_mean = np.array([_f(r1[("", k)][3]) for k in pct]) / (total_n / 100.0)
        glob_d = [_f(r2[(d, "")][13]) / _f(r1[(d, "")][3]) for d in dec]
        glob_p = [_f(r2[("", k)][13]) / _f(r1[("", k)][3]) for k in pct]
        types = {"individual": (0, 2, 4, 6), "joint_one_income": (8, 10), "joint_two_incomes": (12,)}
        for name, cols in types.items():
            for i, d in enumerate(dec):
                n = sum(r3[d][c] for c in cols)
                amount = sum(r3[d][c + 1] for c in cols)
                if n <= 0:
                    continue
                mean = amount / n
                if i < 9:
                    for x in _spread(lows[i], highs[i], mean, POINTS):
                        out.append([reg, name, round(float(x) * glob_d[i], 2), round(n / POINTS, 3)])
                    continue
                w = _tilt(p_mean, float(np.clip(mean, p_mean[0] * 1.001, p_mean[-1] * 3.0)))
                scale = mean / float(w @ p_mean)                  # exact mean of the type in the decile
                for j in range(10):
                    k = 4 if j < 9 else 12
                    for x in _spread(p_lows[j], p_highs[j], float(p_mean[j]), k):
                        out.append([reg, name, round(float(x) * scale * glob_p[j], 2), round(n * w[j] / k, 3)])
    return out


def main() -> int:
    paths = download()
    rows = build(paths)
    dist = distribution(paths)
    val = {(t, r, k): v for t, r, k, v in rows}
    # checks: regions add up to Belgium (the German-speaking community is part of Wallonia)
    for key in ("net_taxable_income", "state_tax", "municipal_tax"):
        parts = sum(val[("C", r, key)] for r in ("FLA", "WAL", "BRU"))
        if abs(parts / val[("C", "BE", key)] - 1.0) > 1e-3:
            print(f"regions do not add up for {key}: {parts} vs {val[('C', 'BE', key)]}", file=sys.stderr)
            return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "region", "key", "value", "year", "source"])
        for r in rows:
            w.writerow([*r, YEAR, "Statbel fiscal income statistics"])
    print(f"written {OUT} ({len(rows)} rows)")
    with DIST.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "type", "income", "returns", "year", "source"])
        for r in dist:
            w.writerow([*r, YEAR, "Statbel fiscal income statistics, tables B.1 B.2 B.3"])
    for reg in SHEETS:
        income = sum(r[2] * r[3] for r in dist if r[0] == reg) / 1e6
        print(f"  {reg}: {sum(r[3] for r in dist if r[0] == reg):,.0f} returns, globally taxable income "
              f"{income:,.0f} mn (table B.2: {val[('B2', reg, 'net_income')]:,.0f} before deductible expenses)")
    print(f"written {DIST} ({len(dist)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
