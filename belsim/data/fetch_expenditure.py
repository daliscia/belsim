"""Download and clean the expenditure / balance / debt calibration tables (Eurostat open API).

    uv run python -m belsim.data.fetch_expenditure

Raw JSON-stat responses go to data/raw/expenditure/, cleaned tables to data/base/expenditure_*.csv.
The model only ever reads data/base/.

Tables written
    expenditure_cofog_2024.csv      gov_10a_exp 2024: COFOG group x ESA transaction x subsector (EUR mn)
    expenditure_main.csv            gov_10a_main 2023-latest: transactions x subsector, incl. the
                                    transfers between subsectors (D7PAY_S13xx, D9PAY_S13xx, D4PAY_S13xx)
    expenditure_debt.csv            gov_10dd_edpt1 2023-latest (debt, deficit, GDP, GNI, by subsector) and
                                    gov_10q_ggdebt 2024-Q4 (debt by instrument / original maturity)
    expenditure_esspros.csv         ESSPROS pension expenditure (spr_exp_pens) and beneficiaries
                                    (spr_pns_ben), 2023-2024
    expenditure_lines_2024.csv      DERIVED: the model's expenditure lines for 2024 (EUR mn), their ESA
                                    components and the amount per government entity
    expenditure_transfers_2024.csv  DERIVED: 4x4 matrix of transfers between entities (D7 + D9 + D4)
    expenditure_scalars.csv         DERIVED: scalars used for calibration (GDP, debt, balances, ...)

The line definitions live in belsim/fiscal/expenditure/lines.py. Three splits are not available
from Eurostat gov_10a_exp and use parameters read from params/expenditure.yaml (2024 entry):
`pension.scheme_spending_share_2024`, `subsidy.service_vouchers_2024` and
`health.ltc_from_health_2024` (long-term care recorded under health, moved to the care line).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import requests
import yaml

from ..core.dims import ENTITIES, ENTITY_ESA
from ..fiscal.expenditure import lines as L

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "base"
RAW_DIR = ROOT / "data" / "raw" / "expenditure"
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
HEADERS = {"User-Agent": "python-requests"}
YEAR = 2024
SECTORS = ["S13"] + [ENTITY_ESA[e] for e in ENTITIES]
ESA_ENTITY = {v: k for k, v in ENTITY_ESA.items()}
INTRA_ITEMS = ("D4", "D7", "D9")  # transactions that include flows to other subsectors


def eurostat(dataset: str, tag: str = "", **filters) -> pd.DataFrame:
    """Fetch a JSON-stat slice for Belgium as a long DataFrame (one column per dimension + value)."""
    params: list[tuple[str, str]] = [("format", "JSON"), ("lang", "EN"), ("geo", "BE")]
    for k, v in filters.items():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            params.append((k, str(x)))
    r = requests.get(API + dataset, params=params, headers=HEADERS, timeout=180)
    r.raise_for_status()
    js = r.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{dataset}{('_' + tag) if tag else ''}.json").write_text(json.dumps(js))
    dims = js["id"]
    cats = []
    for d in dims:
        idx = js["dimension"][d]["category"]["index"]
        cats.append([c for c, _ in sorted(idx.items(), key=lambda kv: kv[1])])
    values = js.get("value", {})
    rows = []
    for flat, combo in enumerate(itertools.product(*cats)):
        v = values.get(str(flat)) if isinstance(values, dict) else values[flat]
        if v is not None:
            rows.append(combo + (float(v),))
    df = pd.DataFrame(rows, columns=dims + ["value"])
    return df.drop(columns=[c for c in ("freq", "geo") if c in df.columns])


def _assumption(name: str):
    spec = yaml.safe_load((ROOT / "params" / "expenditure.yaml").read_text())[name]
    vals = {int(k): v for k, v in spec["values"].items()}
    return vals[min(vals)]


def _carve(parent: dict[str, float], amount: float, prefer: list[str]) -> dict[str, float]:
    """Take `amount` out of the entity amounts of `parent` (mutated): first pro rata from the
    preferred entities, then pro rata from the others. Returns the carved entity amounts."""
    out = {e: 0.0 for e in ENTITIES}
    left = amount
    for group in (prefer, [e for e in ENTITIES if e not in prefer]):
        avail = sum(max(parent[e], 0.0) for e in group)
        if left <= 1e-9 or avail <= 0:
            continue
        take = min(left, avail)
        for e in group:
            x = take * max(parent[e], 0.0) / avail
            out[e] += x
            parent[e] -= x
        left -= take
    return out


def move_ltc(comp: dict, ent: dict, amounts: dict, scale: float = 1.0) -> None:
    """Move long-term care paid under health (D632) from `health_care` to `long_term_care`,
    entity by entity. The Communities' amount is a COFOG cell and gets the vintage scaling of
    the other cells (`scale`); the health-insurance amount comes from the health accounts."""
    for e, amount in amounts.items():
        x = float(amount) * (scale if e == "regions_communities" else 1.0)
        x = min(x, ent["health_care"][e], comp["health_care"]["D632"])
        for line, sign in (("health_care", -1.0), ("long_term_care", 1.0)):
            comp[line]["D632"] += sign * x
            ent[line][e] += sign * x


def build() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- raw downloads ----------------
    intra = [f"{i}_{s}" for i in INTRA_ITEMS for s in SECTORS[1:]]
    cof = eurostat("gov_10a_exp", unit="MIO_EUR", time=YEAR)
    cof = cof[cof["na_item"].isin(L.ITEMS + intra + ["TE"]) & cof["sector"].isin(SECTORS)]
    cof = cof[["cofog99", "na_item", "sector", "value"]].sort_values(["cofog99", "na_item", "sector"])
    cof.to_csv(BASE_DIR / "expenditure_cofog_2024.csv", index=False)

    main = eurostat("gov_10a_main", unit="MIO_EUR", sinceTimePeriod=YEAR - 1)
    main = main[main["sector"].isin(SECTORS)][["na_item", "sector", "time", "value"]]
    main["time"] = main["time"].astype(int)
    main.sort_values(["time", "na_item", "sector"]).to_csv(BASE_DIR / "expenditure_main.csv", index=False)

    edp = eurostat("gov_10dd_edpt1", unit="MIO_EUR", sinceTimePeriod=YEAR - 1)
    edp = edp[["na_item", "sector", "time", "value"]]
    edp["time"] = edp["time"].astype(int)
    qd = eurostat("gov_10q_ggdebt", unit="MIO_EUR", time=f"{YEAR}-Q4")
    qd = qd[["na_item", "sector", "value"]].assign(time=YEAR)
    qd["na_item"] = "Q4_" + qd["na_item"]
    debt = pd.concat([edp, qd[["na_item", "sector", "time", "value"]]])
    debt.sort_values(["time", "na_item", "sector"]).to_csv(BASE_DIR / "expenditure_debt.csv", index=False)

    pens = eurostat("spr_exp_pens", unit="MIO_EUR", sinceTimePeriod=YEAR - 1)
    pens = pens.assign(table="spr_exp_pens", sex="T", unit="MIO_EUR")
    ben = eurostat("spr_pns_ben", sinceTimePeriod=YEAR - 1).assign(table="spr_pns_ben")
    cols = ["table", "spdepb", "spdepm", "sex", "unit", "time", "value"]
    ess = pd.concat([pens[cols], ben[cols]])
    ess["time"] = ess["time"].astype(int)
    ess.to_csv(BASE_DIR / "expenditure_esspros.csv", index=False)

    # ---------------- helpers on the downloads ----------------
    cv = cof.set_index(["cofog99", "na_item", "sector"])["value"].to_dict()
    mv = main.set_index(["time", "na_item", "sector"])["value"].to_dict()
    dv = debt.set_index(["time", "na_item", "sector"])["value"].to_dict()
    ev = ess.set_index(["table", "spdepb", "spdepm", "sex", "time"])["value"].to_dict()

    def m(item: str, sector: str = "S13", year: int = YEAR) -> float:
        return float(mv.get((year, item, sector), 0.0))

    # ---------------- lines from the COFOG matrix ----------------
    groups = sorted({c for c in cof["cofog99"].unique() if len(c) == 6})
    comp = {d.id: {i: 0.0 for i in L.ITEMS} for d in L.PARENTS}
    ent = {d.id: {e: 0.0 for e in ENTITIES} for d in L.PARENTS}
    for g in groups:
        for item in L.ITEMS:
            v = cv.get((g, item, "S13"), 0.0)
            line = L.assign(g, item)
            if line is None:
                if abs(v) > 1e-6:
                    raise RuntimeError(f"unassigned COFOG cell {g} {item} = {v}")
                continue
            comp[line][item] += v
            for e in ENTITIES:
                s = ENTITY_ESA[e]
                x = cv.get((g, item, s), 0.0)
                if item in INTRA_ITEMS:  # keep only what leaves general government
                    x -= sum(cv.get((g, f"{item}_{t}", s), 0.0) for t in SECTORS[1:])
                ent[line][e] += x
    te_cofog = cv[("TOTAL", "TE", "S13")]
    total = sum(sum(c.values()) for c in comp.values())
    assert abs(total - te_cofog) < 5.0, (total, te_cofog)

    # scale to the (more recent vintage) totals of gov_10a_main: interest = D4PAY, rest pro rata
    te, d4 = m("TE"), m("D4PAY")
    k = (te - d4) / (total - comp["interest"]["D4"])
    for lid in comp:
        f = d4 / comp[lid]["D4"] if lid == "interest" else k
        comp[lid] = {i: v * f for i, v in comp[lid].items()}
        tot_e = sum(ent[lid].values())
        tgt = sum(comp[lid].values())
        ent[lid] = {e: (v * tgt / tot_e if tot_e else 0.0) for e, v in ent[lid].items()}

    # ---------------- carve-outs ----------------
    def esspros(table, b, mt, sex="T", year=YEAR):
        return float(ev[(table, b, mt, sex, year)])

    shares = _assumption("pension.scheme_spending_share_2024")
    vouchers = float(_assumption("subsidy.service_vouchers_2024"))
    grapa = esspros("spr_exp_pens", "OLD", "MT")
    pens_total = comp["pensions"]["D62"]
    rest = pens_total - grapa
    ssum = sum(shares.values())
    non_ss = [e for e in ENTITIES if e != "social_security"]
    carve_plan = [  # (child, parent, item, amount, preferred entities) - order matters
        ("pension_grapa", "pensions", "D62", grapa, ["federal"]),
        ("pension_civil_servants", "pensions", "D62", rest * shares["civil_servants"] / ssum, non_ss),
        ("pension_self_employed", "pensions", "D62", rest * shares["self_employed"] / ssum, ["social_security"]),
        ("pension_employees", "pensions", "D62", rest * shares["employees"] / ssum, ["social_security"]),
        ("disability_allowances", "sick_disab", "D62", esspros("spr_exp_pens", "DIS", "MT"), non_ss),
        ("invalidity", "sick_disab", "D62", esspros("spr_exp_pens", "DIS", "NMT"), ["social_security"]),
        ("rcc_swt", "unemployment", "D62", esspros("spr_exp_pens", "ERLM", "TOTAL"), ["social_security"]),
        ("subsidy_service_vouchers", "subsidy_wage", "D3", vouchers, ["regions_communities"]),
        ("eu_contribution", "general_admin", "D7", m("D76PAY"), ["federal"]),
    ]
    for child, parent, item, amount, prefer in carve_plan:
        amount = min(amount, comp[parent][item])
        comp[child] = {i: 0.0 for i in L.ITEMS}
        comp[child][item] = amount
        comp[parent][item] -= amount
        ent[child] = _carve(ent[parent], amount, prefer)
    # long-term care recorded under health (COFOG 07): moved to the long-term care line
    move_ltc(comp, ent, _assumption("health.ltc_from_health_2024"), scale=k)
    # remainders of the vanishing parents
    comp["sickness_primary"], ent["sickness_primary"] = comp.pop("sick_disab"), ent.pop("sick_disab")
    left = comp.pop("pensions")
    left_e = ent.pop("pensions")
    comp["pension_employees"]["D62"] += left["D62"]
    for e in ENTITIES:
        ent["pension_employees"][e] += left_e[e]

    defs = L.all_defs()
    assert set(defs) == set(comp), set(defs) ^ set(comp)
    rows = []
    for lid, d in defs.items():
        row = {"line": lid, "value": sum(comp[lid].values())}
        row.update(comp[lid])
        row.update(ent[lid])
        rows.append(row)
    lines = pd.DataFrame(rows)
    assert abs(lines["value"].sum() - te) < 1e-3
    lines.round(3).to_csv(BASE_DIR / "expenditure_lines_2024.csv", index=False)

    # ---------------- transfers between entities ----------------
    trows = []
    for a in ENTITIES:
        for b in ENTITIES:
            if a == b:
                continue
            sa, sb = ENTITY_ESA[a], ENTITY_ESA[b]
            d7, d9, d4x = m(f"D7PAY_{sb}", sa), m(f"D9PAY_{sb}", sa), m(f"D4PAY_{sb}", sa)
            trows.append({"from": a, "to": b, "value": d7 + d9 + d4x, "D7": d7, "D9": d9, "D4": d4x})
    tr = pd.DataFrame(trows)
    tr.round(3).to_csv(BASE_DIR / "expenditure_transfers_2024.csv", index=False)

    # ---------------- scalars ----------------
    sc: list[tuple[str, float, str, str]] = []

    def add(key, value, unit, source):
        sc.append((key, float(value), unit, source))

    for y in sorted({t for (t, i, s) in dv if i == "B1GQ"}):
        add(f"gdp_{y}", dv[(y, "B1GQ", "S1")], "EUR mn", "gov_10dd_edpt1 B1GQ")
        add(f"gni_{y}", dv[(y, "B5GQ", "S1")], "EUR mn", "gov_10dd_edpt1 B5GQ")
        add(f"debt_{y}", dv[(y, "GD", "S13")], "EUR mn", "gov_10dd_edpt1 GD S13 (Maastricht debt)")
        add(f"balance_{y}", dv[(y, "B9", "S13")], "EUR mn", "gov_10dd_edpt1 B9 S13")
        add(f"te_{y}", m("TE", year=y), "EUR mn", "gov_10a_main TE S13")
        add(f"tr_{y}", m("TR", year=y), "EUR mn", "gov_10a_main TR S13")
        add(f"interest_{y}", m("D4PAY", year=y), "EUR mn", "gov_10a_main D4PAY S13")
        add(f"eu_funded_{y}", m("D7REC_S212", year=y) + m("D9REC_S212", year=y), "EUR mn",
            "gov_10a_main D7REC_S212 + D9REC_S212 (transfers received from EU institutions)")
        for e in ENTITIES:
            s = ENTITY_ESA[e]
            add(f"debt_{e}_{y}", dv[(y, "GD", s)], "EUR mn", f"gov_10dd_edpt1 GD {s} (consolidated within subsector)")
            add(f"balance_{e}_{y}", dv[(y, "B9", s)], "EUR mn", f"gov_10dd_edpt1 B9 {s}")
            add(f"te_{e}_{y}", m("TE", s, y), "EUR mn", f"gov_10a_main TE {s} (non-consolidated)")
            add(f"tr_{e}_{y}", m("TR", s, y), "EUR mn", f"gov_10a_main TR {s} (non-consolidated)")
    short = sum(dv[(YEAR, f"Q4_{i}", "S13")] for i in ("F2", "F31", "F41"))
    add("short_debt_share_2024", short / dv[(YEAR, "Q4_GD", "S13")], "fraction",
        "gov_10q_ggdebt 2024-Q4: (F2 + F31 + F41) / GD, original maturity <= 1 year")
    add("d12_d11_ratio_2024", m("D12PAY") / m("D11PAY"), "fraction",
        "gov_10a_main D12PAY / D11PAY: employer social contributions (actual + imputed) per euro of wages")
    add("d11_2024", m("D11PAY"), "EUR mn", "gov_10a_main D11PAY S13")
    # own (consolidated) revenue per entity = balance + own final expenditure + transfers paid - received
    own_exp = {e: float(lines[e].sum()) for e in ENTITIES}
    paid = tr.groupby("from")["value"].sum().to_dict()
    recv = tr.groupby("to")["value"].sum().to_dict()
    own_rev = {e: dv[(YEAR, "B9", ENTITY_ESA[e])] + own_exp[e] + paid[e] - recv[e] for e in ENTITIES}
    for e in ENTITIES:
        add(f"own_expenditure_{e}_2024", own_exp[e], "EUR mn", "derived: final (consolidated) expenditure of the entity")
        add(f"own_revenue_{e}_2024", own_rev[e], "EUR mn", "derived: B9 + own expenditure + transfers paid - received")
    s1312 = ENTITY_ESA["regions_communities"]
    educ = cv.get(("GF09", "D7_S1313", s1312), 0.0)
    add("regions_to_local_education_share_2024", educ / cv[("TOTAL", "D7_S1313", s1312)], "fraction",
        "gov_10a_exp S1312: D7 to S1313 in COFOG GF09 / total D7 to S1313")
    for b, mt, key in (("OLD", "MT", "grapa"), ("DIS", "MT", "disability_allowance"), ("DIS", "NMT", "invalidity"),
                       ("ERLM", "TOTAL", "rcc"), ("OLD", "TOTAL", "old_age"), ("SRV", "TOTAL", "survivors")):
        add(f"beneficiaries_{key}_2024", esspros("spr_pns_ben", b, mt), "persons", f"spr_pns_ben {b} {mt} 2024")
        add(f"spending_{key}_2024", esspros("spr_exp_pens", b, mt), "EUR mn", f"spr_exp_pens {b} {mt} 2024")
    pd.DataFrame(sc, columns=["key", "value", "unit", "source"]).to_csv(
        BASE_DIR / "expenditure_scalars.csv", index=False)

    print(lines[["line", "value"] + ENTITIES].round(0).to_string())
    print("total", round(lines["value"].sum(), 1), " TE", te)
    print(tr.round(0).to_string())
    print({e: round(v) for e, v in own_rev.items()}, "sum", round(sum(own_rev.values())), "TR", m("TR"))


if __name__ == "__main__":
    build()
