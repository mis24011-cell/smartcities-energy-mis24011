#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_DASHBOARD_v2.py  —  Interactive dashboard (presentation layer, DSS level 4; Ch.6 §6.2)
Thesis MIS24011 — Zygouris Aristotelis

Builds ONE self-contained HTML page from the result tables (no number is typed by
hand): the template dashboard_template.html receives the data as JSON.
The page recomputes, in the browser and with the same formulas as the thesis:
  • composite E+S index for any weight w_E (min–max over the 5 cities, Ch.4 §4.5.2)
  • stock CO₂/m² for any grid emission-factor reduction r:
        CO₂(r) = CO₂(0) · (1 − share_electricity · r)        (exactly linear, Ch.5 §5.1.3)
  • climate scenario view (Ch.5 §5.1.2), energy burden (Ch.5 §5.2),
    forecast skill with 95 % bootstrap intervals (Ch.4 §4.4.3, 08_SIGNIFICANCE_v2),
    diversity factor (Ch.4 §4.4.2).
Output: outputs_v2/dashboard.html
"""
import json
from pathlib import Path

import pandas as pd

import config as C

HERE = Path(__file__).resolve().parent
CITIES = list(C.CITIES)


def csv(name):
    return pd.read_csv(C.TABLES / name, encoding="utf-8-sig")


def build_data():
    esg = csv("table_4_6_esg_indicators.csv").set_index("Πόλη")
    grid = csv("table_5_2_grid_whatif.csv")
    cl = csv("table_5_1_climate.csv")
    bur = csv("table_5_3_energy_burden.csv")
    dfz = csv("table_4_5_diversity.csv")
    sk = csv("table_4_4e_bootstrap_skill.csv")
    te = csv("table_4_4_benchmark_test.csv").set_index("Μοντέλο")
    oo = csv("table_4_4_benchmark_oos.csv").set_index("Μοντέλο")
    scen = [c.split(":")[0] for c in cl.columns if c.endswith("Δ συνολικής κατανάλωσης (%)")]
    d = {"cities": [{"name": c, "zone": C.CITY_ZONE[c]} for c in CITIES], "scenarios": scen,
         "deltaT": {k: list(v) for k, v in C.SCENARIOS.items()}}
    d["esg"] = {c: {"E": float(esg.loc[c, "E: CO₂ (kg/m²·έτος, όλα τα κτίρια)"]),
                    "S": float(esg.loc[c, "S: διάμεσο energy burden Q1 (%)"])} for c in CITIES}
    d["grid"] = {}
    for c in CITIES:
        d["grid"][c] = {}
        for t, k in (("Οικιστικά", "res"), ("Εμπορικά", "com"), ("Σύνολο", "all")):
            r = grid[(grid["Πόλη"] == c) & (grid["Τύπος"] == t)].iloc[0]
            # share of electricity in CO₂ = slope of the exactly linear what-if, taken from
            # the widest step (−75 %) so that rounding of the table (2 dp) is diluted ×0.75·CO₂
            c0, c75 = float(r["CO₂ kg/m² (EF ηλεκτρ. −0%)"]), float(r["CO₂ kg/m² (EF ηλεκτρ. −75%)"])
            share = (c0 - c75) / (0.75 * c0)
            assert abs(share * 100 - float(r["Μερίδιο ηλεκτρισμού στο CO₂ (%)"])) < 0.1, (c, t)
            steps = [float(r[f"CO₂ kg/m² (EF ηλεκτρ. −{x}%)"]) for x in (0, 25, 50, 75)]
            d["grid"][c][k] = {"co2": c0, "share": share, "steps": steps}
    d["climate"] = {}
    for c in CITIES:
        d["climate"][c] = {}
        for t, k in (("residential", "res"), ("commercial", "com")):
            r = cl[(cl["Πόλη"] == c) & (cl["Τύπος"] == t)].iloc[0]
            d["climate"][c][k] = {}
            for s in scen:
                lo, hi = (float(x) for x in r[f"{s}: εύρος very likely (%)"].split(" … "))
                d["climate"][c][k][s] = {"v": float(r[f"{s}: Δ συνολικής κατανάλωσης (%)"]),
                                         "lo": min(lo, hi), "hi": max(lo, hi)}
    d["burden"] = {}
    for c in CITIES:
        d["burden"][c] = {}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            r = bur[(bur["Πόλη"] == c) & (bur["Τεταρτημόριο"] == q)].iloc[0]
            d["burden"][c][q] = {"med": float(r["Διάμεσο energy burden (%)"]),
                                 "hi6": float(r["Υψηλό burden >6% (% νοικ.)"]),
                                 "hi10": float(r["Σοβαρό burden >10% (% νοικ.)"]),
                                 "inc": float(r["Διάμεσο εισόδημα ($)"]),
                                 "cost": float(r["Διάμεσο ετήσιο κόστος ($)"]), "n": int(r["N"])}
    d["df"] = {}
    for c in CITIES:
        d["df"][c] = {}
        for t, k in (("Οικιστικά", "res"), ("Εμπορικά", "com"), ("Σύνολο", "all")):
            r = dfz[(dfz["Πόλη"] == c) & (dfz["Τύπος"] == t)].iloc[0]
            d["df"][c][k] = {"df": float(r["Συντελεστής ετεροχρονισμού DF"]),
                             "peak": str(r["Ώρα αιχμής συνόλου"]),
                             "lf": float(r["Συντελεστής φορτίου (μέσο/αιχμή)"])}
    d["skill"] = []
    for _, r in sk[sk["Τύπος"] == "Όλα"].iterrows():
        d["skill"].append({"set": "test" if r["Σύνολο"].startswith("Έλεγχος") else "oos",
                           "model": r["Μοντέλο"], "v": float(r["Skill"]),
                           "lo": float(r["95% ΔΕ κάτω"]), "hi": float(r["95% ΔΕ άνω"])})
    d["rmse"] = {"test": {m: float(te.loc[m, "RMSE (kWh/h)"]) for m in te.index},
                 "oos": {m: float(oo.loc[m, "RMSE (kWh/h)"]) for m in oo.index}}
    return d


def main():
    tpl = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    html = tpl.replace("/*__DATA__*/null", data)
    assert "/*__DATA__*/" not in html
    out = C.OUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"✓ {out} ({len(html) / 1024:.0f} KB)")
    # Standalone copy (full HTML document) — opens with a double click in any browser,
    # offline as well (fonts fall back to the system sans-serif).
    full = ('<!doctype html>\n<html lang="el"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            + html.replace("</style>", "</style></head><body>", 1) + "\n</body></html>\n")
    out2 = C.OUT_DIR / "dashboard_standalone.html"
    out2.write_text(full, encoding="utf-8")
    print(f"✓ {out2}")


if __name__ == "__main__":
    main()
