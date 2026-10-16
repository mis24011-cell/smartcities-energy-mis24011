#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_SUPPLEMENT_v2.py  —  Supplementary calculations quoted in the thesis text
Thesis MIS24011 — Zygouris Aristotelis (MSc Information Systems, University of Macedonia)

Collects the small calculations that are reported in the text but are not part of the main
pipeline tables. Everything is computed from files that the earlier steps already wrote,
so this script trains nothing and needs only a few seconds.

  A. Load profiles per m² (Ch.4 §4.4.1, Table 4.6, Fig. 4.3)
     mean hourly demand in W/m² = Σ kWh of the buildings of a type / Σ floor area,
     for winter (Dec–Feb) and summer (Jun–Aug); peak hour, peak and minimum value.
  B. Forecast errors (Ch.4 §4.2.4 Table 4.5 and §4.4.5)
     skill of XGBoost / Random Forest per city, MAE per hour of day, residual quantiles,
     share of squared error by building type and by the 40 largest buildings (1 %).
  C. Low-income (Q1) households (Ch.5 §5.2.2, Table 5.9)
     income < $10,000, renters, construction before 1960, heating fuel, median floor area.
  D. Composite E+S index under grid scenarios (Ch.5 §5.4.1, Table 5.12)
     same formula as 06_ESG_DF_DECARB_v2: min–max over the 5 cities, w_E = 0.5,
     scores rounded to 6 decimals before ranking (exact ties share the rank).

Inputs  (outputs_v2/): tables/stock_hourly.parquet, tables/annual_by_building.parquet,
        predictions_test.parquet, tables/energy_burden_by_household.parquet,
        tables/table_4_6_esg_indicators.csv, tables/table_5_2_grid_whatif.csv
Outputs (outputs_v2/tables/): suppl_A_load_profiles.csv, suppl_B_skill_by_city.csv,
        suppl_B_mae_by_hour.csv, suppl_B_error_summary.json, suppl_C_q1_households.csv,
        suppl_D_composite_grid.csv
Run from the project folder (the one that contains outputs_v2), like every other step.
"""
import json

import numpy as np
import pandas as pd

import config as C

T = C.TABLES
MODELS = ["Persistence (t−1h)", "Random Forest", "XGBoost"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# ── A. load profiles per m² ──────────────────────────────────────────
def load_profiles():
    st = pd.read_parquet(T / "stock_hourly.parquet")
    ann = pd.read_parquet(T / "annual_by_building.parquet")
    area = ann.groupby(["city", "dataset_type"])["floor_area_m2"].sum()   # m² of the sample
    rows = []
    for dtype in ("residential", "commercial"):
        for c in C.CITIES:
            r = {"type": dtype, "city": c}
            for tag, months in (("W", (12, 1, 2)), ("S", (6, 7, 8))):
                s = st[(st.city == c) & (st.dataset_type == dtype) & st["timestamp"].dt.month.isin(months)]
                # kWh per hour / m² × 1000 = W/m² (mean hourly demand)
                prof = s.groupby(s["timestamp"].dt.hour)["kwh"].mean() / area[(c, dtype)] * 1000
                r[f"{tag}_hpk"] = int(prof.idxmax())          # hour of the peak
                r[f"{tag}_pk"] = round(float(prof.max()), 1)  # peak W/m²
                r[f"{tag}_min"] = round(float(prof.min()), 1) # minimum W/m²
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(T / "suppl_A_load_profiles.csv", index=False, encoding="utf-8-sig")
    return out


# ── B. forecast errors ───────────────────────────────────────────────
def error_stats():
    d = pd.read_parquet(C.OUT_DIR / "predictions_test.parquet")
    d["h"] = d["timestamp"].dt.hour
    # MAE by hour of day (Fig. 4.7)
    mae_h = pd.DataFrame({m: (d[m] - d["y"]).abs().groupby(d["h"]).mean() for m in MODELS})
    mae_h.round(4).to_csv(T / "suppl_B_mae_by_hour.csv", encoding="utf-8-sig")
    # skill per city (Table 4.5): 1 − RMSE_model / RMSE_persistence of the same city
    rows = []
    for z, g in d.groupby("zone", observed=True):
        rp = rmse(g["y"], g[MODELS[0]])
        rows.append({"zone": str(z), "n": len(g), "rmse_persistence": round(rp, 4),
                     "rmse_xgb": round(rmse(g["y"], g["XGBoost"]), 4),
                     "skill_xgb": round(1 - rmse(g["y"], g["XGBoost"]) / rp, 4),
                     "skill_rf": round(1 - rmse(g["y"], g["Random Forest"]) / rp, 4)})
    pd.DataFrame(rows).to_csv(T / "suppl_B_skill_by_city.csv", index=False, encoding="utf-8-sig")
    # residuals of XGBoost and where the squared error comes from (§4.4.5)
    r = d["y"] - d["XGBoost"]
    se = r ** 2
    big = d.groupby("building_uid", observed=True)["y"].mean().rank(ascending=False) <= 40
    se_b = se.groupby(d["building_uid"]).sum()
    summ = {"residual_mean": float(r.mean()), "residual_median": float(r.median()),
            "q0.005": float(r.quantile(0.005)), "q0.995": float(r.quantile(0.995)),
            "share_abs_le_1kWh_%": float((r.abs() <= 1).mean() * 100),
            "share_abs_le_5kWh_%": float((r.abs() <= 5).mean() * 100),
            "se_share_commercial_%": float(se[d["dataset_type"] == "commercial"].sum() / se.sum() * 100),
            "se_share_top40_buildings_%": float(se_b[big[big].index].sum() / se_b.sum() * 100)}
    json.dump({k: round(v, 4) for k, v in summ.items()},
              open(T / "suppl_B_error_summary.json", "w", encoding="utf-8"), indent=1)
    return pd.DataFrame(rows), summ


# ── C. Q1 households ─────────────────────────────────────────────────
def q1_households():
    hh = pd.read_parquet(T / "energy_burden_by_household.parquet")
    rows = []
    for c in C.CITIES:
        q = hh[(hh["city"] == c) & (hh["quartile"] == "Q1")]
        hf = q["in.heating_fuel"].astype(str)
        rows.append({"city": c, "zone": C.CITY_ZONE[c], "N": len(q),
                     "income_below_10k_%": round((q["income_usd"] < 10000).mean() * 100, 1),
                     "renters_%": round(q["tenure"].astype(str).str.lower().str.contains("rent").mean() * 100, 1),
                     "built_before_1960_%": round(q["in.vintage"].astype(str)
                                                  .isin(["<1940", "1940s", "1950s"]).mean() * 100, 1),
                     "electric_heating_%": round((hf == "Electricity").mean() * 100, 1),
                     "gas_heating_%": round((hf == "Natural Gas").mean() * 100, 1),
                     "median_floor_area_m2": round(float(q["floor_area_m2"].median()), 0)})
    out = pd.DataFrame(rows)
    out.to_csv(T / "suppl_C_q1_households.csv", index=False, encoding="utf-8-sig")
    return out


# ── D. composite E+S under grid scenarios ────────────────────────────
def composite_grid(w_e=0.5):
    esg = pd.read_csv(T / "table_4_6_esg_indicators.csv", encoding="utf-8-sig").set_index("Πόλη")
    grid = pd.read_csv(T / "table_5_2_grid_whatif.csv", encoding="utf-8-sig")
    tot = grid[grid["Τύπος"] == "Σύνολο"].set_index("Πόλη")
    S = esg.loc[C.CITIES, "S: διάμεσο energy burden Q1 (%)"]

    def mm(x):  # min–max, lower value = better = 100
        return 100 * (x.max() - x) / (x.max() - x.min())

    rows = []
    for red in (0, 25, 50, 75):
        E = tot.loc[C.CITIES, f"CO₂ kg/m² (EF ηλεκτρ. −{red}%)"]
        comp = (w_e * mm(E) + (1 - w_e) * mm(S)).round(6)
        rank = comp.rank(ascending=False, method="min").astype(int)
        for c in C.CITIES:
            rows.append({"grid_ef_reduction_%": red, "city": c, "E_co2_kg_m2": float(E[c]),
                         "composite": round(float(comp[c]), 1), "rank": int(rank[c]),
                         "tie": bool((rank == rank[c]).sum() > 1)})
    out = pd.DataFrame(rows)
    out.to_csv(T / "suppl_D_composite_grid.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    C.ensure_dirs()
    a = load_profiles();  print("A. load profiles\n", a.to_string(index=False))
    b, s = error_stats(); print("B. skill by city\n", b.to_string(index=False)); print(s)
    c = q1_households();  print("C. Q1 households\n", c.to_string(index=False))
    d = composite_grid(); print("D. composite under grid scenarios\n",
                                d.pivot(index="city", columns="grid_ef_reduction_%", values="composite"))
    print("✓ 10_SUPPLEMENT_v2 finished")


if __name__ == "__main__":
    main()
