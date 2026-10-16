#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_ESG_DF_DECARB_v2.py  —  Diversity factor, ESG composite, grid what-if
Thesis MIS24011 — Zygouris Aristotelis

Requires: 04_ENERGY_BURDEN_v2 and 05_EDA_v2 outputs.

Fixes vs 06_ADDITIONAL_CHAPTER4_5_FIGURES.py / 07_REMAINING_FIGURES.py
  K4  Diversity factor (IEEE / Willis, Power Distribution Planning):
          DF = Σ_i max_t L_i(t)  /  max_t Σ_i L_i(t)      (DF ≥ 1)
      computed from the hourly data of every building, per zone and type.
      Coincidence factor CF = 1/DF and load factor = mean/peak also given.
  K2  ESG scores/grades were typed by hand. Now (professor #8):
        E = CO2 intensity of the sample (kg CO2/m²·yr, all buildings)
        S = median energy burden of the lowest-income quartile (Q1, %)
            (residential only — the only part with income data)
        G = NOT scored (no governance data) — discussed qualitatively.
      Min–max normalisation over the 5 zones (100 = best), composite
      w·E + (1−w)·S for w ∈ {0.25, 0.50, 0.75}; the ranking under every
      weight is reported. No letter grades.
  K9  GRID_DECARB trajectory was hand-made. Now an explicitly labelled
      WHAT-IF: the electricity emission factor is reduced by r ∈
      GRID_EF_REDUCTION while consumption is kept at 2018 levels.

Outputs (outputs_v2/tables/)
  table_4_5_diversity.csv, table_4_6_esg_indicators.csv,
  table_4_6b_esg_composite.csv, table_5_2_grid_whatif.csv
"""
import numpy as np
import pandas as pd

import config as C
from common import TYPE_GR, update_summary


def diversity(annual, stock):
    rows = []
    for city in C.CITIES:
        s_city = stock[stock["city"] == city]
        for dtype in ("residential", "commercial", "all"):
            a = annual[annual["city"] == city]
            s = s_city
            if dtype != "all":
                a = a[a["dataset_type"] == dtype]
                s = s[s["dataset_type"] == dtype]
            # City-stock estimate (Horvitz–Thompson weights w_city, see 05):
            #   DF = Σ_i w_i·max_t L_i(t) / max_t Σ_i w_i·L_i(t)
            # so that "Σύνολο" mixes dwellings and commercial buildings in the
            # proportion of the real county stock, not of the 600/400 sample
            # (audit 12/10/2026, professor #4). The unweighted sample DF is kept.
            agg = s.groupby("timestamp")["kwh_w"].sum()        # Σ_i w_i L_i(t)
            agg_s = s.groupby("timestamp")["kwh"].sum()        # sample, unweighted
            assert len(agg) == C.HOURS, f"{city}/{dtype}: {len(agg)} hours"
            peak_sum = (a["peak_kwh_h"] * a["w_city"]).sum()   # Σ_i w_i max_t L_i(t)
            peak_agg = agg.max()
            DF = peak_sum / peak_agg
            DF_s = a["peak_kwh_h"].sum() / agg_s.max()
            assert DF >= 1 - 1e-9 and DF_s >= 1 - 1e-9, "DF < 1 is impossible"
            rows.append({"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": TYPE_GR[dtype],
                         "N": len(a),
                         "Σ ατομικών αιχμών, απόθεμα (MW)": round(peak_sum / 1000, 1),
                         "Αιχμή συνόλου, απόθεμα (MW)": round(peak_agg / 1000, 1),
                         "Συντελεστής ετεροχρονισμού DF": round(DF, 3),
                         "DF δείγματος (χωρίς στάθμιση)": round(DF_s, 3),
                         "Συντελεστής ταυτοχρονισμού CF=1/DF": round(1 / DF, 3),
                         "Συντελεστής φορτίου (μέσο/αιχμή)": round(agg.mean() / peak_agg, 3),
                         "Ώρα αιχμής συνόλου": str(agg.idxmax()),
                         "Αιχμή συνόλου (W/m²)": round(
                             peak_agg / (a["floor_area_m2"] * a["w_city"]).sum() * 1000, 2)})
    t = pd.DataFrame(rows)
    t.to_csv(C.TABLES / "table_4_5_diversity.csv", index=False, encoding="utf-8-sig")
    print(t[["Πόλη", "Τύπος", "N", "Συντελεστής ετεροχρονισμού DF", "Ώρα αιχμής συνόλου"]]
          .to_string(index=False))
    return t


def esg(annual):
    burden = pd.read_csv(C.TABLES / "table_5_3_energy_burden.csv")
    q1 = burden[burden["Τεταρτημόριο"] == "Q1"].set_index("Πόλη")
    hi_col = [c for c in burden.columns if c.startswith("Υψηλό burden")][0]
    rows = []
    for city in C.CITIES:
        a = annual[annual["city"] == city]
        w = a["w_city"]                    # city-stock weights (05_EDA_v2.city_weights)
        rows.append({"Πόλη": city, "Ζώνη": C.CITY_ZONE[city],
                     "E: CO₂ (kg/m²·έτος, όλα τα κτίρια)":
                         (a["co2_kg"] * w).sum() / (a["floor_area_m2"] * w).sum(),
                     "S: διάμεσο energy burden Q1 (%)": q1.loc[city, "Διάμεσο energy burden (%)"],
                     f"S (συμπλ.): Q1 {hi_col}": q1.loc[city, hi_col]})
    ind = pd.DataFrame(rows)

    def minmax_best_low(x):
        """100 = lowest value (best), 0 = highest value (worst) among the 5 zones."""
        rng = x.max() - x.min()
        return pd.Series(100.0, index=x.index) if rng == 0 else 100 * (x.max() - x) / rng

    ind["Βαθμός E (0–100)"] = minmax_best_low(ind["E: CO₂ (kg/m²·έτος, όλα τα κτίρια)"])
    ind["Βαθμός S (0–100)"] = minmax_best_low(ind["S: διάμεσο energy burden Q1 (%)"])
    out = ind.round(2)
    out.to_csv(C.TABLES / "table_4_6_esg_indicators.csv", index=False, encoding="utf-8-sig")

    comp = ind[["Πόλη", "Ζώνη"]].copy()
    ranks = {}
    for w in C.ESG_WEIGHTS_E:
        # round before ranking: an exact tie (e.g. 100·0,5 + 0·0,5 = 0·0,5 + 100·0,5)
        # must not be broken by floating-point noise (audit 13/10/2026)
        sc = (w * ind["Βαθμός E (0–100)"] + (1 - w) * ind["Βαθμός S (0–100)"]).round(6)
        comp[f"Σύνθετος (w_E={w:.2f})"] = sc.round(1)
        comp[f"Κατάταξη (w_E={w:.2f})"] = sc.rank(ascending=False, method="min").astype(int)
        ranks[w] = comp[f"Κατάταξη (w_E={w:.2f})"].tolist()
    stable = all(r == ranks[C.ESG_WEIGHTS_E[0]] for r in ranks.values())
    rk_cols = [f"Κατάταξη (w_E={w:.2f})" for w in C.ESG_WEIGHTS_E]
    comp["Ίδια θέση σε όλα τα βάρη"] = np.where(comp[rk_cols].nunique(axis=1) == 1, "ναι", "όχι")
    comp.to_csv(C.TABLES / "table_4_6b_esg_composite.csv", index=False, encoding="utf-8-sig")
    print(comp.to_string(index=False))
    return out, comp, stable


def grid_whatif(annual):
    rows = []
    for city in C.CITIES:
        for dtype in ("residential", "commercial", "all"):
            a = annual[annual["city"] == city]
            if dtype != "all":
                a = a[a["dataset_type"] == dtype]
            w = a["w_city"]                # city-stock weights (only matter for "all")
            A = (a["floor_area_m2"] * w).sum()
            e_el = (a["co2_electricity"] * w).sum()
            e_fuel = (a["co2_kg"] * w).sum() - e_el
            base = (e_el + e_fuel) / A
            r = {"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": TYPE_GR[dtype],
                 "Μερίδιο ηλεκτρισμού στο CO₂ (%)": round(e_el / (e_el + e_fuel) * 100, 1)}
            for red in C.GRID_EF_REDUCTION:
                v = (e_el * (1 - red) + e_fuel) / A
                r[f"CO₂ kg/m² (EF ηλεκτρ. −{red*100:.0f}%)"] = round(v, 2)
                r[f"Μεταβολή CO₂ (%) (EF −{red*100:.0f}%)"] = round((v / base - 1) * 100, 1)
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(C.TABLES / "table_5_2_grid_whatif.csv", index=False, encoding="utf-8-sig")
    return t


def main():
    C.ensure_dirs()
    annual = pd.read_parquet(C.TABLES / "annual_by_building.parquet")
    stock = pd.read_parquet(C.TABLES / "stock_hourly.parquet")
    df = diversity(annual, stock)
    ind, comp, stable = esg(annual)
    grid_whatif(annual)
    update_summary("esg_df", {
        "df_definition": "sum of individual peaks / peak of the sum (hourly, sample)",
        "df_range_all": [float(df.loc[df["Τύπος"] == "Σύνολο", "Συντελεστής ετεροχρονισμού DF"].min()),
                         float(df.loc[df["Τύπος"] == "Σύνολο", "Συντελεστής ετεροχρονισμού DF"].max())],
        "esg_rank_stable_across_weights": bool(stable),
        "esg_weights_E": C.ESG_WEIGHTS_E,
        "grid_ef_reduction_whatif": C.GRID_EF_REDUCTION})
    print("✓ 06_ESG_DF_DECARB_v2 finished")


if __name__ == "__main__":
    main()
