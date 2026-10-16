#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_EDA_v2.py  —  Chapter 3: descriptive analysis on the repaired data
Thesis MIS24011 — Zygouris Aristotelis

Every number in a figure or table is computed here from data/v2 —
nothing is typed by hand (professor comment #15).

Conventions
  • Intensities are AGGREGATE intensities of the sample:
        Σ kWh of the group / Σ m² of the group   (kWh/m²·year)
    and, where stated, the median of the per-building EUI.
  • Fuel mix / CO2 shares are shown NREL-weighted (stock representative,
    professor #3) and unweighted (sample) in table_3_2.
  • Floor areas in m² (1 ft² = 0.09290304 m²).
  • End-use stacks always add up to the site total: what the regex map
    does not cover goes to "Λοιπά" and its share is reported.

Outputs (outputs_v2/)
  tables/table_3_1_sample.csv            sample & floor area per city/type
  tables/table_3_2_energy_by_zone.csv    EUI, fuel mix (w/unw), CO2 intensity
  tables/table_3_3_hdd_cdd.csv           annual & monthly HDD/CDD (base 18 °C)
  tables/table_3_4_enduse.csv            end-use intensities (kWh/m²·yr)
  tables/annual_by_building.parquet      per building annual kWh, CO2, peak (used by 06/07)
  tables/stock_hourly.parquet            Σ hourly kWh per city × type (used by 06/07)
  figures/fig_3_1 … fig_3_19 (see FIGURES at the end of this file)
"""
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config as C
from common import (FUEL_GR, FUELS, TYPE_GR, ZONE_COLORS, end_use_map,
                    fuel_total_col, gr, gr_axis, load_json, log_axis_plain, save_fig, setup_mpl,
                    site_total_col, split_labels, update_summary)

plt = setup_mpl()
FAMILIES = ["Θέρμανση", "Ψύξη", "Αερισμός", "Ζ.Ν.Χ.", "Φωτισμός", "Εξοπλισμός", "Λοιπά"]
FAM_COLORS = ["#d7301f", "#2b8cbe", "#74c476", "#fd8d3c", "#fed976", "#9e9ac8", "#bdbdbd"]
FUEL_COLORS = {"electricity": "#3182bd", "natural_gas": "#e6550d", "propane": "#31a354",
               "fuel_oil": "#756bb1", "other_fuel": "#bcbddc", "district_heating": "#bd0026", "district_cooling": "#74a9cf"}
SEASON = {12: "Χειμώνας", 1: "Χειμώνας", 2: "Χειμώνας", 6: "Καλοκαίρι", 7: "Καλοκαίρι",
          8: "Καλοκαίρι"}
MONTHS_GR = ["Ιαν", "Φεβ", "Μαρ", "Απρ", "Μαΐ", "Ιουν", "Ιουλ", "Αυγ", "Σεπ", "Οκτ", "Νοε", "Δεκ"]


# ═════════════════════════════════════════════════════════════════════
# 1. One pass over the hourly data -> annual per building + stock series
# ═════════════════════════════════════════════════════════════════════
def city_weights(meta, summary):
    """
    Weight of each sampled building for CITY-STOCK totals (audit 12/10/2026,
    professor #4). The sample has 600 res + 400 com per city, but the share
    of each type in the county differs, so pooling the two types with the
    raw NREL weights is biased. Horvitz–Thompson expansion within each
    stratum (city × type), simple random sampling without replacement:
        w_city = nrel_weight × N_models(city, type) / n_sample(city, type)
    N_models = NREL models that pass the zone + county filter (funnel).
    Σ w_city over a stratum estimates the number of real buildings of the
    county; within one type it is proportional to nrel_weight, so per-type
    results do not change — only the pooled "Σύνολο" rows do.
    """
    f = summary["funnel"]["by_city"]
    m = meta[["building_uid", "city", "dataset_type", "nrel_weight"]].copy()
    n = m.groupby(["city", "dataset_type"])["building_uid"].transform("size")
    N = [f[c]["city_res" if d == "residential" else "city_com"]
         for c, d in zip(m["city"], m["dataset_type"])]
    m["w_city"] = m["nrel_weight"].astype("float64") * np.asarray(N, "float64") / n
    return m.set_index("building_uid")["w_city"]


def collect(w_city):
    cols = pq.read_schema(C.BLDG_V2).names
    site = site_total_col(cols)
    fuel = {f: fuel_total_col(cols, f) for f in FUELS}
    fuel = {f: c for f, c in fuel.items() if c}
    eu_map, eu_unmatched = end_use_map(cols)
    energy = [site] + list(fuel.values()) + list(eu_map)

    annual, stock, monthly = [], [], []
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            t = pq.read_table(C.BLDG_V2, columns=["building_uid", "timestamp"] + energy,
                              filters=[("city", "=", city), ("dataset_type", "=", dtype)])
            # annual sums + hourly peak per building (pyarrow: no pandas copy)
            agg = t.group_by("building_uid").aggregate(
                [(c, "sum") for c in energy] + [(site, "max")]).to_pandas()
            agg.columns = [c[:-4] if c.endswith("_sum") else c for c in agg.columns]
            agg = agg.rename(columns={f"{site}_max": "peak_kwh_h"})
            a = pd.DataFrame({"building_uid": agg["building_uid"],
                              "site_kwh": agg[site].astype("float64"),
                              "peak_kwh_h": agg["peak_kwh_h"].astype("float64")})
            for f, c in fuel.items():
                a[f"kwh_{f}"] = agg[c].astype("float64")
            for fam in FAMILIES[:-1]:
                fc = [c for c, v in eu_map.items() if v == fam]
                a[f"eu_{fam}"] = agg[fc].sum(axis=1).astype("float64") if fc else 0.0
            a["eu_Λοιπά"] = a["site_kwh"] - a[[f"eu_{x}" for x in FAMILIES[:-1]]].sum(axis=1)
            a["city"], a["zone"], a["dataset_type"] = city, C.CITY_ZONE[city], dtype
            annual.append(a)

            # Σ over buildings per hour (stock load) + monthly
            ts = t.select(["building_uid", "timestamp", site]).to_pandas()
            w = ts["building_uid"].astype(str).map(w_city)
            if w.isna().any():
                raise AssertionError(f"{city}/{dtype}: building without city weight")
            ts["w_kwh"] = ts[site].astype("float64") * w.to_numpy()
            g = ts.groupby("timestamp")
            s, sw = g[site].sum().astype("float64"), g["w_kwh"].sum()
            stock.append(pd.DataFrame({"timestamp": s.index, "city": city,
                                       "zone": C.CITY_ZONE[city], "dataset_type": dtype,
                                       "kwh": s.values,          # Σ of the sample
                                       "kwh_w": sw.values}))     # city-stock estimate
            del t, ts
            print(f"  read {city:9s} {dtype}")
    annual = pd.concat(annual, ignore_index=True)
    stock = pd.concat(stock, ignore_index=True)
    return annual, stock, fuel, eu_map, eu_unmatched


def main():
    C.ensure_dirs()
    summary = load_json(C.SUMMARY)
    meta = pd.read_parquet(C.META_V2)
    w_city = city_weights(meta, summary)
    annual, stock, fuel, eu_map, eu_unmatched = collect(w_city)

    annual = annual.merge(meta[["building_uid", "floor_area_m2", "nrel_weight"]],
                          on="building_uid", how="left", validate="one_to_one")
    annual["w_city"] = annual["building_uid"].map(w_city).astype("float64")
    assert annual["w_city"].notna().all(), "city weight missing"
    assert annual["floor_area_m2"].notna().all(), "floor area missing after merge"
    assert len(annual) == 5 * (C.N_RES + C.N_COM), "building count"
    annual["eui"] = annual["site_kwh"] / annual["floor_area_m2"]      # kWh/m²·yr

    # CO2 (kg/year) — electricity by zone (eGRID2018), fuels (EPA Hub)
    annual["co2_electricity"] = annual.get("kwh_electricity", 0) * annual["zone"].map(C.EF_ELEC)
    for f, ef in C.EF_FUEL.items():
        annual[f"co2_{f}"] = annual[f"kwh_{f}"] * ef if f"kwh_{f}" in annual else 0.0
    co2_cols = ["co2_electricity"] + [f"co2_{f}" for f in C.EF_FUEL]
    annual["co2_kg"] = annual[co2_cols].sum(axis=1)
    annual["co2_kg_m2"] = annual["co2_kg"] / annual["floor_area_m2"]
    annual.to_parquet(C.TABLES / "annual_by_building.parquet", index=False)
    co2_check(annual, meta)
    stock.to_parquet(C.TABLES / "stock_hourly.parquet", index=False)

    # consistency of the end-use map
    rest = annual["eu_Λοιπά"].sum() / annual["site_kwh"].sum() * 100
    print(f"  end-use 'Λοιπά' (not mapped) = {rest:.2f} % of site energy; unmatched cols: {eu_unmatched}")

    tables(annual, meta, summary, fuel, rest)
    figures(annual, stock, meta, summary, fuel)
    print("✓ 05_EDA_v2 finished")


def co2_check(annual, meta):
    """
    Cross-check of OUR electricity CO2 (kWh × eGRID2018 subregion factor of
    config.py) against NREL's own ComStock output
    out.emissions.electricity.egrid_2018_subregion..co2e_kg (commercial only;
    CO2e vs CO2, so a ratio close to — not exactly — 1 is expected).
    """
    col = "out.emissions.electricity.egrid_2018_subregion..co2e_kg"
    if col not in meta.columns:
        print("  (no NREL eGRID2018 emission column -> CO2 cross-check skipped)")
        return
    m = annual.merge(meta[["building_uid", col]], on="building_uid", how="left")
    m = m[m["dataset_type"] == "commercial"]
    rows = []
    for city in C.CITIES:
        d = m[m["city"] == city]
        ours, nrel = d["co2_electricity"].sum(), pd.to_numeric(d[col], errors="coerce").sum()
        rows.append({"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "N εμπορικά": len(d),
                     "CO₂ ηλεκτρισμού, εδώ (t)": round(ours / 1000, 1),
                     "CO₂e ηλεκτρισμού, NREL eGRID2018 (t)": round(nrel / 1000, 1),
                     "Λόγος εδώ / NREL": round(ours / nrel, 3) if nrel > 0 else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(C.TABLES / "table_3_5_co2_crosscheck_nrel.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))


# ═════════════════════════════════════════════════════════════════════
# 2. Tables
# ═════════════════════════════════════════════════════════════════════
def wsum(d, col, weighted):
    """Σ col, optionally weighted with the city-stock weight (see city_weights)."""
    return float((d[col] * (d["w_city"] if weighted else 1.0)).sum())


def tables(annual, meta, summary, fuel, rest):
    # 3.1 sample + floor area (m²)
    rows = []
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            m = meta[(meta["city"] == city) & (meta["dataset_type"] == dtype)]["floor_area_m2"]
            rows.append({"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": TYPE_GR[dtype],
                         "N κτιρίων": len(m), "Εμβαδόν διάμεσος (m²)": round(m.median(), 1),
                         "Εμβαδόν Q1 (m²)": round(m.quantile(.25), 1),
                         "Εμβαδόν Q3 (m²)": round(m.quantile(.75), 1),
                         "Συνολικό εμβαδόν δείγματος (m²)": round(m.sum())})
    pd.DataFrame(rows).to_csv(C.TABLES / "table_3_1_sample.csv", index=False, encoding="utf-8-sig")

    # 3.2 energy by zone/type
    rows = []
    for city in C.CITIES:
        for dtype in ("residential", "commercial", "all"):
            d = annual[annual["city"] == city]
            if dtype != "all":
                d = d[d["dataset_type"] == dtype]
            r = {"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": TYPE_GR[dtype], "N": len(d),
                 "EUI διάμεσος κτιρίων (kWh/m²·έτος)": round(d["eui"].median(), 1),
                 "EUI συνολική (Σ kWh / Σ m²)": round(d["site_kwh"].sum() / d["floor_area_m2"].sum(), 1),
                 "CO₂ (kg/m²·έτος, Σ/Σ)": round(d["co2_kg"].sum() / d["floor_area_m2"].sum(), 2),
                 "EUI απόθεμα πόλης, σταθμ. (kWh/m²·έτος)": round(
                     wsum(d, "site_kwh", True) / wsum(d, "floor_area_m2", True), 1),
                 "CO₂ απόθεμα πόλης, σταθμ. (kg/m²·έτος)": round(
                     wsum(d, "co2_kg", True) / wsum(d, "floor_area_m2", True), 2),
                 "Μερίδιο οικιστικών στο εμβαδόν, σταθμ. (%)": round(
                     wsum(d[d["dataset_type"] == "residential"], "floor_area_m2", True)
                     / wsum(d, "floor_area_m2", True) * 100, 1),
                 "Μερίδιο οικιστικών στο εμβαδόν, δείγμα (%)": round(
                     d.loc[d["dataset_type"] == "residential", "floor_area_m2"].sum()
                     / d["floor_area_m2"].sum() * 100, 1)}
            for f in fuel:
                for w, tag in ((True, "σταθμ."), (False, "μη σταθμ.")):
                    r[f"{FUEL_GR[f]} % ({tag})"] = round(
                        wsum(d, f"kwh_{f}", w) / wsum(d, "site_kwh", w) * 100, 1)
            dist = [f"kwh_{f}" for f in ("district_heating", "district_cooling") if f in fuel]
            r["Ενέργεια χωρίς συντελεστή CO₂ (%)"] = round(
                d[dist].sum().sum() / d["site_kwh"].sum() * 100, 2) if dist else 0.0
            rows.append(r)
    t32 = pd.DataFrame(rows)
    t32.to_csv(C.TABLES / "table_3_2_energy_by_zone.csv", index=False, encoding="utf-8-sig")

    # 3.3 HDD / CDD (base 18 °C) from the AMY2018 weather actually used
    wx = weather()
    rows = []
    for city in C.CITIES:
        w = wx[wx["city"] == city]
        r = {"Πόλη": city, "Ζώνη": C.CITY_ZONE[city],
             "Μέση θερμοκρασία (°C)": round(w["temperature_c"].mean(), 1),
             "HDD18 (°C·ημέρες)": round(w["hdd"].sum() / 24),
             "CDD18 (°C·ημέρες)": round(w["cdd"].sum() / 24)}
        for sc, (best, _, _) in C.SCENARIOS.items():
            T = w["temperature_c"] + best
            r[f"HDD18 {sc}"] = round(np.maximum(0, 18 - T).sum() / 24)
            r[f"CDD18 {sc}"] = round(np.maximum(0, T - 18).sum() / 24)
        rows.append(r)
    pd.DataFrame(rows).to_csv(C.TABLES / "table_3_3_hdd_cdd.csv", index=False, encoding="utf-8-sig")

    # 3.4 end uses (kWh/m²·yr, aggregate) — sums to EUI by construction
    rows = []
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            d = annual[(annual["city"] == city) & (annual["dataset_type"] == dtype)]
            A = d["floor_area_m2"].sum()
            r = {"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": TYPE_GR[dtype]}
            for fam in FAMILIES:
                r[f"{fam} (kWh/m²)"] = round(d[f"eu_{fam}"].sum() / A, 1)
            r["Σύνολο (kWh/m²)"] = round(d["site_kwh"].sum() / A, 1)
            rows.append(r)
    pd.DataFrame(rows).to_csv(C.TABLES / "table_3_4_enduse.csv", index=False, encoding="utf-8-sig")

    update_summary("eda", {
        "enduse_unmapped_pct": round(rest, 3),
        "eui_median_res": round(annual.loc[annual.dataset_type == "residential", "eui"].median(), 1),
        "eui_median_com": round(annual.loc[annual.dataset_type == "commercial", "eui"].median(), 1),
        "area_median_res_m2": round(meta.loc[meta.dataset_type == "residential", "floor_area_m2"].median(), 1),
        "area_median_com_m2": round(meta.loc[meta.dataset_type == "commercial", "floor_area_m2"].median(), 1),
        "ef_electricity_kg_kwh": C.EF_ELEC, "ef_fuel_kg_kwh": C.EF_FUEL})


def weather():
    """Weather exactly as joined in 00b (hour-starting), with hdd/cdd."""
    wx = pd.read_parquet(C.WEATHER_V2)          # written by 00b: shifted + hourly means
    wx["hdd"] = np.maximum(0.0, 18.0 - wx["temperature_c"])
    wx["cdd"] = np.maximum(0.0, wx["temperature_c"] - 18.0)
    return wx


# ═════════════════════════════════════════════════════════════════════
# 3. Figures
# ═════════════════════════════════════════════════════════════════════
def figures(annual, stock, meta, summary, fuel):
    fig_funnel(summary)
    fig_split()
    fig_hdd_cdd()
    for dtype, n_char, n_mix, n_co2, n_zone, n_temp, n_eu in (
            ("residential", "3_6", "3_7", "3_8", "3_9", "3_10", "3_18"),
            ("commercial", "3_11", "3_12", "3_13", "3_14", "3_15", "3_19")):
        a = annual[annual["dataset_type"] == dtype]
        m = meta[meta["dataset_type"] == dtype]
        fig_characteristics(m, dtype, f"fig_{n_char}_{dtype[:3]}_characteristics.png")
        fig_fuel_mix(a, dtype, fuel, f"fig_{n_mix}_{dtype[:3]}_fuel_mix.png")
        fig_co2(a, dtype, f"fig_{n_co2}_{dtype[:3]}_co2_fuel.png")
        fig_eui_zone(a, dtype, f"fig_{n_zone}_{dtype[:3]}_eui_by_zone.png")
        fig_temporal(stock[stock["dataset_type"] == dtype], a, dtype,
                     f"fig_{n_temp}_{dtype[:3]}_temporal.png")
        fig_enduse(a, dtype, f"fig_{n_eu}_enduse_{dtype}.png")
        sample_profile(m, a, dtype, "3_20" if dtype == "residential" else "3_21")
    data_quality()


def fig_funnel(summary):
    """Fig 3.1 — NREL national -> climate zone -> county -> sample (log scale)."""
    f = summary["funnel"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, dtype, key, n_s in ((axes[0], "residential", "res", C.N_RES),
                                (axes[1], "commercial", "com", C.N_COM)):
        x = np.arange(len(C.CITIES))
        stages = [("Ζώνη ASHRAE", [f["by_city"][c][f"zone_{key}"] for c in C.CITIES]),
                  ("Κομητεία πόλης", [f["by_city"][c][f"city_{key}"] or np.nan for c in C.CITIES]),
                  ("Δείγμα", [n_s] * len(C.CITIES))]
        for i, (lab, vals) in enumerate(stages):
            b = ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=lab,
                       color=["#c6dbef", "#6baed6", "#08519c"][i])
            for rect, v in zip(b, vals):
                if np.isfinite(v):
                    ax.text(rect.get_x() + rect.get_width() / 2, v * 1.08, gr(v, 0),
                            ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_yscale("log")
        allv = [v for _, vals in stages for v in vals if np.isfinite(v)]
        ax.set_ylim(min(allv) * 0.5, max(allv) * 4)          # room for the labels
        log_axis_plain(ax)
        ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES])
        tot = f[f"nrel_total_{key}"]
        ax.set_title(f"{TYPE_GR[dtype]} — σύνολο NREL: {gr(tot, 0)} κτίρια")
        ax.set_ylabel("Αριθμός κτιρίων (λογαριθμική κλίμακα)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Χωνί δειγματοληψίας — δείγμα {gr(5 * (C.N_RES + C.N_COM), 0)} κτιρίων "
                 f"({C.N_RES} οικιστικά + {C.N_COM} εμπορικά ανά πόλη)")
    save_fig(fig, "fig_3_1_sampling_funnel.png")


def fig_split():
    """Fig 3.4 — daily calendar of train / validation / test (+ warm-up week)."""
    days = pd.date_range("2018-01-01", "2018-12-31", freq="D")
    lab = split_labels(pd.Series(days))
    code = lab.map({"train": 0, "val": 1, "test": 2}).to_numpy().astype(float)
    code[:7] = 3                                    # first 168 h: lag warm-up (dropped)
    grid = np.full((12, 31), np.nan)
    for d, c in zip(days, code):
        grid[d.month - 1, d.day - 1] = c
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#9ecae1", "#fdae6b", "#e6550d", "#d9d9d9"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_yticks(range(12), MONTHS_GR)
    ax.set_xticks(range(0, 31, 2), [str(i + 1) for i in range(0, 31, 2)])
    ax.set_xlabel("Ημέρα του μήνα")
    ax.grid(False)
    n = pd.Series(code).value_counts()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap(0), label=f"Εκπαίδευση ({int(n.get(0, 0))} ημ.)"),
                       Patch(color=cmap(1), label=f"Επικύρωση ({int(n.get(1, 0))} ημ.)"),
                       Patch(color=cmap(2), label=f"Έλεγχος ({int(n.get(2, 0))} ημ.)"),
                       Patch(color=cmap(3), label="Προθέρμανση lag (7 ημ.)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    ax.set_title(f"Χρονικός διαχωρισμός: τελευταίες {C.TEST_DAYS} ημέρες κάθε μήνα = έλεγχος, "
                 f"οι {C.VAL_DAYS} προηγούμενες = επικύρωση")
    save_fig(fig, "fig_3_4_temporal_split.png")


def fig_hdd_cdd():
    """Fig 3.17 — monthly HDD/CDD heatmaps; Fig 3.5 — annual HDD/CDD per scenario."""
    wx = weather()
    wx["month"] = wx["timestamp"].dt.month
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.8))
    for ax, col, cm, ttl in ((axes[0], "hdd", "Blues", "HDD18"), (axes[1], "cdd", "Reds", "CDD18")):
        m = (wx.groupby(["city", "month"])[col].sum() / 24).unstack().reindex(C.CITIES)
        im = ax.imshow(m.values, cmap=cm, aspect="auto")
        ax.set_yticks(range(5), [f"{c} ({C.CITY_ZONE[c]})" for c in C.CITIES])
        ax.set_xticks(range(12), MONTHS_GR, fontsize=8)
        ax.grid(False)
        for i in range(5):
            for j in range(12):
                v = m.values[i, j]
                ax.text(j, i, gr(v, 0), ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanmax(m.values) * 0.6 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("°C·ημέρες")
        ax.set_title(f"{ttl} ανά μήνα (AMY2018, βάση 18 °C)")
    save_fig(fig, "fig_3_17_hdd_cdd_heatmap.png")

    t = pd.read_csv(C.TABLES / "table_3_3_hdd_cdd.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    labels = ["2018"] + list(C.SCENARIOS)
    for ax, kind in ((axes[0], "HDD18"), (axes[1], "CDD18")):
        x = np.arange(len(C.CITIES))
        w = 0.8 / len(labels)
        for i, lab in enumerate(labels):
            col = f"{kind} (°C·ημέρες)" if lab == "2018" else f"{kind} {lab}"
            ax.bar(x + (i - len(labels) / 2 + 0.5) * w, t[col], w, label=lab,
                   color=plt.cm.viridis(i / (len(labels) - 1)))
        ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES])
        ax.set_ylabel("°C·ημέρες")
        gr_axis(ax)
        ax.set_title(f"{kind} (βάση 18 °C)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Βαθμοημέρες AMY2018 και με ΔT κάθε σεναρίου (κεντρική εκτίμηση IPCC AR6, "
                 "μεταβολή έναντι 1995–2014)", fontsize=10)
    save_fig(fig, "fig_3_5_hdd_cdd_scenarios.png")


def fig_characteristics(m, dtype, name):
    """Fig 3.6 / 3.11 — floor area (m²) distribution and building types."""
    tcol = "in.geometry_building_type_recs" if dtype == "residential" else "in.comstock_building_type"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    bins = np.logspace(np.log10(m["floor_area_m2"].min()), np.log10(m["floor_area_m2"].max()), 30)
    ax.hist(m["floor_area_m2"], bins=bins, color="#3182bd", edgecolor="white")
    ax.set_xscale("log")
    log_axis_plain(ax, "x")
    # wide ranges (commercial: 93 … 92.903 m², three decades) -> only 1·10^k
    # ticks, otherwise the labels 10.000/20.000/50.000 overlap
    if m["floor_area_m2"].max() / m["floor_area_m2"].min() > 300:
        from matplotlib.ticker import LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.axvline(m["floor_area_m2"].median(), color="k", ls="--",
               label=f"διάμεσος {gr(m['floor_area_m2'].median(), 0)} m²")
    ax.set_xlabel("Εμβαδόν (m², λογαριθμική κλίμακα)")
    ax.set_ylabel("Αριθμός κτιρίων")
    ax.legend(fontsize=8)
    ax.set_title("Κατανομή εμβαδού")
    ax = axes[1]
    data = [m.loc[m["city"] == c, "floor_area_m2"] for c in C.CITIES]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, 6), [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES], fontsize=8)
    ax.set_ylabel("Εμβαδόν (m²)")
    ax.set_title("Εμβαδόν ανά πόλη (χωρίς ακραίες τιμές)")
    gr_axis(ax)
    ax = axes[2]
    if tcol in m.columns and m[tcol].notna().any():
        vc = m[tcol].astype(str).value_counts().head(12)[::-1]
        ax.barh(vc.index, vc.values / len(m) * 100, color="#6baed6")
        ax.set_xlabel("% του δείγματος")
        ax.set_title("Τύπος κτιρίου")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_yticks(range(len(vc)), [t if len(t) <= 36 else t[:34] + "…" for t in vc.index])
        gr_axis(ax, "x")
    else:
        ax.axis("off")
    src = load_json(C.SUMMARY).get("area_source_residential", "")
    note = f"Πηγή εμβαδού οικιστικών: {src}" if dtype == "residential" else "Πηγή εμβαδού: in.sqft"
    fig.suptitle(f"{TYPE_GR[dtype]}: χαρακτηριστικά δείγματος (N = {gr(len(m), 0)}) — {note}",
                 fontsize=10)
    save_fig(fig, name)


def fig_fuel_mix(a, dtype, fuel, name):
    """Fig 3.7 / 3.12 — fuel shares per zone (NREL-weighted)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(C.CITIES))
    bottom = np.zeros(len(C.CITIES))
    for f in fuel:
        sh = np.array([wsum(a[a.city == c], f"kwh_{f}", True) /
                       wsum(a[a.city == c], "site_kwh", True) * 100 for c in C.CITIES])
        if sh.max() < 0.05:
            continue
        ax.bar(x, sh, 0.6, bottom=bottom, label=FUEL_GR[f], color=FUEL_COLORS[f])
        for i, v in enumerate(sh):
            if v >= 4:
                ax.text(x[i], bottom[i] + v / 2, f"{gr(v, 1)}%", ha="center", va="center",
                        fontsize=8, color="white")
        bottom += sh
    ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES])
    ax.set_ylabel("% της ετήσιας κατανάλωσης τελικής ενέργειας")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(f"{TYPE_GR[dtype]}: μείγμα καυσίμων (σταθμισμένο με βάρη NREL)")
    save_fig(fig, name)


def fig_co2(a, dtype, name):
    """Fig 3.8 / 3.13 — CO2 intensity (kg/m²·yr) by fuel and zone."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(C.CITIES))
    bottom = np.zeros(len(C.CITIES))
    for f in ["electricity"] + list(C.EF_FUEL):
        v = np.array([a.loc[a.city == c, f"co2_{f}"].sum() / a.loc[a.city == c, "floor_area_m2"].sum()
                      for c in C.CITIES])
        if v.max() <= 0:
            continue
        ax.bar(x, v, 0.6, bottom=bottom, label=FUEL_GR[f], color=FUEL_COLORS[f])
        bottom += v
    for i, v in enumerate(bottom):
        ax.text(x[i], v, gr(v, 1), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, bottom.max() * 1.10)
    ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})\nEF {gr(C.EF_ELEC[C.CITY_ZONE[c]], 3)}"
                      for c in C.CITIES], fontsize=8)
    ax.set_ylabel("kg CO₂ / m² · έτος")
    gr_axis(ax)
    ax.legend(fontsize=8)
    ax.set_title(f"{TYPE_GR[dtype]}: ένταση εκπομπών CO₂ ανά καύσιμο "
                 f"(ηλεκτρισμός: eGRID2018, καύσιμα: EPA)")
    save_fig(fig, name)


def fig_eui_zone(a, dtype, name):
    """Fig 3.9 / 3.14 — per-building annual EUI distribution per zone."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [a.loc[a.city == c, "eui"] for c in C.CITIES]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for p, c in zip(bp["boxes"], C.CITIES):
        p.set_facecolor(ZONE_COLORS[C.CITY_ZONE[c]])
        p.set_alpha(0.6)
    for i, d in enumerate(data):
        ax.text(i + 1, d.median(), gr(d.median(), 0), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(1, 6), [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES])
    ax.set_ylabel("Ετήσια EUI (kWh/m² · έτος)")
    gr_axis(ax)
    ax.set_title(f"{TYPE_GR[dtype]}: ετήσια ένταση κατανάλωσης ανά κτίριο (διάμεσος, IQR)")
    save_fig(fig, name)


def fig_temporal(st, a, dtype, name):
    """Fig 3.10 / 3.15 — monthly EUI and mean daily profile (winter vs summer), per m²."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for c in C.CITIES:
        s = st[st.city == c].copy()
        A = a.loc[a.city == c, "floor_area_m2"].sum()
        col = ZONE_COLORS[C.CITY_ZONE[c]]
        mon = s.groupby(s["timestamp"].dt.month)["kwh"].sum() / A
        axes[0].plot(range(1, 13), mon.values, marker="o", color=col, label=f"{c} ({C.CITY_ZONE[c]})")
        for ax, sea in ((axes[1], "Χειμώνας"), (axes[2], "Καλοκαίρι")):
            ss = s[s["timestamp"].dt.month.map(SEASON) == sea]
            prof = ss.groupby(ss["timestamp"].dt.hour)["kwh"].mean() / A * 1000
            ax.plot(prof.index, prof.values, color=col)
    axes[0].set_xticks(range(1, 13), MONTHS_GR, fontsize=8)
    axes[0].set_ylabel("kWh / m² ανά μήνα")
    axes[0].set_title("Μηνιαία ένταση")
    axes[0].legend(fontsize=7)
    for ax, sea in ((axes[1], "Χειμώνας (Δεκ–Φεβ)"), (axes[2], "Καλοκαίρι (Ιουν–Αυγ)")):
        ax.set_xlabel("Ώρα της ημέρας")
        ax.set_ylabel("Wh / m² ανά ώρα")
        ax.set_xticks(range(0, 24, 3))
        ax.set_title(f"Μέσο ημερήσιο προφίλ — {sea}")
    for ax in axes:
        gr_axis(ax)
    fig.suptitle(f"{TYPE_GR[dtype]}: χρονικά μοτίβα (Σ κατανάλωσης / Σ εμβαδού ανά πόλη)")
    save_fig(fig, name)


def fig_enduse(a, dtype, name):
    """Fig 3.18 / 3.19 — end-use intensities per zone (sum = site EUI)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(C.CITIES))
    bottom = np.zeros(len(C.CITIES))
    for fam, col in zip(FAMILIES, FAM_COLORS):
        v = np.array([a.loc[a.city == c, f"eu_{fam}"].sum() / a.loc[a.city == c, "floor_area_m2"].sum()
                      for c in C.CITIES])
        ax.bar(x, v, 0.6, bottom=bottom, label=fam, color=col)
        bottom += v
    tot = np.array([a.loc[a.city == c, "site_kwh"].sum() / a.loc[a.city == c, "floor_area_m2"].sum()
                    for c in C.CITIES])
    assert np.allclose(bottom, tot, rtol=1e-6), "end-use stack != site EUI"
    for i, v in enumerate(tot):
        ax.text(x[i], v, gr(v, 0), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, tot.max() * 1.10)
    ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES])
    ax.set_ylabel("kWh / m² · έτος")
    gr_axis(ax)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(f"{TYPE_GR[dtype]}: τελικές χρήσεις ενέργειας (άθροισμα = συνολική EUI)")
    save_fig(fig, name)


# ═════════════════════════════════════════════════════════════════════
# 4. Sample profile per city & data quality (Ch.3 §«Δείγμα» / «Ποιότητα»)
# ═════════════════════════════════════════════════════════════════════
GR_LABELS = {
    # residential building type (RECS)
    "Single-Family Detached": "Μονοκατοικία", "Single-Family Attached": "Μεζονέτα / σειρά",
    "Multi-Family with 2 - 4 Units": "Πολυκατοικία 2–4", "Multi-Family with 5+ Units": "Πολυκατοικία 5+",
    "Mobile Home": "Κινητή κατοικία",
    # heating fuel (ResStock / ComStock spellings)
    "Natural Gas": "Φυσικό αέριο", "NaturalGas": "Φυσικό αέριο", "Electricity": "Ηλεκτρισμός",
    "Fuel Oil": "Πετρέλαιο", "FuelOil": "Πετρέλαιο", "Propane": "Προπάνιο", "None": "Χωρίς θέρμανση",
    "Other Fuel": "Άλλο καύσιμο", "Wood": "Ξύλο", "DistrictHeating": "Τηλεθέρμανση",
    # tenure / vacancy
    "Owner": "Ιδιοκατοίκηση", "Renter": "Ενοικίαση", "Not Available": "Κενή κατοικία",
    # ComStock building types
    "SmallOffice": "Μικρό γραφείο", "MediumOffice": "Μεσαίο γραφείο", "LargeOffice": "Μεγάλο γραφείο",
    "RetailStandalone": "Κατάστημα", "RetailStripmall": "Εμπορικό κέντρο", "Warehouse": "Αποθήκη",
    "Hospital": "Νοσοκομείο", "Outpatient": "Κλινική", "PrimarySchool": "Δημοτικό σχολείο",
    "SecondarySchool": "Δευτεροβάθμιο σχολείο", "QuickServiceRestaurant": "Ταχυφαγείο",
    "FullServiceRestaurant": "Εστιατόριο", "SmallHotel": "Μικρό ξενοδοχείο", "LargeHotel": "Μεγάλο ξενοδοχείο",
}
PROFILE_VARS = {
    "residential": [("in.geometry_building_type_recs", "Τύπος κατοικίας"), ("in.vintage", "Περίοδος κατασκευής"),
                    ("in.heating_fuel", "Καύσιμο θέρμανσης"), ("in.tenure", "Καθεστώς χρήσης"),
                    ("in.geometry_stories", "Όροφοι"), ("in.bedrooms", "Υπνοδωμάτια")],
    "commercial": [("in.comstock_building_type", "Τύπος κτιρίου"), ("in.vintage", "Περίοδος κατασκευής"),
                   ("in.heating_fuel", "Καύσιμο θέρμανσης"), ("in.hvac_system_type", "Σύστημα HVAC")],
}


def _order_key(v):
    """Chronological / numeric order for vintage, stories, bedrooms."""
    import re as _re
    m = _re.search(r"\d+", str(v))
    n = int(m.group()) if m else 10 ** 6
    if str(v).startswith("<") or str(v).lower().startswith(("before", "πριν")):
        n -= 1
    return n


def _shares(m, col, top=7):
    """% per city for one categorical column; rare categories -> «Λοιπά»."""
    x = m[col].astype(str).map(lambda v: GR_LABELS.get(v, v)
                               .replace("Before ", "Πριν το ").replace(" to ", "–"))
    keep = x.value_counts().index[:top]
    x = x.where(x.isin(keep), "Λοιπά")
    t = pd.crosstab(m["city"], x, normalize="index").reindex(C.CITIES) * 100
    numeric = col in ("in.vintage", "in.geometry_stories", "in.bedrooms")
    cols = sorted(t.columns, key=_order_key) if numeric else list(t.sum().sort_values(ascending=False).index)
    if "Λοιπά" in cols:
        cols = [c for c in cols if c != "Λοιπά"] + ["Λοιπά"]
    return t[cols]


def sample_profile(m, a, dtype, num):
    """Fig 3.20 / 3.21 + table_3_6: characteristics of the sample per city."""
    vars_ = [(c, l) for c, l in PROFILE_VARS[dtype] if c in m.columns and m[c].notna().any()]
    extra = 1  # last panel: income (res) or floor-area per city (com)
    n = len(vars_) + extra
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    long = []
    for ax, (col, lab) in zip(axes, vars_):
        t = _shares(m, col)
        bottom = np.zeros(len(t))
        cmap = plt.cm.tab20 if t.shape[1] > 10 else plt.cm.tab10
        for i, c in enumerate(t.columns):
            ax.bar(range(len(t)), t[c].values, 0.7, bottom=bottom, label=c, color=cmap(i % cmap.N))
            for j, v in enumerate(t[c].values):
                if v >= 8:
                    ax.text(j, bottom[j] + v / 2, f"{gr(v, 0)}%", ha="center", va="center", fontsize=6.5)
            bottom += t[c].values
        ax.set_xticks(range(len(t)), [f"{c}\n({C.CITY_ZONE[c]})" for c in t.index], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% κτιρίων της πόλης")
        ax.set_title(lab)
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
        long.append(t.reset_index().melt(id_vars="city", var_name="Κατηγορία", value_name="%")
                    .assign(Μεταβλητή=lab))
    ax = axes[len(vars_)]
    if dtype == "residential" and "income_usd" in m.columns:
        occ = m[pd.to_numeric(m["income_usd"], errors="coerce") > 0]
        ax.boxplot([occ.loc[occ.city == c, "income_usd"] / 1000 for c in C.CITIES], showfliers=False)
        ax.set_ylabel("Εισόδημα (χιλ. $, κατοικημένα)")
        ax.set_title("Αντιπροσωπευτικό εισόδημα NREL (κενές κατοικίες εκτός)")
    else:
        ax.boxplot([a.loc[a.city == c, "floor_area_m2"] for c in C.CITIES], showfliers=False)
        ax.set_ylabel("Εμβαδόν (m²)")
        ax.set_title("Εμβαδόν ανά πόλη")
    ax.set_xticks(range(1, 6), [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES], fontsize=8)
    gr_axis(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{TYPE_GR[dtype]}: χαρακτηριστικά δείγματος ανά πόλη (N = {gr(len(m), 0)})")
    save_fig(fig, f"fig_{num}_{dtype[:3]}_sample_by_city.png")
    if long:
        (pd.concat(long).rename(columns={"city": "Πόλη"})
         .assign(**{"%": lambda d: d["%"].round(1)})
         .to_csv(C.TABLES / f"table_3_6_sample_profile_{dtype}.csv", index=False, encoding="utf-8-sig"))


def data_quality():
    """Fig 3.22 + table_3_0: data-quality check of STEP A (before modelling)."""
    path = C.V2_DIR / "qc_by_building.parquet"
    if not path.exists():
        print("  (qc_by_building.parquet not found -> run STEP A again)")
        return
    q = pd.read_parquet(path)
    t = (q.groupby(["city", "dataset_type"])
          .agg(**{"N κτιρίων": ("building_uid", "size"),
                  "Διπλές γραμμές που αφαιρέθηκαν": ("dup_rows", "sum"),
                  "Ώρες με μηδενική κατανάλωση": ("zero_h", "sum"),
                  f"Κτίρια με κενό ≥ {C.QC_MAX_ZERO_RUN_H} h": ("flag_gap", "sum"),
                  f"Κτίρια με παγωμένη σειρά ≥ {C.QC_MAX_FLAT_RUN_H} h": ("flag_flat", "sum"),
                  "Κτίρια με αρνητικές τιμές": ("flag_negative", "sum"),
                  "Κτίρια με ακραία EUI (|z|>3,5)": ("flag_eui_outlier", "sum")})
          .reindex(pd.MultiIndex.from_product([C.CITIES, ["residential", "commercial"]]))
          .reset_index().rename(columns={"level_0": "Πόλη", "level_1": "Τύπος"}))
    t["Τύπος"] = t["Τύπος"].map(TYPE_GR)
    t.to_csv(C.TABLES / "table_3_0_data_quality.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axes[0]
    flags = ["flag_gap", "flag_flat", "flag_negative", "flag_eui_outlier"]
    labels = ["Κενό (μηδενικά)", "Παγωμένη σειρά", "Αρνητικές τιμές", "Ακραία EUI"]
    x = np.arange(len(C.CITIES))
    for i, (f, l) in enumerate(zip(flags, labels)):
        v = [int(q.loc[q.city == c, f].sum()) for c in C.CITIES]
        ax.bar(x + (i - 1.5) * 0.2, v, 0.2, label=l)
    ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES], fontsize=8)
    ax.set_ylabel("Αριθμός κτιρίων")
    ax.set_title("Κτίρια με σημαία ελέγχου ανά πόλη")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.hist(q["max_zero_run"], bins=50, color="#6baed6")
    ax.axvline(C.QC_MAX_ZERO_RUN_H, color="r", ls="--", lw=1, label=f"όριο {C.QC_MAX_ZERO_RUN_H} h")
    ax.set_yscale("symlog")
    ax.set_xlabel("Μέγιστη συνεχόμενη διάρκεια μηδενικής κατανάλωσης (h)")
    ax.set_ylabel("Αριθμός κτιρίων")
    ax.legend(fontsize=7)
    ax.set_title("Έλεγχος κενών στις χρονοσειρές")
    ax = axes[2]
    for dtype, col in (("residential", "#41ab5d"), ("commercial", "#3182bd")):
        z = q.loc[q.dataset_type == dtype, "eui_robust_z"].dropna()
        ax.hist(z, bins=60, alpha=0.6, color=col, label=TYPE_GR[dtype])
    for v in (-3.5, 3.5):
        ax.axvline(v, color="r", ls="--", lw=1)
    ax.set_xlabel("Robust z-score του log(EUI) εντός πόλης × τύπου")
    ax.set_ylabel("Αριθμός κτιρίων")
    ax.legend(fontsize=7)
    ax.set_title("Ακραίες τιμές ετήσιας EUI")
    from matplotlib.ticker import MaxNLocator
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    gr_axis(axes[0], "y", nd=0)
    for a_ in axes[1:]:
        gr_axis(a_, "both")
    fig.suptitle("Έλεγχος ποιότητας δεδομένων ανά κτίριο (Βήμα Α, πριν από την εκπαίδευση)")
    save_fig(fig, "fig_3_22_data_quality.png")


if __name__ == "__main__":
    main()
