#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_FIGURES_v2.py  —  Chapter 4 & 5 figures, ONLY from the v2 tables
Thesis MIS24011 — Zygouris Aristotelis

Nothing is simulated or typed by hand (K1, K2, #15): every figure reads
a CSV/parquet written by 02–06. Run after 06.

Fixes
  K1  Fig 4.1 = the real Optuna trials (optuna_trials_*.csv).
  K8  Seasonal figure: y-axis starts at 0 and is set by the data.
  #12 CO2 heatmap with colour bar and units.
  Residual histogram: x-axis clipped to the 0,5–99,5 percentile and the
      share of residuals outside is printed on the figure.
  Load profiles per m² (comparable between residential & commercial).
"""
import json

import numpy as np
import pandas as pd

import config as C
from common import (FUEL_GR, TYPE_GR, ZONE_COLORS, gr, gr_axis, save_fig, setup_mpl)

plt = setup_mpl()
T = C.TABLES
MODEL_COLORS = {"Persistence (t−1h)": "#969696", "Seasonal Naive (t−168h)": "#bdbdbd",
                "Ridge Regression": "#9e9ac8", "Random Forest": "#41ab5d", "XGBoost": "#e6550d"}
SEASONS = ["Χειμώνας", "Άνοιξη", "Καλοκαίρι", "Φθινόπωρο"]
CITY_LAB = [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES]


def csv(name):
    return pd.read_csv(T / name)


# ── Chapter 4 ────────────────────────────────────────────────────────
def fig_4_1_optuna():
    names = [("unified", "Ενιαίο μοντέλο"), ("res", "Οικιστικά"), ("com", "Εμπορικά")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (n, lab) in zip(axes, names):
        d = csv(f"optuna_trials_{n}.csv")
        d = d[d["state"] == "COMPLETE"].sort_values("number")
        best = json.load(open(C.MODELS / f"best_params_{n}.json", encoding="utf-8"))
        ax.scatter(d["number"], d["value"], s=18, color="#6baed6", label="Δοκιμή (trial)")
        ax.plot(d["number"], d["value"].cummin(), color="#e6550d", lw=2, label="Καλύτερη έως τότε")
        ax.set_xlabel("Αριθμός δοκιμής")
        ax.set_ylabel("RMSE επικύρωσης (kWh/h)")
        ax.set_title(f"{lab}: {len(d)} δοκιμές — βέλτιστο {gr(d['value'].min(), 4)}\n"
                     f"δέντρα {best['n_estimators']}, max_depth {best['max_depth']}, "
                     f"lr {gr(best['learning_rate'], 3)}", fontsize=9)
        gr_axis(ax, nd=4)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
    fig.suptitle("Βελτιστοποίηση υπερπαραμέτρων XGBoost (Optuna TPE) στο σύνολο ΕΠΙΚΥΡΩΣΗΣ")
    save_fig(fig, "fig_4_1_optuna_optimization.png")


def fig_4_2_comparison():
    te, oo = csv("table_4_4_benchmark_test.csv"), csv("table_4_4_benchmark_oos.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    models = te["Μοντέλο"].tolist()
    x = np.arange(len(models))
    for ax, col, ttl in ((axes[0], "RMSE (kWh/h)", "RMSE (μικρότερο = καλύτερο)"),
                         (axes[1], "CV(RMSE) (%)", "CV(RMSE) % (ASHRAE G14)"),
                         (axes[2], "Skill vs Persistence", "Skill score έναντι Persistence")):
        v1 = te.set_index("Μοντέλο").loc[models, col]
        v2 = oo.set_index("Μοντέλο").loc[models, col]
        ax.bar(x - 0.2, v1, 0.4, label="Έλεγχος 2A–5A", color="#3182bd")
        ax.bar(x + 0.2, v2, 0.4, label="Εκτός δείγματος 5B", color="#fd8d3c")
        for xi, a, b in zip(x, v1, v2):
            ax.text(xi - 0.2, a, gr(a, 3), ha="center", va="bottom", fontsize=6.5, rotation=90)
            ax.text(xi + 0.2, b, gr(b, 3), ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(x, [m.replace(" (", "\n(") for m in models], fontsize=7, rotation=30, ha="right")
        ax.set_title(ttl)
        ax.axhline(0, color="k", lw=0.6)
        # headroom for the rotated value labels (they must not reach the title)
        hi, lo = max(v1.max(), v2.max(), 0), min(v1.min(), v2.min(), 0)
        span = hi - lo
        ax.set_ylim(lo - (0.18 * span if lo < 0 else 0), hi + 0.22 * span)
        gr_axis(ax)
    axes[0].legend(fontsize=8, loc="upper right")   # right side is free of labels
    save_fig(fig, "fig_4_2_model_comparison.png")


def fig_4_3_actual_vs_pred():
    p = pd.read_parquet(C.OUT_DIR / "predictions_test.parquet")
    rng = np.random.RandomState(C.SEED)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, dtype in zip(axes, ("residential", "commercial")):
        d = p[p["dataset_type"] == dtype]
        uid = rng.choice(np.sort(d["building_uid"].astype(str).unique()))
        d = d[d["building_uid"].astype(str) == uid].sort_values("timestamp")
        d = d[d["timestamp"].dt.month == 7]                 # one test window (July)
        ax.plot(d["timestamp"], d["y"], color="k", lw=2, label="Πραγματική")
        for m in ("Persistence (t−1h)", "Random Forest", "XGBoost"):
            ax.plot(d["timestamp"], d[m], lw=1.2, color=MODEL_COLORS[m], label=m)
        ax.set_ylabel("kWh / h")
        ax.set_title(f"{TYPE_GR[dtype]}: κτίριο {uid} (τυχαία επιλογή, seed {C.SEED}), "
                     f"εβδομάδα ελέγχου Ιουλίου")
        gr_axis(ax)
        ax.legend(fontsize=8, ncol=4)
    save_fig(fig, "fig_4_3_actual_vs_predicted.png")


def fig_4_4_load_profiles():
    st = pd.read_parquet(T / "stock_hourly.parquet")
    ann = pd.read_parquet(T / "annual_by_building.parquet")
    area = ann.groupby(["city", "dataset_type"])["floor_area_m2"].sum()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for i, dtype in enumerate(("residential", "commercial")):
        for j, (sea, months) in enumerate((("Χειμώνας", (12, 1, 2)), ("Καλοκαίρι", (6, 7, 8)))):
            ax = axes[i, j]
            for c in C.CITIES:
                s = st[(st.city == c) & (st.dataset_type == dtype) &
                       st["timestamp"].dt.month.isin(months)]
                prof = s.groupby(s["timestamp"].dt.hour)["kwh"].mean() / area[(c, dtype)] * 1000
                ax.plot(prof.index, prof.values, color=ZONE_COLORS[C.CITY_ZONE[c]],
                        lw=2, label=f"{c} ({C.CITY_ZONE[c]})")
            ax.set_title(f"{TYPE_GR[dtype]} — {sea}")
            ax.set_ylabel("W / m² (μέση ωριαία ζήτηση)")
            gr_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("Ώρα της ημέρας")
        ax.set_xticks(range(0, 24, 3))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Μέσα ημερήσια προφίλ φορτίου ανά m² (Σ φορτίου / Σ εμβαδού δείγματος)")
    save_fig(fig, "fig_4_4_urban_load_profiles.png")


def fig_4_4b_seasonal():
    s = csv("table_4_4b_seasonal.csv")
    models = list(dict.fromkeys(s["Μοντέλο"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(SEASONS))
    w = 0.8 / len(models)
    for ax, col in ((axes[0], "RMSE (kWh/h)"), (axes[1], "R²")):
        for i, m in enumerate(models):
            v = s[s["Μοντέλο"] == m].set_index("Εποχή").loc[SEASONS, col]
            ax.bar(x + (i - len(models) / 2 + 0.5) * w, v, w, label=m,
                   color=MODEL_COLORS.get(m, None))
        ax.set_xticks(x, SEASONS)
        ax.set_ylabel(col)
        if col == "R²":
            ax.set_ylim(min(0.0, s[col].min() - 0.05), 1.0)   # full honest scale
        else:
            ax.set_ylim(0, s[col].max() * 1.08)
        gr_axis(ax)
    axes[0].legend(fontsize=7)
    axes[1].set_title("R² ανά εποχή (σύνολο ελέγχου)")
    axes[0].set_title("RMSE ανά εποχή (σύνολο ελέγχου)")
    save_fig(fig, "fig_4_4b_seasonal_performance.png")


def fig_4_4c_by_type():
    d = csv("table_4_4c_by_type.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, dtype in zip(axes, ("residential", "commercial")):
        s = d[d["Τύπος"] == dtype]
        labels = [f"{m}\n[{st}]" for m, st in zip(s["Μοντέλο"], s["Σύνολο"])]
        ax.barh(labels, s["CV(RMSE) (%)"], color=["#e6550d" if "unified" in m else "#969696"
                                                  if m == "Persistence" else "#3182bd"
                                                  for m in s["Μοντέλο"]])
        for i, v in enumerate(s["CV(RMSE) (%)"]):
            ax.text(v, i, " " + gr(v, 1), va="center", fontsize=8)
        ax.set_xlabel("CV(RMSE) %")
        ax.set_title(TYPE_GR[dtype])
        ax.tick_params(axis="y", labelsize=7)
        gr_axis(ax, "x")
    fig.suptitle("Ενιαίο έναντι ξεχωριστών μοντέλων ανά τύπο κτιρίου")
    save_fig(fig, "fig_4_4c_unified_vs_separate.png")


def fig_4_5_diversity():
    d = csv("table_4_5_diversity.csv")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(C.CITIES))
    for i, t in enumerate(("Οικιστικά", "Εμπορικά", "Σύνολο")):
        v = d[d["Τύπος"] == t].set_index("Πόλη").loc[C.CITIES, "Συντελεστής ετεροχρονισμού DF"]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, label=t, color=["#41ab5d", "#3182bd", "#636363"][i])
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * 0.27, vi, gr(vi, 2), ha="center", va="bottom", fontsize=7)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_ylim(0, d["Συντελεστής ετεροχρονισμού DF"].max() * 1.25)
    ax.set_xticks(x, CITY_LAB)
    ax.set_ylabel("DF = Σ ατομικών αιχμών / αιχμή συνόλου")
    gr_axis(ax)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.set_title("Συντελεστής ετεροχρονισμού ανά ζώνη (ωριαία δεδομένα όλων των κτιρίων)")
    save_fig(fig, "fig_4_5_diversity_factor.png")


def fig_4_6_importance():
    g = pd.read_csv(T / "feature_importance_gain.csv", index_col=0)["gain_share"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    g2 = g.sort_values()
    axes[0].barh(g2.index, g2.values, color="#3182bd")
    for i, v in enumerate(g2.values):
        axes[0].text(v, i, " " + gr(v, 1) + "%", va="center", fontsize=7)
    axes[0].set_xlabel("% συνολικού gain")
    axes[0].set_title("Σημαντικότητα χαρακτηριστικών (gain, ενιαίο XGBoost)")
    gr_axis(axes[0], "x")
    cum = g.sort_values(ascending=False).cumsum()
    axes[1].plot(range(1, len(cum) + 1), cum.values, marker="o")
    axes[1].axhline(90, color="r", ls="--", lw=0.8)
    n90 = int((cum < 90).sum()) + 1
    axes[1].set_title(f"Αθροιστική σημαντικότητα — {n90} χαρακτηριστικά καλύπτουν το 90%")
    axes[1].set_xlabel("Πλήθος χαρακτηριστικών (φθίνουσα σειρά)")
    axes[1].set_ylabel("% αθροιστικού gain")
    from matplotlib.ticker import MaxNLocator
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    gr_axis(axes[1])
    save_fig(fig, "fig_4_6_feature_importance.png")


def fig_4_7_errors():
    p = pd.read_parquet(C.OUT_DIR / "predictions_test.parquet")
    r = (p["y"] - p["XGBoost"]).to_numpy()
    lo, hi = np.percentile(r, [0.5, 99.5])
    out = np.mean((r < lo) | (r > hi)) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    axes[0].hist(r[(r >= lo) & (r <= hi)], bins=80, color="#3182bd")
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("Υπόλοιπο y − ŷ (kWh/h)")
    axes[0].set_ylabel("Συχνότητα")
    axes[0].set_title(f"Υπόλοιπα XGBoost (εμφάνιση 0,5–99,5 εκατοστημορίου· "
                      f"εκτός άξονα {gr(out, 1)}%)", fontsize=9)
    gr_axis(axes[0], "both")
    hour = p["timestamp"].dt.hour
    for m in ("Persistence (t−1h)", "Random Forest", "XGBoost"):
        mae = (p["y"] - p[m]).abs().groupby(hour).mean()
        axes[1].plot(mae.index, mae.values, marker=".", color=MODEL_COLORS[m], label=m)
    axes[1].set_xlabel("Ώρα της ημέρας")
    axes[1].set_ylabel("MAE (kWh/h)")
    axes[1].set_title("Σφάλμα ανά ώρα (σύνολο ελέγχου)")
    axes[1].legend(fontsize=8)
    gr_axis(axes[1])
    save_fig(fig, "fig_4_7_error_analysis.png")


def fig_4_10_co2_heatmap():
    a = pd.read_parquet(T / "annual_by_building.parquet")
    fuels = ["electricity"] + list(C.EF_FUEL)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, dtype in zip(axes, ("residential", "commercial")):
        d = a[a.dataset_type == dtype]
        M = np.array([[d.loc[d.city == c, f"co2_{f}"].sum() / d.loc[d.city == c, "floor_area_m2"].sum()
                       for f in fuels] for c in C.CITIES])
        im = ax.imshow(M, cmap="OrRd", aspect="auto")
        ax.set_yticks(range(5), [f"{c} ({C.CITY_ZONE[c]})" for c in C.CITIES])
        lab = [FUEL_GR[f].replace(" θέρμανσης", "\nθέρμανσης").replace(" (", "\n(")
               .replace("Φυσικό αέριο", "Φυσικό\nαέριο") for f in fuels]
        ax.set_xticks(range(len(fuels)), lab, fontsize=8)
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, gr(M[i, j], 1), ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > M.max() * 0.6 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("kg CO₂ / m² · έτος")
        ax.set_title(TYPE_GR[dtype])
    fig.suptitle("Ένταση εκπομπών CO₂ ανά ζώνη και καύσιμο (eGRID2018 / EPA)")
    save_fig(fig, "fig_4_10_co2_heatmap.png")


def fig_4_11_esg():
    ind, comp = csv("table_4_6_esg_indicators.csv"), csv("table_4_6b_esg_composite.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(C.CITIES))
    axes[0].bar(x - 0.2, ind["Βαθμός E (0–100)"], 0.4, label="E (CO₂/m²)", color="#41ab5d")
    axes[0].bar(x + 0.2, ind["Βαθμός S (0–100)"], 0.4, label="S (burden Q1)", color="#3182bd")
    axes[0].set_xticks(x, CITY_LAB)
    axes[0].set_ylabel("Κανονικοποιημένος βαθμός (100 = καλύτερη ζώνη)")
    axes[0].set_title("Επιμέρους δείκτες (min–max στις 5 ζώνες)")
    axes[0].legend(fontsize=8)
    for i, w in enumerate(C.ESG_WEIGHTS_E):
        v = comp[f"Σύνθετος (w_E={w:.2f})"]
        axes[1].bar(x + (i - 1) * 0.27, v, 0.27, label=f"w_E = {gr(w, 2)}",
                    color=plt.cm.Greens(0.4 + 0.25 * i))
        rks = comp[f"Κατάταξη (w_E={w:.2f})"]
        for xi, vi, rk in zip(x, v, rks):
            tie = "=" if (rks == rk).sum() > 1 else ""      # exact tie (e.g. 50,0 = 50,0)
            axes[1].text(xi + (i - 1) * 0.27, vi, f"{rk}η{tie}", ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(x, CITY_LAB)
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Σύνθετος δείκτης E+S και κατάταξη ανά βάρος (ανάλυση ευαισθησίας)")
    axes[1].legend(fontsize=8, ncol=3, loc="upper center")
    for ax in axes:
        gr_axis(ax)
    save_fig(fig, "fig_4_11_esg_composite.png")


# ── Chapter 5 ────────────────────────────────────────────────────────
def fig_5_0_climate_check():
    b = pd.read_parquet(T / "table_5_1_climate_by_building.parquet")
    acc = csv("table_5_0_climate_model_accuracy.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dtype in zip(axes, ("residential", "commercial")):
        d = b[b.dataset_type == dtype]
        ax.scatter(d["EUI_actual"], d["EUI_model_base"], s=6, alpha=0.5,
                   c=d["city"].map({c: ZONE_COLORS[C.CITY_ZONE[c]] for c in C.CITIES}))
        m = max(d["EUI_actual"].max(), d["EUI_model_base"].max())
        ax.plot([0, m], [0, m], "k--", lw=0.8)
        r = acc[acc["Μοντέλο"].str.contains(f"annual EUI.*{dtype}")].iloc[0]
        ax.set_title(f"{TYPE_GR[dtype]}: R² {gr(r['R²'], 3)}, CV(RMSE) {gr(r['CV(RMSE) (%)'], 1)}%, "
                     f"NMBE {gr(r['NMBE (%)'], 1)}%", fontsize=9)
        ax.set_xlabel("Ετήσια EUI 2018 NREL (kWh/m²·έτος)")
        ax.set_ylabel("Ετήσια EUI μοντέλου καιρού (kWh/m²·έτος)")
        gr_axis(ax, "both")
    # legend of the city colours (one entry per city)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=ZONE_COLORS[C.CITY_ZONE[c]],
                label=f"{c} ({C.CITY_ZONE[c]})") for c in C.CITIES]
    axes[0].legend(handles=h, fontsize=8, loc="upper left")
    fig.suptitle("Έλεγχος του μοντέλου καιρού (χωρίς lags) πριν από τα κλιματικά σενάρια")
    save_fig(fig, "fig_5_0_climate_model_check.png")


def fig_5_1_climate():
    t = csv("table_5_1_climate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    scen = list(C.SCENARIOS)
    for ax, dtype in zip(axes, ("residential", "commercial")):
        d = t[t["Τύπος"] == dtype].set_index("Πόλη").loc[C.CITIES]
        x = np.arange(len(C.CITIES))
        w = 0.8 / len(scen)
        for i, sc in enumerate(scen):
            v = d[f"{sc}: Δ συνολικής κατανάλωσης (%)"].to_numpy()
            rng = d[f"{sc}: εύρος very likely (%)"].str.split(" … ", expand=True).astype(float)
            err = np.vstack([v - rng[0].to_numpy(), rng[1].to_numpy() - v])
            err = np.abs(err)
            ax.bar(x + (i - len(scen) / 2 + 0.5) * w, v, w, yerr=err, capsize=2,
                   label=sc, color=(plt.cm.Blues if "2-4.5" in sc else plt.cm.Reds)(0.4 + 0.25 * (i % 3)))
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, CITY_LAB)
        ax.set_title(TYPE_GR[dtype])
        ax.set_ylabel("Μεταβολή ετήσιας κατανάλωσης έναντι 2018 (%)")
        gr_axis(ax)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Κλιματικά σενάρια IPCC AR6 (ΔT έναντι 1995–2014, κεντρική εκτίμηση· "
                 "μπάρες σφάλματος = very likely εύρος)", fontsize=10)
    save_fig(fig, "fig_5_1_climate_scenarios.png")


def fig_5_2_grid():
    t = csv("table_5_2_grid_whatif.csv")
    d = t[t["Τύπος"] == "Σύνολο"].set_index("Πόλη").loc[C.CITIES]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(C.CITIES))
    n = len(C.GRID_EF_REDUCTION)
    for i, r in enumerate(C.GRID_EF_REDUCTION):
        v = d[f"CO₂ kg/m² (EF ηλεκτρ. −{r*100:.0f}%)"]
        ax.bar(x + (i - n / 2 + 0.5) * 0.8 / n, v, 0.8 / n,
               label=f"EF ηλεκτρισμού −{r*100:.0f}%", color=plt.cm.Greens(0.9 - 0.2 * i))
    ax.set_xticks(x, [f"{c}\n({C.CITY_ZONE[c]})\nηλεκτρ. {gr(s, 0)}% του CO₂"
                      for c, s in zip(C.CITIES, d["Μερίδιο ηλεκτρισμού στο CO₂ (%)"])], fontsize=8)
    ax.set_ylabel("kg CO₂ / m² · έτος")
    gr_axis(ax)
    ax.legend(fontsize=8)
    ax.set_title("ΥΠΟΘΕΤΙΚΟ ΣΕΝΑΡΙΟ (what-if): μείωση συντελεστή εκπομπών δικτύου, "
                 "κατανάλωση 2018 σταθερή", fontsize=10)
    save_fig(fig, "fig_5_2_grid_whatif.png")


def fig_5_4_burden():
    b = csv("table_5_3_energy_burden.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(C.CITIES))
    hi = [c for c in b.columns if c.startswith("Υψηλό")][0]
    for i, q in enumerate(("Q1", "Q2", "Q3", "Q4")):
        d = b[b["Τεταρτημόριο"] == q].set_index("Πόλη").loc[C.CITIES]
        axes[0].bar(x + (i - 1.5) * 0.2, d["Διάμεσο energy burden (%)"], 0.2, label=q,
                    color=plt.cm.RdYlGn(0.1 + 0.27 * i))
        axes[1].bar(x + (i - 1.5) * 0.2, d[hi], 0.2, label=q, color=plt.cm.RdYlGn(0.1 + 0.27 * i))
    axes[0].axhline(C.BURDEN_HIGH, color="orange", ls="--", lw=1, label=f"υψηλό {gr(C.BURDEN_HIGH, 0)}%")
    axes[0].axhline(C.BURDEN_SEVERE, color="red", ls="--", lw=1, label=f"σοβαρό {gr(C.BURDEN_SEVERE, 0)}%")
    axes[0].set_ylabel("Διάμεσο energy burden (%)")
    axes[1].set_ylabel(f"% νοικοκυριών με burden > {gr(C.BURDEN_HIGH, 0)}%")
    for ax in axes:
        ax.set_xticks(x, CITY_LAB)
        gr_axis(ax)
        ax.legend(fontsize=7)
    fig.suptitle("Οικιστικά: ενεργειακή επιβάρυνση ανά τεταρτημόριο εισοδήματος "
                 "(όλα τα καύσιμα, τιμές EIA SEDS 2018)", fontsize=10)
    save_fig(fig, "fig_5_4_energy_burden.png")


def fig_5_5_dashboard():
    te = csv("table_4_4_benchmark_test.csv").set_index("Μοντέλο")
    oo = csv("table_4_4_benchmark_oos.csv").set_index("Μοντέλο")
    cl = csv("table_5_1_climate.csv")
    b = csv("table_5_3_energy_burden.csv")
    ind = csv("table_4_6_esg_indicators.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    m = ["Persistence (t−1h)", "Random Forest", "XGBoost"]
    ax.bar(np.arange(3) - 0.2, te.loc[m, "CV(RMSE) (%)"], 0.4, label="Έλεγχος")
    ax.bar(np.arange(3) + 0.2, oo.loc[m, "CV(RMSE) (%)"], 0.4, label="OOS 5B")
    ax.set_xticks(range(3), m, fontsize=8)
    ax.set_title("Α. Ακρίβεια πρόβλεψης: CV(RMSE) %")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    sc = [s for s in C.SCENARIOS if "2081-2100" in s]
    for i, s in enumerate(sc):
        for j, dtype in enumerate(("residential", "commercial")):
            d = cl[cl["Τύπος"] == dtype].set_index("Πόλη").loc[C.CITIES]
            ax.plot(range(5), d[f"{s}: Δ συνολικής κατανάλωσης (%)"], marker="os"[j],
                    ls="-" if i == 0 else "--", label=f"{TYPE_GR[dtype]} {s}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(5), CITY_LAB, fontsize=8)
    ax.set_title("Β. Μεταβολή κατανάλωσης 2081–2100 (%)")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    q1 = b[b["Τεταρτημόριο"] == "Q1"].set_index("Πόλη").loc[C.CITIES]
    q4 = b[b["Τεταρτημόριο"] == "Q4"].set_index("Πόλη").loc[C.CITIES]
    ax.bar(np.arange(5) - 0.2, q1["Διάμεσο energy burden (%)"], 0.4, label="Q1", color="#d7301f")
    ax.bar(np.arange(5) + 0.2, q4["Διάμεσο energy burden (%)"], 0.4, label="Q4", color="#1a9850")
    ax.axhline(C.BURDEN_HIGH, color="orange", ls="--", lw=1)
    ax.set_xticks(range(5), CITY_LAB, fontsize=8)
    ax.set_title("Γ. Διάμεσο energy burden (%) — οικιστικά")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.bar(range(5), ind["E: CO₂ (kg/m²·έτος, όλα τα κτίρια)"],
           color=[ZONE_COLORS[z] for z in ind["Ζώνη"]])
    ax.set_xticks(range(5), CITY_LAB, fontsize=8)
    ax.set_title("Δ. Ένταση CO₂ (kg/m²·έτος)")
    for a in axes.flat:
        gr_axis(a)
    fig.suptitle("Σύνοψη αποτελεσμάτων (όλες οι τιμές από τους πίνακες outputs_v2/tables)")
    save_fig(fig, "fig_5_5_master_dashboard.png")


def main():
    C.ensure_dirs()
    for f in (fig_4_1_optuna, fig_4_2_comparison, fig_4_3_actual_vs_pred, fig_4_4_load_profiles,
              fig_4_4b_seasonal, fig_4_4c_by_type, fig_4_5_diversity, fig_4_6_importance,
              fig_4_7_errors, fig_4_10_co2_heatmap, fig_4_11_esg, fig_5_0_climate_check,
              fig_5_1_climate, fig_5_2_grid, fig_5_4_burden, fig_5_5_dashboard):
        print(f.__name__)
        f()
    print("✓ 07_FIGURES_v2 finished")


if __name__ == "__main__":
    main()
