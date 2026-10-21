#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_REVISION_FIGURES_v2.py  —  Tables and figures of the revision (supervisor, 20/10/2026)
Thesis MIS24011 — Zygouris Aristotelis

Reads ONLY the outputs of 11_REVISION_v2.py (outputs_v2/revision/) and of
05_EDA_v2 (annual_by_building.parquet); nothing is typed by hand.
Run after 11_REVISION_v2.py:   python v2/12_REVISION_FIGURES_v2.py

Part A (comment 1 — peaks and diversity per fuel)
  rev_table_4_6_profiles_electricity.csv  hour and height of the peak of the mean
                                          daily ELECTRICITY profile per m² (Table 4.6)
  rev_table_4_7_df_by_fuel.csv            DF, stock peak and peak hour, electricity
                                          and natural gas (Table 4.7)
  figures/fig_4_4_urban_load_profiles_electricity.png   (Figure 4.3)
  figures/fig_4_5_diversity_factor_by_fuel.png          (Figure 4.4)

Part B (comment 5 — size-independent accuracy, Table 4.8δ)
  rev_table_4_8d_rmae_by_type.csv, rev_rmae_overall.csv, rev_rmae_by_city_type.csv,
  rev_rmae_by_building.parquet
  For every building b and model m:  rMAE = MAE_b(model) / MAE_b(persistence t-1h)
  (relative MAE, Davydenko and Fildes 2013; equivalent to MASE with the one-step naive
  forecast as scale, Hyndman and Koehler 2006, on the same evaluation hours).
  Each building counts once whatever its size -> median, geometric mean, IQR and the
  share of buildings with rMAE < 1 (model better than persistence).

Part C (comment 7 — Q1 burden and very low incomes, Table 5.10α)
  rev_table_5_10a_burden_low_income.csv
  A  main result (as Table 5.8)
  B  income floored at $10.000 (upper limit of the lowest ACS bin) -> lower bound
  C  homes of the lowest bin (< $10.000) excluded

Part D (comment 4 — is the tuned RF significantly different from XGBoost?)
  rev_table_4_8e_significance_tuned_rf.csv
  Same code as 08_SIGNIFICANCE_v2 (stratified cluster bootstrap over buildings, B = 2.000,
  seed 42; Diebold–Mariano with HLN correction, lag 24 h) on the predictions of stage F
  (rev_F_predictions_test/oos.parquet, which add "Random Forest (Optuna)").
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from common import TYPE_GR, ZONE_COLORS, gr, gr_axis, setup_mpl

plt = setup_mpl()
REV = C.OUT_DIR / "revision"
FIG = REV / "figures"
WINTER, SUMMER = (12, 1, 2), (6, 7, 8)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=C.DPI)
    plt.close(fig)
    print("  ->", FIG / name)


def profiles(st, area, fuel):
    """Mean hourly load per m² of the SAMPLE of each city × type (as Fig. 4.3):
    W/m² = mean over the days of Σ_i L_i(h) / Σ_i A_i × 1000."""
    rows, prof = [], {}
    for dtype in ("residential", "commercial"):
        for c in C.CITIES:
            s = st[(st.city == c) & (st.dataset_type == dtype)]
            r = {"Τύπος": TYPE_GR[dtype], "Πόλη": f"{C.CITY_ZONE[c]} {c}"}
            for tag, months in (("Χειμ.", WINTER), ("Καλ.", SUMMER)):
                x = s[s["timestamp"].dt.month.isin(months)]
                p = x.groupby(x["timestamp"].dt.hour)[f"{fuel}_kwh"].mean() / area[(c, dtype)] * 1000
                prof[(dtype, c, tag)] = p
                r[f"{tag}: ώρα αιχμής"] = f"{int(p.idxmax()):02d}:00"
                r[f"{tag}: αιχμή (W/m²)"] = round(float(p.max()), 1)
                r[f"{tag}: ελάχιστο (W/m²)"] = round(float(p.min()), 1)
            rows.append(r)
    return pd.DataFrame(rows), prof


def part_A():
    st = pd.read_parquet(REV / "rev_A_stock_hourly_by_fuel.parquet")
    pk = pd.read_parquet(REV / "rev_A_peaks_by_building.parquet")
    area = pk.groupby(["city", "dataset_type"])["floor_area_m2"].sum()

    # ── Table 4.6 (electricity) + Figure 4.3 ───────────────────────────
    t46, prof = profiles(st, area, "electricity")
    t46.to_csv(REV / "rev_table_4_6_profiles_electricity.csv", index=False, encoding="utf-8-sig")
    print(t46.to_string(index=False))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    for i, dtype in enumerate(("residential", "commercial")):
        for j, (sea, tag) in enumerate((("Χειμώνας", "Χειμ."), ("Καλοκαίρι", "Καλ."))):
            ax = axes[i, j]
            for c in C.CITIES:
                p = prof[(dtype, c, tag)]
                ax.plot(p.index, p.values, color=ZONE_COLORS[C.CITY_ZONE[c]], lw=2,
                        label=f"{c} ({C.CITY_ZONE[c]})")
            ax.set_title(f"{TYPE_GR[dtype]} — {sea}")
            ax.set_ylabel("W / m² (μέση ωριαία ζήτηση)")
            ax.set_ylim(bottom=0)
            gr_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("Ώρα της ημέρας")
        ax.set_xticks(range(0, 24, 3))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Μέσα ημερήσια προφίλ ηλεκτρικού φορτίου ανά m² (Σ φορτίου / Σ εμβαδού δείγματος)")
    _save(fig, "fig_4_4_urban_load_profiles_electricity.png")

    # ── Table 4.7 (DF per fuel) ────────────────────────────────────────
    d = pd.read_csv(REV / "rev_A_diversity_by_fuel.csv")
    fmt_t = lambda s: pd.Timestamp(s).strftime("%d/%m %H:%M")
    rows = []
    for c in C.CITIES:
        for typ in ("Οικιστικά", "Εμπορικά", "Σύνολο"):
            e = d[(d["Πόλη"] == c) & (d["Τύπος"] == typ) & (d["Καύσιμο"] == "Ηλεκτρισμός")].iloc[0]
            g = d[(d["Πόλη"] == c) & (d["Τύπος"] == typ) & (d["Καύσιμο"] == "Φυσικό αέριο")].iloc[0]
            rows.append({"Πόλη": f"{C.CITY_ZONE[c]} {c}", "Τύπος": typ,
                         "Ηλ.: αιχμή συνόλου (MW)": e["Αιχμή συνόλου, απόθεμα (MW)"],
                         "Ηλ.: DF": e["DF"], "Ηλ.: ώρα αιχμής": fmt_t(e["Ώρα αιχμής συνόλου"]),
                         "Ηλ.: αιχμή Ιουν–Σεπ (MW)": e["Αιχμή καλοκαιριού Ιουν–Σεπ (MW)"],
                         "Ηλ.: αιχμή Δεκ–Φεβ (MW)": e["Αιχμή χειμώνα Δεκ–Φεβ (MW)"],
                         "Ηλ.: συντ. φορτίου": e["Συντελεστής φορτίου (μέσο/αιχμή)"],
                         "Αέριο: αιχμή συνόλου (MW)": g["Αιχμή συνόλου, απόθεμα (MW)"],
                         "Αέριο: DF": g["DF"], "Αέριο: ώρα αιχμής": fmt_t(g["Ώρα αιχμής συνόλου"]),
                         "Ηλ.: DF δείγματος": e["DF δείγματος (χωρίς στάθμιση)"]})
    t47 = pd.DataFrame(rows)
    t47.to_csv(REV / "rev_table_4_7_df_by_fuel.csv", index=False, encoding="utf-8-sig")
    print(t47.to_string(index=False))

    # ── Figure 4.4: DF electricity vs natural gas ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    x = np.arange(len(C.CITIES))
    lab = [f"{c}\n({C.CITY_ZONE[c]})" for c in C.CITIES]
    for ax, fuel in zip(axes, ("Ηλεκτρισμός", "Φυσικό αέριο")):
        for i, typ in enumerate(("Οικιστικά", "Εμπορικά", "Σύνολο")):
            v = (d[(d["Καύσιμο"] == fuel) & (d["Τύπος"] == typ)]
                 .set_index("Πόλη").loc[C.CITIES, "DF"])
            ax.bar(x + (i - 1) * 0.27, v, 0.27, label=typ, color=["#41ab5d", "#3182bd", "#636363"][i])
            for xi, vi in zip(x, v):
                ax.text(xi + (i - 1) * 0.27, vi, gr(vi, 2), ha="center", va="bottom", fontsize=7)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, lab)
        ax.set_title(fuel)
        gr_axis(ax)
    axes[0].set_ylabel("DF = Σ ατομικών αιχμών / αιχμή συνόλου")
    axes[0].set_ylim(0, d[d["Καύσιμο"] != "Συνολική ενέργεια"]["DF"].max() * 1.18)
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("Συντελεστής ετεροχρονισμού ανά καύσιμο (ωριαία δεδομένα όλων των κτιρίων, απόθεμα πόλης)")
    _save(fig, "fig_4_5_diversity_factor_by_fuel.png")


# ═════════════════════════ Part B — rMAE per building ═════════════════════════
MODELS = ["Seasonal Naive (t−168h)", "Ridge Regression", "Random Forest", "XGBoost"]
PERS = "Persistence (t−1h)"
CITY = {"2A": "Houston", "3A": "Atlanta", "4A": "Baltimore", "5A": "Chicago", "5B": "Denver"}


def _per_building(path, setname):
    """MAE of every model and of persistence per building, then the ratio."""
    d = pd.read_parquet(path)
    e = pd.DataFrame({m: (d[m].astype("float64") - d["y"]).abs() for m in [PERS] + MODELS})
    for k in ("building_uid", "zone", "dataset_type"):
        e[k] = d[k].astype(str)
    mae = e.groupby(["building_uid", "zone", "dataset_type"], observed=True).mean().reset_index()
    for m in MODELS:
        mae[f"rMAE|{m}"] = mae[m] / mae[PERS]
    mae["Σύνολο"] = setname
    return mae


def _summarise(b, by):
    rows = []
    for key, g in b.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        for m in MODELS:
            r = g[f"rMAE|{m}"].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(dict(zip(by, key), **{
                "Μοντέλο": m, "Κτίρια": len(r),
                "Διάμεσος rMAE": round(r.median(), 3),
                "Γεωμ. μέσος rMAE": round(float(np.exp(np.log(r[r > 0]).mean())), 3),
                "IQR": f"{r.quantile(.25):.3f} … {r.quantile(.75):.3f}",
                "% κτιρίων καλύτερα από persistence": round(100 * (r < 1).mean(), 1)}))
    return pd.DataFrame(rows)


def part_B():
    b = pd.concat([_per_building(C.OUT_DIR / "predictions_test.parquet", "Έλεγχος (2A–5A)"),
                   _per_building(C.OUT_DIR / "predictions_oos.parquet", "Denver (5B, εκτός δείγματος)")])
    b["Τύπος"] = b["dataset_type"].map(TYPE_GR)
    b["Πόλη"] = b["zone"].map(CITY)
    b.to_parquet(REV / "rev_rmae_by_building.parquet", index=False)
    for by, f in ((["Σύνολο"], "rev_rmae_overall.csv"),
                  (["Σύνολο", "Τύπος"], "rev_table_4_8d_rmae_by_type.csv"),
                  (["Σύνολο", "Πόλη", "Τύπος"], "rev_rmae_by_city_type.csv")):
        t = _summarise(b, by)
        t.to_csv(REV / f, index=False, encoding="utf-8-sig")
        print(t.to_string(index=False))


# ═════════════════════════ Part C — Q1 burden sensitivity ═════════════════════
def part_C(floor=10000.0):
    h = pd.read_parquet(C.TABLES / "energy_burden_by_household.parquet")
    h = h[h["burden_pct"].notna() & (h["income_usd"] > 0)].copy()
    rows = []
    for c in C.CITIES:
        q1 = h[(h.city == c) & (h.quartile == "Q1")]
        low = q1.income_usd < floor
        bB = q1.cost_usd / np.maximum(q1.income_usd, floor) * 100     # variant B
        bC = q1.loc[~low, "burden_pct"]                                # variant C
        rows.append({"Πόλη": f"{C.CITY_ZONE[c]} {c}", "N Q1": len(q1),
                     "Q1 < $10.000 (%)": round(100 * low.mean(), 1),
                     "Διάμεσο εισόδημα Q1 ($)": round(q1.income_usd.median()),
                     "A: διάμεσο (%)": round(q1.burden_pct.median(), 2),
                     "A: > 6 % (%)": round(100 * (q1.burden_pct > 6).mean(), 1),
                     "B: εισόδημα ≥ $10.000, διάμεσο (%)": round(bB.median(), 2),
                     "B: > 6 % (%)": round(100 * (bB > 6).mean(), 1),
                     "Γ: χωρίς < $10.000, διάμεσο (%)": round(bC.median(), 2),
                     "Γ: > 6 % (%)": round(100 * (bC > 6).mean(), 1),
                     "Γ: N": int((~low).sum())})
    t = pd.DataFrame(rows)
    t.to_csv(REV / "rev_table_5_10a_burden_low_income.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))


# ═════════════════════════ Part D — significance, tuned RF ════════════════════
def part_D():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sig", Path(__file__).resolve().parent / "08_SIGNIFICANCE_v2.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    RFO = "Random Forest (Optuna)"
    S.MODELS = ["Random Forest", RFO, "XGBoost"]          # module globals used by its functions
    S.PAIRS = [("XGBoost", RFO), (RFO, "Random Forest"), ("XGBoost", "Random Forest")]
    rng = np.random.default_rng(C.SEED)
    rows = []
    for name, f in (("Έλεγχος (2A–5A)", "rev_F_predictions_test.parquet"),
                    ("Denver (5B)", "rev_F_predictions_oos.parquet")):
        g, hourly = S.building_sse(REV / f)
        for sub, gg in (("Όλα", g), ("Οικιστικά", g[g["type"] == "residential"]),
                        ("Εμπορικά", g[g["type"] == "commercial"])):
            point = S.skill(gg[[S.PERS] + S.MODELS].sum())
            bs = S.bootstrap(gg.reset_index(drop=True), rng)
            for a, b in S.PAIRS:
                ja, jb = S.MODELS.index(a), S.MODELS.index(b)
                diff = bs[:, ja] - bs[:, jb]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                r = {"Σύνολο": name, "Τύπος": sub, "Σύγκριση": f"{a} − {b}",
                     "Skill A": round(point[a], 4), "Skill B": round(point[b], 4),
                     "Διαφορά skill": round(point[a] - point[b], 4),
                     "95% ΔΕ κάτω": round(lo, 4), "95% ΔΕ άνω": round(hi, 4),
                     "p (bootstrap)": round(p, 4)}
                if sub == "Όλα":                         # DM on the hourly differential of all buildings
                    st_, p_dm = S.diebold_mariano(hourly[(a, b)])
                    r.update({"DM (HLN)": round(st_, 3), "p (DM)": float(f"{p_dm:.3g}")})
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(REV / "rev_table_4_8e_significance_tuned_rf.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console (cp1253)
    FIG.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1:] or ["A", "B", "C", "D"]
    for p in parts:
        {"A": part_A, "B": part_B, "C": part_C, "D": part_D}[p.upper()]()
    print("✓ 12_REVISION_FIGURES_v2 finished")
