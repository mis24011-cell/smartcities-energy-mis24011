#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02c_SEASONAL_BY_TYPE_v2.py  —  Season × building type, unified vs separate models
Thesis MIS24011 — Zygouris Aristotelis

Run AFTER 02_ML_v2 (uses the saved models and models/building_scale.parquet;
only predicts — a few minutes).
Seasonal skill by building type (thesis Ch.4 §4.6.3, Table 4.14):
  for every building type and season (test set, zones 2A–5A) and for the
  out-of-sample zone (Denver 5B):
     Persistence (t−1h) · XGBoost unified · XGBoost separate (res / com)
  with R², RMSE, MAE, MAPE (floor 0,1 kWh), CV(RMSE), NMBE and the skill
  score against persistence of the same subset.
Outputs: tables/table_4_4d_seasonal_by_type.csv, figures/fig_4_4d_seasonal_by_type.png
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBRegressor

import config as C
from common import TYPE_GR, gr, metrics, save_fig, scale_rows, scale_X, setup_mpl

plt = setup_mpl()
FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend",
    "temperature_c", "rel_humidity", "solar_ghi_wm2", "hdd", "cdd",
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h",
    "is_holiday", "is_commercial", "floor_area_log"]          # identical to 02_ML_v2
SEASON = {12: "Χειμώνας", 1: "Χειμώνας", 2: "Χειμώνας", 3: "Άνοιξη", 4: "Άνοιξη",
          5: "Άνοιξη", 6: "Καλοκαίρι", 7: "Καλοκαίρι", 8: "Καλοκαίρι",
          9: "Φθινόπωρο", 10: "Φθινόπωρο", 11: "Φθινόπωρο"}
SEASONS = ["Χειμώνας", "Άνοιξη", "Καλοκαίρι", "Φθινόπωρο"]


def load_model(name):
    m = XGBRegressor()
    m.load_model(str(C.MODELS / f"xgb_{name}.json"))
    return m


def main():
    C.ensure_dirs()
    cols = pq.read_schema(C.FEAT_V2).names
    target = next(c for c in cols if c.startswith("out.site_energy.total"))
    use = ["building_uid", "timestamp", "zone", "dataset_type", "split", target] + FEATURES
    df = pq.read_table(C.FEAT_V2, columns=use,
                       filters=[("split", "=", "test")]).to_pandas()   # test days of all zones
    df["zone"] = df["zone"].astype(str)
    df["building_uid"] = df["building_uid"].astype(str)
    df["dataset_type"] = df["dataset_type"].astype(str)
    df["Εποχή"] = df["timestamp"].dt.month.map(SEASON)
    df["Σύνολο"] = np.where(df["zone"] == C.OOS_ZONE, "OOS 5B", "test 2A–5A")

    # same per-building scaling as 02_ML_v2 (#18): s_b from the train days
    sc = pd.read_parquet(C.MODELS / "building_scale.parquet")
    s = scale_rows(df, sc.set_index("building_uid")["scale_kwh"])
    X = scale_X(df[FEATURES].to_numpy(np.float32), s, FEATURES)
    y = df[target].to_numpy(np.float32)
    uni = load_model("unified").predict(X) * s                     # back to kWh
    sep = np.empty_like(uni)
    for dtype, name in (("residential", "res"), ("commercial", "com")):
        m = (df["dataset_type"] == dtype).to_numpy()
        sep[m] = load_model(name).predict(X[m]) * s[m]
    pers = df["lag_1h"].to_numpy()

    rows = []
    for (sset, dtype, season), g in df.groupby(["Σύνολο", "dataset_type", "Εποχή"], sort=False):
        i = g.index.to_numpy()
        rp = float(np.sqrt(np.mean((y[i] - pers[i]) ** 2)))
        for name, p in (("Persistence (t−1h)", pers), ("XGBoost unified", uni),
                        (f"XGBoost {'res' if dtype == 'residential' else 'com'} (ξεχωριστό)", sep)):
            rows.append({"Σύνολο": sset, "Τύπος": TYPE_GR[dtype], "Εποχή": season,
                         **metrics(y[i], p[i], name, rp)})
    t = pd.DataFrame(rows)
    order = {s: k for k, s in enumerate(SEASONS)}
    t = t.sort_values(["Σύνολο", "Τύπος", "Εποχή", "Μοντέλο"],
                      key=lambda c: c.map(order) if c.name == "Εποχή" else c)
    t.to_csv(C.TABLES / "table_4_4d_seasonal_by_type.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))

    # Figure: CV(RMSE) % heatmap, test set (rows = type × model, columns = season)
    s = t[t["Σύνολο"] == "test 2A–5A"].copy()
    s["Γραμμή"] = s["Τύπος"] + " — " + s["Μοντέλο"]
    piv = s.pivot_table(index="Γραμμή", columns="Εποχή", values="CV(RMSE) (%)")[SEASONS]
    sk = s.pivot_table(index="Γραμμή", columns="Εποχή", values="Skill vs Persistence")[SEASONS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for ax, M, ttl, cm in ((axes[0], piv, "CV(RMSE) % (μικρότερο = καλύτερο)", "RdYlGn_r"),
                           (axes[1], sk, "Skill έναντι Persistence (μεγαλύτερο = καλύτερο)", "RdYlGn")):
        im = ax.imshow(M.values, cmap=cm, aspect="auto")
        ax.set_yticks(range(len(M)), M.index, fontsize=8)
        ax.set_xticks(range(len(SEASONS)), SEASONS)
        ax.grid(False)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, gr(M.values[a, b], 2 if "Skill" in ttl else 1),
                        ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(ttl, fontsize=10)
    fig.suptitle("Απόδοση ανά εποχή και τύπο κτιρίου — ενιαίο έναντι ξεχωριστών μοντέλων (σύνολο ελέγχου)")
    save_fig(fig, "fig_4_4d_seasonal_by_type.png")
    print("✓ 02c_SEASONAL_BY_TYPE_v2 finished")


if __name__ == "__main__":
    main()
