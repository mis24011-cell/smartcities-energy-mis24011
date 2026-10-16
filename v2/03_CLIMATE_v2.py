#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_CLIMATE_v2.py  —  Climate scenarios through a weather-driven ML model
Thesis MIS24011 — Zygouris Aristotelis

Why a separate model (professor comment #5):
  The forecasting model of Ch.4 is dominated by lag_1h/lag_168h (≈66 % of
  the gain). Under a scenario those lags would still carry the OBSERVED
  2018 consumption, so the model cannot respond to a warmer climate.
  For the scenarios we therefore train an XGBoost model WITHOUT
  autoregressive inputs:  weather + calendar + building characteristics.
  This is the standard "weather-normalised" set-up for climate impact
  studies; its accuracy is reported (hourly and annual) so the reader can
  judge it.

Fixes vs 03_CLIMATE_SCENARIOS_INDIVIDUAL.py
  K5  EUI was the MEDIAN OF HOURLY values (rounded to 0.0).
      -> annual EUI per building (kWh/m²/year) = Σ hourly / floor area.
  K11 residential floor area now exists (00b).
  ΔT  IPCC AR6 Table SPM.1 values (config.py), periods 2021–2040,
      2041–2060, 2081–2100, with best estimate AND very likely range.

Outputs: tables/table_5_1_climate.csv, table_5_1_climate_by_building.parquet,
         table_5_0_climate_model_accuracy.csv, models/xgb_climate.json
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBRegressor

import config as C
from common import (STATIC_CAT, STATIC_NUM, is_holiday, metrics, site_total_col, split_labels,
                    static_features, update_summary)

# Audit 12/10/2026: the first version knew only zone, type and floor area,
# i.e. it predicted roughly the group average (annual R² 0,45 / 0,27). The
# static characteristics of each building (vintage, heating fuel, building
# type, HVAC system) are now inputs, so the per-building response to a
# warmer climate depends on the building. zone_enc removed (weather already
# describes the climate of each city).
CLIM_FEATURES = (["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek",
                  "is_weekend", "is_holiday", "temperature_c", "rel_humidity",
                  "solar_ghi_wm2", "hdd", "cdd", "is_commercial", "floor_area_log"]
                 + STATIC_NUM + STATIC_CAT)
CLIM_PARAMS = dict(max_depth=10, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                   enable_categorical=True, max_cat_to_onehot=1)


def prepare(df, target):
    """Build the 14 weather/calendar/building features (float32 to save memory)."""
    f4 = np.float32
    ts = df["timestamp"]
    h, mth = ts.dt.hour.to_numpy(), ts.dt.month.to_numpy()
    df["hour_sin"] = np.sin(2 * np.pi * h / 24).astype(f4)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24).astype(f4)
    df["month_sin"] = np.sin(2 * np.pi * mth / 12).astype(f4)
    df["month_cos"] = np.cos(2 * np.pi * mth / 12).astype(f4)
    df["dayofweek"] = ts.dt.dayofweek.astype(f4)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(f4)
    df["is_holiday"] = is_holiday(ts).astype(f4)
    df["temperature_c"] = df["temperature_c"].astype(f4)
    df["rel_humidity"] = df["rel_humidity"].astype(f4)
    df["solar_ghi_wm2"] = df["solar_ghi_wm2"].astype(f4)
    df["hdd"] = np.maximum(0.0, 18.0 - df["temperature_c"]).astype(f4)
    df["cdd"] = np.maximum(0.0, df["temperature_c"] - 18.0).astype(f4)
    df["is_commercial"] = (df["dataset_type"].astype(str) == "commercial").astype(f4)
    df["floor_area_log"] = np.log(df["floor_area_m2"]).astype(f4)
    df["y_wh_m2"] = (df[target] / df["floor_area_m2"] * 1000.0).astype(f4)   # hourly EUI, Wh/m²
    df["split"] = split_labels(ts).values
    return df.drop(columns=[target])


def shifted(X, dT):
    """Scenario: +dT on temperature; hdd/cdd recomputed; everything else fixed."""
    Xs = X.copy()
    Xs["temperature_c"] = X["temperature_c"] + dT
    Xs["hdd"] = np.maximum(0.0, 18.0 - Xs["temperature_c"])
    Xs["cdd"] = np.maximum(0.0, Xs["temperature_c"] - 18.0)
    return Xs


def main():
    C.ensure_dirs()
    cols = pq.read_schema(C.BLDG_V2).names
    target = site_total_col(cols)
    use = ["building_uid", "timestamp", "city", "zone", "dataset_type", "floor_area_m2",
           "temperature_c", "rel_humidity", "solar_ghi_wm2", target]
    df = pq.read_table(C.BLDG_V2, columns=use,
                       read_dictionary=["building_uid", "city", "zone", "dataset_type"]).to_pandas()
    df = prepare(df, target)
    # static building characteristics (metadata, no energy information)
    st = static_features(pd.read_parquet(C.META_V2))
    codes = df["building_uid"].cat.codes.to_numpy()          # row -> building (no strings)
    cats = df["building_uid"].cat.categories.astype(str)
    if st.reindex(cats).isna().any().any():
        raise AssertionError("static features missing for some buildings")
    for c in STATIC_NUM:
        df[c] = st[c].reindex(cats).to_numpy("float32")[codes]
    for c in STATIC_CAT:
        cc = pd.Categorical(st[c].astype(str).reindex(cats),
                            categories=list(st[c].cat.categories)).codes
        df[c] = pd.Categorical.from_codes(cc[codes], categories=list(st[c].cat.categories))
    assert df[CLIM_FEATURES + ["y_wh_m2"]].notna().all().all(), "NaN in climate features"

    # ── 1. Train the weather-driven model (all 5 cities) ─────────────
    # NOTE: unlike the forecasting models of Ch.4, this model is a scenario
    # tool applied to all five cities, so Denver is included in training
    # (stated in Ch.5). Its accuracy is reported per city below.
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    m = XGBRegressor(n_estimators=C.MAX_TREES, tree_method="hist", random_state=C.SEED,
                     n_jobs=-1, early_stopping_rounds=C.EARLY_STOPPING_ROUNDS,
                     eval_metric="rmse", **CLIM_PARAMS)
    m.fit(tr[CLIM_FEATURES], tr["y_wh_m2"], eval_set=[(va[CLIM_FEATURES], va["y_wh_m2"])],
          verbose=False)
    n_trees = int(m.best_iteration) + 1
    final = XGBRegressor(n_estimators=n_trees, tree_method="hist", random_state=C.SEED,
                         n_jobs=-1, **CLIM_PARAMS)
    trva = df[df["split"] != "test"]
    final.fit(trva[CLIM_FEATURES], trva["y_wh_m2"], verbose=False)
    final.save_model(str(C.MODELS / "xgb_climate.json"))

    acc = [metrics(te["y_wh_m2"], final.predict(te[CLIM_FEATURES]), "Climate model — hourly EUI (Wh/m²), test")]
    for dtype in ("residential", "commercial"):
        s = te[te["dataset_type"] == dtype]
        acc.append(metrics(s["y_wh_m2"], final.predict(s[CLIM_FEATURES]),
                           f"Climate model — hourly EUI, test, {dtype}"))
    for city in C.CITIES:                      # accuracy per city (test days)
        s = te[te["city"].astype(str) == city]
        acc.append(metrics(s["y_wh_m2"], final.predict(s[CLIM_FEATURES]),
                           f"Climate model — hourly EUI, test, {city}"))

    # Extrapolation check: trees cannot extrapolate beyond the training range,
    # so report the share of hours whose shifted temperature exceeds the
    # hottest temperature seen in training (and each city's own 2018 maximum).
    t_max_train = float(trva["temperature_c"].max())
    one = df.groupby("city", observed=True)["building_uid"].first().astype(str)
    wx = df.loc[df["building_uid"].isin(list(one)), ["city", "temperature_c"]]
    ex = []
    for city in C.CITIES:
        T = wx.loc[wx["city"].astype(str) == city, "temperature_c"].to_numpy("float64")
        r = {"Πόλη": city, "Μέγιστη T 2018 (°C)": round(T.max(), 1),
             "Μέγιστη T εκπαίδευσης (°C)": round(t_max_train, 1)}
        for sc, (best, lo, hi) in C.SCENARIOS.items():
            for tag, dT in (("κεντρική", best), ("άνω όριο", hi)):
                r[f"{sc} {tag}: % ωρών > T εκπαίδευσης"] = round(100 * np.mean(T + dT > t_max_train), 2)
                r[f"{sc} {tag}: % ωρών > T 2018 πόλης"] = round(100 * np.mean(T + dT > T.max()), 2)
        ex.append(r)
    pd.DataFrame(ex).to_csv(C.TABLES / "table_5_0b_extrapolation.csv", index=False,
                            encoding="utf-8-sig")
    del tr, va, te, trva

    # ── 2. Baseline + scenarios, per city (memory-friendly) ──────────
    rows_b = []
    for city in C.CITIES:
        sub = df[df["city"].astype(str) == city]
        X = sub[CLIM_FEATURES]
        uid = sub["building_uid"].astype(str).to_numpy()      # plain string keys

        def annual(values):
            """Σ hourly Wh/m² per building -> kWh/m²·year (indexed by uid)."""
            return pd.Series(np.asarray(values, dtype="float64")).groupby(uid).sum() / 1000.0

        ann = (sub[["dataset_type", "floor_area_m2"]].assign(building_uid=uid)
               .drop_duplicates("building_uid").set_index("building_uid"))
        ann["dataset_type"] = ann["dataset_type"].astype(str)
        ann["city"] = city
        ann["EUI_actual"] = annual(sub["y_wh_m2"])
        ann["EUI_model_base"] = annual(final.predict(X))
        for sc, (best, lo, hi) in C.SCENARIOS.items():
            for tag, dT in (("best", best), ("low", lo), ("high", hi)):
                ann[f"{sc}|{tag}"] = annual(final.predict(shifted(X, dT)))
        assert ann.notna().all().all(), f"{city}: NaN in annual EUI table"
        rows_b.append(ann.rename_axis("building_uid").reset_index())
        print(f"  ✓ {city}")
    byb = pd.concat(rows_b, ignore_index=True)
    byb.to_parquet(C.TABLES / "table_5_1_climate_by_building.parquet", index=False)

    # Annual accuracy of the climate model (does it reproduce 2018 per building?)
    for dtype in ("residential", "commercial"):
        s = byb[byb["dataset_type"] == dtype]
        acc.append(metrics(s["EUI_actual"], s["EUI_model_base"],
                           f"Climate model — annual EUI (kWh/m²·yr), {dtype}"))
    pd.DataFrame(acc).to_csv(C.TABLES / "table_5_0_climate_model_accuracy.csv",
                             index=False, encoding="utf-8-sig")

    # ── 3. Table 5.1 ─────────────────────────────────────────────────
    out = []
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            s = byb[(byb["city"] == city) & (byb["dataset_type"] == dtype)]
            row = {"Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": dtype,
                   "N κτιρίων": len(s),
                   "EUI 2018 πραγματικό, διάμεσος (kWh/m²·έτος)": round(s["EUI_actual"].median(), 1),
                   "EUI 2018 μοντέλο, διάμεσος (kWh/m²·έτος)": round(s["EUI_model_base"].median(), 1)}
            for sc in C.SCENARIOS:
                d = (s[f"{sc}|best"] / s["EUI_model_base"] - 1) * 100
                stock = (np.sum(s[f"{sc}|best"] * s["floor_area_m2"]) /
                         np.sum(s["EUI_model_base"] * s["floor_area_m2"]) - 1) * 100
                lo = (np.sum(s[f"{sc}|low"] * s["floor_area_m2"]) /
                      np.sum(s["EUI_model_base"] * s["floor_area_m2"]) - 1) * 100
                hi = (np.sum(s[f"{sc}|high"] * s["floor_area_m2"]) /
                      np.sum(s["EUI_model_base"] * s["floor_area_m2"]) - 1) * 100
                row[f"{sc}: ΔEUI διάμεσος κτιρίων (%)"] = round(d.median(), 2)
                row[f"{sc}: ΔEUI IQR (%)"] = f"{d.quantile(.25):.2f} … {d.quantile(.75):.2f}"
                row[f"{sc}: Δ συνολικής κατανάλωσης (%)"] = round(stock, 2)
                row[f"{sc}: εύρος very likely (%)"] = f"{lo:.2f} … {hi:.2f}"
            out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(C.TABLES / "table_5_1_climate.csv", index=False, encoding="utf-8-sig")
    print(t.iloc[:, :8].to_string(index=False))
    update_summary("climate", {"deltas_vs_1995_2014": C.SCENARIOS,
                               "model_trees": n_trees, "features": CLIM_FEATURES,
                               "params": CLIM_PARAMS})
    print("✓ 03_CLIMATE_v2 finished")


if __name__ == "__main__":
    main()
