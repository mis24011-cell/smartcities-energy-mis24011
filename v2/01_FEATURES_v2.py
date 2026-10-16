#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_FEATURES_v2.py  —  Feature engineering on the repaired data
Thesis MIS24011 — Zygouris Aristotelis

Changes vs 01_FEATURES_INDIVIDUAL.py
  • key = building_uid (ResStock/ComStock ids collide)          [ID]
  • floor_area_log from the repaired m² (no fillna(150))        [K11]
  • split label train/val/test stored once, used by every model [K3]
  • no energy columns of the same hour in the features          [#1]
Target: out.site_energy.total.energy_consumption (kWh in the hour).
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as C
from common import is_holiday, check, site_total_col, split_labels, update_summary

FEATURES = [
    # calendar (cyclic encoding)
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend",
    # weather (NREL AMY2018, °C)
    "temperature_c", "rel_humidity", "solar_ghi_wm2", "hdd", "cdd",
    # autoregressive — computed per building, strictly from the past
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h",
    # building
    # building / calendar extras ("zone_enc" kept in the file for reference,
    # NOT used by the models: an ordinal city code that Denver never takes in
    # training — audit 12/10/2026; replaced by the holiday flag)
    "is_holiday", "zone_enc", "is_commercial", "floor_area_log",
]


def log(m):
    print(m)


def build(df, target):
    df = df.sort_values(["building_uid", "timestamp"]).reset_index(drop=True)
    ts = df["timestamp"]
    hour, month = ts.dt.hour, ts.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["is_holiday"] = is_holiday(ts).values              # US federal holidays 2018

    # Base 18 °C "degree-hours" (hourly hdd/cdd). Annual degree-days for
    # Ch.3 tables = Σ degree-hours / 24 (computed in 05_EDA).
    df["hdd"] = np.maximum(0.0, 18.0 - df["temperature_c"])
    df["cdd"] = np.maximum(0.0, df["temperature_c"] - 18.0)

    g = df.groupby("building_uid", sort=False)[target]
    df["lag_1h"] = g.shift(1)
    df["lag_24h"] = g.shift(24)
    df["lag_168h"] = g.shift(168)
    prev = g.shift(1)                                    # only past values
    pg = prev.groupby(df["building_uid"], sort=False)
    df["rolling_mean_24h"] = pg.transform(lambda x: x.rolling(24, min_periods=24).mean())
    df["rolling_std_24h"] = pg.transform(lambda x: x.rolling(24, min_periods=24).std())

    df["zone_enc"] = df["zone"].map(C.ZONE_ENC).astype("int8")
    df["is_commercial"] = (df["dataset_type"] == "commercial").astype("int8")
    df["floor_area_log"] = np.log(df["floor_area_m2"])
    df["eui_wh_m2"] = df[target] / df["floor_area_m2"] * 1000.0   # for 03_CLIMATE
    df["split"] = split_labels(ts).values
    return df


def main():
    C.ensure_dirs()
    cols = pq.read_schema(C.BLDG_V2).names
    target = site_total_col(cols)
    log(f"Target: {target}")
    keep_in = ["building_uid", "bldg_id", "timestamp", "city", "zone", "dataset_type",
               "floor_area_m2", "nrel_weight", "temperature_c", "rel_humidity",
               "solar_ghi_wm2", target]

    writer, n_rows, n_warm = None, 0, 0
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            df = pq.read_table(C.BLDG_V2, columns=keep_in,
                               filters=[("city", "=", city), ("dataset_type", "=", dtype)]).to_pandas()
            check(df["floor_area_m2"].gt(0).all(), f"{city}/{dtype}: floor area > 0", log)
            df = build(df, target)
            before = len(df)
            df = df.dropna(subset=["lag_168h", "rolling_std_24h"])   # warm-up week
            n_warm += before - len(df)
            out_cols = (["building_uid", "bldg_id", "timestamp", "city", "zone",
                         "dataset_type", "floor_area_m2", "nrel_weight", "split",
                         "eui_wh_m2", target] + FEATURES)
            df = df[out_cols]
            for c in FEATURES + ["eui_wh_m2", target, "floor_area_m2"]:
                df[c] = df[c].astype("float32")
            check(df[FEATURES].notna().all().all(), f"{city}/{dtype}: no NaN in features", log)
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(C.FEAT_V2, tbl.schema, compression="snappy")
            else:
                tbl = tbl.cast(writer.schema)
            writer.write_table(tbl)
            n_rows += len(df)
            log(f"  {city:9s} {dtype:11s}: {len(df):,} rows")
            del df, tbl
    writer.close()

    n_b = 5 * (C.N_RES + C.N_COM)
    check(n_warm == n_b * 168, f"warm-up rows removed = {n_b} × 168", log)
    update_summary("features", {"rows_after_warmup": n_rows, "warmup_rows": n_warm,
                                "n_features": len(FEATURES), "features": FEATURES,
                                "target": target})
    log(f"✓ {C.FEAT_V2}: {n_rows:,} rows, {len(FEATURES)} features")


if __name__ == "__main__":
    main()
