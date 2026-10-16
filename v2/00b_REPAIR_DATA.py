#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00b_REPAIR_DATA.py  —  STEP A (≈10–20 min, no timeseries re-download)
Thesis MIS24011 — Zygouris Aristotelis

Fixes the data problems found in the final check WITHOUT downloading the
5.000 timeseries again:

  K11  Residential floor area: ResStock "in.geometry_floor_area" is a
       categorical bin ("1000-1499"), so floor_area_m2 was NaN for all
       3.000 homes.  -> numeric area from NREL metadata (or bin mid-point).
  ID   bldg_id is NOT unique across ResStock and ComStock (8 collisions).
       The original merge on bldg_id duplicated rows and mixed series.
       -> new key  building_uid = "R<bldg_id>" / "C<bldg_id>".
  WX   Weather was joined one hour late (EPW stamps are hour-ending).
       -> weather(t+1h) is joined to energy hour t.
  #3   NREL sampling weights are stored (professor comment #3).
  #15  Every count quoted in the text is written to data_summary.json.

Outputs (originals are NOT modified):
  data/v2/metadata_v2.parquet, data/v2/buildings_v2.parquet,
  data/v2/data_summary.json,   data/v2/diagnostics.txt   <- send this file

Usage:
  python 00b_REPAIR_DATA.py
  (testing only) python 00b_REPAIR_DATA.py --local-res X.parquet --local-com Y.parquet
"""
import argparse
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config as C
from common import (FUELS, check, end_use_map, fuel_total_col, save_json,
                    site_total_col)

C.ensure_dirs()
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def flush_log():
    C.DIAG_TXT.write_text("\n".join(_log_lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# 1. NREL national metadata (only the columns we need)
# ─────────────────────────────────────────────────────────────────────
def load_nrel_meta(version, local_path=None):
    """Read baseline metadata from S3 (or a local copy for testing)."""
    if local_path:
        tbl = pq.read_table(local_path)
    else:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED),
                          region_name="us-west-2")
        key = f"{C.BASE}/{version}/metadata/baseline.parquet"
        log(f"  downloading s3://{C.BUCKET}/{key} ...")
        body = s3.get_object(Bucket=C.BUCKET, Key=key)["Body"].read()
        tbl = pq.read_table(io.BytesIO(body))
    cols = tbl.column_names
    keep_rx = re.compile(r"(^bldg_id$|weight|sqft|floor_area|climate_zone|county|"
                         r"gisjoin|nhgis|puma|^in\.state$|income|emission|co2|bill|"
                         r"building_type|vacan|tenure|occupants)", re.I)
    keep = [c for c in cols if keep_rx.search(c)]
    pmeta = tbl.schema.pandas_metadata or {}
    idx_cols = [c for c in pmeta.get("index_columns", []) if isinstance(c, str) and c in cols]
    keep = list(dict.fromkeys(idx_cols + keep))
    df = tbl.select(keep).to_pandas()
    if "bldg_id" in df.columns:
        df = df.set_index("bldg_id")
    log(f"    {version}: index name = {df.index.name} (00_DOWNLOAD used this index as bldg_id)")
    df.index = df.index.astype("int64")
    return df, cols


def strip_kwh(c):
    """ResStock names: out.x.y.energy_consumption ; ComStock: ... .kwh  -> one name."""
    return c[:-4] if c.startswith("out.") and c.endswith("energy_consumption.kwh") else c


def harmonise_units_suffix(df):
    """
    The raw file is a concat of ResStock (no suffix) and ComStock ('.kwh')
    columns, so each building has values in only ONE of the two variants.
    Merge them into the un-suffixed name (both are kWh in the hour).
    """
    for c in [c for c in df.columns if strip_kwh(c) != c]:
        base = strip_kwh(c)
        if base in df.columns:
            both = df[base].notna() & df[c].notna()
            if both.any():   # must never disagree
                d = (df.loc[both, base] - df.loc[both, c]).abs().max()
                if d > 1e-6:
                    raise ValueError(f"{base} and {c} disagree (max |Δ| = {d})")
            df[base] = df[base].combine_first(df[c])
            df = df.drop(columns=c)
        else:
            df = df.rename(columns={c: base})
    return df


def longest_run(mask):
    """Length of the longest run of True values in a 1-D boolean array."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def qc_batch(df, site, energy_cols, n_raw_by_b):
    """
    Data-quality metrics per building (STEP A, before any modelling):
      zero_h          hours with site energy = 0
      max_zero_run    longest run of consecutive zero hours  (00_DOWNLOAD filled
                      missing hours with 0 -> long runs reveal download gaps)
      max_flat_run    longest run of identical NON-zero consecutive values
                      (frozen / repeated series)
      neg_values      negative values in any energy column (should be 0)
      dup_rows        duplicated rows removed (old bldg_id merge)
      site_minus_fuels_pct  annual site − Σ fuel totals (consistency, %)
    """
    rows = []
    for uid, g in df.groupby("building_uid", sort=False):
        a = g[site].to_numpy()
        flat = np.r_[False, (np.diff(a) == 0) & (a[1:] > 0)]
        rows.append({"building_uid": uid,
                     "zero_h": int((a <= 0).sum()),
                     "max_zero_run": longest_run(a <= 0),
                     "max_flat_run": longest_run(flat) + 1 if flat.any() else 0,
                     "neg_values": int((g[energy_cols].to_numpy() < 0).sum()),
                     "dup_rows": int(n_raw_by_b.get(uid, C.HOURS) - len(g)),
                     "site_kwh": float(a.sum())})
    return rows


def zone_city_counts(meta, zone_col, zone, city):
    """Replicates the filter of 00_DOWNLOAD (zone field + county/FIPS)."""
    z = meta[meta[zone_col].astype(str).str.startswith(zone)]
    pattern = "|".join(C.CITY_FIPS[city] + [C.CITY_GISJOIN[city]])
    geo_cols = [c for c in z.columns if any(k in c.lower() for k in
                ("nhgis", "gisjoin", "fips", "county")) and not c.lower().startswith("calc.")]
    for gc in geo_cols:
        hit = z[z[gc].astype(str).str.contains(pattern, case=False, na=False)]
        if len(hit) >= 20:
            # which county codes actually matched (audit 12/10: only the GISJOIN)
            return len(z), len(hit), gc, hit[gc].astype(str).value_counts().to_dict()
    return len(z), None, None, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-res"); ap.add_argument("--local-com")
    args = ap.parse_args()

    log("=" * 70)
    log("00b_REPAIR_DATA — diagnostics & repair")
    log("=" * 70)

    # ── Original sample metadata ─────────────────────────────────────
    meta = pd.read_parquet(C.RAW_META)
    log(f"\n[1] Original metadata: {len(meta):,} rows")
    meta["building_uid"] = np.where(meta["dataset_type"] == "residential", "R", "C") \
        + meta["bldg_id"].astype("int64").astype(str)
    n_collide = int(meta["bldg_id"].duplicated(keep=False).sum() // 2)
    log(f"    bldg_id collisions between ResStock/ComStock: {n_collide}")
    check(meta["building_uid"].is_unique, "building_uid is unique", log)
    check(len(meta) == 5 * (C.N_RES + C.N_COM),
          f"sample size = {5 * (C.N_RES + C.N_COM)}", log)

    # ── NREL national metadata ───────────────────────────────────────
    log("\n[2] NREL national metadata")
    nres, res_all_cols = load_nrel_meta(C.RES_VER, args.local_res)
    ncom, com_all_cols = load_nrel_meta(C.COM_VER, args.local_com)
    log(f"    ResStock: {len(nres):,} buildings, {len(res_all_cols)} columns")
    log(f"    ComStock: {len(ncom):,} buildings, {len(com_all_cols)} columns")
    for name, cols in (("ResStock", res_all_cols), ("ComStock", com_all_cols)):
        for kw in ("sqft", "floor_area", "weight", "emission", "bill", "income"):
            hits = [c for c in cols if kw in c.lower()]
            log(f"    {name} columns with '{kw}': {hits[:25]}{' ...' if len(hits) > 25 else ''}")

    # Sampling funnel (professor #15: no hard-coded numbers in Fig. 3.1)
    res_zone = next(c for c in nres.columns if "ashrae_iecc_climate_zone_2004" in c)
    com_zone = next((c for c in ncom.columns if "ashrae_iecc_climate_zone_2006" in c),
                    next(c for c in ncom.columns if "climate_zone" in c.lower()
                         and "building_america" not in c.lower()))
    funnel = {"nrel_total_res": int(len(nres)), "nrel_total_com": int(len(ncom)),
              "by_city": {}}
    for zone, city in C.ZONE_CITY.items():
        zr, cr, gr_, vr = zone_city_counts(nres, res_zone, zone, city)
        zc, cc, gc_, vc = zone_city_counts(ncom, com_zone, zone, city)
        log(f"    {city:9s} matched county codes: res {vr} | com {vc}")
        funnel["by_city"][city] = {"zone_res": zr, "city_res": cr,
                                   "zone_com": zc, "city_com": cc,
                                   "geo_col_res": gr_, "geo_col_com": gc_}
        log(f"    {city:9s} zone filter res={zr:,} com={zc:,} | +county res={cr} com={cc}")

    # ── Floor area (K11) ─────────────────────────────────────────────
    log("\n[3] Floor area")
    res = meta[meta["dataset_type"] == "residential"].copy()
    com = meta[meta["dataset_type"] == "commercial"].copy()
    check(res["bldg_id"].isin(nres.index).all(), "all sampled residential ids exist in ResStock metadata", log)
    check(com["bldg_id"].isin(ncom.index).all(), "all sampled commercial ids exist in ComStock metadata", log)

    num_res_area = [c for c in nres.columns if ("sqft" in c.lower() or "floor_area" in c.lower())
                    and pd.api.types.is_numeric_dtype(nres[c])]
    log(f"    numeric residential area candidates: {num_res_area}")
    # prefer in.sqft (ResStock sampled floor area, ft²) if present
    num_res_area = sorted(num_res_area, key=lambda c: (c != "in.sqft", c))
    if num_res_area:
        col = num_res_area[0]
        res["area_sqft"] = res["bldg_id"].map(nres[col]).astype(float)
        area_source_res = f"NREL numeric column '{col}'"
        # validation: the numeric area must fall inside its own category bin
        lo_hi = res["in.geometry_floor_area"].astype(str).str.extract(r"^(\d+)[-+](\d*)$")
        lo = pd.to_numeric(lo_hi[0], errors="coerce")
        hi = pd.to_numeric(lo_hi[1], errors="coerce").fillna(np.inf)
        inside = (res["area_sqft"] >= lo) & (res["area_sqft"] <= hi)
        log(f"    '{col}' inside its in.geometry_floor_area bin: "
            f"{int(inside.sum())}/{int(lo.notna().sum())}")
        check(inside[lo.notna()].mean() > 0.95,
              f"numeric area '{col}' consistent with the categorical bin", log)
    else:
        bins = res["in.geometry_floor_area"].astype(str)
        unknown = sorted(set(bins) - set(C.RES_AREA_BIN_MIDPOINT_SQFT))
        check(not unknown, f"all residential area bins known (unknown: {unknown})", log)
        res["area_sqft"] = bins.map(C.RES_AREA_BIN_MIDPOINT_SQFT).astype(float)
        area_source_res = "bin mid-points of in.geometry_floor_area (config.py)"
    com["area_sqft"] = pd.to_numeric(com["in.sqft"], errors="coerce")
    meta2 = pd.concat([res, com])
    meta2["floor_area_m2"] = meta2["area_sqft"] * C.SQFT_TO_M2
    log(f"    residential area source: {area_source_res}")
    check(meta2["floor_area_m2"].notna().all() and (meta2["floor_area_m2"] > 0).all(),
          f"floor_area_m2 > 0 for all {len(meta2):,} buildings", log)
    area_stats = meta2.groupby("dataset_type")["floor_area_m2"].describe(
        percentiles=[.25, .5, .75]).round(1)
    log(area_stats.to_string())

    # ── Weights, income, emissions, bills ───────────────────────────
    log("\n[4] Weights / income / NREL emission & bill outputs")
    wcol_r = next((c for c in nres.columns if c.lower() == "weight"), None)
    wcol_c = next((c for c in ncom.columns if c.lower() == "weight"), None)
    meta2["nrel_weight"] = np.nan
    if wcol_r:
        m = meta2["dataset_type"] == "residential"
        meta2.loc[m, "nrel_weight"] = meta2.loc[m, "bldg_id"].map(nres[wcol_r]).values
    if wcol_c:
        m = meta2["dataset_type"] == "commercial"
        meta2.loc[m, "nrel_weight"] = meta2.loc[m, "bldg_id"].map(ncom[wcol_c]).values
    log(f"    weight columns: res={wcol_r} com={wcol_c}")
    log(meta2.groupby("dataset_type")["nrel_weight"].describe().round(2).to_string())

    inc = pd.to_numeric(meta2.loc[meta2["dataset_type"] == "residential", "income_usd"],
                        errors="coerce")
    n_inc0 = int((inc.fillna(0) <= 0).sum())
    log(f"    residential income <= 0 or missing: {n_inc0} (excluded in 04, not imputed)")
    log(f"    distinct income values: {inc.nunique()}")
    # Income 0 in ResStock = VACANT dwelling (no household): check against
    # tenure (sample metadata) and the NREL vacancy column, per city.
    rmask = meta2["dataset_type"] == "residential"
    vcol = next((c for c in nres.columns if "vacan" in c.lower()), None)
    if vcol:
        meta2.loc[rmask, "vacancy_status"] = meta2.loc[rmask, "bldg_id"].map(nres[vcol]).astype(str).values
    if "in.tenure" in meta2.columns:
        meta2.loc[rmask, "tenure"] = meta2.loc[rmask, "in.tenure"].astype(str).values
    rr = meta2[rmask].assign(income0=(inc.fillna(0) <= 0).values)
    log(f"    NREL vacancy column: {vcol}")
    by = [c for c in ("vacancy_status", "tenure") if c in rr.columns]
    for c in by:
        log(f"    income<=0 × {c}:")
        log(pd.crosstab(rr[c], rr["income0"]).to_string())
    log("    income<=0 per city:")
    log(rr.groupby("city")["income0"].agg(["sum", "count"]).to_string())
    vac_by_city = rr.groupby("city")["income0"].sum().astype(int).to_dict()

    extra_cols = {}
    for name, nm, dtype in (("res", nres, "residential"), ("com", ncom, "commercial")):
        for c in nm.columns:
            lc = c.lower()
            if ("emission" in lc or "co2" in lc or "bill" in lc) and \
                    pd.api.types.is_numeric_dtype(nm[c]):
                extra_cols.setdefault(c, []).append(dtype)
                m = meta2["dataset_type"] == dtype
                meta2.loc[m, c] = meta2.loc[m, "bldg_id"].map(nm[c]).values
    log(f"    numeric emission/bill columns stored: {len(extra_cols)}")
    for c, d in sorted(extra_cols.items()):
        log(f"      {c}  [{', '.join(d)}]")

    # county code of every sampled building (documents the geographic filter)
    meta2["county_code"] = None
    for dtype, nm in (("residential", nres), ("commercial", ncom)):
        gcol = next((v["geo_col_res" if dtype == "residential" else "geo_col_com"]
                     for v in funnel["by_city"].values()), None)
        if gcol and gcol in nm.columns:
            m = meta2["dataset_type"] == dtype
            meta2.loc[m, "county_code"] = meta2.loc[m, "bldg_id"].map(nm[gcol]).astype(str).values
    log("    sampled buildings per county code:")
    log(meta2.groupby(["city", "dataset_type"])["county_code"].value_counts().to_string())

    meta2 = meta2.drop(columns=["area_sqft"])
    meta2.to_parquet(C.META_V2, index=False)
    log(f"\n    -> {C.META_V2}")

    # ── Timeseries repair (per city × type, low memory) ──────────────
    log("\n[5] Timeseries repair")
    schema = pq.read_schema(C.RAW_BLDG)
    raw_cols = schema.names
    n_kwh = sum(strip_kwh(c) != c for c in raw_cols)
    log(f"    columns with '.kwh' suffix (ComStock naming) merged into un-suffixed: {n_kwh}")
    cols = list(dict.fromkeys(strip_kwh(c) for c in raw_cols))
    site = site_total_col(cols)
    fuel_tot = {f: fuel_total_col(cols, f) for f in FUELS}
    log(f"    site total column: {site}")
    log(f"    fuel totals: {fuel_tot}")
    eu_map, eu_unmatched = end_use_map(cols)
    log(f"    end-use columns mapped: {len(eu_map)} | unmatched: {eu_unmatched}")
    for c, fam in sorted(eu_map.items()):
        log(f"      {fam:11s} <- {c}")

    energy_cols = [c for c in cols if c.startswith("out.") and "energy_consumption" in c]
    # fixed column order for every chunk (ParquetWriter needs one schema)
    out_cols = (["building_uid", "bldg_id", "timestamp", "city", "zone", "dataset_type",
                 "floor_area_m2", "income_usd", "nrel_weight", "temperature_c",
                 "rel_humidity", "wind_speed_ms", "solar_ghi_wm2"] + energy_cols)
    wx = pd.read_parquet(C.WEATHER)
    # weather.parquet is stamped hour-ENDING (01:00 … 2019-01-01 00:00, checked
    # on the real file) -> shift to hour-starting; unify the time unit to ns
    wx["timestamp"] = (pd.to_datetime(wx["timestamp"]).astype("datetime64[ns]")
                       - pd.Timedelta(hours=1))
    wx = wx[["timestamp", "city", "temperature_c", "rel_humidity",
             "wind_speed_ms", "solar_ghi_wm2"]].sort_values(["city", "timestamp"])
    # Temperature / humidity / wind in the EPW-based files are INSTANTANEOUS
    # values at the stamp (end of the hour); EnergyPlus interpolates between
    # them. The best estimate for the hour [h, h+1) is the mean of the values
    # at h and h+1 (audit 12/10/2026). GHI is already an hourly total -> kept.
    # For 00:00 of 1 January the value at h is unknown -> the h+1 value is used.
    for c in ("temperature_c", "rel_humidity", "wind_speed_ms"):
        prev = wx.groupby("city")[c].shift(1)
        wx[c] = 0.5 * (wx[c] + prev.fillna(wx[c]))
    check(wx.groupby("city")["timestamp"].nunique().eq(C.HOURS).all(),
          "weather: 8.760 unique hours per city", log)
    check((wx["timestamp"].min() == pd.Timestamp("2018-01-01 00:00")) and
          (wx["timestamp"].max() == pd.Timestamp("2018-12-31 23:00")),
          "weather after shift spans 2018-01-01 00:00 … 2018-12-31 23:00", log)
    wx.to_parquet(C.WEATHER_V2, index=False)          # used by 05_EDA (HDD/CDD tables)

    # Buildings are read in batches of BATCH (only the needed columns).
    # Nothing is lost: every row and every value (float64, as published by
    # NREL) is written unchanged; batching only limits the memory used.
    BATCH = int(os.environ.get("V2_BATCH", 100))   # buildings per read
    read_cols = ["bldg_id", "timestamp"] + [c for c in raw_cols
                                           if c.startswith("out.") and "energy_consumption" in c]
    wx_cols = ["temperature_c", "rel_humidity", "wind_speed_ms", "solar_ghi_wm2"]
    mi = meta2.set_index("building_uid")[["floor_area_m2", "income_usd", "nrel_weight"]]

    writer, rows_total, consistency, zero_hours, qc_rows = None, 0, [], [], []
    for city in C.CITIES:
        wxc = wx[wx["city"] == city].set_index("timestamp")[wx_cols]
        for dtype in ("residential", "commercial"):
            prefix = "R" if dtype == "residential" else "C"
            sel = meta2["city"].eq(city) & meta2["dataset_type"].eq(dtype)
            ids = sorted(int(x) for x in meta2.loc[sel, "bldg_id"])
            n_raw = n_rows = n_b = 0
            site_sum = fuel_sum = 0.0
            zero_all = []
            for k in range(0, len(ids), BATCH):
                batch = ids[k:k + BATCH]
                t = pq.read_table(C.RAW_BLDG, columns=read_cols,
                                  filters=[("city", "=", city), ("dataset_type", "=", dtype),
                                           ("bldg_id", "in", batch)])
                df = t.to_pandas()
                del t
                n_raw += len(df)
                df = harmonise_units_suffix(df)
                df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
                n_raw_by_b = (prefix + df["bldg_id"].astype("int64").astype(str)).value_counts().to_dict()
                dup = df.duplicated(subset=["bldg_id", "timestamp"])  # old merge duplicates
                if dup.any():
                    df = df[~dup.to_numpy()].copy()
                df["building_uid"] = prefix + df["bldg_id"].astype("int64").astype(str)
                df["city"], df["zone"], df["dataset_type"] = city, C.CITY_ZONE[city], dtype
                # weather: index lookup (no merge copy)
                pos = wxc.index.get_indexer(df["timestamp"])
                if (pos < 0).any():
                    raise AssertionError(f"{city}/{dtype}: timestamps without weather")
                for c in wx_cols:
                    df[c] = wxc[c].to_numpy()[pos]
                for c in mi.columns:
                    df[c] = df["building_uid"].map(mi[c]).astype("float64")
                if df[site].isna().any():
                    raise AssertionError(f"{city}/{dtype}: site total missing in some hours")
                for c in energy_cols:                  # column absent for this type -> 0 kWh
                    if c not in df.columns:
                        df[c] = 0.0
                    df[c] = df[c].fillna(0.0).astype("float64")   # full NREL precision kept
                df = df[out_cols].sort_values(["building_uid", "timestamp"])
                per_b = df.groupby("building_uid").size()
                if not per_b.eq(C.HOURS).all():
                    bad = per_b[per_b != C.HOURS]
                    raise AssertionError(f"{city}/{dtype}: buildings without 8.760 h: {bad.head().to_dict()}")
                n_b += len(per_b)
                qc_rows.extend(qc_batch(df, site, energy_cols, n_raw_by_b))
                z = (df[site] <= 0).groupby(df["building_uid"]).sum()
                zero_all.append(z)
                site_sum += float(df[site].astype("float64").sum())
                fuel_sum += float(sum(df[c].astype("float64").sum() for c in fuel_tot.values() if c))
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(C.BLDG_V2, table.schema, compression="snappy")
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                n_rows += len(df)
                del df, table
            check(n_b == len(ids), f"{city}/{dtype}: {n_b}/{len(ids)} buildings × 8.760 h", log)
            check(True, f"{city}/{dtype}: site total present every hour, weather joined", log)
            z = pd.concat(zero_all)
            zero_hours.append((city, dtype, int(z.sum()), int((z > 24).sum()), int(z.max())))
            log(f"    {city:9s} {dtype:11s}: hours with site energy = 0 -> total {int(z.sum())}, "
                f"buildings with >24 such hours: {int((z > 24).sum())}, max per building {int(z.max())}")
            rel = (site_sum - fuel_sum) / site_sum
            consistency.append((city, dtype, round(rel * 100, 3)))
            log(f"    {city:9s} {dtype:11s}: raw rows {n_raw:,} -> {n_rows:,} "
                f"| site − Σfuels = {rel*100:.3f}%")
            rows_total += n_rows
            flush_log()
    writer.close()

    # ── [6] Data-quality report (per building) ───────────────────────
    log("\n[6] Data quality per building (before modelling)")
    qc = pd.DataFrame(qc_rows).merge(
        meta2[["building_uid", "city", "dataset_type", "floor_area_m2"]
              + (["vacancy_status"] if "vacancy_status" in meta2.columns else [])],
        on="building_uid", how="left", validate="one_to_one")
    qc["eui_kwh_m2"] = qc["site_kwh"] / qc["floor_area_m2"]
    # robust z-score of log(EUI) within city × type (simulation outliers are
    # reported, not removed: they are legitimate members of the NREL stock)
    le = np.log(qc["eui_kwh_m2"].clip(lower=1e-6))
    grp = qc.groupby(["city", "dataset_type"])
    med = grp["eui_kwh_m2"].transform(lambda x: np.log(x.clip(lower=1e-6)).median())
    mad = grp["eui_kwh_m2"].transform(lambda x: (np.log(x.clip(lower=1e-6))
                                                 - np.log(x.clip(lower=1e-6)).median()).abs().median())
    qc["eui_robust_z"] = 0.6745 * (le - med) / mad.replace(0, np.nan)
    qc["flag_gap"] = qc["max_zero_run"] >= C.QC_MAX_ZERO_RUN_H
    qc["flag_flat"] = qc["max_flat_run"] >= C.QC_MAX_FLAT_RUN_H
    qc["flag_negative"] = qc["neg_values"] > 0
    qc["flag_eui_outlier"] = qc["eui_robust_z"].abs() > 3.5
    qc.to_parquet(C.V2_DIR / "qc_by_building.parquet", index=False)
    summ = qc.groupby(["city", "dataset_type"]).agg(
        N=("building_uid", "size"), zero_hours=("zero_h", "sum"),
        max_zero_run=("max_zero_run", "max"), gap_flags=("flag_gap", "sum"),
        flat_flags=("flag_flat", "sum"), negative_flags=("flag_negative", "sum"),
        eui_outliers=("flag_eui_outlier", "sum"), dup_rows=("dup_rows", "sum"),
        eui_median=("eui_kwh_m2", "median"))
    summ.to_csv(C.V2_DIR / "qc_summary.csv", encoding="utf-8-sig")
    log(f"    thresholds: zero run ≥ {C.QC_MAX_ZERO_RUN_H} h, flat run ≥ {C.QC_MAX_FLAT_RUN_H} h, |robust z(log EUI)| > 3,5")
    log(summ.round(1).to_string())
    worst = qc.sort_values(["max_zero_run", "max_flat_run"], ascending=False).head(15)
    log("    worst 15 buildings (longest zero / flat runs):")
    log(worst[["building_uid", "city", "dataset_type", "zero_h", "max_zero_run",
               "max_flat_run", "neg_values", "eui_kwh_m2", "eui_robust_z"]].round(2).to_string(index=False))
    if "vacancy_status" in qc.columns:
        log("    EUI median by vacancy status (residential):")
        log(qc[qc["dataset_type"] == "residential"].groupby("vacancy_status")["eui_kwh_m2"]
            .describe(percentiles=[.5]).round(1).to_string())
    n_flag = int((qc["flag_gap"] | qc["flag_flat"] | qc["flag_negative"]).sum())
    log(f"    buildings with a technical flag (gap / flat / negative): {n_flag} -> decision before STEP B")

    check(rows_total == len(meta2) * C.HOURS,
          f"total rows = {len(meta2):,} × 8.760 = {len(meta2)*C.HOURS:,}", log)
    log(f"    -> {C.BLDG_V2}")

    # ── Summary for the text ─────────────────────────────────────────
    summary = {
        "n_buildings": int(len(meta2)),
        "n_by_city_type": meta2.groupby(["city", "dataset_type"]).size()
                               .unstack().to_dict(orient="index"),
        "bldg_id_collisions": n_collide,
        "rows_hourly": int(rows_total),
        "area_source_residential": area_source_res,
        "area_m2": area_stats.to_dict(orient="index"),
        "income_le0_excluded": n_inc0,
        "income_le0_by_city": vac_by_city,
        "zero_site_hours": zero_hours,
        "qc_technical_flags": n_flag,
        "qc_eui_outliers": int(qc["flag_eui_outlier"].sum()),
        "funnel": funnel,
        "site_minus_fuels_pct": consistency,
        "emission_bill_columns": sorted(extra_cols),
        "weights_found": {"res": wcol_r, "com": wcol_c},
        "weather_alignment": "EPW hour-ending stamp t+1h joined to energy hour t",
    }
    save_json(summary, C.SUMMARY)
    log("\n✓ STEP A finished. Please send data/v2/diagnostics.txt before STEP B.")
    flush_log()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:      # always keep the log, even on failure
        log(f"\n!!! ERROR: {type(e).__name__}: {e}")
        flush_log()
        raise
