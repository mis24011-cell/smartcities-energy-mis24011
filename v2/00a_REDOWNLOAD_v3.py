#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00a_REDOWNLOAD_v3.py  —  STEP 0: re-download the SAME 5.000 buildings with the
                          correct 15-min -> hourly aggregation  (≈ 1–2 h)
Thesis MIS24011 — Zygouris Aristotelis

Why (found 12/10/2026)
  NREL stamps every 15-min value at the END of its interval: the value at
  10:15 is the energy of 10:00–10:15 (ResStock FAQ). The original downloader
  used resample("h") (closed="left"), so the hour labelled 10:00 summed the
  stamps 10:00, 10:15, 10:30, 10:45 = energy of 09:45–10:45 (15 min early),
  the first hour of the year had only 3 quarters and the last quarter of the
  year (stamp 2019-01-01 00:00) was dropped.
  Correct:  hour h = Σ of the stamps (h, h+1h]  ->  closed="right", label="left".

What this script does
  • Takes the building list from data/individual_final/metadata.parquet
    (the sample is NOT drawn again: same 5.000 bldg_id, same cities/types).
  • Detects the timestamp convention IN EACH FILE (first/last stamp) and
    stops if a file does not have the expected 35.040 quarter-hours.
  • Aggregates to 8.760 hours per building, no fillna (a gap = STOP).
  • Keeps every out.*energy_consumption* column (not *intensity*) at full
    float64 precision.
  • Writes one parquet file per (city, type) chunk through ONE ParquetWriter
    (low memory) to data/individual_final/buildings_v3.parquet.
    The original buildings.parquet is NOT touched (kept as archive).
  • Resumable: finished buildings are cached in data/individual_final/_v3_cache/
    (one small parquet per building) — if the run stops, start it again.
Run from the repository root:   python v2/00a_REDOWNLOAD_v3.py   (or v2\\RUN_ALL.bat)
"""
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config

import config as C

WORKERS = int(os.environ.get("V3_WORKERS", 16))
CACHE = C.DATA_DIR / "_v3_cache"
OUT = C.RAW_BLDG_V3
HOURLY = pd.date_range("2018-01-01 00:00", "2018-12-31 23:00", freq="h")
VERSION = {"residential": C.RES_VER, "commercial": C.COM_VER}
STATE = C.STATE


def s3_client():
    return boto3.client("s3", region_name="us-west-2",
                        config=Config(signature_version=UNSIGNED,
                                      max_pool_connections=WORKERS + 4,
                                      retries={"max_attempts": 10, "mode": "standard"}))


def to_hourly(df):
    """
    15-min -> hourly (kWh are summed). Convention detected from the stamps:
      period-ENDING  (first 00:15, last 2019-01-01 00:00) -> closed="right"
      period-START   (first 00:00, last 2018-12-31 23:45) -> closed="left"
    Returns (hourly frame with exactly 8.760 rows, convention string).
    """
    ts = pd.DatetimeIndex(pd.to_datetime(df.index))
    if ts.tz is not None:                       # NREL files are naive local standard time
        raise ValueError(f"unexpected time zone {ts.tz}")
    if len(ts) != 35040 or ts.has_duplicates:
        raise ValueError(f"{len(ts)} stamps (expected 35.040, unique)")
    first, last = ts.min(), ts.max()
    if first == pd.Timestamp("2018-01-01 00:15") and last == pd.Timestamp("2019-01-01 00:00"):
        conv, closed = "period-ending", "right"
    elif first == pd.Timestamp("2018-01-01 00:00") and last == pd.Timestamp("2018-12-31 23:45"):
        conv, closed = "period-start", "left"
    else:
        raise ValueError(f"unknown stamps {first} … {last}")
    df.index = ts
    n_q = pd.Series(1, index=ts).resample("h", closed=closed, label="left").sum()
    if len(n_q) != 8760 or not n_q.eq(4).all():
        raise ValueError("an hour does not contain exactly 4 quarter-hours")
    h = df.resample("h", closed=closed, label="left").sum(min_count=1)
    h = h.reindex(HOURLY)
    if h.isna().all(axis=1).any():
        raise ValueError("empty hour after aggregation")
    return h, conv


def download_one(s3, bldg_id, dtype, city):
    """Download one building, aggregate, cache. Returns (bldg_id, dtype, convention)."""
    f = CACHE / f"{dtype[0].upper()}{bldg_id}.parquet"
    if f.exists():
        return bldg_id, dtype, "cached"
    key = (f"{C.BASE}/{VERSION[dtype]}/timeseries_individual_buildings/by_state/"
           f"upgrade=0/state={STATE[city]}/{bldg_id}-0.parquet")
    body = s3.get_object(Bucket=C.BUCKET, Key=key)["Body"].read()
    df = pd.read_parquet(io.BytesIO(body))
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    keep = [c for c in df.columns if "energy_consumption" in c and "intensity" not in c]
    if not keep:
        raise ValueError("no energy columns")
    h, conv = to_hourly(df[keep].astype("float64"))
    h = h.rename_axis("timestamp").reset_index()
    h.insert(0, "bldg_id", np.int64(bldg_id))
    h.to_parquet(f, index=False)
    return bldg_id, dtype, conv


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(C.RAW_META)[["bldg_id", "city", "dataset_type"]]
    meta["bldg_id"] = meta["bldg_id"].astype("int64")
    assert len(meta) == 5 * (C.N_RES + C.N_COM), f"metadata has {len(meta)} buildings"
    print(f"buildings to download: {len(meta):,} (same list as the original download)")

    # ── 1. download (parallel, resumable) ───────────────────────────
    s3 = s3_client()
    t0, conv_count, errors = time.time(), {}, []
    tasks = list(meta.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut = {ex.submit(download_one, s3, int(r.bldg_id), r.dataset_type, r.city): r
               for r in tasks}
        for k, f in enumerate(as_completed(fut), 1):
            r = fut[f]
            try:
                _, _, conv = f.result()
                conv_count[conv] = conv_count.get(conv, 0) + 1
            except Exception as e:                      # noqa: BLE001
                errors.append((r.dataset_type, int(r.bldg_id), repr(e)))
            if k % 250 == 0 or k == len(tasks):
                print(f"  {k:,}/{len(tasks):,}  ({(time.time() - t0) / 60:.1f} min)  "
                      f"errors: {len(errors)}", flush=True)
    print(f"timestamp conventions: {conv_count}")
    if errors:
        for e in errors[:20]:
            print("  ERROR", e)
        sys.exit(f"STOP: {len(errors)} buildings failed — run the script again "
                 f"(finished buildings are cached).")

    # ── 2. combine into one parquet (one schema, low memory) ─────────
    cols = set()
    for f in CACHE.glob("*.parquet"):
        cols.update(pq.read_schema(f).names)
    energy = sorted(c for c in cols if c not in ("bldg_id", "timestamp"))
    schema = pa.schema([("bldg_id", pa.int64()), ("timestamp", pa.timestamp("ns")),
                        ("city", pa.string()), ("zone", pa.string()),
                        ("dataset_type", pa.string())] + [(c, pa.float64()) for c in energy])
    tmp = OUT.with_suffix(".tmp")
    writer = pq.ParquetWriter(tmp, schema, compression="snappy")
    n_rows = 0
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            ids = sorted(meta.loc[(meta.city == city) & (meta.dataset_type == dtype), "bldg_id"])
            n_ct = 0
            for k in range(0, len(ids), 50):          # 50 buildings per write (low memory)
                frames = []
                for b in ids[k:k + 50]:
                    d = pd.read_parquet(CACHE / f"{dtype[0].upper()}{b}.parquet")
                    if len(d) != C.HOURS:
                        sys.exit(f"STOP: {dtype} {b} has {len(d)} hours")
                    frames.append(d)
                d = pd.concat(frames, ignore_index=True)
                d["timestamp"] = pd.to_datetime(d["timestamp"]).astype("datetime64[ns]")
                d["city"], d["zone"], d["dataset_type"] = city, C.CITY_ZONE[city], dtype
                for c in energy:                   # column absent for this type -> NaN
                    if c not in d.columns:         # (00b turns it into 0 kWh)
                        d[c] = np.nan
                writer.write_table(pa.Table.from_pandas(d[schema.names], schema=schema,
                                                        preserve_index=False))
                n_ct += len(d)
                del d, frames
            n_rows += n_ct
            print(f"  {city:9s} {dtype:11s}: {len(ids)} buildings, {n_ct:,} rows", flush=True)
    writer.close()
    assert n_rows == len(meta) * C.HOURS
    os.replace(tmp, OUT)
    if os.environ.get("V3_KEEP_CACHE") != "1":       # free the disk (~ several GB)
        for f in CACHE.glob("*.parquet"):
            f.unlink()
        CACHE.rmdir()
    print(f"\n✓ {OUT}: {n_rows:,} rows ({OUT.stat().st_size / 1e9:.2f} GB), "
          f"{len(energy)} energy columns. Original buildings.parquet untouched.")
    print("  Next: 00b_REPAIR_DATA.py (reads buildings_v3.parquet)")


if __name__ == "__main__":
    main()
