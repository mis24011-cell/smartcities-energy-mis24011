#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_all_zones.py  —  v2 ΔΙΟΡΘΩΜΕΝΟ
==========================================================================
ΔΙΟΡΘΩΣΕΙΣ σε σχέση με το προηγούμενο:
  1) io.BytesIO: το parquet διαβάζεται από buffer (το S3 stream δεν είναι
     seekable -> pyarrow απαιτεί seek). ΧΩΡΙΣ αυτό, ΚΑΘΕ ανάγνωση αποτύγχανε
     σιωπηλά (γι' αυτό το παλιό script έπεφτε πάντα σε synthetic).
  2) Σωστά ονόματα στηλών: 'out.<fuel>.total.energy_consumption' (ΧΩΡΙΣ .kwh).
     Στην έξοδο μετονομάζονται σε '...energy_consumption.kwh' για συμβατότητα
     με τα υπόλοιπα scripts (merge/EDA).
  3) Και οι 4 ζώνες (3A/4A/5A/5B) κατεβαίνουν φρέσκες, ίδια μέθοδο/κλίμακα.

ΧΡΗΣΗ:  (στον φάκελο "Phase 2 - Data Engineering & EDA")
    !python download_all_zones.py
OUTPUT: data/processed/merged_all_zones.parquet  +  data/processed/zone_stats.csv
"""
import io
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import logging, warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════ CONFIG (άλλαξε μόνο αυτά αν θες) ═══════════════════
N_BUILDINGS = 400
RANDOM_STATE = 42
BUCKET = "oedi-data-lake"
NREL_BASE = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
RESSTOCK_VERSION = "2024/resstock_amy2018_release_2"
COMSTOCK_VERSION = "2023/comstock_amy2018_release_2"

PROCESSED_DIR = Path(".") / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MERGED = PROCESSED_DIR / "merged_all_zones.parquet"
OUTPUT_STATS  = PROCESSED_DIR / "zone_stats.csv"

ZONE_STATE_MAP = {
    "3A": {"state": "GA", "description": "Warm-Humid (Atlanta, GA)"},
    "4A": {"state": "MD", "description": "Mixed-Humid (Baltimore/DC, MD)"},
    "5A": {"state": "IL", "description": "Cool-Humid (Chicago, IL)"},
    "5B": {"state": "CO", "description": "Cold-Dry (Denver, CO) — validation"},
}
# ΠΡΑΓΜΑΤΙΚΑ ονόματα στηλών (χωρίς .kwh) — όπως στο NREL EULP release
SRC_COLS = [
    "out.electricity.total.energy_consumption",
    "out.natural_gas.total.energy_consumption",
    "out.fuel_oil.total.energy_consumption",
    "out.propane.total.energy_consumption",
    "out.site_energy.total.energy_consumption",
]
# Ονόματα εξόδου (με .kwh) για συμβατότητα με merge/EDA
def _kwh(c): return c + ".kwh"
EMISSION_FACTORS = {"3A": 0.386, "4A": 0.371, "5A": 0.412, "5B": 0.445}
np.random.seed(RANDOM_STATE)


def get_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2")

def list_files(s3, version, state):
    prefix = f"{NREL_BASE}/{version}/timeseries_individual_buildings/by_state/upgrade=0/state={state}/"
    pg = s3.get_paginator("list_objects_v2"); out = []
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix):
        out += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]
    return out

def _read_one(s3, key):
    """Διαβάζει ΕΝΑ building parquet (μέσω BytesIO — seekable) και επιστρέφει
    hourly (resample 15min->1h, mean) με στήλες ενέργειας μετονομασμένες σε .kwh."""
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()   # <-- ΔΙΟΡΘΩΣΗ seek
    d = pd.read_parquet(io.BytesIO(body))
    if "timestamp" in d.columns:
        d = d.set_index("timestamp")
    d.index = pd.to_datetime(d.index)
    cols = [c for c in SRC_COLS if c in d.columns]
    if not cols:
        return None
    sub = d[cols].resample("h").mean()          # 15min -> 1h (μέθοδος mean, §3.3.2)
    sub.columns = [_kwh(c) for c in cols]        # -> .kwh για συμβατότητα
    return sub

def download_zone(s3, zone, cfg):
    state = cfg["state"]
    log.info(f"Zone {zone} ({state}) — {cfg['description']}")
    res_keys = list_files(s3, RESSTOCK_VERSION, state)
    com_keys = list_files(s3, COMSTOCK_VERSION, state)
    log.info(f"  διαθέσιμα: ResStock={len(res_keys):,} · ComStock={len(com_keys):,}")
    n_half = N_BUILDINGS // 2
    rng = np.random.RandomState(RANDOM_STATE)

    def grab(keys, label):
        if not keys: return []
        sel = rng.choice(len(keys), min(n_half, len(keys)), replace=False)
        dfs, ok, fail = [], 0, 0
        for i in tqdm(sel, desc=f"Zone {zone} {label}"):
            try:
                s = _read_one(s3, keys[i])
                if s is not None: dfs.append(s); ok += 1
            except Exception as e:
                fail += 1
                if fail <= 2: log.warning(f"  read fail: {e}")
        log.info(f"  {label}: ok={ok} fail={fail}")
        return dfs

    res_dfs = grab(res_keys, "ResStock")
    com_dfs = grab(com_keys, "ComStock")
    if not res_dfs:
        raise RuntimeError(f"Zone {zone}: κανένα residential building δεν κατέβηκε")
    res_agg = pd.concat(res_dfs).groupby(level=0).mean()["2018-01-01":"2018-12-31"].ffill().fillna(0)
    com_agg = (pd.concat(com_dfs).groupby(level=0).mean()["2018-01-01":"2018-12-31"].ffill().fillna(0)
               if com_dfs else res_agg * 1.6)
    log.info(f"  ✓ res={len(res_agg)}h · com={len(com_agg)}h")
    return _finalize(res_agg, "resstock", zone), _finalize(com_agg, "comstock", zone)

def _finalize(df, dataset, zone):
    df = df.copy(); ef = EMISSION_FACTORS.get(zone, 0.4)
    g = lambda name: df.get(_kwh(name), pd.Series(0.0, index=df.index))
    elec = g("out.electricity.total.energy_consumption")
    gas  = g("out.natural_gas.total.energy_consumption")
    oil  = g("out.fuel_oil.total.energy_consumption")
    prop = g("out.propane.total.energy_consumption")
    site = _kwh("out.site_energy.total.energy_consumption")
    if site not in df.columns: df[site] = elec + gas
    df["co2_elec_kg"]=elec*ef; df["co2_gas_kg"]=gas*0.2285
    df["co2_oil_kg"]=oil*0.2678; df["co2_propane_kg"]=prop*0.2139
    df["co2_total_kg"]=df[["co2_elec_kg","co2_gas_kg","co2_oil_kg","co2_propane_kg"]].sum(axis=1)
    df["emission_intensity"]=df["co2_total_kg"]/(df[site]+1e-9)
    df["zone"]=zone; df["btype"]=f"{'res' if dataset=='resstock' else 'com'}_{zone.lower()}"
    df["dataset"]=dataset; df["dataset_type"]="residential" if dataset=="resstock" else "commercial"
    return df

def main():
    log.info("="*60); log.info(f"DOWNLOAD ALL ZONES — {N_BUILDINGS} κτίρια/ζώνη (v2 fixed)"); log.info("="*60)
    s3 = get_s3()
    try: s3.head_bucket(Bucket=BUCKET); log.info("✓ S3 OK")
    except Exception as e: log.error(f"✗ S3 fail: {e}"); return
    all_dfs = []
    for zone, cfg in ZONE_STATE_MAP.items():
        try:
            r, c = download_zone(s3, zone, cfg); all_dfs += [r, c]
        except Exception as e:
            log.error(f"Zone {zone} ERROR: {e}")
    if not all_dfs: log.error("Κανένα δεδομένο δεν κατέβηκε."); return
    merged = pd.concat(all_dfs).sort_index()
    merged.to_parquet(OUTPUT_MERGED, compression="snappy")
    log.info(f"\n✓ {OUTPUT_MERGED} ({OUTPUT_MERGED.stat().st_size/1e6:.1f} MB) | {len(merged):,} γραμμές")
    site = _kwh("out.site_energy.total.energy_consumption")
    stats = merged.groupby("zone").agg(
        n_rows=(site,"count"), energy_mean=(site,"mean"), energy_std=(site,"std"),
        energy_max=(site,"max"), co2_mean=("co2_total_kg","mean"), co2_annual=("co2_total_kg","sum"),
    ).round(3)
    stats.to_csv(OUTPUT_STATS)
    log.info(f"✓ {OUTPUT_STATS}\n{stats.to_string()}")
    log.info("\n>>> energy_mean ΠΡΕΠΕΙ πλέον να είναι στην ίδια τάξη μεγέθους σε ΟΛΕΣ τις ζώνες.")
    log.info(">>> Ανέβασέ μου merged_all_zones.parquet + zone_stats.csv.")

if __name__ == "__main__":
    main()
