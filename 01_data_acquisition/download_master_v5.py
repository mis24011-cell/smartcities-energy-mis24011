#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_master_v4.py — ΠΛΗΡΕΣ end-use breakdown (δυναμική ανακάλυψη στηλών)
==========================================================================
Ίδιο δείγμα με v3 (600 res + 400 com/ζώνη = 60/40, seed=42, ΠΑΡΑΛΛΗΛΟ), αλλά:
  • Δεν βασίζεται σε σταθερά ονόματα end-use (που διαφέρουν ResStock↔ComStock).
  • Ανακαλύπτει ΔΥΝΑΜΙΚΑ όλες τις *.energy_consumption στήλες κάθε κτιρίου και
    τις ομαδοποιεί σε ΚΟΙΝΕΣ κατηγορίες: heating/cooling/lighting/equipment/
    ventilation/hotwater/other — για elec & gas. → πλήρη Σχ. 3.18/3.19.
  • Κρατά και τα 5 totals + CO₂ (όπως v3).
OUTPUT: data/processed/merged_master.parquet  (ΑΝΤΙΚΑΘΙΣΤΑ — υπερσύνολο του v3)
Checkpoints: data/processed/checkpoints_v5/  (νέος φάκελος → δεν κάνει resume από v3)
"""
import io, boto3, pandas as pd, numpy as np
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─────────────── CONFIG ───────────────
N_RES_PER_ZONE = 600
N_COM_PER_ZONE = 400
RANDOM_STATE   = 42
WORKERS        = 8
BATCH          = WORKERS * 6
BUCKET = "oedi-data-lake"
NREL_BASE = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
RESSTOCK_VERSION = "2024/resstock_amy2018_release_2"
COMSTOCK_VERSION = "2023/comstock_amy2018_release_2"
PROCESSED_DIR = Path(".") / "data" / "processed"
CKPT_DIR      = PROCESSED_DIR / "checkpoints_v5"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True); CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MERGED = PROCESSED_DIR / "merged_master.parquet"
OUTPUT_STATS  = PROCESSED_DIR / "zone_stats_master.csv"
ZONE_STATE_MAP = {"3A":{"state":"GA"}, "4A":{"state":"MD"}, "5A":{"state":"IL"}, "5B":{"state":"CO"}}

TOTAL_COLS = [
    "out.electricity.total.energy_consumption", "out.natural_gas.total.energy_consumption",
    "out.fuel_oil.total.energy_consumption", "out.propane.total.energy_consumption",
    "out.site_energy.total.energy_consumption",
]
def _kwh(c): return c + ".kwh"
OUT_TOTALS = [_kwh(c) for c in TOTAL_COLS]

# Κοινές κατηγορίες end-use (ίδιες για res & com)
EU_CATS = ["heating","cooling","lighting","equipment","ventilation","hotwater","other"]
EU_COLS = [f"eu.{c}.elec" for c in EU_CATS] + [f"eu.{c}.gas" for c in EU_CATS]
ALL_OUT_COLS = OUT_TOTALS + EU_COLS

EF_ELEC = {"3A":0.362, "4A":0.331, "5A":0.285, "5B":0.412}   # Cambium 2022
EF_GAS, EF_OIL, EF_PROPANE = 0.1824, 0.2650, 0.2238          # EPA 2023
HOURLY_INDEX = pd.date_range("2018-01-01 00:00", "2018-12-31 23:00", freq="h")

# ── Κατηγοριοποίηση μιας end-use στήλης σε (κατηγορία, καύσιμο) ─────────────────
_EQUIP_KW = ("plug","equipment","appliance","refrigerator","freezer","dishwasher",
             "clothes","cooking","range","oven","tv","pool","hot_tub","spa","well",
             "vehicle","ev_charg","misc","pv","battery")
def categorize(col):
    c = col.lower()
    if not c.endswith(".energy_consumption"): return None
    if ".total." in c or ".net." in c: return None
    fuel = "elec" if "electricity" in c else ("gas" if "natural_gas" in c else None)
    if fuel is None: return None
    if "cooling" in c: cat="cooling"
    elif ("hot_water" in c) or ("water_systems" in c) or ("water_heat" in c): cat="hotwater"
    elif "heating" in c and "water" not in c: cat="heating"
    elif "lighting" in c: cat="lighting"
    elif any(k in c for k in _EQUIP_KW): cat="equipment"
    elif ("fan" in c) or ("vent" in c): cat="ventilation"
    else: cat="other"
    return f"eu.{cat}.{fuel}"

def get_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=WORKERS+4),
                        region_name="us-west-2")
def list_files(s3, version, state):
    prefix = f"{NREL_BASE}/{version}/timeseries_individual_buildings/by_state/upgrade=0/state={state}/"
    pg = s3.get_paginator("list_objects_v2"); out = []
    for page in pg.paginate(Bucket=BUCKET, Prefix=prefix):
        out += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]
    return out

def read_one_hourly(s3, key):
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    d = pd.read_parquet(io.BytesIO(body))
    if "timestamp" in d.columns: d = d.set_index("timestamp")
    d.index = pd.to_datetime(d.index)
    out = pd.DataFrame(0.0, index=d.index, columns=ALL_OUT_COLS)
    got = False
    # totals
    for c in TOTAL_COLS:
        if c in d.columns: out[_kwh(c)] = d[c].fillna(0); got = True
    # end-uses -> δυναμική ομαδοποίηση
    for col in d.columns:
        cat = categorize(col)
        if cat is not None:
            out[cat] = out[cat] + d[col].fillna(0); got = True
    if not got: return None
    return out.resample("h").mean().reindex(index=HOURLY_INDEX, columns=ALL_OUT_COLS).fillna(0.0)

def aggregate_unit(s3, zone, dtype, version, state, seed, n_target):
    keys = list_files(s3, version, state)
    if not keys: return None
    rng = np.random.RandomState(seed)
    sel = rng.choice(len(keys), min(n_target, len(keys)), replace=False)
    sel_keys = [keys[i] for i in sel]
    acc = pd.DataFrame(0.0, index=HOURLY_INDEX, columns=ALL_OUT_COLS); ok = fail = 0
    pbar = tqdm(total=len(sel_keys), desc=f"{zone} {dtype:11s}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for start in range(0, len(sel_keys), BATCH):
            batch = sel_keys[start:start+BATCH]
            futs = {ex.submit(read_one_hourly, s3, k): k for k in batch}
            for fut in as_completed(futs):
                try:
                    b = fut.result()
                    if b is not None: acc += b; ok += 1
                    else: fail += 1
                except Exception: fail += 1
                pbar.update(1)
    pbar.close()
    if ok == 0: return None
    return _finalize(acc / ok, zone, dtype, ok)

def _finalize(df, zone, dtype, n_buildings):
    df = df.copy(); ef_e = EF_ELEC.get(zone, 0.35)
    elec=df[_kwh("out.electricity.total.energy_consumption")]; gas=df[_kwh("out.natural_gas.total.energy_consumption")]
    oil=df[_kwh("out.fuel_oil.total.energy_consumption")]; prop=df[_kwh("out.propane.total.energy_consumption")]
    site=_kwh("out.site_energy.total.energy_consumption")
    if (df[site]==0).all(): df[site]=elec+gas+oil+prop
    df["co2_elec_kg"]=elec*ef_e; df["co2_gas_kg"]=gas*EF_GAS
    df["co2_oil_kg"]=oil*EF_OIL; df["co2_propane_kg"]=prop*EF_PROPANE
    df["co2_total_kg"]=df[["co2_elec_kg","co2_gas_kg","co2_oil_kg","co2_propane_kg"]].sum(axis=1)
    df["emission_intensity"]=df["co2_total_kg"]/(df[site]+1e-9)
    df["zone"]=zone; df["dataset"]="resstock" if dtype=="residential" else "comstock"
    df["dataset_type"]=dtype; df["btype"]=f"{'res' if dtype=='residential' else 'com'}_{zone.lower()}"
    df["n_buildings"]=n_buildings
    return df

def main():
    tot=(N_RES_PER_ZONE+N_COM_PER_ZONE)*len(ZONE_STATE_MAP)
    log.info("="*64)
    log.info(f"MASTER v5 (καθαρά end-uses) — {N_RES_PER_ZONE} res + {N_COM_PER_ZONE} com/ζώνη = {tot} κτ. · {WORKERS} workers")
    log.info("="*64)
    s3 = get_s3()
    try: s3.head_bucket(Bucket=BUCKET); log.info("✓ S3 προσβάσιμο")
    except Exception as e: log.error(f"✗ S3 fail: {e}"); return
    units=[]
    for zi,(zone,cfg) in enumerate(ZONE_STATE_MAP.items()):
        units.append((zone,"residential",RESSTOCK_VERSION,cfg["state"],RANDOM_STATE+zi*2+0,N_RES_PER_ZONE))
        units.append((zone,"commercial", COMSTOCK_VERSION,cfg["state"],RANDOM_STATE+zi*2+1,N_COM_PER_ZONE))
    for zone,dtype,version,state,seed,n_target in units:
        ckpt=CKPT_DIR/f"{zone}_{dtype}.parquet"
        if ckpt.exists(): log.info(f"↩︎  {zone}/{dtype}: checkpoint — παράλειψη."); continue
        log.info(f"► {zone}/{dtype} ({state}) — στόχος {n_target}")
        res=aggregate_unit(s3,zone,dtype,version,state,seed,n_target)
        if res is not None: res.to_parquet(ckpt,compression="snappy"); log.info(f"  💾 {ckpt.name}")
    parts=sorted(CKPT_DIR.glob("*.parquet"))
    if not parts: log.error("Κανένα checkpoint."); return
    merged=pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    merged.to_parquet(OUTPUT_MERGED,compression="snappy")
    log.info(f"\n✓ {OUTPUT_MERGED} ({OUTPUT_MERGED.stat().st_size/1e6:.1f} MB) | {len(merged):,} γραμμές · {len(merged.columns)} στήλες")
    site=_kwh("out.site_energy.total.energy_consumption")
    stats=merged.groupby(["zone","dataset_type"]).agg(
        n_buildings=("n_buildings","first"), energy_mean=(site,"mean"),
        energy_max=(site,"max"), co2_mean=("co2_total_kg","mean"), co2_annual=("co2_total_kg","sum")).round(3)
    stats.to_csv(OUTPUT_STATS)
    log.info(f"✓ {OUTPUT_STATS}\n{stats.to_string()}")
    # έλεγχος end-uses
    for dt in ["residential","commercial"]:
        d=merged[merged.dataset_type==dt]
        nz=[c for c in EU_COLS if d[c].sum()>0]
        log.info(f"[{dt}] end-use κατηγορίες με δεδομένα: {len(nz)}/{len(EU_COLS)} → {[c.split('.')[1] for c in nz]}")
    log.info("\n>>> Ανέβασέ μου: merged_master.parquet + zone_stats_master.csv")

if __name__ == "__main__":
    main()
