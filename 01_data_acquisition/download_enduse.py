#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_enduse.py — Κατέβασμα ΤΕΛΙΚΩΝ ΧΡΗΣΕΩΝ (end-uses) για τα Σχ. 3.18 & 3.19
==========================================================================
Ίδια μεθοδολογία/δείγμα με το download_all_zones.py (seed=42, 400 κτίρια/ζώνη),
αλλά κρατά τις στήλες ΑΝΑ ΤΕΛΙΚΗ ΧΡΗΣΗ (θέρμανση/ψύξη/φωτισμός/εξοπλισμός)
που χρειάζονται για τα δύο end-use διαγράμματα.

ΧΡΗΣΗ (τοπικά, στον φάκελο της Φάσης 2):  python download_enduse.py
OUTPUT: data/processed/enduse_all_zones.parquet
"""
import io, boto3, pandas as pd, numpy as np
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log=logging.getLogger()

N_BUILDINGS=400; RANDOM_STATE=42
BUCKET="oedi-data-lake"
NREL_BASE="nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
RES_VER="2024/resstock_amy2018_release_2"; COM_VER="2023/comstock_amy2018_release_2"
OUT=Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
ZONE_STATE={"3A":"GA","4A":"MD","5A":"IL","5B":"CO"}

# ΤΕΛΙΚΕΣ ΧΡΗΣΕΙΣ — κύριες συνιστώσες (ηλεκτρισμός + αέριο) ανά χρήση
ENDUSE=[
 "out.electricity.heating.energy_consumption","out.electricity.cooling.energy_consumption",
 "out.electricity.interior_lighting.energy_consumption","out.electricity.interior_equipment.energy_consumption",
 "out.electricity.fans.energy_consumption","out.electricity.water_systems.energy_consumption",
 "out.natural_gas.heating.energy_consumption","out.natural_gas.water_systems.energy_consumption",
]
def s3c(): return boto3.client("s3",config=Config(signature_version=UNSIGNED),region_name="us-west-2")
def list_files(s3,ver,st):
    pfx=f"{NREL_BASE}/{ver}/timeseries_individual_buildings/by_state/upgrade=0/state={st}/"
    out=[]; pg=s3.get_paginator("list_objects_v2")
    for p in pg.paginate(Bucket=BUCKET,Prefix=pfx):
        out+=[o["Key"] for o in p.get("Contents",[]) if o["Key"].endswith(".parquet")]
    return out
def read_one(s3,key):
    body=s3.get_object(Bucket=BUCKET,Key=key)["Body"].read()
    d=pd.read_parquet(io.BytesIO(body))
    if "timestamp" in d.columns: d=d.set_index("timestamp")
    d.index=pd.to_datetime(d.index)
    cols=[c for c in ENDUSE if c in d.columns]
    if not cols: return None
    return d[cols].resample("h").mean()
def grab(s3,keys,zone,label,rng):
    if not keys: return []
    sel=rng.choice(len(keys),min(N_BUILDINGS//2,len(keys)),replace=False); dfs=[]
    for i in tqdm(sel,desc=f"{zone} {label}"):
        try:
            s=read_one(s3,keys[i])
            if s is not None: dfs.append(s)
        except Exception: pass
    return dfs
def main():
    s3=s3c()
    try: s3.head_bucket(Bucket=BUCKET); log.info("S3 OK")
    except Exception as e: log.error(f"S3 fail: {e}"); return
    all_dfs=[]
    for zone,st in ZONE_STATE.items():
        rng=np.random.RandomState(RANDOM_STATE)
        for ver,lbl,dt in [(RES_VER,"ResStock","residential"),(COM_VER,"ComStock","commercial")]:
            keys=list_files(s3,ver,st)
            dfs=grab(s3,keys,zone,lbl,rng)
            if not dfs: continue
            agg=pd.concat(dfs).groupby(level=0).mean()["2018-01-01":"2018-12-31"].ffill().fillna(0)
            agg["zone"]=zone; agg["dataset_type"]=dt
            all_dfs.append(agg)
            log.info(f"  {zone} {lbl}: {len(agg)}h, {len([c for c in agg.columns if 'energy' in c])} end-use cols")
    merged=pd.concat(all_dfs).sort_index()
    fp=OUT/"enduse_all_zones.parquet"; merged.to_parquet(fp,compression="snappy")
    log.info(f"✓ {fp} ({fp.stat().st_size/1e6:.1f} MB) | {len(merged):,} γραμμές")
    log.info(">>> Ανέβασέ μου το enduse_all_zones.parquet για τα Σχ. 3.18 & 3.19.")
if __name__=="__main__": main()
