#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00c_VERIFY_ANNUAL.py  —  STEP A2: completeness check against NREL (≈5 min)
Thesis MIS24011 — Zygouris Aristotelis

Purpose
  The original 00_DOWNLOAD filled missing hours with 0 (reindex + fillna).
  A building with long runs of zero consumption is therefore either
    (a) a genuine simulation result (e.g. vacant dwelling, no HVAC), or
    (b) an incomplete download.
  NREL publishes the ANNUAL site energy of every building in the baseline
  metadata. If Σ of our hourly values equals NREL's annual value, the
  hourly series is complete; if it is lower, hours are missing.

Output
  data/v2/annual_check.csv     one row per building (ours vs NREL, % diff)
  appended to data/v2/diagnostics.txt
Run AFTER 00b, from the repository root:  python v2/00c_VERIFY_ANNUAL.py
"""
import argparse
import io
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from common import site_total_col

TOL_PCT = 1.0          # |difference| above 1 % -> building flagged


def nrel_annual(version, local_path=None):
    """Annual site energy (kWh) per bldg_id from NREL baseline metadata."""
    if local_path:
        src = local_path
    else:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2")
        key = f"{C.BASE}/{version}/metadata/baseline.parquet"
        print(f"  downloading s3://{C.BUCKET}/{key} ...")
        src = io.BytesIO(s3.get_object(Bucket=C.BUCKET, Key=key)["Body"].read())
    names = pq.read_schema(src).names
    def ok(c):
        lc = c.lower()
        return (c.startswith("out.") and "energy_consumption" in lc and "intensity" not in lc
                and "savings" not in lc and ".net." not in lc and "co2" not in lc)
    site = [c for c in names if ok(c) and "site_energy" in c and ".total." in c]
    site = sorted(site, key=lambda c: (not c.endswith("kwh"), c))
    if site:
        cols = [site[0]]
    else:   # no site total -> sum of the fuel totals
        cols = [c for c in names if ok(c) and re.match(
            r"^out\.(electricity|natural_gas|propane|fuel_oil|other_fuel|district_heating|"
            r"district_cooling)\.total\.energy_consumption", c)]
    if not cols:
        cand = [c for c in names if c.startswith("out.") and "energy" in c.lower()][:60]
        raise SystemExit(f"STOP: no annual energy column in {version}. Candidates: {cand}")
    if hasattr(src, "seek"):
        src.seek(0)
    want = cols + (["bldg_id"] if "bldg_id" in names else [])
    df = pq.read_table(src, columns=want, use_pandas_metadata=True).to_pandas()
    if "bldg_id" in df.columns:
        df = df.set_index("bldg_id")
    df.index = df.index.astype("int64")          # same convention as 00_DOWNLOAD / 00b
    label = cols[0] if len(cols) == 1 else "Σ " + " + ".join(cols)
    print(f"  {version}: {label} ({len(df):,} buildings)")
    return df[cols].astype("float64").sum(axis=1, min_count=1), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-res"); ap.add_argument("--local-com")
    args = ap.parse_args()
    lines = ["", "[7] Completeness check: Σ hourly (ours) vs NREL annual site energy"]

    site = site_total_col(pq.read_schema(C.BLDG_V2).names)
    t = pq.read_table(C.BLDG_V2, columns=["building_uid", site])
    ours = t.group_by("building_uid").aggregate([(site, "sum")]).to_pandas()
    ours.columns = ["building_uid", "ours_kwh"]
    meta = pd.read_parquet(C.META_V2)[["building_uid", "bldg_id", "city", "dataset_type"]
                                      + [c for c in ("vacancy_status",) if c in
                                         pq.read_schema(C.META_V2).names]]
    df = meta.merge(ours, on="building_uid", how="left", validate="one_to_one")

    res, rc = nrel_annual(C.RES_VER, args.local_res)
    com, cc = nrel_annual(C.COM_VER, args.local_com)
    lines.append(f"    NREL columns: res={rc} | com={cc}")
    df["nrel_kwh"] = np.where(df["dataset_type"] == "residential",
                              df["bldg_id"].map(res), df["bldg_id"].map(com))
    df["diff_pct"] = (df["ours_kwh"] - df["nrel_kwh"]) / df["nrel_kwh"].where(df["nrel_kwh"] != 0) * 100
    both_zero = (df["ours_kwh"].abs() < 1e-6) & (df["nrel_kwh"].abs() < 1e-6)
    df.loc[both_zero, "diff_pct"] = 0.0
    df["flag_incomplete"] = df["diff_pct"].abs() > TOL_PCT

    qc_path = C.V2_DIR / "qc_by_building.parquet"
    if qc_path.exists():
        qc = pd.read_parquet(qc_path)[["building_uid", "zero_h", "max_zero_run"]]
        df = df.merge(qc, on="building_uid", how="left")
    df.to_csv(C.V2_DIR / "annual_check.csv", index=False, encoding="utf-8-sig")

    s = df.groupby(["city", "dataset_type"]).agg(
        N=("building_uid", "size"), missing_nrel=("nrel_kwh", lambda x: int(x.isna().sum())),
        median_abs_diff_pct=("diff_pct", lambda x: float(x.abs().median())),
        max_abs_diff_pct=("diff_pct", lambda x: float(x.abs().max())),
        flagged=("flag_incomplete", "sum"))
    ratio = (df["ours_kwh"] / df["nrel_kwh"]).replace([np.inf, -np.inf], np.nan)
    lines.append("    median ratio ours / NREL by type (≈1 expected; a constant ≠1 = unit difference): "
                 + str(ratio.groupby(df["dataset_type"]).median().round(4).to_dict()))
    lines.append(f"    tolerance: |Σ ours − NREL annual| > {TOL_PCT} % -> incomplete")
    lines.append(s.round(4).to_string())
    bad = df[df["flag_incomplete"] | (df.get("max_zero_run", 0) >= C.QC_MAX_ZERO_RUN_H)]
    cols = [c for c in ("building_uid", "city", "dataset_type", "vacancy_status", "ours_kwh",
                        "nrel_kwh", "diff_pct", "zero_h", "max_zero_run") if c in bad.columns]
    lines.append(f"    buildings flagged (incomplete or long zero run): {len(bad)}")
    if len(bad):
        lines.append(bad[cols].round(2).to_string(index=False))
    lines.append(f"    -> {C.V2_DIR / 'annual_check.csv'}")

    # Which columns carry the negative values flagged in [6]? (PV generation
    # is reported by NREL as negative consumption -> expected, not an error)
    if qc_path.exists():
        qn = pd.read_parquet(qc_path)
        neg_uids = qn.loc[qn["neg_values"] > 0, "building_uid"].tolist()
        if neg_uids:
            names = pq.read_schema(C.BLDG_V2).names
            ecols = [c for c in names if c.startswith("out.") and "energy_consumption" in c]
            tn = pq.read_table(C.BLDG_V2, columns=["building_uid"] + ecols,
                               filters=[("building_uid", "in", neg_uids)]).to_pandas()
            negc = (tn[ecols] < 0).sum()
            negc = negc[negc > 0].sort_values(ascending=False)
            lines.append(f"    negative values in {len(neg_uids)} buildings, by column (hours):")
            lines.append(negc.to_string())
            lines.append(f"    is the site total ever negative? {bool((tn[site] < 0).any())}")
    txt = "\n".join(lines)
    print(txt)
    with open(C.DIAG_TXT, "a", encoding="utf-8") as f:
        f.write(txt + "\n")
    n_bad = int(df["flag_incomplete"].sum()) + int(df["nrel_kwh"].isna().sum())
    if n_bad:
        raise SystemExit(f"STOP: {n_bad} buildings differ from NREL by more than {TOL_PCT} % "
                         f"(or have no NREL value) — see data/v2/annual_check.csv")
    print("\n✓ STEP A2 finished. Send data/v2/diagnostics.txt and data/v2/annual_check.csv")


if __name__ == "__main__":
    main()
