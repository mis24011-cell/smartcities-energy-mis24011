# -*- coding: utf-8 -*-
"""
recalibrate_and_regen.py — Βήμα 7 (re-run) στα ΠΡΑΓΜΑΤΙΚΑ δεδομένα.
------------------------------------------------------------------------------
ΣΚΟΠΟΣ: το synthetic baseline (awp.parquet) υπερεκτιμά την κατανάλωση
(οικιστικά ~7-9×, εμπορικά ~1,5-2×). Το βαθμονομούμε ώστε οι ΜΕΣΕΣ τιμές
ανά ζώνη×τύπο να ταυτίζονται με τα ΠΡΑΓΜΑΤΙΚΑ (merged_all_zones), διατηρώντας:
  • την ετερογένεια ανά κτίριο (πολλαπλασιαστική κλιμάκωση ανά ομάδα),
  • τα χαρακτηριστικά κτιρίου (vintage/μόνωση/εισόδημα) για τα social/ESG figures,
  • τη ΜΕΘΟΔΟ 2×3 convex-cooling (coeffs, HDD/CDD baseline αμετάβλητα — κλίμα).
Επιπλέον κρατάμε το ΠΡΑΓΜΑΤΙΚΟ μερίδιο ηλεκτρισμού (elec/site) & αερίου ανά
ζώνη×τύπο ώστε οι εκπομπές CO₂ να είναι συνεπείς με το πραγματικό μείγμα καυσίμων.

ΕΞΟΔΟΣ: awp_real_baseline.parquet (baseline), scenarios_2x3_baseline.parquet (7 σενάρια).
"""
import pandas as pd, numpy as np
from climate_scenarios_v2 import build_scenarios, local_delta

SITE="out.site_energy.total.energy_consumption.kwh"
ELEC="out.electricity.total.energy_consumption.kwh"
GAS="out.natural_gas.total.energy_consumption.kwh"
ENERGY_COMPONENTS=["annual_kwh","heating_kwh","cooling_kwh","lighting_kwh","equipment_kwh"]

# ---------- 1) ΠΡΑΓΜΑΤΙΚΑ μεγέθη ανά ζώνη×τύπο ----------
def real_targets(p="merged_all_zones.parquet"):
    m=pd.read_parquet(p)
    g=m.groupby(["zone","dataset_type"]).agg(site=(SITE,"sum"),elec=(ELEC,"sum"),gas=(GAS,"sum"))
    g["elec_share"]=(g["elec"]/g["site"]).clip(0,1)          # μερίδιο ηλεκτρισμού (για CO₂)
    return g  # index: (zone, 'residential'/'commercial')

# ---------- 2) Βαθμονόμηση synthetic baseline -> real means ----------
def recalibrate(awp_path="awp.parquet", out="awp_real_baseline.parquet"):
    a=pd.read_parquet(awp_path)
    b=a[a.scenario.str.contains("Baseline")].copy().reset_index(drop=True)
    tgt=real_targets()
    # μέσες synthetic τιμές ανά ομάδα (για τον λόγο κλιμάκωσης)
    grp=b.groupby(["climate_zone","building_type"])
    awp_annual=grp["annual_kwh"].transform("mean")
    awp_gas=grp["natural_gas_kwh"].transform("mean").replace(0,np.nan)
    # στόχοι ανά γραμμή (map μέσω πολυ-index)
    key=list(zip(b["climate_zone"],b["building_type"]))
    tgt_site=np.array([tgt.loc[k,"site"] for k in key]) / grp["annual_kwh"].transform("size").values  # mean real site
    # ΔΙΟΡΘΩΣΗ: το tgt.site είναι ΑΘΡΟΙΣΜΑ ωριαίων -> ήδη «ανά μέσο κτίριο/έτος», όχι /n
    tgt_site=np.array([tgt.loc[k,"site"] for k in key])
    tgt_gas =np.array([tgt.loc[k,"gas"]  for k in key])
    esh     =np.array([tgt.loc[k,"elec_share"] for k in key])
    # πολλαπλασιαστικοί συντελεστές (διατηρούν ετερογένεια)
    f_energy = tgt_site / awp_annual.values
    f_gas    = tgt_gas  / awp_gas.values
    f_gas    = np.where(np.isnan(f_gas), 1.0, f_gas)
    for c in ENERGY_COMPONENTS:
        b[c]=b[c].values * f_energy
    b["natural_gas_kwh"]=b["natural_gas_kwh"].values * f_gas
    b["elec_share"]=esh                                  # για CO₂ (πραγματικό μείγμα)
    # ενημέρωση EUI baseline (αν υπάρχει area)
    if "floor_area_m2" in b.columns:
        b["eui_baseline"]=b["annual_kwh"]/b["floor_area_m2"].replace(0,np.nan)
    b.to_parquet(out,index=False)
    return b, tgt

# ---------- 3) Παραγωγή σεναρίων 2×3 στο βαθμονομημένο baseline ----------
def regen_scenarios(baseline_df, out="scenarios_2x3_baseline.parquet"):
    HOR={y:{p:local_delta(p,y) for p in ["SSP2-4.5","SSP5-8.5"]} for y in ["2040","2050","2100"]}
    df=build_scenarios(baseline_df, HOR)
    # διάδοση elec_share (σταθερό ανά κτίριο) στα σενάρια
    if "elec_share" not in df.columns:
        share=baseline_df.set_index("building_id")["elec_share"].to_dict()
        df["elec_share"]=df["building_id"].map(share)
    df.to_parquet(out,index=False)
    return df, HOR

if __name__=="__main__":
    b,tgt=recalibrate()
    print("✓ awp_real_baseline.parquet:",b.shape)
    print("\nΈλεγχος μέσων (real target vs βαθμονομημένο):")
    chk=b.groupby(["climate_zone","building_type"]).agg(
        cal_site=("annual_kwh","mean"), cal_gas=("natural_gas_kwh","mean")).round(0)
    tgt2=tgt[["site","gas"]].rename(columns={"site":"real_site","gas":"real_gas"}).round(0)
    tgt2.index=tgt2.index.set_names(["climate_zone","building_type"])
    print(chk.join(tgt2))
    df,HOR=regen_scenarios(b)
    print("\n✓ scenarios_2x3_baseline.parquet:",df.shape,"| σενάρια:",sorted(df.scenario.unique()))
    print("HORIZONS deltas:",HOR)
