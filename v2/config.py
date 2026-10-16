# -*- coding: utf-8 -*-
"""
config.py — Shared configuration for the corrected pipeline (v2)
Thesis MIS24011 — Zygouris Aristotelis

Every number that comes from an external source is defined ONCE here,
together with its source, so that text, tables and figures can never
disagree with each other again (professor comment #15).
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Paths (run all scripts from the repository root, which contains data/ and outputs_v2/)
# ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/individual_final")
RAW_BLDG_V1 = DATA_DIR / "buildings.parquet"      # original download (archive, NOT used)
# v3 (12/10/2026): same 5.000 buildings, 15-min -> hourly with the NREL
# period-ending convention (00a_REDOWNLOAD_v3.py). Every later step reads this.
RAW_BLDG_V3 = DATA_DIR / "buildings_v3.parquet"
RAW_BLDG    = RAW_BLDG_V3
RAW_META    = DATA_DIR / "metadata.parquet"       # original metadata (kept untouched)
WEATHER     = DATA_DIR / "weather.parquet"

V2_DIR      = Path("data/v2")                     # every corrected file goes here
BLDG_V2     = V2_DIR / "buildings_v2.parquet"
META_V2     = V2_DIR / "metadata_v2.parquet"
FEAT_V2     = V2_DIR / "ml_features_v2.parquet"
WEATHER_V2  = V2_DIR / "weather_v2.parquet"       # weather exactly as joined (hour-starting, hourly means)
SUMMARY     = V2_DIR / "data_summary.json"        # all counts quoted in the text
DIAG_TXT    = V2_DIR / "diagnostics.txt"          # human-readable quality checks

OUT_DIR     = Path("outputs_v2")
TABLES      = OUT_DIR / "tables"
FIGS        = OUT_DIR / "figures"
MODELS      = OUT_DIR / "models"

# ─────────────────────────────────────────────────────────────────────
# NREL OEDI S3 (anonymous access)
# ─────────────────────────────────────────────────────────────────────
BUCKET  = "oedi-data-lake"
BASE    = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock"
RES_VER = "2024/resstock_amy2018_release_2"
COM_VER = "2023/comstock_amy2018_release_2"

# ─────────────────────────────────────────────────────────────────────
# Study design
# ─────────────────────────────────────────────────────────────────────
SEED      = 42
N_RES     = 600          # sampled residential buildings per city
N_COM     = 400          # sampled commercial buildings per city
HOURS     = 8760         # hours in 2018
OOS_ZONE  = "5B"         # Denver: never used for training/tuning
ZONES     = ["2A", "3A", "4A", "5A", "5B"]
CITIES    = ["Houston", "Atlanta", "Baltimore", "Chicago", "Denver"]
ZONE_CITY = dict(zip(ZONES, CITIES))
CITY_ZONE = dict(zip(CITIES, ZONES))
ZONE_ENC  = {"2A": 0, "3A": 1, "4A": 2, "5A": 3, "5B": 4}
STATE     = {"Houston": "TX", "Atlanta": "GA", "Baltimore": "MD",
             "Chicago": "IL", "Denver": "CO"}
CITY_FIPS = {"Houston": ["48201"], "Atlanta": ["13121"],
             "Baltimore": ["24510"], "Chicago": ["17031"],
             "Denver": ["08031"]}
# NOTE (audit 12/10/2026): the original download matched the county column
# against "FIPS|GISJOIN". The NREL county fields hold GISJOIN codes
# (e.g. G1301210), so only the GISJOIN of each city matched: the sample
# is Fulton County (Atlanta) and Baltimore City (Baltimore) only —
# DeKalb (13089) and Baltimore County (24005) were never included.
# The lists above describe what was actually sampled (funnel unchanged).
CITY_GISJOIN = {"Houston": "G4802010", "Atlanta": "G1301210",
                "Baltimore": "G2405100", "Chicago": "G1700310",
                "Denver": "G0800310"}

SQFT_TO_M2 = 0.09290304   # exact conversion factor (1 ft² = 0.09290304 m²)

# Residential floor area in ResStock metadata is categorical (sqft bins).
# Preferred source: a numeric column (e.g. "in.sqft") if it exists in the
# NREL metadata — 00b_REPAIR_DATA.py checks this and reports which was used.
# FALLBACK ONLY: bin mid-points (assumption, must be stated in §3.3).
RES_AREA_BIN_MIDPOINT_SQFT = {
    "0-499": 250, "500-749": 625, "750-999": 875, "1000-1499": 1250,
    "1500-1999": 1750, "2000-2499": 2250, "2500-2999": 2750,
    "3000-3999": 3500, "4000+": 4500,   # 4000+: open bin -> 4500 assumed
}

# ─────────────────────────────────────────────────────────────────────
# Train / validation / test split (professor comment #10 + leakage fix)
#   test       = last 7 days of every month      (final evaluation only)
#   validation = the 7 days immediately before   (Optuna + early stopping)
#   train      = all remaining days
# ─────────────────────────────────────────────────────────────────────
TEST_DAYS = 7
VAL_DAYS  = 7

# MAPE denominator floor (kWh) — avoids division by ~0 at night.
MAPE_FLOOR_KWH = 0.1

# Optuna
N_TRIALS              = 30
EARLY_STOPPING_ROUNDS = 30
MAX_TREES             = 1500   # upper bound; early stopping on validation picks the number
TUNE_FRACTION         = 0.25   # share of training buildings used inside Optuna trials

# ─────────────────────────────────────────────────────────────────────
# Climate scenarios — IPCC AR6 WGI, Table SPM.1 (GSAT change vs
# 1850–1900, best estimate and very likely range). Converted to change
# vs 1995–2014 by subtracting 0.85 °C, as instructed in the caption of
# Table SPM.1. AMY2018 is used as the present-day baseline.
# Source: IPCC (2021), Summary for Policymakers, Table SPM.1.
# ─────────────────────────────────────────────────────────────────────
SPM1_VS_1850 = {   # (best, low, high)  °C vs 1850–1900
    ("SSP2-4.5", "2021-2040"): (1.5, 1.2, 1.8),
    ("SSP2-4.5", "2041-2060"): (2.0, 1.6, 2.5),
    ("SSP2-4.5", "2081-2100"): (2.7, 2.1, 3.5),
    ("SSP5-8.5", "2021-2040"): (1.6, 1.3, 1.9),
    ("SSP5-8.5", "2041-2060"): (2.4, 1.9, 3.0),
    ("SSP5-8.5", "2081-2100"): (4.4, 3.3, 5.7),
}
OFFSET_1995_2014 = 0.85
SCENARIOS = {f"{s} {p}": tuple(round(v - OFFSET_1995_2014, 2) for v in vals)
             for (s, p), vals in SPM1_VS_1850.items()}
# -> e.g. "SSP2-4.5 2081-2100": (1.85, 1.25, 2.65)

# ─────────────────────────────────────────────────────────────────────
# Emission factors (CO2 only), year 2018 to match the AMY2018 simulations
# Verified 11/10/2026.
#
# Electricity — US EPA eGRID2018 (released 2020), Table 1, subregion
#   CO2 TOTAL OUTPUT emission rate (lb/MWh) -> kg/kWh (× 0.45359237 / 1000).
#   Houston ERCT 931.7 · Atlanta SRSO 1,027.9 · Baltimore RFCE 716.0 ·
#   Chicago RFCW 1,166.1 · Denver RMPA 1,273.6   (U.S. 947.2)
#   Grid losses (~4.9 %) are NOT added (eGRID output rate, stated in text).
# Fuels — US EPA GHG Emission Factors Hub, stationary combustion
#   (kg CO2 per mmBtu, HHV) -> kg/kWh (÷ 293.07107 kWh per mmBtu):
#   natural gas 53.06 · propane 62.87 · distillate fuel oil No.2 73.96
# District heating/cooling have no factor -> excluded from CO2 and their
# share of site energy is reported (05_EDA, table_3_2).
# ─────────────────────────────────────────────────────────────────────
LB_TO_KG = 0.45359237
KWH_PER_MMBTU = 293.07107
EGRID2018_LB_PER_MWH = {"Houston": ("ERCT", 931.7), "Atlanta": ("SRSO", 1027.9),
                        "Baltimore": ("RFCE", 716.0), "Chicago": ("RFCW", 1166.1),
                        "Denver": ("RMPA", 1273.6)}
EF_ELEC = {CITY_ZONE[c]: round(v * LB_TO_KG / 1000, 4)
           for c, (_, v) in EGRID2018_LB_PER_MWH.items()}          # kg CO2/kWh by zone
EF_FUEL = {"natural_gas": 53.06 / KWH_PER_MMBTU,                     # 0.1810
           "propane": 62.87 / KWH_PER_MMBTU,                         # 0.2145
           "fuel_oil": 73.96 / KWH_PER_MMBTU,                        # 0.2524
           # ComStock "other_fuel" = propane + fuel oil (not separated hourly):
           # the fuel-oil factor is used (upper bound); its share is reported.
           "other_fuel": 73.96 / KWH_PER_MMBTU}

# ─────────────────────────────────────────────────────────────────────
# Residential energy prices 2018 — US EIA, State Energy Data System
# (SEDS), "Residential sector energy price estimates", $/million Btu,
# row 2018 (accessed 11/10/2026). State averages are applied to the
# city (limitation stated in §5.3). $/kWh = ($/mmBtu) / 293.07107.
#   state: NG, distillate fuel oil, HGL(propane), electricity
#   TX 11.11 17.28 28.36 32.82 · GA 13.61 18.88 31.11 33.62
#   MD 11.29 20.69 34.46 38.97 · IL  7.92 19.29 18.69 37.41
#   CO  7.23 19.34 23.93 35.60
# (check: electricity 32.82 $/mmBtu = 11.20 ¢/kWh for TX)
# ─────────────────────────────────────────────────────────────────────
PRICES_VERIFIED = True
SEDS2018_USD_PER_MMBTU = {
    "Houston":   {"natural_gas": 11.11, "fuel_oil": 17.28, "propane": 28.36, "electricity": 32.82},
    "Atlanta":   {"natural_gas": 13.61, "fuel_oil": 18.88, "propane": 31.11, "electricity": 33.62},
    "Baltimore": {"natural_gas": 11.29, "fuel_oil": 20.69, "propane": 34.46, "electricity": 38.97},
    "Chicago":   {"natural_gas":  7.92, "fuel_oil": 19.29, "propane": 18.69, "electricity": 37.41},
    "Denver":    {"natural_gas":  7.23, "fuel_oil": 19.34, "propane": 23.93, "electricity": 35.60},
}
PRICE_USD_PER_KWH = {c: {f: v / KWH_PER_MMBTU for f, v in d.items()}
                     for c, d in SEDS2018_USD_PER_MMBTU.items()}

# Data-quality thresholds (STEP A report; nothing is removed automatically)
QC_MAX_ZERO_RUN_H = 24    # ≥ 24 consecutive hours with site energy = 0 -> possible download gap
QC_MAX_FLAT_RUN_H = 48    # ≥ 48 consecutive identical non-zero hours -> frozen series

# Energy-burden thresholds — Drehobl, Ross & Ayala (2020), ACEEE:
# high burden > 6 %, severe burden > 10 %.
BURDEN_HIGH, BURDEN_SEVERE = 6.0, 10.0

# Energy cost source for 04_ENERGY_BURDEN.  After STEP A, look in
# diagnostics.txt ("numeric emission/bill columns stored"). If a ResStock
# annual-bill column (USD/year, all fuels) is listed, write its exact name
# here; otherwise leave None and verify PRICE_USD_PER_KWH instead.
BILL_COLUMN = None

# ESG composite: weights used in the sensitivity analysis (professor #8)
ESG_WEIGHTS_E = [0.25, 0.50, 0.75]     # weight of E; S gets (1 - w)

# Grid-decarbonisation WHAT-IF (Ch.5): relative reduction of the
# electricity emission factor. These are ASSUMPTIONS for a sensitivity
# analysis, not forecasts — they are labelled as such in tables/figures.
GRID_EF_REDUCTION = [0.0, 0.25, 0.50, 0.75]

# Figures
DPI = 300

# ─────────────────────────────────────────────────────────────────────
# Smoke test (developer only): THESIS_SMOKE_TEST=1 runs the whole
# pipeline on a tiny synthetic sample. NEVER set it for the real run.
# ─────────────────────────────────────────────────────────────────────
SMOKE_TEST = os.environ.get("THESIS_SMOKE_TEST") == "1"
if SMOKE_TEST:
    N_RES, N_COM, N_TRIALS, MAX_TREES = 6, 4, 3, 200


def ensure_dirs():
    """Create every output folder (idempotent)."""
    for d in (V2_DIR, OUT_DIR, TABLES, FIGS, MODELS):
        d.mkdir(parents=True, exist_ok=True)
