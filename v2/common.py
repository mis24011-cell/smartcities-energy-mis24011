# -*- coding: utf-8 -*-
"""
common.py — Helper functions shared by all v2 scripts.
Keeping them in one place guarantees that every script uses the SAME
split, the SAME metrics and the SAME column logic.
"""
import json
import re

import numpy as np
import pandas as pd

import config as C

# ─────────────────────────────────────────────────────────────────────
# Column discovery (fail loudly instead of silently picking a wrong one)
# ─────────────────────────────────────────────────────────────────────
FUELS = ["electricity", "natural_gas", "propane", "fuel_oil", "other_fuel",
         "district_heating", "district_cooling"]   # other_fuel = ComStock (propane + fuel oil)


def fuel_total_col(columns, fuel):
    """Return the '<fuel>.total.energy_consumption' column, or None."""
    pat = re.compile(rf"^out\.{fuel}\.total\.energy_consumption(\.kwh)?$")
    hits = [c for c in columns if pat.match(c)]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous total column for {fuel}: {hits}")
    return hits[0] if hits else None


def site_total_col(columns):
    """Return the site-energy total column. Raises if missing."""
    pat = re.compile(r"^out\.site_energy\.total\.energy_consumption(\.kwh)?$")
    hits = [c for c in columns if pat.match(c)]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one site_energy total column, found {hits}")
    return hits[0]


# End-use families. Each regex is matched against the END-USE token of a
# column named  out.<fuel>.<end_use>.energy_consumption
# Anything not matched is reported and goes to "Λοιπά" so that the stacked
# end-use bars ALWAYS add up to the site total (checked in 05_EDA).
END_USE_FAMILIES = [   # ORDER MATTERS: first match wins (checked on the real
    # column list of the download, diagnostics 11/10/2026)
    ("Εξοπλισμός", r"pool|spa|well_pump|grill|range|clothes|dish|refrig|freezer|plug|"
                   r"equipment|television|computer|vehicle|(^|_)ev(_|$)"),
    ("Ζ.Ν.Χ.",     r"hot_water|water_systems|water_heat"),
    ("Ψύξη",       r"cool|chiller|heat_rejection"),
    ("Αερισμός",   r"heat_recovery|ceiling_fan|^fans$|ventilation|mech_vent|^pumps$"),
    ("Θέρμανση",   r"heat|boiler|furnace|fireplace"),
    ("Φωτισμός",   r"light"),
]


def end_use_map(columns):
    """
    Map every out.<fuel>.<end_use>.energy_consumption column (excluding
    totals and net/PV) to an end-use family. Returns (mapping, unmatched).
    """
    mapping, unmatched = {}, []
    pat = re.compile(r"^out\.([a-z_]+)\.([a-z0-9_]+)\.energy_consumption(\.kwh)?$")
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        fuel, eu = m.group(1), m.group(2)
        if eu == "total" or fuel in ("site_energy",) or eu in ("net",) \
                or "pv" in eu or "generation" in eu or "battery" in eu:
            continue
        fam = next((name for name, rx in END_USE_FAMILIES if re.search(rx, eu)), None)
        if fam is None:
            unmatched.append(c)
        else:
            mapping[c] = fam
    return mapping, unmatched


# ─────────────────────────────────────────────────────────────────────
# Split
# ─────────────────────────────────────────────────────────────────────
def split_labels(ts):
    """
    ts: pandas Series of timestamps.
    Returns a Series with values 'train' / 'val' / 'test'.
      test = last TEST_DAYS days of each month
      val  = the VAL_DAYS days immediately before the test window
      train= everything else
    """
    day = ts.dt.day
    dim = ts.dt.days_in_month
    lab = np.where(day > dim - C.TEST_DAYS, "test",
          np.where(day > dim - C.TEST_DAYS - C.VAL_DAYS, "val", "train"))
    return pd.Series(lab, index=ts.index)


# ─────────────────────────────────────────────────────────────────────
# Metrics (one definition for every table)
# ─────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, name, rmse_persistence=None):
    """
    R², RMSE, MAE, MAPE (denominator floored at MAPE_FLOOR_KWH),
    CV(RMSE) and NMBE as defined in ASHRAE Guideline 14-2014, and the
    skill score vs persistence: 1 - RMSE_model / RMSE_persistence.
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    err = y - p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), C.MAPE_FLOOR_KWH)) * 100)
    ybar = float(y.mean())
    cv = rmse / ybar * 100 if ybar != 0 else np.nan
    nmbe = float(np.sum(err)) / (len(y) * ybar) * 100 if ybar != 0 else np.nan
    out = {"Μοντέλο": name, "N": int(len(y)), "R²": round(r2, 4),
           "RMSE (kWh/h)": round(rmse, 4), "MAE (kWh/h)": round(mae, 4),
           "MAPE (%)": round(mape, 2), "CV(RMSE) (%)": round(cv, 2),
           "NMBE (%)": round(nmbe, 2)}
    if rmse_persistence:
        out["Skill vs Persistence"] = round(1 - rmse / rmse_persistence, 4)
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-building scaling for the global ML models (Step B, fix #18)
# ─────────────────────────────────────────────────────────────────────
# Why: the hourly load of the sample spans five orders of magnitude
# (≈0,01 kWh in a dwelling … ≈12.900 kWh in the largest office). XGBoost
# ("hist" method) cuts every feature into 256 quantile bins; because
# 99,6 % of the rows are below 1.000 kWh, the top bin of lag_1h covered
# 1.025 … 12.900 kWh, so the model could not tell 1.500 kWh from
# 5.000 kWh in the lag and under-predicted the peaks of the large
# commercial buildings (first run of 12/10/2026, Table 4.4: unified
# XGBoost worse than Random Forest; 10 buildings = 66 % of its SSE).
# Remedy (standard for "global" forecasting models across many series,
# Salinas et al. 2020; Montero-Manso & Hyndman 2021): every series is
# divided by its own level s_b = mean target over the TRAIN days of
# building b (no val/test information), the model learns the relative
# dynamics, and the prediction is multiplied back by s_b. All metrics
# are still reported in kWh. The same scaled inputs are used by Ridge,
# Random Forest and XGBoost, so the comparison stays fair; Persistence
# and Seasonal Naive use the raw lags.
SCALED_FEATURES = ["lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h"]


def building_scale(df, target):
    """s_b = mean target of building b over its 'train' rows (float64).
    Buildings with a train mean ≤ 1e-6 kWh (never in the real data) get 1."""
    tr = df["split"].astype(str) == "train"
    s = (df.loc[tr, target].astype("float64")
           .groupby(df.loc[tr, "building_uid"].astype(str), sort=False).mean())
    return s.where(s > 1e-6, 1.0).rename("scale_kwh")


def scale_rows(df, scale):
    """Per-row scale vector (float64), aligned with df."""
    s = df["building_uid"].astype(str).map(scale)
    if s.isna().any():
        raise ValueError(f"{int(s.isna().sum())} rows without a building scale")
    return s.to_numpy("float64")


def scale_X(X, s, features):
    """Divide the level-dependent columns of X (float32 array) by s.
    (pandas may hand back a read-only view -> copy only in that case)."""
    if not X.flags.writeable:
        X = X.copy()
    for c in SCALED_FEATURES:
        j = features.index(c)
        X[:, j] = (X[:, j] / s).astype(np.float32)
    return X


# ─────────────────────────────────────────────────────────────────────
# Calendar: US federal holidays 2018 (pandas built-in calendar, reproducible)
# ─────────────────────────────────────────────────────────────────────
def is_holiday(ts):
    """1 if the date is a US federal holiday (observed), else 0 (int8)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(start="2018-01-01", end="2018-12-31")
    return ts.dt.normalize().isin(hol).astype("int8")


# ─────────────────────────────────────────────────────────────────────
# Static building characteristics (for the weather-driven climate model)
# ─────────────────────────────────────────────────────────────────────
VINTAGE_YEAR = {   # midpoint of each NREL vintage bin (ResStock + ComStock labels)
    "<1940": 1935, "1940s": 1945, "1950s": 1955, "1960s": 1965, "1970s": 1975,
    "1980s": 1985, "1990s": 1995, "2000s": 2005, "2010s": 2015,
    "Before 1946": 1940, "1946 to 1959": 1953, "1960 to 1969": 1965,
    "1970 to 1979": 1975, "1980 to 1989": 1985, "1990 to 1999": 1995,
    "2000 to 2012": 2006, "2013 to 2018": 2016}
STATIC_NUM = ["vintage_year", "heat_gas", "heat_elec", "heat_oil_propane", "heat_other"]
STATIC_CAT = ["btype", "hvac"]


def static_features(meta):
    """
    One row per building_uid with numeric / categorical characteristics from
    the NREL metadata (no energy information -> no leakage):
      vintage_year  (bin midpoint), heating fuel one-hot, building type and
      HVAC system as pandas categoricals (XGBoost enable_categorical).
    """
    m = meta.set_index("building_uid")
    out = pd.DataFrame(index=m.index)
    vy = m["in.vintage"].astype(str).map(VINTAGE_YEAR)
    if vy.isna().any():
        raise ValueError(f"unknown vintage labels: {sorted(m.loc[vy.isna(), 'in.vintage'].astype(str).unique())}")
    out["vintage_year"] = vy.astype("float32")
    hf = m["in.heating_fuel"].astype(str).str.lower().str.replace(" ", "", regex=False)
    out["heat_gas"] = hf.eq("naturalgas").astype("float32")
    out["heat_elec"] = hf.eq("electricity").astype("float32")
    out["heat_oil_propane"] = hf.isin(["fueloil", "propane"]).astype("float32")
    out["heat_other"] = (1 - out[["heat_gas", "heat_elec", "heat_oil_propane"]].sum(axis=1)).astype("float32")
    bt = np.where(m["dataset_type"].eq("residential"),
                  "R:" + m["in.geometry_building_type_recs"].astype(str),
                  "C:" + m["in.comstock_building_type"].astype(str))
    out["btype"] = pd.Categorical(bt)
    hv = np.where(m["dataset_type"].eq("residential"), "Residential",
                  m["in.hvac_system_type"].astype(str))
    out["hvac"] = pd.Categorical(hv)
    return out


# ─────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────
def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_summary(key, value):
    """Add/overwrite one entry of data_summary.json (all numbers for the text)."""
    s = load_json(C.SUMMARY) if C.SUMMARY.exists() else {}
    s[key] = value
    save_json(s, C.SUMMARY)


def gr(x, nd=2):
    """Greek number formatting: 1.234,56 (thousands '.', decimals ',')."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def weighted_median(values, weights):
    v = np.asarray(values, float); w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if len(v) == 0:
        return np.nan
    o = np.argsort(v); v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def check(cond, msg, log):
    """Assertion that is also written to the diagnostics log."""
    log(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers (Greek labels, Greek decimal comma, one style)
# ─────────────────────────────────────────────────────────────────────
TYPE_GR = {"residential": "Οικιστικά", "commercial": "Εμπορικά", "all": "Σύνολο"}
FUEL_GR = {"electricity": "Ηλεκτρισμός", "natural_gas": "Φυσικό αέριο",
           "propane": "Προπάνιο", "fuel_oil": "Πετρέλαιο θέρμανσης",
           "other_fuel": "Άλλα καύσιμα (ComStock)",
           "district_heating": "Τηλεθέρμανση", "district_cooling": "Τηλεψύξη"}
ZONE_COLORS = {"2A": "#d7301f", "3A": "#fc8d59", "4A": "#41ab5d",
               "5A": "#2b8cbe", "5B": "#6a51a3"}


def setup_mpl():
    """Non-interactive backend + a font that has Greek glyphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "savefig.bbox": "tight",
                         "figure.constrained_layout.use": True})
    return plt


def gr_axis(ax, axis="y", nd=None):
    """Greek number format on an axis (1.234,5). nd=None -> automatic decimals."""
    from matplotlib.ticker import FuncFormatter

    def f(x, _):
        if nd is not None:
            return gr(x, nd)
        if abs(x - round(x)) < 1e-9:
            return gr(x, 0)
        return gr(x, 2 if abs(x) < 10 else 1)
    fmt = FuncFormatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


def log_axis_plain(ax, axis="y"):
    """Log-scale axis with plain Greek-formatted numbers instead of 10^x."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda x, _: gr(x, 0)))
    a.set_minor_formatter(NullFormatter())


def save_fig(fig, name, log=print):
    path = C.FIGS / name
    fig.savefig(path, dpi=C.DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"  -> {path}")
