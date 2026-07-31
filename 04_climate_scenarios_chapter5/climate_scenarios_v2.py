# -*- coding: utf-8 -*-
# ============================================================================
# climate_scenarios_v2.py  (v2 — CONVEX cooling)
# ----------------------------------------------------------------------------
# Σχέδιο: 2 pathways (SSP2-4.5, SSP5-8.5) × 3 ορίζοντες (2040/2050/2100)
#   + Baseline (2018). Διαχωρίζει pathway από έτος (IPCC AR6, WG1 2021).
#
# ΜΕΘΟΔΟΣ (ανακατασκευασμένη & επικυρωμένη, MAE=0 στα υπάρχοντα):
#   • Degree-days: Normal(μ=base_temp, σ) με κατώφλι 18°C (ASHRAE 90.1-2019).
#       σ ανά κτίριο προσαρμόζεται ώστε να αναπαράγει το HDD_baseline.
#   • ΘΕΡΜΑΝΣΗ (γραμμική): heating_% = coeff_heating × ΔT (HDD ~γραμμικό στο εύρος).
#   • ΨΥΞΗ (CONVEX): cooling ∝ (CDD_projected − CDD_baseline), δηλ. ακολουθεί τη
#       ΜΗ-γραμμική αύξηση των βαθμοημερών ψύξης. Καλιμπράρεται ανά κτίριο ώστε
#       στο ΔT=REF_DT (2,5°C) να ταυτίζεται με το προηγούμενο γραμμικό μοντέλο —
#       άρα συνεπές στο κοντινό μέλλον, αλλά εκτοξεύεται ρεαλιστικά στο 2100.
#       (Fallback σε γραμμικό όπου η CDD παραμένει αμελητέα — ψυχρά κτίρια.)
#
# DELTAS (τοπικά, από baseline 2018): IPCC AR6 warming levels − ~1,0°C × REGIONAL_FACTOR.
# Αναφορές: IPCC AR6 WGI (2021)· Hawkins and Sutton (2009)· ASHRAE 90.1-2019.
# ============================================================================
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

TC = 18.0
DAYS = 365
REGIONAL_FACTOR = 1.40      # υπερθέρμανση ξηράς ΗΠΑ vs global (IPCC AR6). 1.0 = global.
REF_DT = 2.5                # σημείο calibration ψύξης (ίσο με το παλιό 2040)
CDD_FLOOR = 5.0             # κάτω από αυτό -> γραμμικό fallback για την ψύξη

_GLOBAL_VS_1850 = {
    ("SSP2-4.5", "2040"): 1.5, ("SSP5-8.5", "2040"): 1.7,
    ("SSP2-4.5", "2050"): 2.0, ("SSP5-8.5", "2050"): 2.4,
    ("SSP2-4.5", "2100"): 2.7, ("SSP5-8.5", "2100"): 4.4,
}
WARMING_2018 = 1.0

def local_delta(pathway, year, regional=REGIONAL_FACTOR):
    return round((_GLOBAL_VS_1850[(pathway, year)] - WARMING_2018) * regional, 3)

def _dd(mu, sigma, Tc=TC, days=DAYS):
    """HDD, CDD από Normal(μ, σ) (vectorized)."""
    z = (Tc - mu) / sigma
    hdd = days * (sigma * norm.pdf(z) + (Tc - mu) * norm.cdf(z))
    cdd = days * (sigma * norm.pdf(z) - (Tc - mu) * (1 - norm.cdf(z)))
    return hdd, cdd

def fit_sigma(mu, hdd_baseline):
    try:
        return brentq(lambda s: _dd(mu, s)[0] - hdd_baseline, 0.3, 40.0)
    except Exception:
        return np.nan

def _prepare(df):
    """Προσθέτει βοηθητικές στήλες: _sigma, _cool_per_cdd (convex calibration)."""
    df = df.copy()
    df["_sigma"] = [fit_sigma(m, h) for m, h in zip(df["base_temp_mean"], df["HDD_baseline"])]
    mu = df["base_temp_mean"].values; sig = df["_sigma"].values
    _, cdd_ref = _dd(mu + REF_DT, sig)                 # CDD στο σημείο calibration
    dcdd_ref = cdd_ref - df["CDD_baseline"].values     # μεταβολή CDD στο REF_DT
    # fractional αύξηση ψύξης ανά μονάδα CDD ώστε στο REF_DT να δίνει coeff_cooling*REF_DT %
    # (υπολογίζεται ΜΟΝΟ όπου η CDD είναι μη-αμελητέα -> αλλού γραμμικό fallback)
    safe = dcdd_ref >= CDD_FLOOR
    cpc = np.full_like(dcdd_ref, np.nan, dtype=float)
    cpc[safe] = (df["coeff_cooling"].values[safe] * REF_DT / 100.0) / dcdd_ref[safe]
    df["_cool_per_cdd"] = cpc                            # nan -> linear fallback
    return df

def _project(df, dT):
    mu = df["base_temp_mean"].values; sig = df["_sigma"].values
    hdd_p, cdd_p = _dd(mu + dT, sig)
    # Θέρμανση: γραμμικό
    heat_p = df["heating_kwh"].values * (1 + df["coeff_heating"].values * dT / 100.0)
    # Ψύξη: convex (CDD-based) με γραμμικό fallback
    cpc = df["_cool_per_cdd"].values
    cool_frac_cdd = cpc * (cdd_p - df["CDD_baseline"].values)
    cool_frac_lin = df["coeff_cooling"].values * dT / 100.0
    cool_frac = np.where(np.isnan(cpc), cool_frac_lin, cool_frac_cdd)
    cool_p = df["cooling_kwh"].values * (1 + cool_frac)
    light = df["lighting_kwh"].values; equip = df["equipment_kwh"].values
    # ΣΩΣΤΗ φυσική: μόνο θέρμανση & ψύξη αποκρίνονται στη θερμοκρασία. Το ΣΥΝΟΛΟ
    # ξεκινά από το πραγματικό annual_kwh και μεταβάλλεται μόνο κατά τα Δ(θέρμ.), Δ(ψύξη)
    # -> έτσι το baseline (dT=0) δίνει ΑΚΡΙΒΩΣ 0% μεταβολή (τα σταθερά φορτία δεν αλλάζουν).
    annual_p = df["annual_kwh"].values + (heat_p - df["heating_kwh"].values) + (cool_p - df["cooling_kwh"].values)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdd_chg = np.where(df["CDD_baseline"].values > 0, (cdd_p / df["CDD_baseline"].values - 1) * 100, np.nan)
    return dict(
        temp_mean_projected=mu + dT, HDD_projected=hdd_p, CDD_projected=cdd_p,
        HDD_change_pct=(hdd_p / df["HDD_baseline"].values - 1) * 100, CDD_change_pct=cdd_chg,
        heating_projected=heat_p, cooling_projected=cool_p,
        lighting_projected=light, equipment_projected=equip, annual_kwh_projected=annual_p,
        heating_delta_pct=df["coeff_heating"].values * dT,
        cooling_delta_pct=cool_frac * 100,
        consumption_delta_pct=(annual_p / df["annual_kwh"].values - 1) * 100,
    )

def build_scenarios(baseline_df, horizons):
    """horizons: {year: {pathway: dT}} -> long DataFrame (Baseline + 2×3)."""
    b = _prepare(baseline_df)
    out = []
    base = b.copy(); base["scenario"] = "Baseline (2018)"; base["scenario_year"] = 2018; base["delta_temp"] = 0.0
    for k, v in _project(base, 0.0).items(): base[k] = v
    out.append(base)
    for year, paths in horizons.items():
        for pathway, dT in paths.items():
            s = b.copy(); s["scenario"] = f"{pathway} ({year})"; s["scenario_year"] = int(year); s["delta_temp"] = dT
            for k, v in _project(s, dT).items(): s[k] = v
            out.append(s)
    return pd.concat(out, ignore_index=True).drop(columns=["_sigma", "_cool_per_cdd"])

if __name__ == "__main__":
    base = pd.read_parquet("awp.parquet")
    base = base[base.scenario.str.contains("Baseline")].copy().reset_index(drop=True)
    HOR = {y: {p: local_delta(p, y) for p in ["SSP2-4.5", "SSP5-8.5"]} for y in ["2040", "2050", "2100"]}
    df = build_scenarios(base, HOR)
    df.to_parquet("scenarios_2x3_baseline.parquet", index=False)
    print("✓ scenarios_2x3_baseline.parquet", df.shape, "| σενάρια:", df.scenario.nunique())
