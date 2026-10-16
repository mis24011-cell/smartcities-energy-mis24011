#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_ENERGY_BURDEN_v2.py  —  Residential energy burden (all fuels)
Thesis MIS24011 — Zygouris Aristotelis

Energy burden = annual energy cost of the household / household income
(Drehobl, Ross & Ayala, 2020). Residential buildings only — ComStock has
no income field.

Fixes vs 04_ENERGY_BURDEN_INDIVIDUAL.py
  K6   Cost now includes ALL fuels (electricity, natural gas, propane,
       fuel oil), not electricity only.
  K12  Households with income <= 0 are EXCLUDED (count reported), not
       moved to the city median.
       Income = in.representative_income (NREL, drawn from ACS income
       bins) — described as such in the text, not as "measured income".
  Quartiles: within each city; ties in the binned income are broken by
       rank(method="first") so that each quartile holds 25 % of homes.
Cost source (in this order):
  1. NREL utility-bill output found by 00b (config.BILL_COLUMN), or
  2. EIA SEDS 2018 residential prices per state and fuel (config.py) —
     default; state averages applied to the city (limitation).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from common import FUELS, fuel_total_col, load_json, update_summary


def main():
    C.ensure_dirs()
    meta = pd.read_parquet(C.META_V2)
    res = meta[meta["dataset_type"] == "residential"].copy()
    res["income_usd"] = pd.to_numeric(res["income_usd"], errors="coerce")

    cols = pq.read_schema(C.BLDG_V2).names
    fcols = {f: fuel_total_col(cols, f) for f in ("electricity", "natural_gas",
                                                  "propane", "fuel_oil")}
    use = ["building_uid", "dataset_type"] + [c for c in fcols.values() if c]
    ts = pq.read_table(C.BLDG_V2, columns=use,
                       filters=[("dataset_type", "=", "residential")]).to_pandas()
    ann = ts.groupby("building_uid")[[c for c in fcols.values() if c]].sum()
    ann.columns = [f"kwh_{f}" for f, c in fcols.items() if c]
    res = res.set_index("building_uid").join(ann)

    bill_col = getattr(C, "BILL_COLUMN", None)
    if bill_col:
        if bill_col not in res.columns:
            raise ValueError(f"BILL_COLUMN '{bill_col}' not in metadata_v2")
        res["cost_usd"] = pd.to_numeric(res[bill_col], errors="coerce")
        cost_source = f"NREL output column {bill_col}"
    else:
        if not C.PRICES_VERIFIED:
            raise SystemExit(
                "STOP: no NREL bill column selected (config.BILL_COLUMN) and prices in "
                "config.PRICE_USD_PER_KWH are not verified. Fill/verify them from EIA "
                "2018 and set PRICES_VERIFIED = True.")
        res["cost_usd"] = 0.0
        for f in ("electricity", "natural_gas", "propane", "fuel_oil"):
            if f"kwh_{f}" not in res.columns:
                continue
            price = res["city"].map({c: C.PRICE_USD_PER_KWH[c][f] for c in C.CITIES})
            used = res[f"kwh_{f}"] > 0
            if used.any() and price[used].isna().any():
                raise SystemExit(f"STOP: missing price for {f} in a city that uses it")
            res["cost_usd"] += res[f"kwh_{f}"].fillna(0) * price.fillna(0)
        cost_source = "EIA SEDS 2018 residential prices by state and fuel (config.py)"

    # Every residential kWh must have a price: district energy is not priced
    other = [fuel_total_col(cols, c) for c in ("district_heating", "district_cooling")]
    other = [c for c in other if c]
    if other:
        d = pq.read_table(C.BLDG_V2, columns=other,
                          filters=[("dataset_type", "=", "residential")]).to_pandas()
        if float(d.sum().sum()) > 0:
            raise SystemExit(f"STOP: residential district energy found in {other} (no price)")
    # cost must cover ALL residential site energy (Σ priced fuels ≈ site total)
    site = [c for c in cols if c.startswith("out.site_energy.total")][0]
    s_site = pq.read_table(C.BLDG_V2, columns=["building_uid", site],
                           filters=[("dataset_type", "=", "residential")]).to_pandas()
    s_site = s_site.groupby("building_uid")[site].sum()
    gap = float((s_site.sum() - ann.sum().sum()) / s_site.sum() * 100)
    print(f"  residential site energy not covered by priced fuels: {gap:.3f} %")
    if abs(gap) > 0.5:
        raise SystemExit(f"STOP: priced fuels cover site energy only up to {gap:.2f} % gap")
    n_all = len(res)
    # income <= 0 in ResStock = vacant dwelling (tenure "Not Available"):
    # there is no household, so no energy burden. Excluded, counted per city.
    excl = res["income_usd"].isna() | (res["income_usd"] <= 0)
    excl_city = excl.groupby(res["city"]).sum().astype(int).to_dict()
    print(f"  excluded (income <= 0 / vacant) per city: {excl_city}")
    res = res[~excl].copy()
    res["burden_pct"] = res["cost_usd"] / res["income_usd"] * 100
    res["quartile"] = res.groupby("city")["income_usd"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))

    rows = []
    for city in C.CITIES:
        for q in ("Q1", "Q2", "Q3", "Q4"):
            s = res[(res["city"] == city) & (res["quartile"] == q)]
            rows.append({"Πόλη": city, "Τεταρτημόριο": q, "N": len(s),
                         "Εξαιρέθηκαν στην πόλη (εισόδημα 0/κενές)": excl_city.get(city, 0),
                         "Διάμεσο εισόδημα ($)": round(float(s["income_usd"].median()), 0),
                         "Διάμεσο ετήσιο κόστος ($)": round(float(s["cost_usd"].median()), 0),
                         "Διάμεσο energy burden (%)": round(s["burden_pct"].median(), 2),
                         f"Υψηλό burden >{C.BURDEN_HIGH:.0f}% (% νοικ.)":
                             round((s["burden_pct"] > C.BURDEN_HIGH).mean() * 100, 1),
                         f"Σοβαρό burden >{C.BURDEN_SEVERE:.0f}% (% νοικ.)":
                             round((s["burden_pct"] > C.BURDEN_SEVERE).mean() * 100, 1)})
    t = pd.DataFrame(rows)
    t.to_csv(C.TABLES / "table_5_3_energy_burden.csv", index=False, encoding="utf-8-sig")

    # Robustness check: NREL's own annual utility bills (ResStock output
    # out.bills.all_fuels.usd — utility rates incl. fixed charges, as modelled
    # by NREL) instead of EIA SEDS 2018 energy prices. Same households, same
    # quartiles. Reported as a sensitivity table, not as the main result.
    nb = "out.bills.all_fuels.usd"
    if nb in res.columns and res[nb].notna().any():
        res["burden_nrel_pct"] = pd.to_numeric(res[nb], errors="coerce") / res["income_usd"] * 100
        rb = []
        for city in C.CITIES:
            for q in ("Q1", "Q2", "Q3", "Q4"):
                s = res[(res["city"] == city) & (res["quartile"] == q)]
                rb.append({"Πόλη": city, "Τεταρτημόριο": q, "N": len(s),
                           "Burden EIA SEDS 2018, διάμεσος (%)": round(s["burden_pct"].median(), 2),
                           "Burden λογαριασμοί NREL, διάμεσος (%)": round(s["burden_nrel_pct"].median(), 2),
                           f"> {C.BURDEN_HIGH:.0f}% EIA (% νοικ.)": round((s["burden_pct"] > C.BURDEN_HIGH).mean() * 100, 1),
                           f"> {C.BURDEN_HIGH:.0f}% NREL (% νοικ.)": round((s["burden_nrel_pct"] > C.BURDEN_HIGH).mean() * 100, 1)})
        pd.DataFrame(rb).to_csv(C.TABLES / "table_5_3b_burden_sensitivity_nrel_bills.csv",
                                index=False, encoding="utf-8-sig")
        print("  sensitivity table with NREL bills written")
    res.reset_index().to_parquet(C.TABLES / "energy_burden_by_household.parquet", index=False)
    print(t.to_string(index=False))
    update_summary("energy_burden", {"households_total": int(n_all),
                                     "excluded_income_le0": int(excl.sum()),
                                     "excluded_by_city": excl_city,
                                     "households_used": int(len(res)),
                                     "cost_source": cost_source,
                                     "site_energy_not_priced_pct": round(gap, 3)})
    print("✓ 04_ENERGY_BURDEN_v2 finished")


if __name__ == "__main__":
    main()
