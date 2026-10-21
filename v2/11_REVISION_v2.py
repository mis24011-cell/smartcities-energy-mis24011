#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_REVISION_v2.py  —  Additional analyses requested by the supervisor (20/10/2026)
Thesis MIS24011 — Zygouris Aristotelis

Run from the thesis_code_final folder (like every v2 script):
    python v2/11_REVISION_v2.py A        (≈5–10 min)
    python v2/11_REVISION_v2.py B        (≈1–2 h)
    python v2/11_REVISION_v2.py C        (≈1–3 h, stops tuning after RF_TUNE_MINUTES)
    python v2/11_REVISION_v2.py D        (≈1–1,5 h)
    python v2/11_REVISION_v2.py E        (≈30 min, diagnostic of B without month features)
    python v2/11_REVISION_v2.py F        (≈30 min, predictions of the tuned RF, needs C)
or simply  v2\\RUN_REVISION.bat  (runs A, B, C, D in this order, one log per stage).

Nothing of the main results is overwritten: every output goes to
outputs_v2/revision/  (tables + models).  The main scripts 00–10 are unchanged.

Stages
------
A  Diversity factor and peak hour PER FUEL  (comment 1)
   The thesis computed DF on the total site energy (electricity + natural gas
   + other fuels). The electricity grid only sees electricity, so DF, the peak
   hour and the stock peak (MW) are recomputed for
       electricity  |  natural gas  |  site energy (for comparison)
   with exactly the same definition and city-stock weights w_i as 06_ESG_DF:
       DF = Σ_i w_i·max_t L_i(t)  /  max_t Σ_i w_i·L_i(t)
   Also saved: the hourly weighted stock load per fuel (small file, used for the
   figures and for the winter/summer peaks) and each building's peak per fuel.

B  Strictly chronological split  (comment 2)
   In the main protocol the test set is the last 7 days of every month, so the
   training set contains days AFTER each test week (e.g. early February when the
   end of January is tested) and so does the per-building level s_b.
   Here: train = January–September, test = October–December (no day of the
   test period, and nothing after it, is used for training or for s_b).
   Same features, same scaling, same models; XGBoost uses the hyper-parameters
   already selected in 02_ML_v2 (best_params_unified.json, no re-tuning on the
   test months). The XGBoost / RF results of the main protocol on the SAME hours
   (their test days of Oct–Dec) are given next to them for comparison.

C  Random Forest with hyper-parameter tuning  (comment 4)
   In 02_ML_v2 XGBoost was tuned with Optuna (30 trials, validation window)
   while Random Forest kept fixed settings (50 trees, depth 12). Here RF gets
   the same treatment: Optuna on the same 25 % of the training buildings and the
   same validation window, objective = RMSE in kWh; then refit on train+val and
   evaluation on the test days and on Denver, as in Table 4.4.

D  Climate (weather-driven) model WITHOUT the month variables  (comment 6)
   month_sin / month_cos may absorb part of the seasonal temperature effect and
   damp the response to a warmer climate. 03_CLIMATE_v2 is re-run with the two
   month features removed; Table 5.1 (and 5.0 accuracy) are written to the
   revision folder so they can be compared with the main result.

Outputs (outputs_v2/revision/)
  A: rev_A_diversity_by_fuel.csv, rev_A_stock_hourly_by_fuel.parquet,
     rev_A_peaks_by_building.parquet
  B: rev_B_chrono_benchmark.csv, rev_B_chrono_same_hours.csv,
     rev_B_chrono_rmae.csv, rev_B_predictions.parquet
  C: rev_C_rf_optuna_trials.csv, rev_C_rf_best_params.json, rev_C_rf_benchmark.csv
  D: rev_D_climate_no_month.csv, rev_D_climate_accuracy_no_month.csv,
     rev_D_climate_by_building_no_month.parquet
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from common import (fuel_total_col, metrics, scale_rows, scale_X, site_total_col)

REV = C.OUT_DIR / "revision"                 # all outputs of this script
RF_TUNE_TRIALS = 15                          # Optuna trials for Random Forest (stage C)
RF_TUNE_MINUTES = 120                        # hard time limit of the RF tuning
if C.SMOKE_TEST:                             # developer test on synthetic data
    RF_TUNE_TRIALS, RF_TUNE_MINUTES = 3, 5
CITY_OF_ZONE = C.ZONE_CITY


def _load_module(fname, alias):
    """Import a v2 script whose file name starts with a digit (e.g. 02_ML_v2.py)."""
    path = Path(__file__).with_name(fname)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save(df, name):
    """CSV with BOM (opens correctly in Greek Excel)."""
    df.to_csv(REV / name, index=False, encoding="utf-8-sig")
    _log(f"saved {name}")


# ═════════════════════════════════════════════════════════════════════
# A — Diversity factor per fuel
# ═════════════════════════════════════════════════════════════════════
def stage_A():
    cols = pq.read_schema(C.BLDG_V2).names
    fuel_cols = {"site": site_total_col(cols),
                 "electricity": fuel_total_col(cols, "electricity"),
                 "natural_gas": fuel_total_col(cols, "natural_gas")}
    if not fuel_cols["electricity"] or not fuel_cols["natural_gas"]:
        raise ValueError(f"electricity / natural gas total column missing: {fuel_cols}")
    # city-stock weights w_i and floor areas, exactly as written by 05_EDA_v2
    ann = pd.read_parquet(C.TABLES / "annual_by_building.parquet",
                          columns=["building_uid", "w_city", "floor_area_m2"])
    w_city = ann.set_index("building_uid")["w_city"]
    area = ann.set_index("building_uid")["floor_area_m2"]

    hourly, peaks = [], []
    for city in C.CITIES:
        for dtype in ("residential", "commercial"):
            t = pq.read_table(C.BLDG_V2,
                              columns=["building_uid", "timestamp"] + list(fuel_cols.values()),
                              filters=[("city", "=", city), ("dataset_type", "=", dtype)])
            d = t.to_pandas()
            d["building_uid"] = d["building_uid"].astype(str)
            w = d["building_uid"].map(w_city)
            if w.isna().any():
                raise AssertionError(f"{city}/{dtype}: building without city weight")
            w = w.to_numpy("float64")
            h = pd.DataFrame({"timestamp": d["timestamp"]})
            for f, c in fuel_cols.items():
                x = d[c].astype("float64")
                d[f"{f}_w"] = x.to_numpy() * w            # w_i · L_i(t)
                d[f] = x
            g = d.groupby("timestamp")
            hs = pd.DataFrame({"timestamp": g.size().index})
            for f in fuel_cols:
                hs[f"{f}_kwh_w"] = g[f"{f}_w"].sum().to_numpy()   # Σ_i w_i L_i(t) (stock)
                hs[f"{f}_kwh"] = g[f].sum().to_numpy()            # Σ_i L_i(t) (sample)
            hs["city"], hs["dataset_type"] = city, dtype
            assert len(hs) == C.HOURS, f"{city}/{dtype}: {len(hs)} hours"
            hourly.append(hs)
            # each building's own peak per fuel (numerator of DF)
            pk = d.groupby("building_uid")[list(fuel_cols)].max().add_prefix("peak_").reset_index()
            pk["city"], pk["dataset_type"] = city, dtype
            peaks.append(pk)
            del t, d
            _log(f"A: read {city:9s} {dtype}")
    hourly = pd.concat(hourly, ignore_index=True)
    peaks = pd.concat(peaks, ignore_index=True)
    peaks["w_city"] = peaks["building_uid"].map(w_city).astype("float64")
    peaks["floor_area_m2"] = peaks["building_uid"].map(area).astype("float64")
    hourly.to_parquet(REV / "rev_A_stock_hourly_by_fuel.parquet", index=False)
    peaks.to_parquet(REV / "rev_A_peaks_by_building.parquet", index=False)

    rows = []
    gr = {"residential": "Οικιστικά", "commercial": "Εμπορικά", "all": "Σύνολο"}
    fgr = {"electricity": "Ηλεκτρισμός", "natural_gas": "Φυσικό αέριο", "site": "Συνολική ενέργεια"}
    for city in C.CITIES:
        for dtype in ("residential", "commercial", "all"):
            hs = hourly[hourly["city"] == city]
            pk = peaks[peaks["city"] == city]
            if dtype != "all":
                hs, pk = hs[hs["dataset_type"] == dtype], pk[pk["dataset_type"] == dtype]
            for f in ("electricity", "natural_gas", "site"):
                agg = hs.groupby("timestamp")[f"{f}_kwh_w"].sum()       # stock load per hour
                agg_s = hs.groupby("timestamp")[f"{f}_kwh"].sum()
                peak_sum = float((pk[f"peak_{f}"] * pk["w_city"]).sum())
                peak_agg = float(agg.max())
                if peak_agg <= 0:          # e.g. no gas at all in a stratum
                    continue
                DF = peak_sum / peak_agg
                DF_s = float(pk[f"peak_{f}"].sum() / agg_s.max())
                assert DF >= 1 - 1e-9, f"DF < 1 ({city}/{dtype}/{f})"
                t_pk = pd.Timestamp(agg.idxmax())
                summer = agg[agg.index.month.isin([6, 7, 8, 9])]
                winter = agg[agg.index.month.isin([12, 1, 2])]
                rows.append({
                    "Πόλη": city, "Ζώνη": C.CITY_ZONE[city], "Τύπος": gr[dtype],
                    "Καύσιμο": fgr[f], "N": len(pk),
                    "Σ ατομικών αιχμών, απόθεμα (MW)": round(peak_sum / 1000, 1),
                    "Αιχμή συνόλου, απόθεμα (MW)": round(peak_agg / 1000, 1),
                    "DF": round(DF, 3), "DF δείγματος (χωρίς στάθμιση)": round(DF_s, 3),
                    "Συντελεστής φορτίου (μέσο/αιχμή)": round(float(agg.mean()) / peak_agg, 3),
                    "Ώρα αιχμής συνόλου": str(t_pk),
                    "Μήνας αιχμής": int(t_pk.month), "Ώρα": int(t_pk.hour),
                    "Αιχμή καλοκαιριού Ιουν–Σεπ (MW)": round(float(summer.max()) / 1000, 1),
                    "Ώρα αιχμής καλοκαιριού": str(summer.idxmax()),
                    "Αιχμή χειμώνα Δεκ–Φεβ (MW)": round(float(winter.max()) / 1000, 1),
                    "Ώρα αιχμής χειμώνα": str(winter.idxmax()),
                    "Αιχμή συνόλου (W/m²)": round(
                        peak_agg / float((pk["floor_area_m2"] * pk["w_city"]).sum()) * 1000, 2)})
    t = pd.DataFrame(rows)
    _save(t, "rev_A_diversity_by_fuel.csv")
    print(t[["Πόλη", "Τύπος", "Καύσιμο", "DF", "Αιχμή συνόλου, απόθεμα (MW)",
             "Ώρα αιχμής συνόλου"]].to_string(index=False))


# ═════════════════════════════════════════════════════════════════════
# Shared loading for stages B and C (same arrays as 02_ML_v2)
# ═════════════════════════════════════════════════════════════════════
def _load_features():
    ml = _load_module("02_ML_v2.py", "ml02")
    df, target = ml.load()                       # same columns / dtypes as 02_ML_v2
    return ml, df, target


def _rmae_table(d, models, setname):
    """Per-building relative MAE (MAE model / MAE persistence), summarised so that
    every building counts once (size-independent; professor comment 5)."""
    e = pd.DataFrame({m: (d[m].astype("float64") - d["y"]).abs() for m in models})
    e["building_uid"] = d["building_uid"].astype(str).to_numpy()
    e["Τύπος"] = d["dataset_type"].astype(str).map({"residential": "Οικιστικά",
                                                     "commercial": "Εμπορικά"}).to_numpy()
    mae = e.groupby(["building_uid", "Τύπος"]).mean().reset_index()
    rows = []
    for typ, g in [("Όλα", mae)] + list(mae.groupby("Τύπος")):
        for m in models[1:]:
            r = (g[m] / g[models[0]]).replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"Σύνολο": setname, "Τύπος": typ, "Μοντέλο": m, "Κτίρια": len(r),
                         "Διάμεσος rMAE": round(r.median(), 3),
                         "% κτιρίων καλύτερα από persistence": round(100 * (r < 1).mean(), 1)})
    return rows


# ═════════════════════════════════════════════════════════════════════
# B — Strictly chronological split (train Jan–Sep, test Oct–Dec)
# ═════════════════════════════════════════════════════════════════════
def stage_B(drop=(), tag="B"):
    """drop: features left out (stage E = no month_sin/cos); tag: prefix of the output files."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    ml, df, target = _load_features()
    FEATURES = [f for f in ml.FEATURES if f not in drop]
    month = df["timestamp"].dt.month.to_numpy()
    df["split_chrono"] = np.where(month <= 9, "train", "test")
    is_oos = (df["zone"] == C.OOS_ZONE).to_numpy()
    tr = df[(~is_oos) & (df["split_chrono"] == "train")]
    te = df[(~is_oos) & (df["split_chrono"] == "test")]
    oos = df[is_oos & (df["split_chrono"] == "test")]
    _log(f"{tag}: train (Jan–Sep) {len(tr):,} | test (Oct–Dec) {len(te):,} | Denver Oct–Dec {len(oos):,}")

    # s_b = mean load of building b over Jan–Sep ONLY (Denver: its own Jan–Sep, as in 02)
    trs = df[df["split_chrono"] == "train"]
    scale = trs[target].astype("float64").groupby(trs["building_uid"].astype(str)).mean()
    scale = scale.where(scale > 1e-6, 1.0).rename("scale_kwh")
    s_tr, s_te, s_oo = (scale_rows(d, scale) for d in (tr, te, oos))
    X = lambda d, s: scale_X(d[FEATURES].to_numpy(np.float32), s, FEATURES)
    Xtr, Xte, Xoo = X(tr, s_tr), X(te, s_te), X(oos, s_oo)
    ytr, yte, yoo = (d[target].to_numpy(np.float32) for d in (tr, te, oos))
    ys_tr = (ytr / s_tr).astype(np.float32)

    p_te = {"Persistence (t−1h)": te["lag_1h"].to_numpy(),
            "Seasonal Naive (t−168h)": te["lag_168h"].to_numpy()}
    p_oo = {"Persistence (t−1h)": oos["lag_1h"].to_numpy(),
            "Seasonal Naive (t−168h)": oos["lag_168h"].to_numpy()}
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, ys_tr)
    p_te["Ridge Regression"], p_oo["Ridge Regression"] = ridge.predict(Xte) * s_te, ridge.predict(Xoo) * s_oo
    _log(f"{tag}: Ridge done")
    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=C.SEED, n_jobs=-1)
    rf.fit(Xtr, ys_tr)
    p_te["Random Forest"], p_oo["Random Forest"] = rf.predict(Xte) * s_te, rf.predict(Xoo) * s_oo
    del rf
    _log(f"{tag}: Random Forest done")
    best = json.loads((C.MODELS / "best_params_unified.json").read_text(encoding="utf-8"))
    params = {k: v for k, v in best.items() if k not in ("val_rmse", "tuning_minutes")}
    xgb = XGBRegressor(tree_method="hist", random_state=C.SEED, n_jobs=-1, **params)
    xgb.fit(Xtr, ys_tr, verbose=False)
    p_te["XGBoost"], p_oo["XGBoost"] = xgb.predict(Xte) * s_te, xgb.predict(Xoo) * s_oo
    _log(f"{tag}: XGBoost done")

    rows = []
    for setname, y, P in (("Έλεγχος Οκτ–Δεκ (2A–5A)", yte, p_te), ("Denver Οκτ–Δεκ (5B)", yoo, p_oo)):
        rp = float(np.sqrt(np.mean((y - P["Persistence (t−1h)"]) ** 2)))
        for n, p in P.items():
            rows.append({"Σύνολο": setname, **metrics(y, p, n, rp)})
    _save(pd.DataFrame(rows), f"rev_{tag}_chrono_benchmark.csv")

    # predictions (also the input of the per-building table)
    out_te = te[["building_uid", "timestamp", "zone", "dataset_type"]].copy()
    out_te["y"] = yte
    for n, p in p_te.items():
        out_te[n] = np.asarray(p, dtype="float32")
    out_oo = oos[["building_uid", "timestamp", "zone", "dataset_type"]].copy()
    out_oo["y"] = yoo
    for n, p in p_oo.items():
        out_oo[n] = np.asarray(p, dtype="float32")
    pd.concat([out_te.assign(set="test"), out_oo.assign(set="oos")]).to_parquet(
        REV / f"rev_{tag}_predictions.parquet", index=False)
    models = list(p_te)
    _save(pd.DataFrame(_rmae_table(out_te, models, "Έλεγχος Οκτ–Δεκ (2A–5A)")
                       + _rmae_table(out_oo, models, "Denver Οκτ–Δεκ (5B)")),
          f"rev_{tag}_chrono_rmae.csv")

    # same hours, two protocols: the main-protocol test days of Oct–Dec
    rows = []
    for setname, f, d_new in (("Έλεγχος (2A–5A)", "predictions_test.parquet", out_te),
                              ("Denver (5B)", "predictions_oos.parquet", out_oo)):
        old = pd.read_parquet(C.OUT_DIR / f)
        old = old[old["timestamp"].dt.month >= 10]
        key = ["building_uid", "timestamp"]
        old["building_uid"] = old["building_uid"].astype(str)
        d_new = d_new.assign(building_uid=d_new["building_uid"].astype(str))
        m = old.merge(d_new, on=key, suffixes=("_main", "_chrono"))
        assert len(m) == len(old), "hours of the main test set missing in the chronological run"
        rp = float(np.sqrt(np.mean((m["y_main"] - m["Persistence (t−1h)_main"]) ** 2)))
        for n in ("Random Forest", "XGBoost"):
            rows.append({"Σύνολο": setname, "Πρωτόκολλο": "κύριο (τελευταίες 7 ημ./μήνα)",
                         **metrics(m["y_main"], m[f"{n}_main"], n, rp)})
            rows.append({"Σύνολο": setname, "Πρωτόκολλο": "χρονολογικό (εκπαίδευση Ιαν–Σεπ)",
                         **metrics(m["y_main"], m[f"{n}_chrono"], n, rp)})
    _save(pd.DataFrame(rows), f"rev_{tag}_chrono_same_hours.csv")


# ═════════════════════════════════════════════════════════════════════
# C — Random Forest tuned with Optuna (same protocol as XGBoost)
# ═════════════════════════════════════════════════════════════════════
def stage_C():
    import optuna
    from sklearn.ensemble import RandomForestRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ml, df, target = _load_features()
    FEATURES = ml.FEATURES
    is_oos = df["zone"] == C.OOS_ZONE
    dev = df[~is_oos]
    oos = df[is_oos & (df["split"] == "test")]
    tr, va, te = (dev[dev["split"] == s] for s in ("train", "val", "test"))
    scale = pd.read_parquet(C.MODELS / "building_scale.parquet").set_index("building_uid")["scale_kwh"]
    scale.index = scale.index.astype(str)
    s_tr, s_va, s_te, s_oo = (scale_rows(d, scale) for d in (tr, va, te, oos))
    X = lambda d, s: scale_X(d[FEATURES].to_numpy(np.float32), s, FEATURES)
    Xtr, Xva, Xte, Xoo = X(tr, s_tr), X(va, s_va), X(te, s_te), X(oos, s_oo)
    ytr, yva, yte, yoo = (d[target].to_numpy(np.float32) for d in (tr, va, te, oos))
    ys_tr, ys_va = (ytr / s_tr).astype(np.float32), (yva / s_va).astype(np.float32)

    # same 25 % of training buildings as the XGBoost tuning (same seed / rule)
    g_tr, g_va = tr["building_uid"].astype(str).to_numpy(), va["building_uid"].astype(str).to_numpy()
    rng = np.random.RandomState(C.SEED)
    uids = np.unique(g_tr)
    pick = rng.choice(uids, int(len(uids) * C.TUNE_FRACTION), replace=False)
    mtr, mva = np.isin(g_tr, pick), np.isin(g_va, pick)
    _log(f"C: tuning on {len(pick):,}/{len(uids):,} buildings, {mtr.sum():,} train rows")

    def objective(trial):
        # search space bounded for memory/time on 26 M rows (depth ≤ 16 keeps every
        # tree ≤ 65.536 leaves; max_samples = share of rows drawn for each tree)
        p = dict(n_estimators=trial.suggest_int("n_estimators", 50, 150, step=25),
                 max_depth=trial.suggest_int("max_depth", 8, 16),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
                 max_features=trial.suggest_float("max_features", 0.3, 1.0),
                 max_samples=trial.suggest_float("max_samples", 0.05, 0.3))
        m = RandomForestRegressor(random_state=C.SEED, n_jobs=-1, **p)
        m.fit(Xtr[mtr], ys_tr[mtr])
        pred = m.predict(Xva[mva])
        return float(np.sqrt(np.mean((s_va[mva] * (ys_va[mva].astype("float64") - pred)) ** 2)))

    # baseline = the fixed settings of 02_ML_v2, evaluated on the same subset first
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    # first trial = the settings of 02_ML_v2 (50 trees, depth 12, all features)
    study.enqueue_trial({"n_estimators": 50, "max_depth": 12, "min_samples_leaf": 1,
                         "max_features": 1.0, "max_samples": 0.3})
    t0 = time.time()
    study.optimize(objective, n_trials=RF_TUNE_TRIALS, timeout=RF_TUNE_MINUTES * 60,
                   callbacks=[lambda st, t: _log(f"C: trial {t.number:2d} val RMSE {t.value:.4f} {t.params}")])
    study.trials_dataframe().to_csv(REV / "rev_C_rf_optuna_trials.csv", index=False,
                                    encoding="utf-8-sig")
    best = dict(study.best_params)
    info = {**best, "val_rmse": study.best_value, "trials": len(study.trials),
            "tuning_minutes": round((time.time() - t0) / 60, 1)}
    (REV / "rev_C_rf_best_params.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _log(f"C: best {info}")

    # refit on ALL train+val buildings, evaluate as Table 4.4
    rf = RandomForestRegressor(random_state=C.SEED, n_jobs=-1, **best)
    rf.fit(np.vstack([Xtr, Xva]), np.concatenate([ys_tr, ys_va]))
    p_te, p_oo = rf.predict(Xte) * s_te, rf.predict(Xoo) * s_oo
    old_te = pd.read_parquet(C.OUT_DIR / "predictions_test.parquet")
    old_oo = pd.read_parquet(C.OUT_DIR / "predictions_oos.parquet")
    # same row order as 02_ML_v2 (same filter on the same file) — checked here
    assert np.allclose(old_te["y"].to_numpy(), yte) and np.allclose(old_oo["y"].to_numpy(), yoo)
    rows = []
    for setname, y, old, p in (("Έλεγχος (2A–5A)", yte, old_te, p_te), ("Denver (5B)", yoo, old_oo, p_oo)):
        rp = float(np.sqrt(np.mean((y - old["Persistence (t−1h)"].to_numpy()) ** 2)))
        rows.append({"Σύνολο": setname, **metrics(y, old["Random Forest"], "Random Forest (σταθερές ρυθμίσεις)", rp)})
        rows.append({"Σύνολο": setname, **metrics(y, p, "Random Forest (Optuna)", rp)})
        rows.append({"Σύνολο": setname, **metrics(y, old["XGBoost"], "XGBoost (Optuna)", rp)})
    _save(pd.DataFrame(rows), "rev_C_rf_benchmark.csv")
    d_te = old_te[["building_uid", "dataset_type", "y", "Persistence (t−1h)"]].assign(**{"RF Optuna": p_te})
    d_oo = old_oo[["building_uid", "dataset_type", "y", "Persistence (t−1h)"]].assign(**{"RF Optuna": p_oo})
    _save(pd.DataFrame(_rmae_table(d_te, ["Persistence (t−1h)", "RF Optuna"], "Έλεγχος (2A–5A)")
                       + _rmae_table(d_oo, ["Persistence (t−1h)", "RF Optuna"], "Denver (5B)")),
          "rev_C_rf_rmae.csv")


# ═════════════════════════════════════════════════════════════════════
# D — Climate model without month_sin / month_cos
# ═════════════════════════════════════════════════════════════════════
def stage_D():
    cl = _load_module("03_CLIMATE_v2.py", "clim03")
    cl.CLIM_FEATURES = [f for f in cl.CLIM_FEATURES if f not in ("month_sin", "month_cos")]
    cl.update_summary = lambda *a, **k: None          # do not touch data_summary.json
    d = REV / "D_no_month"
    try:                                               # Windows console (cp1253) and «✓»
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # Time: the no-month model is trained once (deterministic, seed 42) and reused if it
    # exists; only the CENTRAL ΔT of each scenario is projected (sensitivity check, the
    # "very likely" ranges are those of the main model, Table 5.5).
    saved = d / "models" / "xgb_climate.json"
    if saved.exists():
        cl.REUSE_MODEL = saved
        _log(f"D: re-using the trained no-month model {saved}")
    cl.CENTRAL_ONLY = True
    # redirect the outputs of 03 to the revision folder (config globals read at run time)
    old_paths = (C.TABLES, C.MODELS)
    C.TABLES, C.MODELS = d / "tables", d / "models"
    try:
        for p in (C.TABLES, C.MODELS):
            p.mkdir(parents=True, exist_ok=True)
        _log(f"D: climate model with {len(cl.CLIM_FEATURES)} features (no month): {cl.CLIM_FEATURES}")
        cl.main()
        for src, dst in (("table_5_1_climate.csv", "rev_D_climate_no_month.csv"),
                         ("table_5_0_climate_model_accuracy.csv", "rev_D_climate_accuracy_no_month.csv"),
                         ("table_5_1_climate_by_building.parquet", "rev_D_climate_by_building_no_month.parquet"),
                         ("table_5_0b_extrapolation.csv", "rev_D_extrapolation_no_month.csv")):
            (REV / dst).write_bytes((C.TABLES / src).read_bytes())
        _log("D: copied to revision/")
    finally:
        # restore the normal folders, otherwise a following stage (E, F) would look for
        # best_params_unified.json / building_scale.parquet in D_no_month/models
        C.TABLES, C.MODELS = old_paths


# ═════════════════════════════════════════════════════════════════════
# E — Chronological split WITHOUT the month variables (diagnostic of B)
# ═════════════════════════════════════════════════════════════════════
def stage_E():
    """In stage B the months October–December never appear in training, so
    month_sin / month_cos take values the trees have not seen (December:
    sin = 0, cos = 1). Stage B showed a negative bias of XGBoost in Oct–Dec.
    Same run without the two month features shows how much of the drop is due
    to this extrapolation of the calendar and not to the split itself."""
    stage_B(drop=("month_sin", "month_cos"), tag="E")


# ═════════════════════════════════════════════════════════════════════
# F — Tuned Random Forest: save the hourly predictions (for the
#     significance test of the XGBoost – tuned RF difference, as in 08)
# ═════════════════════════════════════════════════════════════════════
def stage_F():
    """Refit RF with the parameters found in stage C (rev_C_rf_best_params.json)
    on train+val and save its predictions next to the saved predictions of
    02_ML_v2 (same rows, same order). The bootstrap / Diebold–Mariano test is
    then run with the SAME code as 08_SIGNIFICANCE_v2 (12_REVISION_FIGURES)."""
    from sklearn.ensemble import RandomForestRegressor
    ml, df, target = _load_features()
    FEATURES = ml.FEATURES
    is_oos = df["zone"] == C.OOS_ZONE
    dev = df[~is_oos]
    oos = df[is_oos & (df["split"] == "test")]
    tr, va, te = (dev[dev["split"] == s] for s in ("train", "val", "test"))
    scale = pd.read_parquet(C.MODELS / "building_scale.parquet").set_index("building_uid")["scale_kwh"]
    scale.index = scale.index.astype(str)
    s_tr, s_va, s_te, s_oo = (scale_rows(d, scale) for d in (tr, va, te, oos))
    X = lambda d, s: scale_X(d[FEATURES].to_numpy(np.float32), s, FEATURES)
    Xtrva = np.vstack([X(tr, s_tr), X(va, s_va)])
    ys = np.concatenate([tr[target].to_numpy(np.float32) / s_tr, va[target].to_numpy(np.float32) / s_va])
    best = json.loads((REV / "rev_C_rf_best_params.json").read_text(encoding="utf-8"))
    params = {k: best[k] for k in ("n_estimators", "max_depth", "min_samples_leaf",
                                   "max_features", "max_samples")}
    rf = RandomForestRegressor(random_state=C.SEED, n_jobs=-1, **params)
    rf.fit(Xtrva.astype(np.float32), ys.astype(np.float32))
    del Xtrva, ys
    _log(f"F: tuned RF refit {params}")
    for f_old, d, s, name in (("predictions_test.parquet", te, s_te, "rev_F_predictions_test.parquet"),
                              ("predictions_oos.parquet", oos, s_oo, "rev_F_predictions_oos.parquet")):
        old = pd.read_parquet(C.OUT_DIR / f_old)
        assert np.allclose(old["y"].to_numpy(), d[target].to_numpy(np.float32)), "row order differs"
        old["Random Forest (Optuna)"] = (rf.predict(X(d, s)) * s).astype("float32")
        old.to_parquet(REV / name, index=False)
        _log(f"F: saved {name}")


if __name__ == "__main__":
    # Windows console (cp1253) cannot print «✓» etc. when output is redirected
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    REV.mkdir(parents=True, exist_ok=True)
    stages = sys.argv[1:] or ["A", "B", "C", "D"]
    for s in stages:
        t0 = time.time()
        _log(f"===== stage {s} started")
        {"A": stage_A, "B": stage_B, "C": stage_C, "D": stage_D, "E": stage_E, "F": stage_F}[s.upper()]()
        _log(f"===== stage {s} finished in {(time.time() - t0) / 60:.1f} min")
    print("✓ 11_REVISION_v2 finished")
