#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_ML_v2.py  —  Model benchmarking without test-set leakage
Thesis MIS24011 — Zygouris Aristotelis

Fixes vs 02_ML_INDIVIDUAL.py / 02b_ML_SEPARATE_MODELS*.py
  K3  Optuna objective AND early stopping used the TEST set.
      -> Optuna/early stopping on a VALIDATION window (7 days before
         the test window of every month). Test is touched once, at the end.
  K1  Fig. 4.1 was simulated. -> every Optuna trial is saved to CSV and
      plotted from that file (07_FIGURES_v2.py).
  #2  Persistence baseline + skill score vs persistence for every model.
  Final models are refit on train+val with the selected n_estimators.
  #18 (after the first full run, 12/10/2026) per-building scaling:
      target and level-dependent lags are divided by the building's
      mean train-period load s_b; predictions are multiplied back by s_b.
      Reason and sources in common.py (SCALED_FEATURES). Scales saved to
      models/building_scale.parquet (used again by 02c).

Outputs (outputs_v2/):
  tables/table_4_4_benchmark_test.csv     (all models, test set, zones 2A–5A)
  tables/table_4_4_benchmark_oos.csv      (same models, Denver 5B)
  tables/table_4_4b_seasonal.csv          (every model × season, test set)
  tables/table_4_4c_by_type.csv           (unified + separate models × type)
  tables/optuna_trials_<name>.csv, models/best_params_<name>.json
  tables/feature_importance_gain.csv
  models/xgb_<name>.json
  predictions_test.parquet / predictions_oos.parquet (for the figures)
"""
import time

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config as C
from common import (building_scale, metrics, save_json, scale_rows,
                    scale_X, update_summary)

optuna.logging.set_verbosity(optuna.logging.WARNING)
FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek", "is_weekend",
    "temperature_c", "rel_humidity", "solar_ghi_wm2", "hdd", "cdd",
    "lag_1h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h",
    "is_holiday", "is_commercial", "floor_area_log"]
SEASON = {12: "Χειμώνας", 1: "Χειμώνας", 2: "Χειμώνας", 3: "Άνοιξη", 4: "Άνοιξη",
          5: "Άνοιξη", 6: "Καλοκαίρι", 7: "Καλοκαίρι", 8: "Καλοκαίρι",
          9: "Φθινόπωρο", 10: "Φθινόπωρο", 11: "Φθινόπωρο"}


def load():
    cols = pq.read_schema(C.FEAT_V2).names
    target = next(c for c in cols if c.startswith("out.site_energy.total"))
    use = ["building_uid", "timestamp", "zone", "dataset_type", "split", target] + FEATURES
    df = pq.read_table(C.FEAT_V2, columns=use,
                       read_dictionary=["building_uid", "zone", "dataset_type", "split"]).to_pandas()
    return df, target


def tune_xgb(X_tr, y_tr, X_va, y_va, name, groups=None, groups_va=None, s_va=None):
    """
    Optuna on the validation window. Returns best params (+ n_estimators).
    For run-time reasons the trials use a random TUNE_FRACTION of the
    BUILDINGS (groups); the validation rows of the SAME buildings are used,
    so that early stopping and the Optuna objective see the same building
    population as the training subset. The final model uses all of
    train+val.
    """
    if groups is not None and C.TUNE_FRACTION < 1:
        rng = np.random.RandomState(C.SEED)
        uids = np.unique(groups)
        pick = rng.choice(uids, int(len(uids) * C.TUNE_FRACTION), replace=False)
        mask = np.isin(groups, pick)
        X_tr, y_tr = X_tr[mask], y_tr[mask]
        if groups_va is not None:
            mva = np.isin(groups_va, pick)
            X_va, y_va = X_va[mva], y_va[mva]
            if s_va is not None:
                s_va = s_va[mva]
        print(f"  [{name}] tuning on {len(pick):,}/{len(uids):,} training buildings")
    def objective(trial):
        p = dict(
            max_depth=trial.suggest_int("max_depth", 4, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        )
        m = XGBRegressor(n_estimators=C.MAX_TREES, tree_method="hist", random_state=C.SEED,
                         n_jobs=-1, early_stopping_rounds=C.EARLY_STOPPING_ROUNDS,
                         eval_metric="rmse", **p)
        # early stopping on the same criterion as the objective: with weights
        # s_b² the weighted RMSE of the scaled target ∝ RMSE in kWh
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False,
              sample_weight_eval_set=None if s_va is None else [np.asarray(s_va) ** 2])
        trial.set_user_attr("n_estimators", int(m.best_iteration) + 1)
        pred = m.predict(X_va)
        if s_va is None:
            return float(np.sqrt(np.mean((y_va - pred) ** 2)))
        # objective in kWh (the unit of every reported metric)
        return float(np.sqrt(np.mean((s_va * (y_va.astype("float64") - pred)) ** 2)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=C.N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda st, tr: print(f"    trial {tr.number:2d}: val RMSE {tr.value:.4f}")])
    trials = study.trials_dataframe()
    trials.to_csv(C.TABLES / f"optuna_trials_{name}.csv", index=False, encoding="utf-8-sig")
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    best["val_rmse"] = study.best_value
    best["tuning_minutes"] = round((time.time() - t0) / 60, 1)
    save_json(best, C.MODELS / f"best_params_{name}.json")
    print(f"  [{name}] best val RMSE = {study.best_value:.4f} | {best}")
    return best


def fit_final_xgb(best, X, y):
    p = {k: v for k, v in best.items() if k not in ("val_rmse", "tuning_minutes")}
    m = XGBRegressor(tree_method="hist", random_state=C.SEED, n_jobs=-1, **p)
    m.fit(X, y, verbose=False)
    return m


def main():
    C.ensure_dirs()
    df, target = load()
    is_oos = df["zone"] == C.OOS_ZONE
    # Denver (5B) is never used for training/tuning. It is scored on its TEST
    # days only (same 84 days as the other cities): the building scale s_b of
    # a Denver building is the mean of its own train-day load, so scoring the
    # train days would reuse that information (audit 12/10/2026).
    dev = df[~is_oos]
    oos = df[is_oos & (df["split"] == "test")]
    tr, va, te = (dev[dev["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | OOS {len(oos):,}")
    update_summary("split_rows", {"train": len(tr), "val": len(va), "test": len(te),
                                  "oos": len(oos),
                                  "buildings_dev": int(dev["building_uid"].nunique()),
                                  "buildings_oos": int(oos["building_uid"].nunique())})

    # Per-building level s_b from the TRAIN days only (#18; see common.py)
    scale = building_scale(df, target)
    pd.DataFrame({"building_uid": scale.index, "scale_kwh": scale.values}).to_parquet(
        C.MODELS / "building_scale.parquet", index=False)
    update_summary("building_scale_kwh", {"min": float(scale.min()),
                                          "median": float(scale.median()),
                                          "max": float(scale.max())})
    print(f"building scale s_b (kWh/h): min {scale.min():.4f} | median {scale.median():.3f}"
          f" | max {scale.max():.1f}")

    # Convert each block to numpy ONCE (memory/time; ~40M rows).
    # y = raw kWh (for the metrics), ys = y / s_b (what the models learn),
    # X = features with the level-dependent columns divided by s_b.
    X = lambda d, s: scale_X(d[FEATURES].to_numpy(np.float32), s, FEATURES)
    y = lambda d: d[target].to_numpy(np.float32)
    s_tr, s_va, s_te, s_oo = (scale_rows(d, scale) for d in (tr, va, te, oos))
    Xtr, Xva, Xte, Xoo = X(tr, s_tr), X(va, s_va), X(te, s_te), X(oos, s_oo)
    ytr, yva, yte, yoo = y(tr), y(va), y(te), y(oos)
    ys_tr, ys_va = (ytr / s_tr).astype(np.float32), (yva / s_va).astype(np.float32)
    Xtrva, ys_trva = np.vstack([Xtr, Xva]), np.concatenate([ys_tr, ys_va])
    back_te = lambda p: np.asarray(p, dtype="float64") * s_te       # back to kWh
    back_oo = lambda p: np.asarray(p, dtype="float64") * s_oo

    preds_te = {"Persistence (t−1h)": te["lag_1h"].to_numpy(),
                "Seasonal Naive (t−168h)": te["lag_168h"].to_numpy()}
    preds_oos = {"Persistence (t−1h)": oos["lag_1h"].to_numpy(),
                 "Seasonal Naive (t−168h)": oos["lag_168h"].to_numpy()}

    # Ridge (standardised features)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(Xtrva, ys_trva)
    preds_te["Ridge Regression"] = back_te(ridge.predict(Xte))
    preds_oos["Ridge Regression"] = back_oo(ridge.predict(Xoo))

    # Random Forest (same settings as the submitted version)
    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=C.SEED, n_jobs=-1)
    rf.fit(Xtrva, ys_trva)
    preds_te["Random Forest"] = back_te(rf.predict(Xte))
    preds_oos["Random Forest"] = back_oo(rf.predict(Xoo))
    del rf

    # XGBoost unified
    best = tune_xgb(Xtr, ys_tr, Xva, ys_va, "unified", tr["building_uid"].to_numpy(),
                    va["building_uid"].to_numpy(), s_va)
    xgb = fit_final_xgb(best, Xtrva, ys_trva)
    xgb.save_model(str(C.MODELS / "xgb_unified.json"))
    preds_te["XGBoost"] = back_te(xgb.predict(Xte))
    preds_oos["XGBoost"] = back_oo(xgb.predict(Xoo))
    gain = pd.Series(xgb.get_booster().get_score(importance_type="gain"))
    gain.index = [FEATURES[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k
                  for k in gain.index]
    gain = (gain / gain.sum()).reindex(FEATURES).fillna(0).sort_values(ascending=False)
    gain.rename("gain_share").to_csv(C.TABLES / "feature_importance_gain.csv",
                                     encoding="utf-8-sig")

    # ── Tables ───────────────────────────────────────────────────────
    rmse_p_te = np.sqrt(np.mean((yte - preds_te["Persistence (t−1h)"]) ** 2))
    rmse_p_oos = np.sqrt(np.mean((yoo - preds_oos["Persistence (t−1h)"]) ** 2))
    t_te = pd.DataFrame([metrics(yte, p, n, rmse_p_te) for n, p in preds_te.items()])
    t_oos = pd.DataFrame([metrics(yoo, p, n, rmse_p_oos) for n, p in preds_oos.items()])
    for t, f in ((t_te, "table_4_4_benchmark_test.csv"), (t_oos, "table_4_4_benchmark_oos.csv")):
        t["Κατάταξη (RMSE)"] = t["RMSE (kWh/h)"].rank(method="min").astype(int)
        t.to_csv(C.TABLES / f, index=False, encoding="utf-8-sig")
        print(t.to_string(index=False))

    seas = te["timestamp"].dt.month.map(SEASON).to_numpy()
    rows = []
    for n, p in preds_te.items():
        for s in ("Χειμώνας", "Άνοιξη", "Καλοκαίρι", "Φθινόπωρο"):
            m = seas == s
            rp = float(np.sqrt(np.mean((yte[m] - preds_te["Persistence (t−1h)"][m]) ** 2)))
            rows.append({"Εποχή": s, **metrics(yte[m], p[m], n, rp)})
    pd.DataFrame(rows).to_csv(C.TABLES / "table_4_4b_seasonal.csv", index=False,
                              encoding="utf-8-sig")

    # ── Per building type: unified vs separate models ────────────────
    rows = []
    for dtype in ("residential", "commercial"):
        mte = (te["dataset_type"] == dtype).to_numpy()
        moo = (oos["dataset_type"] == dtype).to_numpy()
        # persistence RMSE of this building type -> skill score per type
        rp_te = float(np.sqrt(np.mean((yte[mte] - preds_te["Persistence (t−1h)"][mte]) ** 2)))
        rp_oo = float(np.sqrt(np.mean((yoo[moo] - preds_oos["Persistence (t−1h)"][moo]) ** 2)))
        rows.append({"Τύπος": dtype, "Σύνολο": "test",
                     **metrics(yte[mte], preds_te["XGBoost"][mte], "XGBoost unified", rp_te)})
        rows.append({"Τύπος": dtype, "Σύνολο": "OOS 5B",
                     **metrics(yoo[moo], preds_oos["XGBoost"][moo], "XGBoost unified", rp_oo)})
        rows.append({"Τύπος": dtype, "Σύνολο": "test",
                     **metrics(yte[mte], preds_te["Persistence (t−1h)"][mte], "Persistence", rp_te)})
        name = "res" if dtype == "residential" else "com"
        mtr = (tr["dataset_type"] == dtype).to_numpy()
        mva = (va["dataset_type"] == dtype).to_numpy()
        b = tune_xgb(Xtr[mtr], ys_tr[mtr], Xva[mva], ys_va[mva], name,
                     tr["building_uid"].to_numpy()[mtr], va["building_uid"].to_numpy()[mva],
                     s_va[mva])
        m = fit_final_xgb(b, np.vstack([Xtr[mtr], Xva[mva]]),
                          np.concatenate([ys_tr[mtr], ys_va[mva]]))
        m.save_model(str(C.MODELS / f"xgb_{name}.json"))
        rows.append({"Τύπος": dtype, "Σύνολο": "test",
                     **metrics(yte[mte], m.predict(Xte[mte]) * s_te[mte], f"XGBoost {name}",
                               rp_te)})
        rows.append({"Τύπος": dtype, "Σύνολο": "OOS 5B",
                     **metrics(yoo[moo], m.predict(Xoo[moo]) * s_oo[moo], f"XGBoost {name}",
                               rp_oo)})
    pd.DataFrame(rows).to_csv(C.TABLES / "table_4_4c_by_type.csv", index=False,
                              encoding="utf-8-sig")

    # ── Predictions for the figures ──────────────────────────────────
    out = te[["building_uid", "timestamp", "zone", "dataset_type"]].copy()
    out["y"] = yte
    for n, p in preds_te.items():
        out[n] = np.asarray(p, dtype="float32")
    out.to_parquet(C.OUT_DIR / "predictions_test.parquet", index=False)
    out = oos[["building_uid", "timestamp", "zone", "dataset_type"]].copy()
    out["y"] = yoo
    for n, p in preds_oos.items():
        out[n] = np.asarray(p, dtype="float32")
    out.to_parquet(C.OUT_DIR / "predictions_oos.parquet", index=False)
    print("✓ 02_ML_v2 finished")


if __name__ == "__main__":
    main()
