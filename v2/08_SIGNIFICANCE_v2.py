#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_SIGNIFICANCE_v2.py  —  Statistical significance of the model comparison (Ch.4 §4.4.3)
Thesis MIS24011 — Zygouris Aristotelis

Why: Table 4.8 gives point estimates of the skill score. A reader (or examiner)
can ask whether e.g. XGBoost 0,153 vs Random Forest 0,185 on Denver is a real
difference or noise. Two complementary checks, both on the SAVED predictions of
02_ML_v2 (no model is re-trained):

1. Cluster (building) bootstrap — Efron & Tibshirani (1993).
   The unit of independent information is the BUILDING (hours of one building are
   strongly autocorrelated), so buildings are resampled with replacement, within
   each city × type stratum (the sampling design of Ch.3: 600 res + 400 com per
   city). For every resample:  skill_m = 1 − sqrt(Σ SSE_m / Σ SSE_persistence)
   (identical to 1 − RMSE_m / RMSE_p because N cancels). B = 2.000, seed 42.
   Reported: 95 % percentile intervals of each skill and of the paired
   differences (XGBoost − RF, XGBoost − Ridge, RF − Ridge) and a two-sided
   bootstrap p-value  2·min(P(Δ ≤ 0), P(Δ ≥ 0)).

2. Diebold–Mariano test (Diebold & Mariano, 1995) with the Harvey et al. (1997)
   small-sample correction, on the HOURLY loss differential of the whole sample:
   d_t = Σ_b (e_A,b,t² − e_B,b,t²). T = 2.016 test hours (84 days × 24);
   Newey–West long-run variance with lag h = 24 (one day).

Outputs (outputs_v2/):
  tables/table_4_4e_bootstrap_skill.csv
  tables/table_4_4f_bootstrap_differences.csv
  tables/table_4_4g_diebold_mariano.csv
  figures/fig_4_2b_skill_ci.png
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as C
from common import gr, gr_axis, save_fig, setup_mpl

B = 2000
MODELS = ["Ridge Regression", "Random Forest", "XGBoost"]
PERS = "Persistence (t−1h)"
PAIRS = [("XGBoost", "Random Forest"), ("XGBoost", "Ridge Regression"),
         ("Random Forest", "Ridge Regression")]


def building_sse(path):
    """One row per building: stratum + Σ squared error of every model (float64)."""
    p = pd.read_parquet(path)
    cols = [PERS] + MODELS
    y = p["y"].to_numpy("float64")
    sse = pd.DataFrame({m: (y - p[m].to_numpy("float64")) ** 2 for m in cols})
    sse["building_uid"] = p["building_uid"].astype(str).to_numpy()
    sse["stratum"] = (p["zone"].astype(str) + "|" + p["dataset_type"].astype(str)).to_numpy()
    sse["type"] = p["dataset_type"].astype(str).to_numpy()
    g = sse.groupby(["building_uid", "stratum", "type"], sort=True)[cols].sum().reset_index()
    # hourly loss differentials for Diebold–Mariano (summed over buildings)
    hourly = {}
    for a, b in PAIRS + [(m, PERS) for m in MODELS]:
        d = (y - p[a].to_numpy("float64")) ** 2 - (y - p[b].to_numpy("float64")) ** 2
        hourly[(a, b)] = pd.Series(d).groupby(p["timestamp"].to_numpy()).sum().sort_index().to_numpy()
    return g, hourly


def skill(sse_tot):
    return {m: 1 - np.sqrt(sse_tot[m] / sse_tot[PERS]) for m in MODELS}


def bootstrap(g, rng):
    """Stratified cluster bootstrap over buildings → array (B, len(MODELS)) of skills."""
    strata = [np.flatnonzero(g["stratum"].to_numpy() == s) for s in sorted(g["stratum"].unique())]
    M = g[[PERS] + MODELS].to_numpy()
    out = np.empty((B, len(MODELS)))
    for k in range(B):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in strata])
        tot = M[idx].sum(axis=0)
        out[k] = 1 - np.sqrt(tot[1:] / tot[0])
    return out


def diebold_mariano(d, h=24):
    """DM statistic with Newey–West variance (lag h) and Harvey–Leybourne–Newbold correction."""
    T = len(d)
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / T
    lrv = gamma0 + 2 * sum((1 - j / (h + 1)) * np.dot(dc[j:], dc[:-j]) / T for j in range(1, h + 1))
    dm = d.mean() / np.sqrt(lrv / T)
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)       # HLN (1997) correction
    stat = dm * k
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return stat, p


def main():
    C.ensure_dirs()
    rng = np.random.default_rng(C.SEED)
    rows_s, rows_d, rows_dm = [], [], []
    sets = {"Έλεγχος 2A–5A": C.OUT_DIR / "predictions_test.parquet",
            "Denver 5B (OOS)": C.OUT_DIR / "predictions_oos.parquet"}
    boots = {}
    for name, path in sets.items():
        g, hourly = building_sse(path)
        subsets = {"Όλα": g, "Οικιστικά": g[g["type"] == "residential"],
                   "Εμπορικά": g[g["type"] == "commercial"]}
        for sub, gg in subsets.items():
            point = skill(gg[[PERS] + MODELS].sum())
            bs = bootstrap(gg.reset_index(drop=True), rng)
            if sub == "Όλα":
                boots[name] = bs
            for j, m in enumerate(MODELS):
                lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
                rows_s.append({"Σύνολο": name, "Τύπος": sub, "Μοντέλο": m, "Κτίρια": len(gg),
                               "Skill": round(point[m], 4), "95% ΔΕ κάτω": round(lo, 4),
                               "95% ΔΕ άνω": round(hi, 4)})
            for a, b in PAIRS:
                ja, jb = MODELS.index(a), MODELS.index(b)
                diff = bs[:, ja] - bs[:, jb]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                rows_d.append({"Σύνολο": name, "Τύπος": sub, "Σύγκριση": f"{a} − {b}",
                               "Διαφορά skill": round(point[a] - point[b], 4),
                               "95% ΔΕ κάτω": round(lo, 4), "95% ΔΕ άνω": round(hi, 4),
                               "p (bootstrap)": round(p, 4),
                               "Σημαντική (5%)": "ναι" if (lo > 0 or hi < 0) else "όχι"})
        for (a, b), d in hourly.items():
            stat, p = diebold_mariano(d)
            rows_dm.append({"Σύνολο": name, "Σύγκριση": f"{a} vs {b}", "Ώρες T": len(d),
                            "Μέση διαφορά απώλειας ανά ώρα (10³ kWh²)": round(d.mean() / 1e3, 2),
                            "DM (HLN)": round(stat, 3), "p": float(f"{p:.3g}"),
                            "Καλύτερο": a if d.mean() < 0 else b})
        print(f"  ✓ {name}: {len(g):,} buildings")
    ts, td, tdm = pd.DataFrame(rows_s), pd.DataFrame(rows_d), pd.DataFrame(rows_dm)
    ts.to_csv(C.TABLES / "table_4_4e_bootstrap_skill.csv", index=False, encoding="utf-8-sig")
    td.to_csv(C.TABLES / "table_4_4f_bootstrap_differences.csv", index=False, encoding="utf-8-sig")
    tdm.to_csv(C.TABLES / "table_4_4g_diebold_mariano.csv", index=False, encoding="utf-8-sig")
    print(ts.to_string(index=False)); print(td.to_string(index=False)); print(tdm.to_string(index=False))

    # ── Figure 4.2b: skill with 95 % intervals + paired differences ──────
    plt = setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    col = {"Έλεγχος 2A–5A": "#3182bd", "Denver 5B (OOS)": "#fd8d3c"}
    for k, (name, off) in enumerate((("Έλεγχος 2A–5A", -0.12), ("Denver 5B (OOS)", 0.12))):
        s = ts[(ts["Σύνολο"] == name) & (ts["Τύπος"] == "Όλα")].set_index("Μοντέλο").loc[MODELS]
        y = np.arange(len(MODELS)) + off
        axes[0].errorbar(s["Skill"], y, xerr=[s["Skill"] - s["95% ΔΕ κάτω"], s["95% ΔΕ άνω"] - s["Skill"]],
                         fmt="o", ms=7, capsize=4, lw=1.6, color=col[name], label=name)
        for yi, v, hi in zip(y, s["Skill"], s["95% ΔΕ άνω"]):       # value right of the interval
            axes[0].text(hi + 0.008, yi, gr(v, 3), va="center", fontsize=8)
    axes[0].set_yticks(range(len(MODELS)), MODELS)
    axes[0].set_ylim(-0.5, len(MODELS) - 0.5)
    axes[0].set_xlim(right=0.5)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("Skill έναντι persistence (95 % διάστημα, bootstrap ανά κτίριο)")
    axes[0].set_title("Skill ανά μοντέλο")
    axes[0].legend(fontsize=8, loc="lower right")
    gr_axis(axes[0], "x")
    labels = [f"{a} − {b}" for a, b in PAIRS]
    for name, off in (("Έλεγχος 2A–5A", -0.12), ("Denver 5B (OOS)", 0.12)):
        d = td[(td["Σύνολο"] == name) & (td["Τύπος"] == "Όλα")].set_index("Σύγκριση").loc[labels]
        y = np.arange(len(labels)) + off
        axes[1].errorbar(d["Διαφορά skill"], y,
                         xerr=[d["Διαφορά skill"] - d["95% ΔΕ κάτω"], d["95% ΔΕ άνω"] - d["Διαφορά skill"]],
                         fmt="s", ms=6, capsize=4, lw=1.6, color=col[name], label=name)
    axes[1].set_yticks(range(len(labels)), [l.replace("Regression", "").replace(" − ", "\n− ") for l in labels])
    axes[1].axvline(0, color="k", lw=0.8, ls="--")
    axes[1].set_ylim(-0.5, len(labels) - 0.5)
    axes[1].set_xlabel("Διαφορά skill (διάστημα που δεν περιέχει το 0 = σημαντική διαφορά)")
    axes[1].set_title("Διαφορές ανά ζεύγος μοντέλων")
    gr_axis(axes[1], "x")
    save_fig(fig, "fig_4_2b_skill_ci.png")
    print("✓ 08_SIGNIFICANCE_v2 finished")


if __name__ == "__main__":
    main()
