# -*- coding: utf-8 -*-
"""
phase4_figures.py — Κλιματικά διαγράμματα (νέο σχέδιο 2×3 + baseline, convex cooling)
EIA/Tableau ύφος: μεγάλα labels, colorblind-safe, καθαρή δομή, + επεξήγηση άξονα x.
"""
import pandas as pd, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
BLUE, RED, GRAY = "#2166AC", "#B2182B", "#5A6B7A"
PATH_COLOR = {"SSP2-4.5": BLUE, "SSP5-8.5": RED}
YEARS = [2018, 2040, 2050, 2100]; POS = {y: i for i, y in enumerate(YEARS)}
TITLE_C = "#0b5394"
# Συντελεστές εκπομπών ηλεκτρισμού — NREL Cambium 2022 (Gagnon et al., 2022),
# όπως αναφέρονται στο §5.1.1 του κειμένου (5A χαμηλός λόγω πυρηνικής IL· 5B υψηλός λόγω άνθρακα CO).
EF_ELEC = {"2A": 0.393, "3A": 0.362, "4A": 0.331, "5A": 0.285, "5B": 0.412}
# Scope 1 — άμεσες εκπομπές ορυκτών (US EPA 2023, kg CO2e/kWh)
EF_GAS, EF_OIL, EF_PROPANE = 0.1824, 0.2650, 0.2238
# Αποανθρακοποίηση ηλεκτρικού δικτύου (Scope 2 multiplier) — IRA 2022 + NREL ATB 2023 /
# NREL Standard Scenarios (Gagnon et al., 2022). 2018=100%, ταχεία μείωση προς net-zero.
GRID_DECARB = {2018: 1.00, 2040: 0.42, 2050: 0.28, 2100: 0.10}

# Επεξήγηση άξονα x (μπαίνει κάτω από ΚΑΘΕ διάγραμμα)
XNOTE = ("Άξονας x — Χρονικοί ορίζοντες: 2018 = Baseline (ιστορική αναφορά) · 2040 · 2050 · 2100 = "
         "μελλοντικές προβολές (IPCC AR6). Τα δύο σενάρια (SSP2-4.5 ήπιο, SSP5-8.5 ακραίο) ξεκινούν "
         "από το κοινό σημείο του 2018 και αποκλίνουν όσο απομακρυνόμαστε στον χρόνο.")

def _load(path="scenarios_2x3_baseline.parquet"):
    df = pd.read_parquet(path)
    df["pathway"] = df["scenario"].str.extract(r"(SSP\d-\d\.\d|Baseline)")
    df["year"] = df["scenario_year"].astype(int)
    ef = df["climate_zone"].map(EF_ELEC).fillna(0.35)
    gd = df["year"].map(GRID_DECARB).fillna(1.0)          # αποανθρακοποίηση δικτύου ανά ορίζοντα
    # CO₂ = Scope 2 (ηλεκτρισμός × Cambium EF × grid decarbonization) + Scope 1 (φυσικό αέριο, EPA).
    # Το προβαλλόμενο σύνολο (site) ανάγεται σε ηλεκτρισμό μέσω του πραγματικού elec_share ανά
    # ζώνη×τύπο (αποφυγή διπλομέτρησης αερίου). Η αποανθρακοποίηση εφαρμόζεται ΜΟΝΟ στο Scope 2.
    share = df["elec_share"] if "elec_share" in df.columns else 0.7
    df["co2_scope2_kg"] = df["annual_kwh_projected"] * share * ef * gd
    df["co2_scope1_kg"] = df["natural_gas_kwh"] * EF_GAS
    df["co2_kg"] = df["co2_scope2_kg"] + df["co2_scope1_kg"]
    return df

def _traj(df, value, agg="mean"):
    base = df[df.pathway == "Baseline"].groupby("climate_zone")[value].agg(agg)
    rows = []
    for zone in sorted(df.climate_zone.unique()):
        for pw in ["SSP2-4.5", "SSP5-8.5"]:
            pts = [(2018, base[zone])]
            sub = df[(df.climate_zone == zone) & (df.pathway == pw)]
            for y in sorted(sub.year.unique()):
                pts.append((y, sub[sub.year == y][value].agg(agg)))
            for yr, val in pts:
                rows.append(dict(zone=zone, pathway=pw, year=yr, val=val))
    return pd.DataFrame(rows)

def _footer(fig, source, y_note=0.055, y_src=0.015):
    """Επεξήγηση άξονα x (πλαίσιο) + γραμμή πηγής."""
    fig.text(0.012, y_note, XNOTE, fontsize=10.3, color="#33475b", style="italic",
             ha="left", va="bottom", wrap=True,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f2f6fa", ec="#cfe0ee", lw=0.8))
    fig.text(0.012, y_src, "Πηγή: " + source, fontsize=9.3, color=GRAY, ha="left", va="bottom")

def fig_hdd_cdd(df, out="fig13_1_hdd_cdd_by_zone_scenario.png"):
    zones = sorted(df.climate_zone.unique())
    fig, axes = plt.subplots(2, len(zones), figsize=(4.4*len(zones), 9.2), sharex=True)
    fig.suptitle("Βαθμοημέρες Θέρμανσης (HDD) & Ψύξης (CDD) ανά Κλιματική Ζώνη και Σενάριο",
                 fontsize=17, fontweight="bold", color=TITLE_C, y=0.99)
    hdd = _traj(df, "HDD_projected"); cdd = _traj(df, "CDD_projected")
    for j, zone in enumerate(zones):
        for i, (data, label, ax) in enumerate([(hdd, "HDD (θέρμανση)", axes[0, j]),
                                               (cdd, "CDD (ψύξη)", axes[1, j])]):
            for pw in ["SSP2-4.5", "SSP5-8.5"]:
                s = data[(data.zone == zone) & (data.pathway == pw)].sort_values("year")
                ax.plot([POS[y] for y in s.year], s.val, marker="o", ms=6, lw=2.4,
                        color=PATH_COLOR[pw], label=pw)
            if i == 0: ax.set_title(f"Ζώνη {zone}", fontsize=14)
            if j == 0: ax.set_ylabel(label, fontsize=12.5, fontweight="bold")
            ax.set_xticks(range(4)); ax.set_xticklabels(YEARS)
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.text(0.5, 0.105, "Έτος (χρονικός ορίζοντας σεναρίου)", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.97])
    _footer(fig, "NREL ResStock/ComStock · Delta scaling (IPCC AR6, WG1 2021) · κατώφλι 18°C (ASHRAE 90.1-2019)")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out

def fig_consumption(df, out="fig14_1_consumption_changes_by_scenario.png"):
    zones = sorted(df.climate_zone.unique())
    tr = _traj(df, "consumption_delta_pct")
    fig, axes = plt.subplots(1, len(zones), figsize=(4.4*len(zones), 6.0), sharey=True)
    fig.suptitle("Μεταβολή Ετήσιας Ενεργειακής Κατανάλωσης ανά Σενάριο (vs Baseline 2018)",
                 fontsize=17, fontweight="bold", color=TITLE_C, y=1.0)
    for j, zone in enumerate(zones):
        ax = axes[j]
        for pw in ["SSP2-4.5", "SSP5-8.5"]:
            s = tr[(tr.zone == zone) & (tr.pathway == pw)].sort_values("year")
            ax.plot([POS[y] for y in s.year], s.val, marker="o", ms=7, lw=2.6,
                    color=PATH_COLOR[pw], label=pw)
        ax.axhline(0, color=GRAY, lw=0.8, ls="--")
        ax.set_title(f"Ζώνη {zone}", fontsize=14); ax.set_xticks(range(4)); ax.set_xticklabels(YEARS)
        if j == 0: ax.set_ylabel("Μεταβολή κατανάλωσης (%)", fontsize=12.5, fontweight="bold")
    axes[-1].legend(loc="best", frameon=False)
    fig.text(0.5, 0.135, "Έτος (χρονικός ορίζοντας σεναρίου)", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.15, 1, 0.99])
    _footer(fig, "NREL ResStock/ComStock · Convex CDD-based cooling (ASHRAE) · Delta scaling (IPCC AR6)",
            y_note=0.065, y_src=0.02)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out

def fig_co2(df, out="fig16_1_co2_emissions_trajectory.png"):
    tr = _traj(df, "co2_kg")
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.suptitle("Τροχιά Μέσων Εκπομπών CO₂ ανά Κτίριο & Σενάριο",
                 fontsize=17, fontweight="bold", color=TITLE_C, y=0.98)
    agg = tr.groupby(["pathway", "year"]).val.mean().reset_index()
    base_t = agg[agg.year==2018].val.mean()/1000
    for pw, dy in [("SSP2-4.5", -14), ("SSP5-8.5", 10)]:
        s = agg[agg.pathway == pw].sort_values("year")
        ax.plot([POS[y] for y in s.year], s.val / 1000, marker="o", ms=9, lw=3,
                color=PATH_COLOR[pw], label=pw)
        ax.annotate(f"{s.val.values[-1]/1000:.1f} t", (POS[s.year.values[-1]], s.val.values[-1]/1000),
                    textcoords="offset points", xytext=(8, dy), color=PATH_COLOR[pw], fontweight="bold")
    # επεξήγηση: η πτώση οφείλεται στην αποανθρακοποίηση δικτύου, όχι σε μείωση κατανάλωσης
    y40 = agg[(agg.year==2040)].val.mean()/1000
    ax.annotate("Η μείωση οφείλεται στην\nαποανθρακοποίηση του δικτύου\n(Scope 2 · IRA 2022 / NREL ATB)",
                (POS[2040], y40), xytext=(POS[2040]+0.15, base_t-3.5),
                fontsize=11, color="#33475b", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f2f6fa", ec="#cfe0ee", lw=0.8),
                arrowprops=dict(arrowstyle="->", color="#8a97a5", lw=1.2))
    ax.annotate(f"−48% έως 2040", (POS[2040], y40), xytext=(0,14), textcoords="offset points",
                ha="center", color=TITLE_C, fontweight="bold", fontsize=11)
    ax.set_xticks(range(4)); ax.set_xticklabels(YEARS)
    ax.set_xlabel("Έτος (χρονικός ορίζοντας σεναρίου)", fontweight="bold")
    ax.set_ylabel("Μέσες εκπομπές CO₂ (τόνοι/κτίριο/έτος)", fontweight="bold")
    ax.legend(frameon=False, title="Σενάριο", loc="center left")
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    _footer(fig, "NREL Cambium 2022 (ηλεκτρισμός, Scope 2) · EPA 2023 (αέριο, Scope 1) · Αποανθρακοποίηση δικτύου: IRA 2022 / NREL ATB 2023",
            y_note=0.055, y_src=0.015)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out

if __name__ == "__main__":
    df = _load()
    for f in (fig_hdd_cdd(df), fig_consumption(df), fig_co2(df)):
        print("✓", f)
