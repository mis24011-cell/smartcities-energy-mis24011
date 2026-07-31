# -*- coding: utf-8 -*-
"""
phase4_social.py — Vulnerability (fig15) & Energy Burden (fig18)
Μέθοδος από notebooks 15 & 18, προσαρμοσμένη στο νέο 2×3 σχέδιο
(worst case = SSP5-8.5 2100). EIA/Tableau ύφος + επεξήγηση άξονα.
"""
import pandas as pd, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
import seaborn as sns
from phase4_figures import _load, BLUE, RED, GRAY, TITLE_C, _footer  # κοινό ύφος

WORST = "SSP5-8.5 (2100)"

# ---------- PVI (Physical Vulnerability Index) — notebook 15 ----------
def add_pvi(df):
    df = df.copy()
    df["vintage_score"] = np.clip(100 - (df["vintage"]-1960)/(2015-1960)*80, 10, 90)
    df["insulation_score"] = df["insulation_r_value"].map({11:80,19:50,30:25,38:10}).fillna(40)
    df["window_score"] = df["window_type"].map({"single":80,"double":40,"triple":10}).fillna(40)
    df["hvac_score"] = df["hvac_type"].map({"heat_pump":15,"central_ac":20,"gas_furnace":70}).fillna(50)
    df["PVI"] = (df.vintage_score*0.35 + df.insulation_score*0.30 +
                 df.window_score*0.20 + df.hvac_score*0.15)
    return df

def fig_vulnerability(df, out="fig15_1_vulnerability_heatmap.png"):
    worst = add_pvi(df[df.scenario==WORST].copy())
    piv_cons = worst.pivot_table(values="consumption_delta_pct", index="climate_zone",
                                 columns="building_type", aggfunc="mean")
    piv_pvi = worst.pivot_table(values="PVI", index="climate_zone",
                                columns="building_type", aggfunc="mean")
    piv_cons.columns=[c.capitalize() for c in piv_cons.columns]
    piv_pvi.columns=[c.capitalize() for c in piv_pvi.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.suptitle("Ανάλυση Τρωτότητας ανά Κλιματική Ζώνη & Τύπο Κτιρίου — υπό SSP5-8.5 (2100)",
                 fontsize=16, fontweight="bold", color=TITLE_C, y=1.0)
    sns.heatmap(piv_cons, ax=axes[0], cmap="RdYlGn_r", annot=True, fmt=".1f",
                cbar_kws={"label":"Μεταβολή κατανάλωσης (%)"}, linewidths=0.6,
                annot_kws={"size":13,"weight":"bold"})
    axes[0].set_title("Μεταβολή Κατανάλωσης (%)", fontsize=13, fontweight="bold")
    sns.heatmap(piv_pvi, ax=axes[1], cmap="RdYlGn_r", annot=True, fmt=".0f",
                cbar_kws={"label":"Physical Vulnerability Index (0-100)"}, linewidths=0.6,
                annot_kws={"size":13,"weight":"bold"})
    axes[1].set_title("Δείκτης Φυσικής Τρωτότητας (PVI)", fontsize=13, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Τύπος Κτιρίου", fontsize=12.5); ax.set_ylabel("Κλιματική Ζώνη", fontsize=12.5)
        ax.tick_params(labelsize=12)
    fig.tight_layout(rect=[0,0.11,1,0.97])
    fig.text(0.012, 0.055,
             "Πώς διαβάζεται: κάθε κελί = μέση τιμή για τον συνδυασμό ζώνης×τύπου. Κόκκινο = υψηλότερη "
             "τρωτότητα/επίπτωση. PVI: παλαιότητα (35%), μόνωση (30%), υαλοπίνακες (20%), HVAC (15%).",
             fontsize=10.3, color="#33475b", style="italic", ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f2f6fa", ec="#cfe0ee", lw=0.8))
    fig.text(0.012, 0.015, "Πηγή: NREL ResStock/ComStock · μεθοδολογία PVI: Drehobl et al. (2020), ACEEE",
             fontsize=9.3, color=GRAY, ha="left")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out

# ---------- Energy Burden — notebook 18 ----------
ELEC_RATE=0.16; GAS_RATE=12.50/293.07
def add_burden(df):
    df=df.copy()
    elec=(df.get("annual_kwh_projected",df["annual_kwh"]) - df["natural_gas_kwh"].clip(lower=0)).clip(lower=0)
    cost=elec*ELEC_RATE + df["natural_gas_kwh"]*GAS_RATE
    df["energy_burden_pct"]=(cost/df["median_income"]*100).clip(0,50)
    df["high_burden"]=df["energy_burden_pct"]>6
    return df

def fig_energy_burden(df, out="fig18_1_energy_burden_analysis.png"):
    res=df[(df.building_type=="residential") & (df.income_quartile.notna())].copy()
    res=add_burden(res)
    scen=["Baseline (2018)","SSP2-4.5 (2100)","SSP5-8.5 (2100)"]
    qcol={1:"#D1495B",2:"#EDAE49",3:"#66A182",4:"#2166AC"}
    fig,axes=plt.subplots(1,2,figsize=(16,6.2))
    fig.suptitle("Ενεργειακή Φτώχεια (Energy Burden) — Κοινωνική Διάσταση ESG",
                 fontsize=16,fontweight="bold",color=TITLE_C,y=1.0)
    # Plot 1: boxplot ανά income quartile (baseline)
    ax=axes[0]; base=res[res.scenario=="Baseline (2018)"]
    data=[base[base.income_quartile==q].energy_burden_pct.values for q in [1,2,3,4]]
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color="black",lw=1.6),
                  flierprops=dict(marker="o",ms=3,alpha=0.25))
    for patch,q in zip(bp["boxes"],[1,2,3,4]): patch.set_facecolor(qcol[q]); patch.set_alpha(0.75)
    ax.axhline(6,color="#E67E22",ls="--",lw=1.8,label="Όριο υψηλής (6%)")
    ax.axhline(10,color="#C0392B",ls="--",lw=1.8,label="Όριο ακραίας (10%)")
    ax.set_title("Κατανομή ανά Εισοδηματική Κατηγορία — Baseline 2018",fontsize=13,fontweight="bold")
    ax.set_xlabel("Εισοδηματική Κατηγορία (1=χαμηλότερη → 4=υψηλότερη)",fontsize=12.5)
    ax.set_ylabel("Energy Burden (% εισοδήματος)",fontsize=12.5)
    ax.set_xticklabels(["Q1","Q2","Q3","Q4"]); ax.legend(fontsize=11,frameon=False)
    # Plot 2: % νοικοκυριών με high burden ανά σενάριο & quartile
    ax=axes[1]
    hb=res.groupby(["scenario","income_quartile"]).high_burden.mean().mul(100).reset_index()
    xlab=["Baseline\n2018","SSP2-4.5\n2100","SSP5-8.5\n2100"]
    for q in [1,2,3,4]:
        vals=[hb[(hb.scenario==s)&(hb.income_quartile==q)].high_burden.values for s in scen]
        vals=[v[0] if len(v) else 0 for v in vals]
        ax.plot(range(3),vals,marker="o",ms=9,lw=2.6,color=qcol[q],
                label=f"Q{q} ({'Χαμηλό' if q==1 else 'Μεσ.-Χαμ.' if q==2 else 'Μεσ.-Υψ.' if q==3 else 'Υψηλό'})")
    ax.set_xticks(range(3)); ax.set_xticklabels(xlab,fontsize=11)
    ax.set_title("% Νοικοκυριών με Υψηλό Burden (>6%)",fontsize=13,fontweight="bold")
    ax.set_ylabel("% Νοικοκυριών",fontsize=12.5); ax.legend(fontsize=10.5,frameon=False,title="Εισόδημα")
    fig.tight_layout(rect=[0,0.10,1,0.97])
    fig.text(0.012,0.05,
             "Energy Burden = ετήσιο κόστος ενέργειας ÷ εισόδημα × 100. Άξονας x δεξιά: σενάρια (2018 baseline → "
             "SSP2-4.5/SSP5-8.5 στο 2100). Χαμηλότερα εισοδήματα (Q1) υφίστανται δυσανάλογα υψηλότερο βάρος.",
             fontsize=10.3,color="#33475b",style="italic",ha="left",va="bottom",
             bbox=dict(boxstyle="round,pad=0.4",fc="#f2f6fa",ec="#cfe0ee",lw=0.8))
    fig.text(0.012,0.012,"Πηγή: NREL ResStock/ComStock · τιμές EIA 2023 · μεθοδολογία ACEEE (Drehobl et al., 2020)",
             fontsize=9.3,color=GRAY,ha="left")
    fig.savefig(out,bbox_inches="tight"); plt.close(fig); return out

if __name__=="__main__":
    df=_load()
    print("✓",fig_vulnerability(df)); print("✓",fig_energy_burden(df))
