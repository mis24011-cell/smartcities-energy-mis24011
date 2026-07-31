# -*- coding: utf-8 -*-
"""
phase3_eda_zones.py — EDA Κεφ. 3 / βήμα 9: ΓΕΩΓΡΑΦΙΚΗ ανάλυση 4 ζωνών ASHRAE
σε ΠΡΑΓΜΑΤΙΚΑ δεδομένα (merged_all_zones.parquet).
Panels:
  (1) Ετήσια κατανάλωση site energy ανά ζώνη (res vs com)
  (2) Εποχιακή υπογραφή: μηνιαίο site energy ανά ζώνη (οικιστικά)
  (3) Θερμικός χάρτης Ώρα×Μήνας ηλεκτρικού φορτίου (μέση 4 ζωνών, οικιστικά)
  (4) Ένταση εκπομπών CO2 (kgCO2/kWh) ανά ζώνη
EIA/Tableau ύφος + επεξηγηματικό footer για τον άξονα x.
"""
import pandas as pd, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from phase4_figures import BLUE, RED, GRAY, TITLE_C

mpl.rcParams.update({
    "figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans","font.size":13,
    "axes.titlesize":15,"axes.titleweight":"bold","axes.labelsize":13,
    "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,
})
RES_C, COM_C = "#2166AC", "#E67E22"
SITE="out.site_energy.total.energy_consumption.kwh"
ELEC="out.electricity.total.energy_consumption.kwh"
# Χρώμα ανά ζώνη (ψυχρότερο κλίμα -> ψυχρότερο χρώμα)
ZONE_C={"3A":"#D1495B","4A":"#EDAE49","5A":"#2166AC","5B":"#66A182"}
ZONE_LBL={"3A":"3A · Atlanta","4A":"4A · Baltimore","5A":"5A · Chicago","5B":"5B · Denver"}
ZONES=["3A","4A","5A","5B"]
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]

def _load(p="merged_all_zones.parquet"):
    m=pd.read_parquet(p); m.index=pd.to_datetime(m.index)
    m["hour"]=m.index.hour; m["month"]=m.index.month
    return m

def fig_zones(m, out="fig_eda_zones_ashrae.png"):
    res=m[m.dataset_type=="residential"]; com=m[m.dataset_type=="commercial"]
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    fig.suptitle("Γεωγραφική Διερεύνηση — 4 Κλιματικές Ζώνες ASHRAE (πραγματικά δεδομένα NREL, 2018)",
                 fontsize=17,fontweight="bold",color=TITLE_C,y=0.985)

    # (1) Ετήσια κατανάλωση ανά ζώνη (άθροισμα ωριαίων = ετήσιο ανά «μέσο κτίριο»)
    ax=axes[0,0]
    r_ann=[res[res.zone==z][SITE].sum() for z in ZONES]
    c_ann=[com[com.zone==z][SITE].sum() for z in ZONES]
    x=np.arange(len(ZONES)); w=0.38
    ax.bar(x-w/2,r_ann,w,color=RES_C,label="Οικιστικά (ResStock)")
    ax.bar(x+w/2,c_ann,w,color=COM_C,label="Εμπορικά (ComStock)")
    for xi,v in zip(x-w/2,r_ann): ax.text(xi,v,f"{v/1000:.1f}k",ha="center",va="bottom",fontsize=9.5,color=RES_C,fontweight="bold")
    for xi,v in zip(x+w/2,c_ann): ax.text(xi,v,f"{v/1000:.1f}k",ha="center",va="bottom",fontsize=9.5,color=COM_C,fontweight="bold")
    ax.set_title("Ετήσια Κατανάλωση ανά Ζώνη (kWh/μέσο κτίριο)",fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([ZONE_LBL[z] for z in ZONES],fontsize=10.5,rotation=12)
    ax.set_ylabel("kWh/έτος"); ax.legend(frameon=False)

    # (2) Εποχιακή υπογραφή: μηνιαίο site energy ανά ζώνη (οικιστικά)
    ax=axes[0,1]
    for z in ZONES:
        mo=res[res.zone==z].groupby("month")[SITE].mean()
        ax.plot(range(1,13),mo.values,lw=2.8,marker="o",ms=6,color=ZONE_C[z],label=ZONE_LBL[z])
    ax.set_title("Εποχιακή Υπογραφή — Μέσο Ωριαίο Site Energy (οικιστικά)",fontsize=14)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS,fontsize=10); ax.set_xlabel("Μήνας")
    ax.set_ylabel("kWh/ώρα (μέσος όρος)"); ax.legend(frameon=False,fontsize=10.5)

    # (3) Heatmap Ώρα×Μήνας ηλεκτρικού φορτίου (μέση 4 ζωνών, οικιστικά, normalized)
    ax=axes[1,0]
    piv=res.pivot_table(values=ELEC,index="hour",columns="month",aggfunc="mean")
    pivn=(piv-piv.values.min())/(piv.values.max()-piv.values.min())
    im=ax.imshow(pivn.values,aspect="auto",origin="lower",cmap="YlOrRd",
                 extent=[0.5,12.5,-0.5,23.5])
    ax.set_title("Θερμικός Χάρτης Ηλεκτρικού Φορτίου — Ώρα × Μήνας (οικιστικά, norm.)",fontsize=13.5)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS,fontsize=9.5)
    ax.set_yticks(range(0,24,3)); ax.set_xlabel("Μήνας",labelpad=2); ax.set_ylabel("Ώρα ημέρας")
    ax.grid(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02); cb.set_label("Κανονικοπ. φορτίο [0-1]",fontsize=10.5)

    # (4) Ένταση εκπομπών CO2 ανά ζώνη (kgCO2/kWh site)
    ax=axes[1,1]
    ci=[m[m.zone==z]["co2_total_kg"].sum()/(m[m.zone==z][SITE].sum()+1e-9) for z in ZONES]
    xz=np.arange(len(ZONES))
    bars=ax.bar(xz,ci,color=[ZONE_C[z] for z in ZONES],alpha=0.9)
    for b,v in zip(bars,ci): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.3f}",ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax.set_title("Ένταση Εκπομπών CO₂ ανά Ζώνη",fontsize=14)
    ax.set_ylabel("kgCO₂ / kWh (site)")
    ax.set_xticks(xz); ax.set_xticklabels([ZONE_LBL[z] for z in ZONES],fontsize=10.5,rotation=12)

    fig.tight_layout(rect=[0,0.075,1,0.955])
    fig.text(0.012,0.052,
        "Άξονας x: 4 ζώνες ASHRAE διατεταγμένες από θερμότερη (3A Atlanta) προς ψυχρότερη (5A Chicago). "
        "Η 5A εμφανίζει τη μέγιστη κατανάλωση (φορτίο θέρμανσης). Ο θερμικός χάρτης αποκαλύπτει δύο καθεστώτα: "
        "υψηλό χειμερινό φορτίο (Ιαν–Φεβ, όλο το 24ωρο) και θερινή μεσημβρινή αιχμή ψύξης (Ιουν–Σεπ, ~12–18h).",
        fontsize=10.3,color="#33475b",style="italic",ha="left",va="bottom",wrap=True,
        bbox=dict(boxstyle="round,pad=0.4",fc="#f2f6fa",ec="#cfe0ee",lw=0.8))
    fig.text(0.012,0.012,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018) · 400 κτίρια/ζώνη · ωριαία aggregated profiles",
             fontsize=9.3,color=GRAY,ha="left")
    fig.savefig(out,bbox_inches="tight"); plt.close(fig); return out

if __name__=="__main__":
    m=_load(); print("✓",fig_zones(m))
