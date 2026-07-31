# -*- coding: utf-8 -*-
"""
phase3_eda.py — EDA Κεφ. 3 σε ΠΡΑΓΜΑΤΙΚΑ δεδομένα (merged_all_zones.parquet)
Συγκριτική ResStock vs ComStock. EIA/Tableau ύφος + επεξήγηση άξονα.
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
RES_C, COM_C = "#2166AC", "#E67E22"   # residential / commercial
SITE="out.site_energy.total.energy_consumption.kwh"
ELEC="out.electricity.total.energy_consumption.kwh"
FUELS={"out.electricity.total.energy_consumption.kwh":"Ηλεκτρισμός",
       "out.natural_gas.total.energy_consumption.kwh":"Φυσικό αέριο",
       "out.fuel_oil.total.energy_consumption.kwh":"Πετρέλαιο",
       "out.propane.total.energy_consumption.kwh":"Προπάνιο"}
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]

def _load(p="merged_all_zones.parquet"):
    m=pd.read_parquet(p); m.index=pd.to_datetime(m.index)
    m["hour"]=m.index.hour; m["month"]=m.index.month; m["dow"]=m.index.dayofweek
    return m

def fig_comparative(m, out="fig_eda_comparative_res_com.png"):
    res=m[m.dataset_type=="residential"]; com=m[m.dataset_type=="commercial"]
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    fig.suptitle("Συγκριτική Διερευνητική Ανάλυση — Οικιστικά (ResStock) vs Εμπορικά (ComStock)",
                 fontsize=17,fontweight="bold",color=TITLE_C,y=0.98)
    # (1) Normalized ωριαίο προφίλ
    ax=axes[0,0]
    for sub,lbl,c in [(res,"Οικιστικά",RES_C),(com,"Εμπορικά",COM_C)]:
        h=sub.groupby("hour")[ELEC].mean(); hn=(h-h.min())/(h.max()-h.min())
        ax.plot(hn.index,hn.values,lw=3,marker="o",ms=5,color=c,label=lbl)
        pk=h.idxmax(); ax.annotate(f"αιχμή {pk}:00",(pk,hn[pk]),xytext=(0,10),textcoords="offset points",ha="center",color=c,fontweight="bold",fontsize=11)
    ax.axvspan(9,17,alpha=0.08,color="gold"); ax.text(13,0.03,"ώρες εργασίας",ha="center",fontsize=10,color="#8a6d00")
    ax.set_title("Κανονικοποιημένο Ωριαίο Ηλεκτρικό Προφίλ",fontsize=14); ax.set_xlabel("Ώρα ημέρας"); ax.set_ylabel("Κανονικοπ. ηλεκτρικό φορτίο [0-1]")
    ax.set_xticks(range(0,24,3)); ax.legend(frameon=False)
    # (2) Μηνιαίο (normalized)
    ax=axes[0,1]
    for sub,lbl,c in [(res,"Οικιστικά",RES_C),(com,"Εμπορικά",COM_C)]:
        mo=sub.groupby("month")[SITE].mean(); mn=(mo-mo.min())/(mo.max()-mo.min())
        ax.plot(range(1,13),mn.values,lw=3,marker="o",ms=7,color=c,label=lbl)
    ax.set_title("Εποχιακό Μοτίβο Συνολικής Ενέργειας (norm.)",fontsize=14); ax.set_xlabel("Μήνας"); ax.set_ylabel("Κανονικοποιημένη κατανάλωση")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS,fontsize=10); ax.legend(frameon=False)
    # (3) Load Duration Curve
    ax=axes[1,0]
    for sub,lbl,c in [(res,"Οικιστικά",RES_C),(com,"Εμπορικά",COM_C)]:
        s=np.sort(sub.groupby(sub.index)[ELEC].mean().values)[::-1]; pct=np.linspace(0,100,len(s))
        s=s/s.max()  # normalized για σύγκριση σχήματος
        ax.plot(pct,s,lw=2.6,color=c,label=lbl)
    ax.set_title("Καμπύλη Διάρκειας Ηλεκτρικού Φορτίου (norm.)",fontsize=14)
    ax.set_xlabel("% ωρών υπέρβασης"); ax.set_ylabel("Κανονικοποιημένο φορτίο"); ax.set_xlim(0,100); ax.legend(frameon=False)
    # (4) Μείγμα καυσίμων (residential — πολυκαυσιμικό)
    ax=axes[1,1]
    rvals=[np.nan_to_num(res[c].mean()) for c in FUELS]; cvals=[np.nan_to_num(com[c].mean()) for c in FUELS]
    rvals=np.array(rvals)/sum(rvals)*100; cvals=np.array(cvals)/sum(cvals)*100
    x=np.arange(len(FUELS)); w=0.38
    ax.bar(x-w/2,rvals,w,color=RES_C,alpha=0.9,label="Οικιστικά")
    ax.bar(x+w/2,cvals,w,color=COM_C,alpha=0.9,label="Εμπορικά")
    ax.set_title("Μείγμα Καυσίμων (% ενέργειας)",fontsize=14); ax.set_xticks(x); ax.set_xticklabels(list(FUELS.values()),fontsize=10.5,rotation=15)
    ax.set_ylabel("% συνολικής ενέργειας"); ax.legend(frameon=False)
    fig.tight_layout(rect=[0,0.055,1,0.96])
    fig.text(0.012,0.052,
        "Σύγκριση σχήματος (κανονικοποιημένα) 4 ζωνών ASHRAE. Κύριο εύρημα «diversity factor»: τα οικιστικά κορυφώνουν το ΒΡΑΔΥ (~19:00) "
        "και τα εμπορικά το ΜΕΣΗΜΕΡΙ (~13:00) — δεν συμπίπτουν, ευκαιρία για διαχείριση δικτύου.",
        fontsize=10.3,color="#33475b",style="italic",ha="left",va="bottom",wrap=True,
        bbox=dict(boxstyle="round,pad=0.4",fc="#f2f6fa",ec="#cfe0ee",lw=0.8))
    fig.text(0.012,0.012,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018) · 4 ζώνες ASHRAE · ωριαία aggregated profiles",fontsize=9.3,color=GRAY,ha="left")
    fig.savefig(out,bbox_inches="tight"); plt.close(fig); return out

if __name__=="__main__":
    m=_load(); print("✓",fig_comparative(m))
