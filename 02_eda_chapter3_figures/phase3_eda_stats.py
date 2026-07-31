# -*- coding: utf-8 -*-
"""
phase3_eda_stats.py — EDA Κεφ. 3 / βήμα 9: ΣΤΑΤΙΣΤΙΚΗ & ΠΟΛΥΜΕΤΑΒΛΗΤΗ δομή
σε ΠΡΑΓΜΑΤΙΚΑ δεδομένα (merged_all_zones.parquet).
Panels:
  (1) Κατανομή ωριαίου ηλεκτρικού φορτίου (log-x) — res vs com (λοξότητα)
  (2) Καθημερινές vs Σαββατοκύριακο — ωριαίο προφίλ (behavioral signature)
  (3) Πίνακας συσχέτισης μεταβλητών (γέφυρα προς feature engineering, Κεφ. 4)
  (4) Μεταβλητότητα φορτίου ανά ώρα (boxplot) — οικιστικά
EIA/Tableau ύφος + επεξηγηματικό footer.
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
GAS="out.natural_gas.total.energy_consumption.kwh"
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]
# Ετικέτες μεταβλητών για τον πίνακα συσχέτισης (σύντομες, EN για συμβατότητα ML)
CORR_VARS={ELEC:"Elec",GAS:"Gas",SITE:"Site","co2_total_kg":"CO2",
           "emission_intensity":"CO2int","hour":"Hour","month":"Month"}

def _load(p="merged_all_zones.parquet"):
    m=pd.read_parquet(p); m.index=pd.to_datetime(m.index)
    m["hour"]=m.index.hour; m["month"]=m.index.month; m["dow"]=m.index.dayofweek
    m["is_weekend"]=m["dow"]>=5
    return m

def fig_stats(m, out="fig_eda_stats_multivariate.png"):
    res=m[m.dataset_type=="residential"]; com=m[m.dataset_type=="commercial"]
    fig,axes=plt.subplots(2,2,figsize=(16,11))
    fig.suptitle("Στατιστική & Πολυμεταβλητή Διερεύνηση — Οικιστικά vs Εμπορικά (πραγματικά δεδομένα)",
                 fontsize=17,fontweight="bold",color=TITLE_C,y=0.985)

    # (1) Κατανομή ωριαίου ηλεκτρικού φορτίου (log-x)
    ax=axes[0,0]
    for sub,lbl,c in [(res,"Οικιστικά",RES_C),(com,"Εμπορικά",COM_C)]:
        v=sub[ELEC].values; v=v[v>0]
        ax.hist(v,bins=np.logspace(np.log10(max(v.min(),1e-2)),np.log10(v.max()),60),
                alpha=0.55,color=c,label=lbl,density=True)
        med=np.median(v); ax.axvline(med,color=c,ls="--",lw=2)
        ax.annotate(f"διάμ. {med:.1f}",(med,0),xytext=(0,6),textcoords="offset points",
                    ha="center",color=c,fontweight="bold",fontsize=10,rotation=90,va="bottom")
    ax.set_xscale("log")
    ax.set_title("Κατανομή Ωριαίου Ηλεκτρικού Φορτίου (log-x)",fontsize=14)
    ax.set_xlabel("Ηλεκτρικό φορτίο [kWh/ώρα] (λογαριθμικός άξονας)"); ax.set_ylabel("Πυκνότητα")
    ax.legend(frameon=False)

    # (2) Καθημερινές vs Σαββατοκύριακο (ωριαίο, κανονικοποιημένο ανά σειρά για σύγκριση σχήματος)
    ax=axes[0,1]
    def _norm(s): return (s-s.min())/(s.max()-s.min())
    for sub,lbl,c in [(res,"Οικιστικά",RES_C),(com,"Εμπορικά",COM_C)]:
        wd=_norm(sub[~sub.is_weekend].groupby("hour")[ELEC].mean())
        we=_norm(sub[sub.is_weekend].groupby("hour")[ELEC].mean())
        ax.plot(wd.index,wd.values,lw=2.8,color=c,label=f"{lbl} — Καθημ.")
        ax.plot(we.index,we.values,lw=2.4,ls="--",color=c,alpha=0.7,label=f"{lbl} — Σ/Κ")
    ax.axvspan(9,17,alpha=0.07,color="gold")
    ax.set_title("Καθημερινές vs Σαββατοκύριακο — Ωριαίο Ηλεκτρικό Προφίλ (norm.)",fontsize=13.5)
    ax.set_xlabel("Ώρα ημέρας"); ax.set_ylabel("Κανονικοπ. ηλεκτρικό φορτίο [0-1]")
    ax.set_xticks(range(0,24,3)); ax.legend(frameon=False,fontsize=9.8,ncol=2)

    # (3) Πίνακας συσχέτισης (Pearson) — οικιστικά
    ax=axes[1,0]
    cols=[c for c in CORR_VARS if c in res.columns]
    C=res[cols].corr(numeric_only=True).values
    labels=[CORR_VARS[c] for c in cols]
    im=ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=11)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",
                    color="white" if abs(C[i,j])>0.55 else "black",fontsize=9.5,fontweight="bold")
    ax.set_title("Πίνακας Συσχέτισης Μεταβλητών (Pearson, οικιστικά)",fontsize=13.5)
    ax.grid(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02); cb.set_label("Συντελεστής r",fontsize=10.5)

    # (4) Μεταβλητότητα φορτίου ανά ώρα (boxplot) — οικιστικά
    ax=axes[1,1]
    hrs=list(range(0,24,2))
    data=[res[res.hour==h][ELEC].values for h in hrs]
    bp=ax.boxplot(data,positions=hrs,widths=1.4,patch_artist=True,showfliers=False,
                  medianprops=dict(color="black",lw=1.5))
    for patch in bp["boxes"]: patch.set_facecolor(RES_C); patch.set_alpha(0.6)
    ax.set_title("Μεταβλητότητα Ηλεκτρικού Φορτίου ανά Ώρα (οικιστικά)",fontsize=13.5)
    ax.set_xlabel("Ώρα ημέρας"); ax.set_ylabel("Ηλεκτρικό φορτίο [kWh/ώρα]")
    ax.set_xticks(range(0,24,3)); ax.set_xticklabels(range(0,24,3))
    ax.set_xlim(-1,24)

    fig.tight_layout(rect=[0,0.075,1,0.955])
    fig.text(0.012,0.052,
        "(1) Άξονας x λογαριθμικός: το φορτίο ακολουθεί δεξιά-λοξή (log-normal) κατανομή — τα εμπορικά έχουν ~50× "
        "υψηλότερη διάμεσο. (2) Κανονικοποιημένα σχήματα: το Σ/Κ εξομαλύνει τη μεσημβρινή αιχμή στα εμπορικά. "
        "(3) Οι εκπομπές CO₂ καθορίζονται από τη συνολική ενέργεια (Site r=0.99)· η ένταση εκπομπών πέφτει με το "
        "μερίδιο αερίου (Gas r=−0.79) — trade-off εξηλεκτρισμού. Η Ώρα συσχετίζεται με το φορτίο (r=0.31) → χρήσιμο feature (Κεφ. 4).",
        fontsize=10.3,color="#33475b",style="italic",ha="left",va="bottom",wrap=True,
        bbox=dict(boxstyle="round,pad=0.4",fc="#f2f6fa",ec="#cfe0ee",lw=0.8))
    fig.text(0.012,0.012,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018) · 4 ζώνες ASHRAE · ωριαία aggregated profiles",
             fontsize=9.3,color=GRAY,ha="left")
    fig.savefig(out,bbox_inches="tight"); plt.close(fig); return out

if __name__=="__main__":
    m=_load(); print("✓",fig_stats(m))
