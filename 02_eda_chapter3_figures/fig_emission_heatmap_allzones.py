# -*- coding: utf-8 -*-
"""Σχ 4.3 — Θερμικός χάρτης ωριαίων εκπομπών CO2 (Ώρα × Μήνας) για ΚΑΙ τις 5 ζώνες.
Απαντά στο αίτημα του χρήστη: στο παλιό κείμενο ο emission heatmap ήταν μόνο για μία
ζώνη· εδώ παράγεται αντίστοιχο small-multiples για όλες τις 5 κλιματικές ζώνες.
District επίπεδο (60% Οικιστικά + 40% Εμπορικά, ULI 2022), baseline 2018 (AMY2018)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans","axes.axisbelow":True})
NAVY="#1f3a56"
CO2="co2_total_kg"
ZONES=["2A","3A","4A","5A","5B"]
CITY={"2A":"Houston","3A":"Atlanta","4A":"Baltimore","5A":"Chicago","5B":"Denver"}
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]
m=pd.read_parquet("merged_master.parquet").sort_index()

def district_co2(zone):
    """CO2 district = 0.6*res + 0.4*com (kg/μέσο κτίριο ανά ώρα)."""
    r=m[(m.zone==zone)&(m.dataset_type=="residential")].sort_index()
    c=m[(m.zone==zone)&(m.dataset_type=="commercial")].sort_index()
    d=pd.DataFrame(index=r.index)
    d["co2"]=0.6*r[CO2].values+0.4*c[CO2].values
    d["hour"]=d.index.hour; d["month"]=d.index.month
    return d

# ── Ενιαία κλίμακα χρωμάτων σε όλες τις ζώνες για δίκαιη οπτική σύγκριση ──
grids={}
vmax=0
for z in ZONES:
    d=district_co2(z)
    g=d.pivot_table(index="hour",columns="month",values="co2",aggfunc="mean")
    g=g.reindex(index=range(24),columns=range(1,13))
    grids[z]=g; vmax=max(vmax,np.nanmax(g.values))

fig,axes=plt.subplots(1,5,figsize=(19,5.2),sharey=True)
im=None
for k,z in enumerate(ZONES):
    ax=axes[k]; g=grids[z]
    im=ax.imshow(g.values,aspect="auto",origin="lower",cmap="inferno",vmin=0,vmax=vmax,
                 extent=[0.5,12.5,-0.5,23.5])
    ax.set_title(f"{z} — {CITY[z]}",fontweight="bold",color=NAVY,fontsize=11)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS,rotation=90,fontsize=7)
    ax.set_xlabel("Μήνας",fontsize=9)
    if k==0:
        ax.set_yticks(range(0,24,3)); ax.set_yticklabels(range(0,24,3),fontsize=8)
        ax.set_ylabel("Ώρα ημέρας",fontsize=9)
fig.suptitle("Θερμικός χάρτης ωριαίων εκπομπών CO₂ αστικής περιοχής (60/40) ανά κλιματική ζώνη — baseline 2018",
             fontweight="bold",color=NAVY,fontsize=13,y=1.02)
cbar=fig.colorbar(im,ax=axes,fraction=0.018,pad=0.012)
cbar.set_label("Μέσες εκπομπές CO₂ (kg / μέσο κτίριο · ώρα)",fontsize=9)
fig.savefig("fig_emission_heatmap_allzones.png",bbox_inches="tight",facecolor="white")
plt.close()
# Στατιστικά ελέγχου
print("✓ Σχ 4.3 emission heatmap all zones | vmax=%.3f kg"%vmax)
for z in ZONES:
    g=grids[z]; tot=district_co2(z)["co2"].sum()
    pk_h=int(np.nanargmax(np.nanmean(g.values,axis=1)))
    pk_m=int(np.nanargmax(np.nanmean(g.values,axis=0)))+1
    print(f"  {z} {CITY[z]:9s}: ετ.άθρ={tot:9.0f} kg | αιχμή ώρα~{pk_h:02d}:00 | αιχμή μήνας={MONTHS[pk_m-1]}")
