# -*- coding: utf-8 -*-
"""Σχ. 3.16 (νέο): Μηνιαίο προφίλ κατανάλωσης ανά κλιματική ζώνη (5 ζώνες)."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25})
m=pd.read_parquet("merged_master.parquet"); m.index=pd.to_datetime(m.index); m["month"]=m.index.month
SITE="out.site_energy.total.energy_consumption.kwh"
NAVY="#1f4e79"; MON=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]
ZONES=["2A","3A","4A","5A","5B"]
ZLAB={"2A":"2A Houston","3A":"3A Atlanta","4A":"4A Baltimore","5A":"5A Chicago","5B":"5B Denver"}
ZCOL={"2A":"#c0435a","3A":"#e08a4f","4A":"#d1a53a","5A":"#4a90c0","5B":"#2e5a86"}
fig,ax=plt.subplots(1,2,figsize=(15,5.6))
fig.suptitle("Χειμερινή αιχμή στις ψυχρές ζώνες (θέρμανση) έναντι θερινής αιχμής στις θερμές (ψύξη)",
             fontsize=15,fontweight="bold",color=NAVY,x=0.012,ha="left",y=1.01)
for j,(dt,title) in enumerate([("residential","Οικιστικά κτίρια (ResStock)"),("commercial","Εμπορικά κτίρια (ComStock)")]):
    a=ax[j]
    for z in ZONES:
        d=m[(m.zone==z)&(m.dataset_type==dt)]
        prof=d.groupby("month")[SITE].mean().reindex(range(1,13))
        a.plot(range(1,13),prof.values,lw=2.6,color=ZCOL[z],label=ZLAB[z],marker="o",ms=4)
    a.set_title(title,fontsize=13,fontweight="bold"); a.set_ylabel("Μέση κατανάλωση (kWh/h)")
    a.set_xticks(range(1,13)); a.set_xticklabels(MON,fontsize=9)
    if j==0: a.legend(fontsize=10,frameon=False,loc="upper center",ncol=2)
fig.tight_layout(rect=[0,0.04,1,0.96])
fig.text(0.012,0.005,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018) · merged_master (5.000 κτίρια) · ιδία επεξεργασία",
         fontsize=9,color="#8a97a5")
fig.savefig("fig_eda_monthly_zones.png",bbox_inches="tight",facecolor="white"); print("✓ fig_eda_monthly_zones.png")
