# -*- coding: utf-8 -*-
"""Σχ. 3.18 (νέο): Κατανομή κατανάλωσης ανά τελική χρήση (end-use) & κλιματική ζώνη."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
m=pd.read_parquet("merged_master.parquet")
NAVY="#1f4e79"; ZONES=["2A","3A","4A","5A","5B"]
# end-use κατηγορίες (elec+gas αθροισμένα ανά χρήση) — τελική ενέργεια
EU=[("heating","Θέρμανση","#c0435a"),("cooling","Ψύξη","#4a90c0"),("hotwater","Ζεστό νερό","#7cb0d6"),
    ("lighting","Φωτισμός","#e0a94b"),("equipment","Συσκευές/Εξοπλ.","#7a9b76"),("ventilation","Αερισμός","#9b8ec4")]
fig,ax=plt.subplots(1,2,figsize=(15,5.8))
fig.suptitle("Στις θερμές ζώνες κυριαρχεί η ψύξη· στις ψυχρές η θέρμανση — end-use ανά κλιματική ζώνη",
             fontsize=15,fontweight="bold",color=NAVY,x=0.012,ha="left",y=1.01)
for j,(dt,title) in enumerate([("residential","Οικιστικά (ResStock)"),("commercial","Εμπορικά (ComStock)")]):
    a=ax[j]; bottom=np.zeros(len(ZONES))
    for key,lbl,col in EU:
        vals=[]
        for z in ZONES:
            d=m[(m.zone==z)&(m.dataset_type==dt)]
            tot_eu=sum(d[f"eu.{k}.elec"].sum()+d[f"eu.{k}.gas"].sum() for k,_,_ in EU)
            v=d[f"eu.{key}.elec"].sum()+d[f"eu.{key}.gas"].sum()
            vals.append(v/tot_eu*100 if tot_eu>0 else 0)
        a.bar(ZONES,vals,bottom=bottom,label=lbl,color=col,edgecolor="white",lw=0.7)
        bottom+=np.array(vals)
    a.set_title(title,fontsize=13,fontweight="bold"); a.set_ylabel("Μερίδιο τελικής ενέργειας (%)")
    a.set_ylim(0,100)
    if j==0: a.legend(fontsize=9.5,frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.28))
fig.tight_layout(rect=[0,0.06,1,0.96])
fig.text(0.012,0.005,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018) · merged_master (end-uses ηλεκτρισμού+αερίου) · ιδία επεξεργασία",
         fontsize=9,color="#8a97a5")
fig.savefig("fig_eda_enduse.png",bbox_inches="tight",facecolor="white"); print("✓ fig_eda_enduse.png")
# ευρήματα
for z in ZONES:
    d=m[(m.zone==z)&(m.dataset_type=="residential")]
    tot=sum(d[f"eu.{k}.elec"].sum()+d[f"eu.{k}.gas"].sum() for k,_,_ in EU)
    heat=(d["eu.heating.elec"].sum()+d["eu.heating.gas"].sum())/tot*100
    cool=d["eu.cooling.elec"].sum()/tot*100
    print(f"  {z} res: θέρμανση {heat:.0f}% · ψύξη {cool:.0f}%")
