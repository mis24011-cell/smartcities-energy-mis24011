# -*- coding: utf-8 -*-
"""Αναπαραγωγή παλιού Σχ 5.2 (CO₂ ανά σενάριο) στα ΝΕΑ δεδομένα (5 ζώνες, SSP2-4.5 & SSP5-8.5, 2040/2050/2100).
Αριστερά: συνολικές εκπομπές (tonnes CO₂e) — μείωση λόγω grid decarbonization παρά την αύξηση κατανάλωσης.
Δεξιά: αποσύνδεση (decoupling) — δείκτης κατανάλωσης ↑ vs δείκτης CO₂ ↓.
Ίδιο μοντέλο CO₂ με phase4_figures (_load): Scope2 ηλεκτρισμός×EF×grid_decarb + Scope1 αέριο×EF."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from phase4_figures import _load, GRID_DECARB
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True})
NAVY="#1f3a56"; BLUE="#2E75B6"; RED="#C00000"; GREEN="#2E9E4F"
df=_load()
SCN=[("Baseline (2018)","Baseline\n2018","#5A6B7A"),
     ("SSP2-4.5 (2040)","SSP2-4.5\n2040","#bcd4ea"),("SSP2-4.5 (2050)","SSP2-4.5\n2050","#6fa8d6"),("SSP2-4.5 (2100)","SSP2-4.5\n2100",BLUE),
     ("SSP5-8.5 (2040)","SSP5-8.5\n2040","#f0b49a"),("SSP5-8.5 (2050)","SSP5-8.5\n2050","#e07a4f"),("SSP5-8.5 (2100)","SSP5-8.5\n2100",RED)]
# σταθερό πλήθος κτιρίων ανά σενάριο;
n_per=df.groupby("scenario").size()
print("buildings/scenario:",n_per.to_dict())
# συνολικές εκπομπές (tonnes) & κατανάλωση (GWh) ανά σενάριο
rows=[]
for scn,_,_ in SCN:
    d=df[df.scenario==scn]
    co2_t=d["co2_kg"].sum()/1000.0
    kwh=d["annual_kwh_projected"].sum() if scn!="Baseline (2018)" else d["annual_kwh"].sum()
    rows.append((scn,co2_t,kwh/1e6))  # tonnes, GWh
R=pd.DataFrame(rows,columns=["scn","co2_t","gwh"])
base_co2=R.iloc[0].co2_t; base_kwh=R.iloc[0].gwh

fig,ax=plt.subplots(1,2,figsize=(16,6))
fig.suptitle("Εκπομπές CO₂ ανά κλιματικό σενάριο — η αποανθρακοποίηση του δικτύου υπερισχύει της αύξησης ζήτησης",
             fontsize=13,fontweight="bold",color=NAVY,y=1.0)
# ── Αριστερά: συνολικές εκπομπές (tonnes) ──
labs=[s[1] for s in SCN]; cols=[s[2] for s in SCN]
b=ax[0].bar(range(len(SCN)),R["co2_t"],color=cols,edgecolor="white")
for i,v in enumerate(R["co2_t"]):
    pct=(v-base_co2)/base_co2*100
    ax[0].text(i,v,f"{v:,.0f}t\n({pct:+.0f}%)" if i>0 else f"{v:,.0f}t",ha="center",va="bottom",fontsize=8.5,fontweight="bold")
ax[0].set_xticks(range(len(SCN)));ax[0].set_xticklabels(labs,fontsize=8.5)
ax[0].set_ylabel("Συνολικές εκπομπές (tonnes CO₂e)")
ax[0].set_title("Συνολικές ετήσιες εκπομπές CO₂ ανά σενάριο",fontweight="bold",color=NAVY,fontsize=11)
ax[0].set_ylim(0,R["co2_t"].max()*1.18)
# ── Δεξιά: decoupling (δείκτης κατανάλωσης ↑ vs CO₂ ↓, base=100) ──
years=[2018,2040,2050,2100]
for pw,col,mk in [("SSP5-8.5",RED,"o"),("SSP2-4.5",BLUE,"s")]:
    co2_idx=[];kwh_idx=[]
    for y in years:
        scn="Baseline (2018)" if y==2018 else f"{pw} ({y})"
        d=df[df.scenario==scn]
        co2_idx.append(d["co2_kg"].sum()/1000.0/base_co2*100)
        kwh_idx.append((d["annual_kwh_projected"].sum() if y!=2018 else d["annual_kwh"].sum())/1e6/base_kwh*100)
    ax[1].plot(years,kwh_idx,color=col,lw=2,ls="--",marker=mk,ms=6,alpha=.6,label=f"{pw} — Κατανάλωση")
    ax[1].plot(years,co2_idx,color=col,lw=2.5,marker=mk,ms=6,label=f"{pw} — CO₂")
ax[1].axhline(100,color="#888",lw=1,ls=":")
ax[1].text(2019,101,"Baseline 2018 = 100",fontsize=8,color="#666")
# grid decarb trajectory (δεύτερος άξονας)
ax2=ax[1].twinx();ax2.grid(False)
gd=[GRID_DECARB[y]*100 for y in years]
ax2.plot(years,gd,color=GREEN,lw=2,ls="-.",marker="^",ms=6,label="Ένταση δικτύου (%)")
ax2.set_ylabel("Ένταση εκπομπών δικτύου (% του 2018)",color=GREEN);ax2.tick_params(axis="y",labelcolor=GREEN);ax2.set_ylim(0,110)
ax[1].set_title("Αποσύνδεση κατανάλωσης–εκπομπών (δείκτης, 2018=100)",fontweight="bold",color=NAVY,fontsize=11)
ax[1].set_xlabel("Χρονικός ορίζοντας");ax[1].set_ylabel("Δείκτης (2018=100)");ax[1].set_xticks(years)
l1,lb1=ax[1].get_legend_handles_labels();l2,lb2=ax2.get_legend_handles_labels()
ax[1].legend(l1+l2,lb1+lb2,fontsize=8,loc="center left")
fig.tight_layout();fig.savefig("v6_5_2_co2_scenario.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 5.2 | base CO2:",round(base_co2),"t")
for _,r in R.iterrows(): print(f"  {r.scn:18s}: {r.co2_t:9.0f} t ({(r.co2_t-base_co2)/base_co2*100:+.0f}%)")
