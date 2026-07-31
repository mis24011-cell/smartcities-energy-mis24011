# -*- coding: utf-8 -*-
"""Συμπλήρωση για ΣΥΜΜΕΤΡΙΑ οικιστικών–εμπορικών: ίδιοι τύποι σχημάτων.
Παράγει: res_char (A, σαν com 3.11) + com_fuel (B) + com_co2 (C) + com_byzone (D) (σαν res 3.5/3.6/3.7)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True})
NAVY="#1f3a56"; SQFT2M2=0.092903
ZCOL={"2A":"#c0435a","3A":"#e08a4f","4A":"#edc84e","5A":"#7ab0d4","5B":"#4574a8"}
CITY={"2A":"Houston","3A":"Atlanta","4A":"Baltimore","5A":"Chicago","5B":"Denver"}
ZONES=["2A","3A","4A","5A","5B"]; NB_R=600; NB_C=400
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]
S2Z={"TX":"2A","GA":"3A","MD":"4A","IL":"5A","CO":"5B"}

# ═══════════ (A) RESIDENTIAL ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ (σαν com 3.11) ═══════════
r=pd.read_parquet("resstock_metadata.parquet")
r=r[r["in.state"].astype(str).isin(S2Z)].copy(); r["zone"]=r["in.state"].astype(str).map(S2Z)
r["m2"]=r["in.sqft"]*SQFT2M2
RTYPE_GR={"Single-Family Detached":"Μονοκατοικία","Single-Family Attached":"Μεζονέτα/σειρά",
 "Mobile Home":"Κινητή κατοικία","2 Unit":"Διπλοκατοικία","3 or 4 Unit":"3–4 διαμ/τα",
 "5 to 9 Unit":"5–9 διαμ/τα","10 to 19 Unit":"10–19 διαμ/τα","20 to 49 Unit":"20–49 διαμ/τα","50 or more Unit":"50+ διαμ/τα"}
r["bt"]=r["in.geometry_building_type_acs"].map(lambda x:RTYPE_GR.get(str(x),str(x)))
fig=plt.figure(figsize=(18,5.4)); fig.suptitle("ResStock — Χαρακτηριστικά οικιστικών κτιρίων (5 πολιτείες / ζώνες 2A/3A/4A/5A/5B)",
             fontsize=13.5,fontweight="bold",color=NAVY,y=1.03)
gs=fig.add_gridspec(1,3,width_ratios=[1.15,0.85,1.15])
ax1=fig.add_subplot(gs[0]); pct=(r["bt"].value_counts()/len(r)*100).sort_values()
ax1.barh(pct.index,pct.values,color="#2E75B6",edgecolor="white")
for i,v in enumerate(pct.values): ax1.text(v,i,f" {v:.1f}%",va="center",fontsize=8)
ax1.set_title("Ποσοστό ανά τύπο κατοικίας",fontweight="bold",color=NAVY,fontsize=11);ax1.set_xlabel("Μερίδιο (%)");ax1.tick_params(axis="y",labelsize=8)
ax2=fig.add_subplot(gs[1]); zc=[r[r.zone==z].shape[0] for z in ZONES]
b=ax2.bar(ZONES,zc,color=[ZCOL[z] for z in ZONES],edgecolor="white")
for rr,v in zip(b,zc): ax2.text(rr.get_x()+rr.get_width()/2,v,f"{v:,}",ha="center",va="bottom",fontsize=8,fontweight="bold")
ax2.set_title("Διαθέσιμα κτίρια ανά ζώνη (ResStock)",fontweight="bold",color=NAVY,fontsize=11);ax2.set_ylabel("Πλήθος κτιρίων")
ax3=fig.add_subplot(gs[2]); ax3.axis("off")
med=r.groupby("bt")["m2"].median().sort_values(ascending=False)
rows=[[t,f"{med[t]:,.0f}"] for t in med.index]
tbl=ax3.table(cellText=rows,colLabels=["Τύπος κατοικίας","Διάμεσο εμβαδόν (m²)"],cellLoc="left",colLoc="center",loc="center")
tbl.auto_set_font_size(False);tbl.set_fontsize(8.5);tbl.scale(1,1.4)
for (rw,cl),cell in tbl.get_celld().items():
    if rw==0: cell.set_facecolor("#1f3a56");cell.set_text_props(color="white",fontweight="bold")
    elif rw%2==0: cell.set_facecolor("#eef3f8")
ax3.set_title("Διάμεσο εμβαδόν ανά τύπο (m²)",fontweight="bold",color=NAVY,fontsize=11,y=0.9)
fig.savefig("v6_3_4_res_char.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ res char (A) | median m²:",round(r["m2"].median()))

# ═══════════ COMMERCIAL B/C/D (σαν res 3.5/3.6/3.7) ═══════════
m=pd.read_parquet("merged_master.parquet").sort_index()
C=m[m.dataset_type=="commercial"].copy()
def czdf(z): return C[C.zone==z].sort_index()
idx=czdf("3A").index
def cagg(col): return pd.concat([czdf(z)[col].reset_index(drop=True) for z in ZONES],axis=1).mean(axis=1)
FUELS=[("Ηλεκτρισμός","out.electricity.total.energy_consumption.kwh","co2_elec_kg","#2E75B6"),
       ("Φυσικό αέριο","out.natural_gas.total.energy_consumption.kwh","co2_gas_kg","#ED7D31"),
       ("Πετρέλαιο","out.fuel_oil.total.energy_consumption.kwh","co2_oil_kg","#8a6a3a"),
       ("Προπάνιο","out.propane.total.energy_consumption.kwh","co2_propane_kg","#9b59b6")]
labs=[f[0] for f in FUELS]; cols=[f[3] for f in FUELS]

# (B) com fuel mix
fig,ax=plt.subplots(1,3,figsize=(17,5)); fig.suptitle("ComStock — Ενεργειακή κατανάλωση ανά καύσιμο (μέσο εμπορικό κτίριο, 5 ζώνες)",fontsize=13.5,fontweight="bold",color=NAVY,y=1.02)
ann=[cagg(f[1]).sum() for f in FUELS]; meanh=[cagg(f[1]).mean() for f in FUELS]
ax[0].pie(ann,labels=labs,colors=cols,autopct="%1.1f%%",startangle=90,textprops={"fontsize":9},wedgeprops={"edgecolor":"white","lw":1})
ax[0].set_title("Ποσοστό ανά καύσιμο (ετήσια)",fontweight="bold",color=NAVY,fontsize=11)
b=ax[1].bar(labs,meanh,color=cols,edgecolor="white");ax[1].set_title("Μ.Ο. ωριαίας κατανάλωσης ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11);ax[1].set_ylabel("kWh/ώρα")
for rr,v in zip(b,meanh): ax[1].text(rr.get_x()+rr.get_width()/2,v,f"{v:.1f}",ha="center",va="bottom",fontweight="bold",fontsize=9)
ax[1].tick_params(axis="x",labelsize=8)
for f in FUELS:
    s=cagg(f[1]);s.index=idx;mo=s.groupby(s.index.month).mean();ax[2].plot(range(1,13),mo.values,marker="o",ms=4,color=f[3],label=f[0],lw=2)
ax[2].set_title("Μηνιαίο προφίλ ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11);ax[2].set_xlabel("Μήνας");ax[2].set_ylabel("kWh/ώρα");ax[2].set_xticks(range(1,13));ax[2].set_xticklabels(MONTHS,fontsize=7);ax[2].legend(fontsize=8)
fig.tight_layout();fig.savefig("v6_3_12_com_fuel.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ com fuel (B):",[f"{l}={v/sum(ann)*100:.1f}%" for l,v in zip(labs,ann)])

# (C) com CO2 by fuel
fig,ax=plt.subplots(1,3,figsize=(17,5)); fig.suptitle("ComStock — Εκπομπές CO₂ ανά καύσιμο (Cambium 2022 / EPA 2023· σύνολο εμπορικού αποθέματος)",fontsize=13.5,fontweight="bold",color=NAVY,y=1.02)
def tot(col): return sum(czdf(z)[col].sum()*NB_C for z in ZONES)/1000.0
co2=[tot(f[2]) for f in FUELS]
ax[0].pie([max(x,1e-6) for x in co2],labels=labs,colors=cols,autopct="%1.1f%%",startangle=90,textprops={"fontsize":9},wedgeprops={"edgecolor":"white","lw":1})
ax[0].set_title("Ποσοστό CO₂ ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
b=ax[1].bar(labs,co2,color=cols,edgecolor="white");ax[1].set_title("Ετήσιες εκπομπές CO₂ (tonnes)",fontweight="bold",color=NAVY,fontsize=11);ax[1].set_ylabel("tonnes CO₂e")
for rr,v in zip(b,co2): ax[1].text(rr.get_x()+rr.get_width()/2,v,f"{v:.0f}t",ha="center",va="bottom",fontweight="bold",fontsize=9)
ax[1].tick_params(axis="x",labelsize=8)
co2h=pd.concat([czdf(z)["co2_total_kg"].reset_index(drop=True)*NB_C for z in ZONES],axis=1).sum(axis=1)
mo=pd.Series(co2h.values,index=idx).groupby(idx.month).sum()/1000.0
ax[2].bar(range(1,13),mo.values,color="#e0705a",edgecolor="white");ax[2].set_title("Μηνιαίες εκπομπές CO₂",fontweight="bold",color=NAVY,fontsize=11)
ax[2].set_xlabel("Μήνας");ax[2].set_ylabel("tonnes CO₂e");ax[2].set_xticks(range(1,13));ax[2].set_xticklabels(MONTHS,fontsize=7)
fig.tight_layout();fig.savefig("v6_3_13_com_co2.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ com CO2 (C):",[f"{l}={v:.0f}t" for l,v in zip(labs,co2)])

# (D) com cons+CO2 by zone
TE="out.site_energy.total.energy_consumption.kwh"; EL="out.electricity.total.energy_consumption.kwh"
fig,ax=plt.subplots(2,2,figsize=(15,10)); fig.suptitle("Εμπορικά κτίρια — Κατανάλωση & CO₂ ανά κλιματική ζώνη ASHRAE (2A/3A/4A/5A/5B)",fontsize=14,fontweight="bold",color=NAVY,y=1.0)
for z in ZONES:
    d=czdf(z);hp=d[EL].groupby(d.index.hour).mean();ax[0,0].plot(hp.index,hp.values,color=ZCOL[z],lw=2,label=f"{z} {CITY[z]}")
ax[0,0].set_title("Ωριαίο προφίλ ηλεκτρισμού ανά ζώνη",fontweight="bold",color=NAVY);ax[0,0].set_xlabel("Ώρα");ax[0,0].set_ylabel("kWh/ώρα");ax[0,0].legend(fontsize=8)
for z in ZONES:
    d=czdf(z);mo=d[TE].groupby(d.index.month).mean();ax[0,1].plot(range(1,13),mo.values,color=ZCOL[z],lw=2,marker="o",ms=4,label=f"{z} {CITY[z]}")
ax[0,1].set_title("Μηνιαία συνολική κατανάλωση ανά ζώνη",fontweight="bold",color=NAVY);ax[0,1].set_xlabel("Μήνας");ax[0,1].set_ylabel("kWh/ώρα");ax[0,1].set_xticks(range(1,13));ax[0,1].set_xticklabels(MONTHS,fontsize=7);ax[0,1].legend(fontsize=8)
meanz=[czdf(z)[TE].mean() for z in ZONES]
b=ax[1,0].bar([f"{z}\n{CITY[z]}" for z in ZONES],meanz,color=[ZCOL[z] for z in ZONES],edgecolor="white");ax[1,0].set_title("Μ.Ο. ωριαίας κατανάλωσης ανά ζώνη",fontweight="bold",color=NAVY);ax[1,0].set_ylabel("kWh/ώρα")
for rr,v in zip(b,meanz): ax[1,0].text(rr.get_x()+rr.get_width()/2,v,f"{v:.0f}",ha="center",va="bottom",fontweight="bold",fontsize=9)
co2z=[czdf(z)["co2_total_kg"].sum()*NB_C/1000.0 for z in ZONES]
b=ax[1,1].bar([f"{z}\n{CITY[z]}" for z in ZONES],co2z,color=[ZCOL[z] for z in ZONES],edgecolor="white");ax[1,1].set_title("Ετήσιες εκπομπές CO₂ ανά ζώνη",fontweight="bold",color=NAVY);ax[1,1].set_ylabel("tonnes CO₂e")
for rr,v in zip(b,co2z): ax[1,1].text(rr.get_x()+rr.get_width()/2,v,f"{v:.0f}t",ha="center",va="bottom",fontweight="bold",fontsize=9)
fig.tight_layout();fig.savefig("v6_3_14_com_byzone.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ com byzone (D) | mean/zone:",[f"{z}={v:.0f}" for z,v in zip(ZONES,meanz)])
print("DONE symmetry")
