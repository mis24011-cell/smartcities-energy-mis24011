# -*- coding: utf-8 -*-
"""Αναπαραγωγή παλιών σχημάτων Thesis_v6 §3.6.2 (Εμπορικά) στα ΝΕΑ δεδομένα (5 ζώνες).
3.11 χαρακτηριστικά (14 τύποι + ζώνη + εμβαδόν m²) · 3.12 χρονικά μοτίβα · 3.13 καύσιμα+CO2 · 3.14 τύποι/ζώνη.
Metadata: comstock_metadata (5 πολιτείες TX/GA/MD/IL/CO → 2A/3A/4A/5A/5B). Προφίλ: merged_master commercial."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True})
NAVY="#1f3a56"; SQFT2M2=0.092903
ZCOL={"2A":"#c0435a","3A":"#e08a4f","4A":"#edc84e","5A":"#7ab0d4","5B":"#4574a8"}
CITY={"2A":"Houston","3A":"Atlanta","4A":"Baltimore","5A":"Chicago","5B":"Denver"}
ZONES=["2A","3A","4A","5A","5B"]
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]
# ── ελληνικές ονομασίες τύπων ──
TYPE_GR={"SmallOffice":"Γραφεία (μικρά)","MediumOffice":"Γραφεία (μεσαία)","LargeOffice":"Γραφεία (μεγάλα)",
 "RetailStandalone":"Λιανικό (μεμονωμένο)","RetailStripmall":"Εμπορικό πάρκο","Warehouse":"Αποθήκες",
 "FullServiceRestaurant":"Εστιατόριο πλήρους","QuickServiceRestaurant":"Ταχυφαγείο",
 "PrimarySchool":"Δημοτικό","SecondarySchool":"Γυμνάσιο/Λύκειο","Hospital":"Νοσοκομείο",
 "Outpatient":"Εξωτερικά ιατρεία","SmallHotel":"Ξενοδοχείο (μικρό)","LargeHotel":"Ξενοδοχείο (μεγάλο)"}
c=pd.read_parquet("comstock_metadata.parquet")
S2Z={"TX":"2A","GA":"3A","MD":"4A","IL":"5A","CO":"5B"}
c=c[c["in.state"].astype(str).isin(S2Z)].copy(); c["zone"]=c["in.state"].astype(str).map(S2Z)
c["m2"]=c["in.sqft"]*SQFT2M2; c["bt"]=c["in.comstock_building_type"]

# ════════════════ Σχ 3.11: Χαρακτηριστικά εμπορικών ════════════════
fig=plt.figure(figsize=(18,5.4))
fig.suptitle("ComStock — Χαρακτηριστικά εμπορικών κτιρίων (5 πολιτείες / ζώνες 2A/3A/4A/5A/5B)",
             fontsize=13.5,fontweight="bold",color=NAVY,y=1.03)
gs=fig.add_gridspec(1,3,width_ratios=[1.15,0.85,1.15])
# P1: % ανά τύπο (horizontal bar)
ax1=fig.add_subplot(gs[0])
vc=c["bt"].value_counts()
pct=(vc/vc.sum()*100).sort_values()
ax1.barh([TYPE_GR.get(t,t) for t in pct.index],pct.values,color="#2E75B6",edgecolor="white")
for i,v in enumerate(pct.values): ax1.text(v,i,f" {v:.1f}%",va="center",fontsize=8)
ax1.set_title("Ποσοστό ανά τύπο κτιρίου",fontweight="bold",color=NAVY,fontsize=11);ax1.set_xlabel("Μερίδιο (%)")
ax1.tick_params(axis="y",labelsize=8)
# P2: διαθέσιμα κτίρια ανά ζώνη
ax2=fig.add_subplot(gs[1])
zc=[c[c.zone==z].shape[0] for z in ZONES]
b=ax2.bar([f"{z}" for z in ZONES],zc,color=[ZCOL[z] for z in ZONES],edgecolor="white")
for r,v in zip(b,zc): ax2.text(r.get_x()+r.get_width()/2,v,f"{v:,}",ha="center",va="bottom",fontsize=8,fontweight="bold")
ax2.set_title("Διαθέσιμα κτίρια ανά ζώνη (ComStock)",fontweight="bold",color=NAVY,fontsize=11);ax2.set_ylabel("Πλήθος κτιρίων")
# P3: ΠΙΝΑΚΑΣ εμβαδού σε m² ανά τύπο (median)
ax3=fig.add_subplot(gs[2]); ax3.axis("off")
med=c.groupby("bt")["m2"].median().sort_values(ascending=False)
rows=[[TYPE_GR.get(t,t),f"{med[t]:,.0f}"] for t in med.index]
tbl=ax3.table(cellText=rows,colLabels=["Τύπος κτιρίου","Διάμεσο εμβαδόν (m²)"],
              cellLoc="left",colLoc="center",loc="center")
tbl.auto_set_font_size(False);tbl.set_fontsize(8);tbl.scale(1,1.28)
for (r,col),cell in tbl.get_celld().items():
    if r==0: cell.set_facecolor("#1f3a56");cell.set_text_props(color="white",fontweight="bold")
    elif r%2==0: cell.set_facecolor("#eef3f8")
ax3.set_title("Διάμεσο εμβαδόν ανά τύπο (m²)",fontweight="bold",color=NAVY,fontsize=11,y=0.92)
fig.savefig("v6_3_11_com_char.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.11 τύποι:",len(vc),"| εμβαδόν median m²:",round(c["m2"].median(),0))

# ════════════════ Σχ 3.14: Τύποι & γεωγραφική κατανομή ════════════════
def grp(x):
    x=str(x).lower()
    if "office" in x: return "Γραφεία"
    if "retail" in x or "strip" in x: return "Λιανικό εμπόριο"
    if "restaurant" in x: return "Εστίαση"
    if "hotel" in x: return "Ξενοδοχεία"
    if "school" in x: return "Εκπαίδευση"
    if "hospital" in x or "outpatient" in x: return "Υγεία"
    if "warehouse" in x: return "Αποθήκες"
    return "Άλλο"
c["grp"]=c["bt"].map(grp)
fig,ax=plt.subplots(1,2,figsize=(16,6),gridspec_kw={"width_ratios":[1,1.3]})
fig.suptitle("ComStock — Τύποι κτιρίων & γεωγραφική κατανομή",fontsize=13.5,fontweight="bold",color=NAVY,y=1.0)
# P1: pie top-8 τύποι
top8=c["bt"].value_counts().head(8)
GCOL=["#2e5077","#7cb0d6","#8a6a3a","#c0435a","#3a7a6a","#e0a94b","#9b8ec4","#999"]
ax[0].pie(top8.values,labels=[TYPE_GR.get(t,t) for t in top8.index],colors=GCOL,autopct="%1.1f%%",
          startangle=90,textprops={"fontsize":8.5},wedgeprops={"edgecolor":"white","lw":1})
ax[0].set_title("Ποσοστό ανά τύπο (Top 8)",fontweight="bold",color=NAVY,fontsize=11)
# P2: stacked bar grouped types ανά ζώνη (πλήθος)
gord=["Γραφεία","Λιανικό εμπόριο","Αποθήκες","Εστίαση","Εκπαίδευση","Υγεία","Ξενοδοχεία","Άλλο"]
GC={"Γραφεία":"#2e5077","Λιανικό εμπόριο":"#7cb0d6","Αποθήκες":"#8a6a3a","Εστίαση":"#c0435a",
    "Εκπαίδευση":"#3a7a6a","Υγεία":"#e0a94b","Ξενοδοχεία":"#9b8ec4","Άλλο":"#bbb"}
bottom=np.zeros(len(ZONES))
for g in gord:
    vals=[c[(c.zone==z)&(c.grp==g)].shape[0] for z in ZONES]
    if sum(vals)==0: continue
    ax[1].bar(ZONES,vals,bottom=bottom,label=g,color=GC[g],edgecolor="white",lw=.6);bottom+=np.array(vals)
ax[1].set_title("Τύποι κτιρίων ανά κλιματική ζώνη",fontweight="bold",color=NAVY,fontsize=11)
ax[1].set_ylabel("Πλήθος κτιρίων");ax[1].legend(fontsize=8,ncol=2,loc="upper right")
fig.tight_layout();fig.savefig("v6_3_14_com_types.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.14 top8 done")

# ════════════════ merged_master commercial για 3.12, 3.13 ════════════════
m=pd.read_parquet("merged_master.parquet").sort_index()
CM=m[m.dataset_type=="commercial"].copy(); NB=400
def czdf(z): return CM[CM.zone==z].sort_index()
idx=czdf("3A").index
def cagg(col): return pd.concat([czdf(z)[col].reset_index(drop=True) for z in ZONES],axis=1).mean(axis=1)
TE="out.site_energy.total.energy_consumption.kwh"
EL="out.electricity.total.energy_consumption.kwh";GAS="out.natural_gas.total.energy_consumption.kwh"
OIL="out.fuel_oil.total.energy_consumption.kwh";PRO="out.propane.total.energy_consumption.kwh"

# ── Σχ 3.12: Χρονικά μοτίβα εμπορικών (business hours) ──
en=cagg(TE); en.index=idx
fig,ax=plt.subplots(2,2,figsize=(15,10))
fig.suptitle("ComStock — Χρονικά μοτίβα εμπορικών κτιρίων (business hours 8:00–18:00)",fontsize=14,fontweight="bold",color=NAVY,y=1.0)
hg=en.groupby(en.index.hour);hm=hg.mean();hs=hg.std()
a=ax[0,0];a.axvspan(8,18,color="#fff4d6",alpha=.7)
a.plot(hm.index,hm.values,color="#ED7D31",lw=2.3);a.fill_between(hm.index,hm-hs,hm+hs,color="#ED7D31",alpha=.15)
a.text(13,hm.max()*0.55,"Business Hours\n8:00–18:00",ha="center",fontsize=9,color="#b5651d")
a.set_title("Ωριαίο προφίλ ηλεκτρισμού",fontweight="bold",color=NAVY);a.set_xlabel("Ώρα");a.set_ylabel("kWh/ώρα")
wd=en[en.index.dayofweek<5].groupby(en[en.index.dayofweek<5].index.hour).mean()
we=en[en.index.dayofweek>=5].groupby(en[en.index.dayofweek>=5].index.hour).mean()
ax[0,1].plot(wd.index,wd.values,color="#ED7D31",lw=2.3,label="Εργάσιμη")
ax[0,1].plot(we.index,we.values,color="#7ab0d4",lw=2,ls="--",label="Σαββατοκύριακο")
ax[0,1].set_title("Εργάσιμη vs Σαββατοκύριακο (έντονη διαφορά)",fontweight="bold",color=NAVY)
ax[0,1].set_xlabel("Ώρα");ax[0,1].set_ylabel("kWh/ώρα");ax[0,1].legend(fontsize=9)
piv=en.groupby([en.index.hour,en.index.dayofweek]).mean().unstack()
im=ax[1,0].imshow(piv.values,aspect="auto",origin="lower",cmap="YlOrRd")
ax[1,0].set_title("Θερμικός χάρτης: Ενέργεια (Ώρα × Ημέρα)",fontweight="bold",color=NAVY)
ax[1,0].set_xticks(range(7));ax[1,0].set_xticklabels(["Δευ","Τρι","Τετ","Πεμ","Παρ","Σαβ","Κυρ"])
ax[1,0].set_ylabel("Ώρα");ax[1,0].set_yticks(range(0,24,3));ax[1,0].set_yticklabels(range(0,24,3))
plt.colorbar(im,ax=ax[1,0],fraction=.046,pad=.04,label="kWh")
mo=[en[en.index.month==k].values for k in range(1,13)]
bp=ax[1,1].boxplot(mo,labels=MONTHS,showfliers=False,patch_artist=True)
for p in bp["boxes"]: p.set_facecolor("#f2b48c");p.set_edgecolor("#b5651d")
for med in bp["medians"]: med.set_color("#7a3e10")
ax[1,1].set_title("Μηνιαία κατανομή συνολικής ενέργειας",fontweight="bold",color=NAVY)
ax[1,1].set_ylabel("kWh/ώρα");ax[1,1].tick_params(axis="x",labelsize=7)
fig.tight_layout();fig.savefig("v6_3_12_com_temporal.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.12 temporal done")

# ── Σχ 3.13: Καύσιμα & CO2 εμπορικών ──
fig,ax=plt.subplots(1,3,figsize=(17,5))
fig.suptitle("ComStock — Καύσιμα & CO₂ (ηλεκτρισμός κυρίαρχος· αέριο κυρίως χειμώνα)",fontsize=13.5,fontweight="bold",color=NAVY,y=1.02)
labs=["Ηλεκτρισμός","Φυσικό αέριο","Πετρέλαιο","Προπάνιο"];cols=["#2E75B6","#ED7D31","#8a6a3a","#9b59b6"]
en_f=[cagg(EL).sum(),cagg(GAS).sum(),cagg(OIL).sum(),cagg(PRO).sum()]
co_f=[sum(czdf(z)["co2_elec_kg"].sum()*NB for z in ZONES),sum(czdf(z)["co2_gas_kg"].sum()*NB for z in ZONES),
      sum(czdf(z)["co2_oil_kg"].sum()*NB for z in ZONES),sum(czdf(z)["co2_propane_kg"].sum()*NB for z in ZONES)]
ax[0].pie(en_f,labels=labs,colors=cols,autopct="%1.1f%%",startangle=90,textprops={"fontsize":9},wedgeprops={"edgecolor":"white","lw":1})
ax[0].set_title("Ποσοστό ενέργειας ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
ax[1].pie(co_f,labels=labs,colors=cols,autopct="%1.1f%%",startangle=90,textprops={"fontsize":9},wedgeprops={"edgecolor":"white","lw":1})
ax[1].set_title("Ποσοστό CO₂ ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
for col,lab,cc in [(EL,"Ηλεκτρισμός","#2E75B6"),(GAS,"Φυσικό αέριο","#ED7D31"),(OIL,"Πετρέλαιο","#8a6a3a"),(PRO,"Προπάνιο","#9b59b6")]:
    s=cagg(col);s.index=idx;mo=s.groupby(s.index.month).mean()
    ax[2].plot(range(1,13),mo.values,marker="o",ms=4,color=cc,label=lab,lw=2)
ax[2].set_title("Μηνιαίο προφίλ ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
ax[2].set_xlabel("Μήνας");ax[2].set_ylabel("kWh/ώρα");ax[2].set_xticks(range(1,13));ax[2].set_xticklabels(MONTHS,fontsize=7);ax[2].legend(fontsize=8)
fig.tight_layout();fig.savefig("v6_3_13_com_fuel.png",bbox_inches="tight",facecolor="white");plt.close()
tot=sum(en_f);print("✓ 3.13 energy%:",[f"{l}={v/tot*100:.1f}%" for l,v in zip(labs,en_f)])
print("DONE commercial")
