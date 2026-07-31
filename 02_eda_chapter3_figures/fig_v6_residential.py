# -*- coding: utf-8 -*-
"""Αναπαραγωγή παλιών σχημάτων Thesis_v6 §3.6.1 (Οικιστικά) στα ΝΕΑ δεδομένα (5.000/5 ζώνες).
3.5 καύσιμα · 3.6 CO2/καύσιμο · 3.7 κατανάλωση+CO2/ζώνη (5 ζώνες) · 3.8 χρονικά μοτίβα.
Πηγή: merged_master.parquet (residential subset, per-average-building· 600 κτίρια/ζώνη)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.25,"axes.axisbelow":True})
NAVY="#1f3a56"
ZCOL={"2A":"#c0435a","3A":"#e08a4f","4A":"#edc84e","5A":"#7ab0d4","5B":"#4574a8"}
CITY={"2A":"Houston","3A":"Atlanta","4A":"Baltimore","5A":"Chicago","5B":"Denver"}
ZONES=["2A","3A","4A","5A","5B"]; NB=600  # residential buildings per zone
FUELS=[("elec","Ηλεκτρισμός","out.electricity.total.energy_consumption.kwh","co2_elec_kg","#2E75B6"),
       ("gas","Φυσικό αέριο","out.natural_gas.total.energy_consumption.kwh","co2_gas_kg","#ED7D31"),
       ("oil","Πετρέλαιο","out.fuel_oil.total.energy_consumption.kwh","co2_oil_kg","#8a6a3a"),
       ("prop","Προπάνιο","out.propane.total.energy_consumption.kwh","co2_propane_kg","#9b59b6")]
MONTHS=["Ιαν","Φεβ","Μαρ","Απρ","Μάι","Ιουν","Ιουλ","Αυγ","Σεπ","Οκτ","Νοε","Δεκ"]
m=pd.read_parquet("merged_master.parquet").sort_index()
R=m[m.dataset_type=="residential"].copy()

def zdf(z): return R[R.zone==z].sort_index()
# aggregate residential (μέσο κτίριο, μέσος όρος 5 ζωνών ίσου βάρους 600/ζώνη)
def agg_hourly(col):
    return pd.concat([zdf(z)[col].reset_index(drop=True) for z in ZONES],axis=1).mean(axis=1)
idx=zdf("3A").index

# ════════════════ Σχ 3.5: Κατανάλωση ανά καύσιμο ════════════════
fig,ax=plt.subplots(1,3,figsize=(17,5))
fig.suptitle("ResStock — Ενεργειακή κατανάλωση ανά καύσιμο (μέσο οικιστικό κτίριο, 5 ζώνες)",
             fontsize=13.5,fontweight="bold",color=NAVY,y=1.02)
ann=[agg_hourly(f[2]).sum() for f in FUELS]      # ετήσια kWh/κτίριο
meanh=[agg_hourly(f[2]).mean() for f in FUELS]    # μέση ωριαία
cols=[f[4] for f in FUELS]; labs=[f[1] for f in FUELS]
ax[0].pie(ann,labels=labs,colors=cols,autopct="%1.1f%%",startangle=90,
          textprops={"fontsize":9},wedgeprops={"edgecolor":"white","lw":1})
ax[0].set_title("Ποσοστό ανά καύσιμο (ετήσια)",fontweight="bold",color=NAVY,fontsize=11)
b=ax[1].bar(labs,meanh,color=cols,edgecolor="white")
ax[1].set_title("Μ.Ο. ωριαίας κατανάλωσης ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
ax[1].set_ylabel("kWh/ώρα")
for r,v in zip(b,meanh): ax[1].text(r.get_x()+r.get_width()/2,v,f"{v:.1f}",ha="center",va="bottom",fontweight="bold",fontsize=9)
ax[1].tick_params(axis="x",labelsize=8)
for f in FUELS:
    mo=agg_hourly(f[2]).groupby(idx.month).mean()
    ax[2].plot(range(1,13),mo.values,marker="o",ms=4,color=f[4],label=f[1],lw=2)
ax[2].set_title("Μηνιαίο προφίλ ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
ax[2].set_xlabel("Μήνας");ax[2].set_ylabel("kWh/ώρα");ax[2].set_xticks(range(1,13));ax[2].set_xticklabels(MONTHS,fontsize=7)
ax[2].legend(fontsize=8)
fig.tight_layout();fig.savefig("v6_3_5_res_fuel.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.5 fuel% :",[f"{l}={v/sum(ann)*100:.1f}%" for l,v in zip(labs,ann)])

# ════════════════ Σχ 3.6: CO2 εκπομπές ανά καύσιμο ════════════════
fig,ax=plt.subplots(1,3,figsize=(17,5))
fig.suptitle("ResStock — Εκπομπές CO₂ ανά καύσιμο (Cambium 2022 / EPA 2023· σύνολο οικιστικού αποθέματος)",
             fontsize=13.5,fontweight="bold",color=NAVY,y=1.02)
# total annual CO2 (tonnes) = Σ_zone per_building_annual_co2 × 600
def tot_co2(col): return sum(zdf(z)[col].sum()*NB for z in ZONES)/1000.0  # tonnes
co2_ann=[tot_co2(f[3]) for f in FUELS]
ax[0].pie(co2_ann,labels=labs,colors=cols,autopct="%1.1f%%",startangle=90,
          textprops={"fontsize":9},wedgeprops={"edgecolor":"white","lw":1})
ax[0].set_title("Ποσοστό CO₂ ανά καύσιμο",fontweight="bold",color=NAVY,fontsize=11)
b=ax[1].bar(labs,co2_ann,color=cols,edgecolor="white")
ax[1].set_title("Ετήσιες εκπομπές CO₂ (tonnes)",fontweight="bold",color=NAVY,fontsize=11);ax[1].set_ylabel("tonnes CO₂e")
for r,v in zip(b,co2_ann): ax[1].text(r.get_x()+r.get_width()/2,v,f"{v:.0f}t",ha="center",va="bottom",fontweight="bold",fontsize=9)
ax[1].tick_params(axis="x",labelsize=8)
# monthly CO2 (tonnes) total
co2h=pd.concat([zdf(z)["co2_total_kg"].reset_index(drop=True)*NB for z in ZONES],axis=1).sum(axis=1)
mo=pd.Series(co2h.values,index=idx).groupby(idx.month).sum()/1000.0
ax[2].bar(range(1,13),mo.values,color="#e0705a",edgecolor="white")
ax[2].set_title("Μηνιαίες εκπομπές CO₂",fontweight="bold",color=NAVY,fontsize=11)
ax[2].set_xlabel("Μήνας");ax[2].set_ylabel("tonnes CO₂e");ax[2].set_xticks(range(1,13));ax[2].set_xticklabels(MONTHS,fontsize=7)
fig.tight_layout();fig.savefig("v6_3_6_res_co2.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.6 CO2 tonnes:",[f"{l}={v:.0f}t" for l,v in zip(labs,co2_ann)])

# ════════════════ Σχ 3.7: Κατανάλωση & CO2 ανά ζώνη (5 ζώνες) ════════════════
TE="out.site_energy.total.energy_consumption.kwh"; EL="out.electricity.total.energy_consumption.kwh"
fig,ax=plt.subplots(2,2,figsize=(15,10))
fig.suptitle("Οικιστικά κτίρια — Κατανάλωση & CO₂ ανά κλιματική ζώνη ASHRAE (2A/3A/4A/5A/5B)",
             fontsize=14,fontweight="bold",color=NAVY,y=1.0)
for z in ZONES:
    d=zdf(z); hp=d[EL].groupby(d.index.hour).mean()
    ax[0,0].plot(hp.index,hp.values,color=ZCOL[z],lw=2,label=f"{z} {CITY[z]}")
ax[0,0].set_title("Ωριαίο προφίλ ηλεκτρισμού ανά ζώνη",fontweight="bold",color=NAVY)
ax[0,0].set_xlabel("Ώρα");ax[0,0].set_ylabel("kWh/ώρα");ax[0,0].legend(fontsize=8)
for z in ZONES:
    d=zdf(z); mo=d[TE].groupby(d.index.month).mean()
    ax[0,1].plot(range(1,13),mo.values,color=ZCOL[z],lw=2,marker="o",ms=4,label=f"{z} {CITY[z]}")
ax[0,1].set_title("Μηνιαία συνολική κατανάλωση ανά ζώνη",fontweight="bold",color=NAVY)
ax[0,1].set_xlabel("Μήνας");ax[0,1].set_ylabel("kWh/ώρα");ax[0,1].set_xticks(range(1,13));ax[0,1].set_xticklabels(MONTHS,fontsize=7);ax[0,1].legend(fontsize=8)
meanz=[zdf(z)[TE].mean() for z in ZONES]
b=ax[1,0].bar([f"{z}\n{CITY[z]}" for z in ZONES],meanz,color=[ZCOL[z] for z in ZONES],edgecolor="white")
ax[1,0].set_title("Μ.Ο. ωριαίας κατανάλωσης ανά ζώνη",fontweight="bold",color=NAVY);ax[1,0].set_ylabel("kWh/ώρα")
for r,v in zip(b,meanz): ax[1,0].text(r.get_x()+r.get_width()/2,v,f"{v:.1f}",ha="center",va="bottom",fontweight="bold",fontsize=9)
co2z=[zdf(z)["co2_total_kg"].sum()*NB/1000.0 for z in ZONES]  # tonnes
b=ax[1,1].bar([f"{z}\n{CITY[z]}" for z in ZONES],co2z,color=[ZCOL[z] for z in ZONES],edgecolor="white")
ax[1,1].set_title("Ετήσιες εκπομπές CO₂ ανά ζώνη",fontweight="bold",color=NAVY);ax[1,1].set_ylabel("tonnes CO₂e")
for r,v in zip(b,co2z): ax[1,1].text(r.get_x()+r.get_width()/2,v,f"{v:.0f}t",ha="center",va="bottom",fontweight="bold",fontsize=9)
fig.tight_layout();fig.savefig("v6_3_7_res_byzone.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.7 mean/zone:",[f"{z}={v:.1f}" for z,v in zip(ZONES,meanz)],"| CO2t:",[f"{z}={v:.0f}" for z,v in zip(ZONES,co2z)])

# ════════════════ Σχ 3.8: Χρονικά μοτίβα (aggregate residential) ════════════════
en=agg_hourly(TE); co=agg_hourly("co2_total_kg")
en.index=idx; co.index=idx
fig,ax=plt.subplots(2,2,figsize=(15,10))
fig.suptitle("ResStock — Χρονικά μοτίβα: συνολική ενέργεια & CO₂ (μέσο οικιστικό κτίριο)",
             fontsize=14,fontweight="bold",color=NAVY,y=1.0)
# panel1: hourly energy & CO2 dual axis + band
hg=en.groupby(en.index.hour); hm=hg.mean(); hs=hg.std()
a=ax[0,0]; a.plot(hm.index,hm.values,color="#2E75B6",lw=2.2,label="Ενέργεια (kWh)")
a.fill_between(hm.index,hm-hs,hm+hs,color="#2E75B6",alpha=.18)
a.set_xlabel("Ώρα");a.set_ylabel("kWh/ώρα",color="#2E75B6");a.tick_params(axis="y",labelcolor="#2E75B6")
a2=a.twinx();a2.grid(False)
ch=co.groupby(co.index.hour).mean();a2.plot(ch.index,ch.values,color="#C00000",lw=2,ls="--",label="CO₂ (kg)")
a2.set_ylabel("kg CO₂/ώρα",color="#C00000");a2.tick_params(axis="y",labelcolor="#C00000")
a.set_title("Ωριαίο προφίλ: Ενέργεια & CO₂",fontweight="bold",color=NAVY)
# panel2: weekday vs weekend
wd=en[en.index.dayofweek<5].groupby(en[en.index.dayofweek<5].index.hour).mean()
we=en[en.index.dayofweek>=5].groupby(en[en.index.dayofweek>=5].index.hour).mean()
ax[0,1].plot(wd.index,wd.values,color="#2E75B6",lw=2.2,label="Εργάσιμη")
ax[0,1].plot(we.index,we.values,color="#ED7D31",lw=2,ls="--",label="Σαββατοκύριακο")
ax[0,1].set_title("Εργάσιμη vs Σαββατοκύριακο",fontweight="bold",color=NAVY)
ax[0,1].set_xlabel("Ώρα");ax[0,1].set_ylabel("kWh/ώρα");ax[0,1].legend(fontsize=9)
# panel3: heatmap hour x dow
piv=en.groupby([en.index.hour,en.index.dayofweek]).mean().unstack()
im=ax[1,0].imshow(piv.values,aspect="auto",origin="lower",cmap="YlOrRd")
ax[1,0].set_title("Θερμικός χάρτης: Ενέργεια (Ώρα × Ημέρα)",fontweight="bold",color=NAVY)
ax[1,0].set_xticks(range(7));ax[1,0].set_xticklabels(["Δευ","Τρι","Τετ","Πεμ","Παρ","Σαβ","Κυρ"])
ax[1,0].set_ylabel("Ώρα");ax[1,0].set_yticks(range(0,24,3));ax[1,0].set_yticklabels(range(0,24,3))
plt.colorbar(im,ax=ax[1,0],fraction=.046,pad=.04,label="kWh")
# panel4: monthly CO2 (per avg building, kg)
moc=co.groupby(co.index.month).sum()
ax[1,1].bar(range(1,13),moc.values,color="#e0705a",edgecolor="white")
ax[1,1].set_title("Μηνιαίες εκπομπές CO₂",fontweight="bold",color=NAVY)
ax[1,1].set_xlabel("Μήνας");ax[1,1].set_ylabel("kg CO₂/κτίριο");ax[1,1].set_xticks(range(1,13));ax[1,1].set_xticklabels(MONTHS,fontsize=7)
fig.tight_layout();fig.savefig("v6_3_8_res_temporal.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ 3.8 temporal done")
print("DONE residential")
