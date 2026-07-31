# -*- coding: utf-8 -*-
"""Κεφ.4 §4.4.1-4.4.2 (district profiles, diversity) + §4.5.2 ESG + §4.5.3 climate stress.
District = 0.6×residential + 0.4×commercial (ULI 2022). 5 ζώνες."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
NAVY="#1f3a56";BLUE="#2E75B6";ORANGE="#ED7D31";GREEN="#2E9E4F";RED="#C00000";TEAL="#2e6b5e"
TE="out.site_energy.total.energy_consumption.kwh";CO2="co2_total_kg"
ZONES=["2A","3A","4A","5A","5B"];CITY={"2A":"Houston","3A":"Atlanta","4A":"Baltimore","5A":"Chicago","5B":"Denver"}
ZCOL={"2A":"#c0435a","3A":"#e08a4f","4A":"#edc84e","5A":"#7ab0d4","5B":"#4574a8"}
m=pd.read_parquet("merged_master.parquet").sort_index()

def district(zone):
    r=m[(m.zone==zone)&(m.dataset_type=="residential")].sort_index()
    c=m[(m.zone==zone)&(m.dataset_type=="commercial")].sort_index()
    d=pd.DataFrame(index=r.index)
    d["energy"]=0.6*r[TE].values+0.4*c[TE].values
    d["co2"]=0.6*r[CO2].values+0.4*c[CO2].values
    d["res"]=r[TE].values; d["com"]=c[TE].values
    return d

# ════ Σχ 4.1: district ωριαία προφίλ (εργάσιμη/ΣΚ) + μηνιαίο ════
fig,ax=plt.subplots(1,2,figsize=(15,5))
for z in ZONES:
    d=district(z); d["hour"]=d.index.hour; d["wd"]=d.index.dayofweek<5
    wd=d[d.wd].groupby("hour")["energy"].mean(); we=d[~d.wd].groupby("hour")["energy"].mean()
    ax[0].plot(wd.index,wd.values,color=ZCOL[z],lw=2,label=f"{z} {CITY[z]}")
    ax[0].plot(we.index,we.values,color=ZCOL[z],lw=1.3,ls="--",alpha=.6)
ax[0].set_title("Ωριαίο προφίλ αστικής περιοχής (60/40) — εργάσιμη (—) vs Σ/Κ (--)",fontweight="bold",color=NAVY)
ax[0].set_xlabel("Ώρα ημέρας");ax[0].set_ylabel("Ενέργεια (kWh/μέσο κτίριο)");ax[0].legend(fontsize=8)
for z in ZONES:
    d=district(z); mo=d.groupby(d.index.month)["energy"].mean()
    ax[1].plot(mo.index,mo.values,color=ZCOL[z],lw=2,marker="o",ms=4,label=f"{z} {CITY[z]}")
ax[1].set_title("Μέσο μηνιαίο προφίλ αστικής περιοχής ανά ζώνη",fontweight="bold",color=NAVY)
ax[1].set_xlabel("Μήνας");ax[1].set_ylabel("Ενέργεια (kWh/μέσο κτίριο)");ax[1].set_xticks(range(1,13));ax[1].legend(fontsize=8)
fig.tight_layout();fig.savefig("ch4_fig_district.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ district Σχ4.1")

# ════ Σχ 4.2: Diversity Factor (stacked area 1 εβδομ. + DF ανά ζώνη) ════
df_res={};
fig,ax=plt.subplots(1,2,figsize=(15,5))
d3=district("3A"); wk=d3.iloc[:168]
ax[0].fill_between(range(168),0,0.6*wk["res"].values,color=BLUE,alpha=.7,label="Οικιστικά (60%)")
ax[0].fill_between(range(168),0.6*wk["res"].values,0.6*wk["res"].values+0.4*wk["com"].values,color=ORANGE,alpha=.7,label="Εμπορικά (40%)")
ax[0].set_title("Σύνθεση φορτίου αστικής περιοχής — 1 εβδομάδα (Ζώνη 3A)",fontweight="bold",color=NAVY)
ax[0].set_xlabel("Ώρα (εβδομάδα)");ax[0].set_ylabel("Ενέργεια (kWh)");ax[0].legend()
DF={}
for z in ZONES:
    d=district(z); pr=d["res"].max(); pc=d["com"].max(); pd_=d["energy"].max()
    DF[z]=(0.6*pr+0.4*pc)/pd_ if pd_>0 else np.nan
ax[1].bar([f"{z}\n{CITY[z]}" for z in ZONES],[DF[z] for z in ZONES],color=[ZCOL[z] for z in ZONES],alpha=.9)
ax[1].axhline(1.0,color=RED,ls="--",lw=1,label="DF=1 (ταυτόχρονες αιχμές)")
ax[1].set_title("Συντελεστής Ποικιλίας (Diversity Factor) ανά ζώνη",fontweight="bold",color=NAVY);ax[1].set_ylabel("Diversity Factor");ax[1].legend(fontsize=9)
for i,z in enumerate(ZONES): ax[1].text(i,DF[z],f"{DF[z]:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
fig.tight_layout();fig.savefig("ch4_fig_diversity.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ diversity Σχ4.2 | DF:",{k:round(v,3) for k,v in DF.items()})

# ════ Πίν 4.6 + fig: ESG scoring ανά ζώνη (district baseline) ════
esg_rows=[]
for z in ZONES:
    d=district(z); ann_e=d["energy"].sum(); ann_c=d["co2"].sum()
    ci=ann_c/ann_e  # kg CO2 / kWh (carbon intensity)
    esg_rows.append({"zone":z,"city":CITY[z],"annual_kwh":ann_e,"annual_co2":ann_c,"carbon_intensity":ci})
edf=pd.DataFrame(esg_rows)
# ESG score: χαμηλότερη carbon intensity + χαμηλότερη ενέργεια = καλύτερο (0-100)
def norm(s,invert=True):
    r=(s-s.min())/(s.max()-s.min()+1e-9); return (1-r if invert else r)
edf["score"]=(0.6*norm(edf["carbon_intensity"])+0.4*norm(edf["annual_kwh"]))*100
def rating(s):
    return "A" if s>=85 else "B" if s>=70 else "C" if s>=55 else "D" if s>=40 else "E" if s>=25 else "F"
edf["rating"]=edf["score"].round().apply(rating)
edf.to_csv("ch4_outputs/esg_by_zone.csv",index=False,float_format="%.4f")
fig,ax=plt.subplots(1,2,figsize=(15,5))
ax[0].bar([f"{r.zone}\n{r.city}" for r in edf.itertuples()],edf["carbon_intensity"],color=[ZCOL[z] for z in ZONES],alpha=.9)
ax[0].set_title("Ένταση άνθρακα district ανά ζώνη (kg CO₂/kWh)",fontweight="bold",color=NAVY);ax[0].set_ylabel("kg CO₂ / kWh")
for i,v in enumerate(edf["carbon_intensity"]): ax[0].text(i,v,f"{v:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
bars=ax[1].bar([f"{r.zone}" for r in edf.itertuples()],edf["score"],color=[ZCOL[z] for z in ZONES],alpha=.9)
ax[1].set_ylim(0,100);ax[1].set_title("ESG Score ανά ζώνη (↑ καλύτερο)",fontweight="bold",color=NAVY);ax[1].set_ylabel("ESG Score (0–100)")
for i,(v,rt) in enumerate(zip(edf["score"],edf["rating"])): ax[1].text(i,v,f"{v:.0f}\n({rt})",ha="center",va="bottom",fontsize=9,fontweight="bold")
fig.tight_layout();fig.savefig("ch4_fig_esg.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ ESG Πίν4.6:\n",edf[["zone","annual_kwh","annual_co2","carbon_intensity","score","rating"]].round(3).to_string(index=False))

# ════ Πίν 4.7 + fig: Climate stress (από scenarios_2x3_baseline) ════
sc=pd.read_parquet("scenarios_2x3_baseline.parquet")
# % μεταβολή κατανάλωσης ανά ζώνη × σενάριο (consumption_delta_pct)
rows=[]
for z in ZONES:
    zc={"2A":"2A","3A":"3A","4A":"4A","5A":"5A","5B":"5B"}[z]
    r={"zone":z,"city":CITY[z]}
    for scn,lbl in [("SSP2-4.5 (2040)","SSP2_2040"),("SSP2-4.5 (2050)","SSP2_2050"),("SSP2-4.5 (2100)","SSP2_2100"),
                    ("SSP5-8.5 (2040)","SSP5_2040"),("SSP5-8.5 (2050)","SSP5_2050"),("SSP5-8.5 (2100)","SSP5_2100")]:
        d=sc[(sc.climate_zone==zc)&(sc.scenario==scn)]
        r[lbl]=round(d["consumption_delta_pct"].mean(),1) if len(d) else np.nan
    rows.append(r)
cs=pd.DataFrame(rows); cs.to_csv("ch4_outputs/climate_stress_by_zone.csv",index=False)
fig,ax=plt.subplots(figsize=(13,5.5))
x=np.arange(len(ZONES));w=.13
scn_cols=[("SSP2_2040","#bcd4ea"),("SSP2_2050","#6fa8d6","" ),("SSP2_2100",BLUE),("SSP5_2040","#f0b49a"),("SSP5_2050","#e07a4f"),("SSP5_2100",RED)]
for i,(c,col,*_) in enumerate(scn_cols):
    ax.bar(x+(i-2.5)*w,cs[c],w,label=c.replace("_"," "),color=col)
ax.set_xticks(x);ax.set_xticklabels([f"{z}\n{CITY[z]}" for z in ZONES]);ax.axhline(0,color="#333",lw=.8)
ax.set_ylabel("Μεταβολή ετήσιας κατανάλωσης (%)");ax.set_title("Climate Stress Testing — μεταβολή κατανάλωσης ανά ζώνη & σενάριο",fontweight="bold",color=NAVY)
ax.legend(fontsize=8,ncol=2)
fig.tight_layout();fig.savefig("ch4_fig_climate_stress.png",bbox_inches="tight",facecolor="white");plt.close()
print("✓ climate stress Πίν4.7:\n",cs.to_string(index=False))
json.dump({"diversity":{k:round(v,4) for k,v in DF.items()},"esg":edf.to_dict("records"),"climate_stress":cs.to_dict("records")},
          open("ch4_outputs/district_esg.json","w"),indent=2,ensure_ascii=False,default=str)
print("DONE")
