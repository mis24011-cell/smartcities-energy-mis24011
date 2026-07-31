# -*- coding: utf-8 -*-
"""
phase4_esg.py — fig17 (Decarbonization Pareto) & fig19 (ESG Dashboard master)
Μέθοδος από notebooks 17 & 19, νέο 2×3 σχέδιο. EIA/Tableau ύφος + επεξήγηση.
"""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from phase4_figures import _load, BLUE, RED, GRAY, TITLE_C, EF_ELEC, EF_GAS, GRID_DECARB
from phase4_social import add_burden
SC3=["Baseline (2018)","SSP2-4.5 (2100)","SSP5-8.5 (2100)"]
SC_LBL={"Baseline (2018)":"Baseline\n2018","SSP2-4.5 (2100)":"SSP2-4.5\n2100","SSP5-8.5 (2100)":"SSP5-8.5\n2100"}
SC_COL={"Baseline (2018)":"#5A6B7A","SSP2-4.5 (2100)":BLUE,"SSP5-8.5 (2100)":RED}

def _prep(df):
    df=df.copy()
    ef=df["climate_zone"].map(EF_ELEC).fillna(0.35)
    gd=df["year"].map(GRID_DECARB).fillna(1.0) if "year" in df.columns else df["scenario_year"].map(GRID_DECARB).fillna(1.0)
    # CO₂ = Scope 2 (ηλεκτρισμός × Cambium EF × αποανθρακοποίηση δικτύου) + Scope 1 (αέριο, EPA)
    share=df["elec_share"] if "elec_share" in df.columns else 0.7
    df["co2_kg"]=df["annual_kwh_projected"]*share*ef*gd + df["natural_gas_kwh"]*EF_GAS
    df["co2_t"]=df["co2_kg"]/1000
    df["ci_kg_m2"]=df["co2_kg"]/df["floor_area_m2"].clip(lower=1)
    return df

# ---------------- fig17: Decarbonization Pareto ----------------
INTERV={
 "BAU":            dict(energy=0.00,co2_add=0.00,cost=0,    equity=0, color="#9E9E9E",mk="o"),
 "Weatherization": dict(energy=0.25,co2_add=0.10,cost=12000,equity=4, color="#2E9E4F",mk="s"),
 "Electrification":dict(energy=0.15,co2_add=0.20,cost=8000, equity=5, color="#2166AC",mk="^"),
 "Deep Retrofit":  dict(energy=0.40,co2_add=0.30,cost=18000,equity=7, color="#8E44AD",mk="D"),
}
def fig_pareto(df, out="fig17_1_decarbonization_pareto.png", price=0.25, disc=0.04):
    base=_prep(df[df.scenario=="Baseline (2018)"])
    n=len(base); tot_kwh=base.annual_kwh.sum(); tot_co2=base.co2_t.sum()
    rows=[]
    for k,p in INTERV.items():
        co2_red=tot_co2*(p["energy"]+p["co2_add"]); co2_pct=co2_red/tot_co2*100 if tot_co2 else 0
        inv=n*p["cost"]/1e6                                   # M€
        save=tot_kwh*p["energy"]*price
        npv=sum(save/(1+disc)**t for t in range(1,21))-inv*1e6
        cpc=inv*1e6/co2_red if co2_red>0 else np.nan
        rows.append(dict(key=k,co2_pct=co2_pct,inv=inv,npv=npv/1e6,cpc=cpc,**p))
    cb=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,2,figsize=(15,6.6))
    fig.suptitle("Ανάλυση Κόστους–Οφέλους Σεναρίων Αποανθρακοποίησης",
                 fontsize=16,fontweight="bold",color=TITLE_C,y=1.0)
    ax=axes[0]
    for _,r in cb.iterrows():
        ax.scatter(r.inv,r.co2_pct,s=230 if r.key!="BAU" else 120,color=r.color,marker=r.mk,zorder=5,edgecolor="white",lw=1.2)
        ax.annotate(r.key,(r.inv,r.co2_pct),xytext=(9,5),textcoords="offset points",fontsize=11,fontweight="bold")
    ax.set_xlabel("Συνολική Επένδυση (εκατ. €)",fontsize=12.5); ax.set_ylabel("Μείωση CO₂ (%)",fontsize=12.5)
    ax.set_title("Pareto Frontier: CO₂ vs Κόστος",fontsize=13,fontweight="bold")
    ax=axes[1]
    for _,r in cb.iterrows():
        if r.key=="BAU": continue
        ax.scatter(r.co2_pct,r.equity,s=r.inv*18+80,color=r.color,alpha=0.85,zorder=5,edgecolor="white",lw=1.2)
        ax.annotate(r.key,(r.co2_pct,r.equity),xytext=(6,6),textcoords="offset points",fontsize=11,fontweight="bold")
    ax.set_xlabel("Μείωση CO₂ (%)",fontsize=12.5); ax.set_ylabel("Δείκτης Ισότητας (0-10)",fontsize=12.5)
    ax.set_title("Απόδοση vs Ισότητα  (μέγεθος = επένδυση)",fontsize=13,fontweight="bold")
    ax.set_xlim(0,80); ax.set_ylim(0,10)
    fig.tight_layout(rect=[0,0.09,1,0.97])
    fig.text(0.012,0.05,
        "Κάθε σημείο = σενάριο παρέμβασης σε ~"+f"{n:,}"+" κτίρια. Αριστερά: μεγαλύτερη μείωση CO₂ ψηλότερα, μικρότερο κόστος αριστερά "
        "(ιδανικό: πάνω-αριστερά). Δεξιά: συνδυασμός κλιματικού & κοινωνικού οφέλους.",
        fontsize=10.3,color="#33475b",style="italic",ha="left",va="bottom",
        bbox=dict(boxstyle="round,pad=0.4",fc="#f2f6fa",ec="#cfe0ee",lw=0.8))
    fig.text(0.012,0.012,"Πηγή: NREL EFS (2021)· ACEEE (2020)· NPV 20ετ., 4%· τιμή 0,25 €/kWh",fontsize=9.3,color=GRAY,ha="left")
    fig.savefig(out,bbox_inches="tight"); plt.close(fig); return out, cb

# ---------------- fig19: ESG Dashboard master ----------------
def _esg_scores(df):
    d=_prep(df); res=add_burden(df[(df.building_type=="residential")&(df.income_quartile.notna())].copy())
    base_ci=d[d.scenario=="Baseline (2018)"].ci_kg_m2.mean()
    out=[]
    for s in SC3:
        ci=d[d.scenario==s].ci_kg_m2.mean()
        e=np.clip(100-(ci/max(base_ci,1))*60,0,100)
        rs=res[res.scenario==s]
        hb=rs.high_burden.mean()*100 if len(rs) else 20
        ext=(rs.energy_burden_pct>10).mean()*100 if len(rs) else 5
        q1=rs[rs.income_quartile==1].energy_burden_pct.median(); q4=rs[rs.income_quartile==4].energy_burden_pct.median()
        eq=q1/max(q4,0.01) if len(rs) else 4
        ssc=np.clip(100-hb*0.5-ext*0.5-(eq-1)*2,0,100)
        g=80.0
        out.append(dict(scenario=s,E=e,S=ssc,G=g,ESG=(e+ssc+g)/3,hb=hb,ci=ci))
    return pd.DataFrame(out), d, res

def fig_esg_dashboard(df, out="fig19_1_ESG_dashboard_master.png"):
    esg,d,res=_esg_scores(df)
    fig=plt.figure(figsize=(17,13))
    fig.suptitle("Ολοκληρωμένο ESG Dashboard — Digital Twin Ανάλυσης Ενέργειας\nEnvironmental · Social · Governance για Έξυπνες Πόλεις",
                 fontsize=16,fontweight="bold",color=TITLE_C,y=0.98)
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.5,wspace=0.32)
    xr=range(len(SC3)); xl=[SC_LBL[s] for s in SC3]; cols=[SC_COL[s] for s in SC3]
    # --- E ---
    ax=fig.add_subplot(gs[0,0])
    co2=[d[d.scenario==s].co2_t.sum() for s in SC3]
    b=ax.bar(xr,co2,color=cols,alpha=0.9,edgecolor="white")
    for bi,v in zip(b,co2): ax.text(bi.get_x()+bi.get_width()/2,v*1.01,f"{v/1000:.0f}k",ha="center",fontsize=10,fontweight="bold")
    ax.set_title("E1 · Τροχιά CO₂ (τόνοι)",fontsize=12,fontweight="bold"); ax.set_xticks(xr); ax.set_xticklabels(xl,fontsize=9.5); ax.grid(axis="y",alpha=0.3)
    ax=fig.add_subplot(gs[0,1])
    zones=["2A","3A","4A","5A","5B"]; x=np.arange(len(zones)); w=0.26
    for j,s in enumerate(SC3):
        vals=[d[(d.scenario==s)&(d.climate_zone==z)].ci_kg_m2.mean() for z in zones]
        ax.bar(x+(j-1)*w,vals,w,color=SC_COL[s],alpha=0.9,label=SC_LBL[s].replace("\n"," "))
    ax.set_title("E2 · Ένταση Άνθρακα (kg/m²)",fontsize=12,fontweight="bold"); ax.set_xticks(x); ax.set_xticklabels([f"Ζ.{z}" for z in zones],fontsize=10); ax.legend(fontsize=8.5,frameon=False); ax.grid(axis="y",alpha=0.3)
    ax=fig.add_subplot(gs[0,2])
    b=ax.barh(xr,esg.E,color=cols,alpha=0.9)
    for bi,v in zip(b,esg.E): ax.text(v+1,bi.get_y()+bi.get_height()/2,f"{v:.0f}",va="center",fontsize=11,fontweight="bold")
    ax.set_title("E Score (0-100 · ↑καλύτερο)",fontsize=12,fontweight="bold"); ax.set_yticks(xr); ax.set_yticklabels([s.replace("\n"," ") for s in xl],fontsize=9.5); ax.set_xlim(0,100)
    # --- S ---
    ax=fig.add_subplot(gs[1,0])
    base=res[res.scenario=="Baseline (2018)"]; data=[base[base.income_quartile==q].energy_burden_pct.values for q in [1,2,3,4]]
    bp=ax.boxplot(data,patch_artist=True,medianprops=dict(color="black",lw=1.4),flierprops=dict(marker="o",ms=2,alpha=0.2))
    for p,c in zip(bp["boxes"],["#D1495B","#EDAE49","#66A182","#2166AC"]): p.set_facecolor(c); p.set_alpha(0.75)
    ax.axhline(6,color="#E67E22",ls="--",lw=1.4); ax.axhline(10,color="#C0392B",ls="--",lw=1.4)
    ax.set_title("S1 · Energy Burden ανά Εισόδημα",fontsize=12,fontweight="bold"); ax.set_xticklabels(["Q1","Q2","Q3","Q4"]); ax.set_ylabel("% εισοδήματος",fontsize=10)
    ax=fig.add_subplot(gs[1,1])
    hb=[res[res.scenario==s].high_burden.mean()*100 for s in SC3]
    b=ax.bar(xr,hb,color=cols,alpha=0.9,edgecolor="white")
    for bi,v in zip(b,hb): ax.text(bi.get_x()+bi.get_width()/2,v+0.8,f"{v:.0f}%",ha="center",fontsize=10,fontweight="bold")
    ax.set_title("S2 · % Νοικοκυριών Υψηλού Burden",fontsize=12,fontweight="bold"); ax.set_xticks(xr); ax.set_xticklabels(xl,fontsize=9.5); ax.grid(axis="y",alpha=0.3)
    ax=fig.add_subplot(gs[1,2])
    b=ax.barh(xr,esg.S,color=cols,alpha=0.9)
    for bi,v in zip(b,esg.S): ax.text(v+1,bi.get_y()+bi.get_height()/2,f"{v:.0f}",va="center",fontsize=11,fontweight="bold")
    ax.set_title("S Score (0-100 · ↑καλύτερο)",fontsize=12,fontweight="bold"); ax.set_yticks(xr); ax.set_yticklabels([s.replace("\n"," ") for s in xl],fontsize=9.5); ax.set_xlim(0,100)
    # --- G + composite ---
    ax=fig.add_subplot(gs[2,0])
    b=ax.barh(xr,esg.G,color=cols,alpha=0.9)
    for bi,v in zip(b,esg.G): ax.text(v+1,bi.get_y()+bi.get_height()/2,f"{v:.0f}",va="center",fontsize=11,fontweight="bold")
    ax.set_title("G Score · Διακυβέρνηση/Ποιότητα",fontsize=12,fontweight="bold"); ax.set_yticks(xr); ax.set_yticklabels([s.replace("\n"," ") for s in xl],fontsize=9.5); ax.set_xlim(0,100)
    ax=fig.add_subplot(gs[2,1:])
    x=np.arange(len(SC3)); w=0.2
    for k,(comp,cc) in enumerate([("E","#2E9E4F"),("S","#2166AC"),("G","#8E44AD"),("ESG","#C0392B")]):
        vals=esg[comp].values
        bars=ax.bar(x+(k-1.5)*w,vals,w,color=cc,alpha=0.9,label=comp+(" (σύνθετο)" if comp=="ESG" else ""))
        for bi,v in zip(bars,vals): ax.text(bi.get_x()+bi.get_width()/2,v+1,f"{v:.0f}",ha="center",fontsize=9,fontweight="bold")
    ax.set_title("Σύνθετες Βαθμολογίες ESG ανά Σενάριο",fontsize=13,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("\n"," ") for s in xl],fontsize=10.5); ax.set_ylim(0,100); ax.set_ylabel("Score (0-100)",fontsize=11); ax.legend(fontsize=10,frameon=False,ncol=4,loc="upper center")
    fig.text(0.012,0.028,
        "E: ένταση άνθρακα (χαμηλότερη=καλύτερη)· S: ενεργειακή φτώχεια & ισότητα· G: ποιότητα δεδομένων/μοντέλου (μεθοδολογικό, ~σταθερό). "
        "Σενάρια: Baseline 2018 & SSP2-4.5/SSP5-8.5 στο 2100. Υψηλότερο score = καλύτερη επίδοση ESG.",
        fontsize=10.3,color="#33475b",style="italic",ha="left",va="bottom",bbox=dict(boxstyle="round,pad=0.4",fc="#f2f6fa",ec="#cfe0ee",lw=0.8))
    fig.text(0.012,0.006,"Πηγή: NREL ResStock/ComStock · μεθοδολογία ESG: Boffo and Patalano (2020)· Pedersen et al. (2021)",fontsize=9.3,color=GRAY,ha="left")
    fig.savefig(out,bbox_inches="tight"); plt.close(fig); return out, esg

if __name__=="__main__":
    df=_load()
    o1,cb=fig_pareto(df); print("✓",o1); print(cb[["key","co2_pct","inv","npv"]].round(1).to_string(index=False))
    o2,esg=fig_esg_dashboard(df); print("✓",o2); print(esg[["scenario","E","S","G","ESG"]].round(1).to_string(index=False))
