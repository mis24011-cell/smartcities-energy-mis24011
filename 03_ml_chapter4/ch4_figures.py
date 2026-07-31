# -*- coding: utf-8 -*-
"""Κεφ.4 figures — ξαναχτίζει features, επανεκπαιδεύει με best_params (χωρίς Optuna),
παράγει όλα τα διαγράμματα. Στιλ: navy, ελληνικά labels."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json, matplotlib as mpl, matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
NAVY="#1f3a56"; BLUE="#2E75B6"; ORANGE="#ED7D31"; GREEN="#2E9E4F"; RED="#C00000"; TEAL="#2e6b5e"
RS=42; np.random.seed(RS)
S=json.load(open("ch4_outputs/ch4_summary.json")); BEST=S["best_params"]; FC=S["features"]
TE="out.site_energy.total.energy_consumption.kwh"; TC="co2_total_kg"
ZE={"2A":0,"3A":1,"4A":2,"5A":3,"5B":4}; ZONES=["2A","3A","4A","5A","5B"]

# ── rebuild features (ίδιο με ch4_run) ──
df=pd.read_parquet("merged_master.parquet").sort_index(); idx=df.index
df["hour"]=idx.hour;df["day_of_week"]=idx.dayofweek;df["month"]=idx.month
df["season"]=df["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
df["is_weekend"]=(idx.dayofweek>=5).astype(int);df["is_business"]=(idx.dayofweek<5).astype(int)
df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24);df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
df["dow_sin"]=np.sin(2*np.pi*df["day_of_week"]/7);df["dow_cos"]=np.cos(2*np.pi*df["day_of_week"]/7)
df["month_sin"]=np.sin(2*np.pi*(df["month"]-1)/12);df["month_cos"]=np.cos(2*np.pi*(df["month"]-1)/12)
h=df["hour"];df["is_evening_peak"]=h.between(18,21).astype(int);df["is_midday_peak"]=h.between(12,15).astype(int);df["is_morning_peak"]=h.between(7,9).astype(int)
df["zone_enc"]=df["zone"].map(ZE);df["is_res"]=(df["dataset_type"]=="residential").astype(int);df["is_com"]=(df["dataset_type"]=="commercial").astype(int);df["is_val_zone"]=(df["zone"]=="5B").astype(int)
gk=["zone","btype"]
for lag in [1,24,168]: df[f"energy_lag_{lag}h"]=df.groupby(gk)[TE].shift(lag)
df["energy_roll_mean_24h"]=df.groupby(gk)[TE].transform(lambda x:x.shift(1).rolling(24,min_periods=1).mean())
df["energy_roll_std_24h"]=df.groupby(gk)[TE].transform(lambda x:x.shift(1).rolling(24,min_periods=1).std()).fillna(0)
df["intensity_x_is_res"]=df["emission_intensity"]*df["is_res"];df["intensity_x_season"]=df["emission_intensity"]*df["season"]
df["zone_x_season"]=df["zone_enc"]*df["season"];df["hour_x_weekend"]=df["hour"]*df["is_weekend"]
dfc=df.dropna(subset=FC+[TE,TC])
ut=dfc.index.unique().sort_values();n=len(ut);t_tr=ut[int(n*.7)];t_va=ut[int(n*.7)+int(n*.15)]
m_tr=dfc.index<t_tr;m_va=(dfc.index>=t_tr)&(dfc.index<t_va);m_te=dfc.index>=t_va;m5b=dfc["zone"]=="5B"
tr0=m_tr&~m5b
Xtr,ytr=dfc.loc[tr0,FC],dfc.loc[tr0,TE];Xva,yva=dfc.loc[m_va&~m5b,FC],dfc.loc[m_va&~m5b,TE]
Xte=dfc.loc[m_te&~m5b,FC];yte=dfc.loc[m_te&~m5b,TE]
fp={**BEST,"tree_method":"hist","random_state":RS,"n_jobs":-1,"verbosity":0}
uni=xgb.XGBRegressor(**fp);uni.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
print("retrained unified R2:",round(uni.score(Xte,yte),4))

# helper: predict a single group's test week
def group_test(zone,dtype):
    g=dfc[(dfc.zone==zone)&(dfc.dataset_type==dtype)&m_te]
    return g.index, g[TE].values, uni.predict(g[FC])

# ════ image19: Optuna history + best params ════
tr=np.array(S.get("_trials") or json.load(open("ch4_outputs/ch4_summary.json")).get("_trials") or [])
trials=np.load("ch4_outputs/preds.npz",allow_pickle=True)["trials"].astype(float)
fig,(a1,a2)=plt.subplots(1,2,figsize=(14,4.6))
a1.plot(range(1,len(trials)+1),trials,"o-",color=BLUE,ms=4,lw=1,alpha=.7)
best_so=np.minimum.accumulate(trials);a1.plot(range(1,len(trials)+1),best_so,color=RED,lw=2,label=f"Best: {trials.min():.3f}")
a1.set_xlabel("Δοκιμή (trial)");a1.set_ylabel("Val RMSE (kWh)");a1.set_title("Ιστορικό βελτιστοποίησης Optuna (30 trials)",fontweight="bold",color=NAVY);a1.legend()
a2.axis("off");rows=[("max_depth",BEST["max_depth"]),("learning_rate",f'{BEST["learning_rate"]:.3f}'),("n_estimators",BEST["n_estimators"]),("subsample",f'{BEST["subsample"]:.2f}'),("colsample_bytree",f'{BEST["colsample_bytree"]:.2f}'),("min_child_weight",BEST["min_child_weight"]),("reg_alpha",f'{BEST["reg_alpha"]:.2f}'),("reg_lambda",f'{BEST["reg_lambda"]:.2f}')]
tb=a2.table(cellText=[[k,str(v)] for k,v in rows],colLabels=["Υπερπαράμετρος","Βέλτιστη τιμή"],loc="center",cellLoc="left")
tb.auto_set_font_size(False);tb.set_fontsize(11);tb.scale(1,1.6)
for (r,c),cell in tb.get_celld().items():
    if r==0: cell.set_facecolor(NAVY);cell.set_text_props(color="white",fontweight="bold")
a2.set_title("Βέλτιστες υπερπαράμετροι XGBoost",fontweight="bold",color=NAVY)
fig.tight_layout();fig.savefig("ch4_fig_optuna.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ optuna")

# ════ image20: XGBoost forecast (actual vs pred, 1 week) + scatter ════
fig,ax=plt.subplots(1,2,figsize=(15,5))
ti,ya,yp=group_test("3A","residential")
a=ax[0];a.plot(ti[:168],ya[:168],color="#333",lw=1.6,label="Πραγματικό",zorder=3)
a.plot(ti[:168],yp[:168],color=RED,lw=1.3,ls="--",label="Πρόβλεψη XGBoost",alpha=.85)
a.set_title("Πρόβλεψη XGBoost — Ζώνη 3A οικιστικά, 1 εβδομάδα test",fontweight="bold",color=NAVY);a.set_ylabel("Ενέργεια (kWh)");a.legend();a.tick_params(axis="x",rotation=25)
a=ax[1];samp=np.random.choice(len(yte),3000,replace=False);a.scatter(yte.values[samp],uni.predict(Xte)[samp],s=6,alpha=.3,color=BLUE)
lim=[0,max(yte.max(),1)];a.plot(lim,lim,color=RED,lw=1.5,ls="--");a.set_xlabel("Πραγματικό (kWh)");a.set_ylabel("Πρόβλεψη (kWh)")
a.set_title(f"Πραγματικό vs Πρόβλεψη (Unified, R²={S['xgb']['XGBoost Unified']['R2']})",fontweight="bold",color=NAVY)
fig.tight_layout();fig.savefig("ch4_fig_xgb_forecast.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ xgb forecast")

# ════ image21: RF — actual vs pred + OOB vs test ════
rf=RandomForestRegressor(n_estimators=300,max_depth=20,max_features="sqrt",min_samples_leaf=5,oob_score=True,random_state=RS,n_jobs=-1)
rf.fit(Xtr,ytr)
fig,ax=plt.subplots(1,2,figsize=(15,5))
a=ax[0];yp_rf=rf.predict(Xte);samp=np.random.choice(len(yte),3000,replace=False)
a.scatter(yte.values[samp],yp_rf[samp],s=6,alpha=.3,color=GREEN);a.plot(lim,lim,color=RED,lw=1.5,ls="--")
a.set_xlabel("Πραγματικό (kWh)");a.set_ylabel("Πρόβλεψη RF (kWh)");a.set_title(f"Random Forest — Πραγματικό vs Πρόβλεψη (R²={S['rf']['RF Unified']['R2']})",fontweight="bold",color=NAVY)
a=ax[1];labels=["Unified","Residential","Commercial","CO₂"]
xr2=[S["xgb"]["XGBoost Unified"]["R2"],S["xgb"]["XGBoost Residential"]["R2"],S["xgb"]["XGBoost Commercial"]["R2"],S["xgb"]["XGBoost CO₂"]["R2"]]
rr2=[S["rf"]["RF Unified"]["R2"],S["rf"]["RF Residential"]["R2"],S["rf"]["RF Commercial"]["R2"],S["rf"]["RF CO2"]["R2"]]
x=np.arange(len(labels));w=.36
a.bar(x-w/2,xr2,w,label="XGBoost",color=BLUE);a.bar(x+w/2,rr2,w,label="Random Forest",color=GREEN)
a.set_xticks(x);a.set_xticklabels(labels);a.set_ylim(.95,1.0);a.set_ylabel("R²");a.set_title("Σύγκριση XGBoost vs Random Forest (R²)",fontweight="bold",color=NAVY);a.legend()
fig.tight_layout();fig.savefig("ch4_fig_rf.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ rf")

# ════ image24: model comparison (RMSE/MAPE/R²) ════
comp=pd.read_csv("ch4_outputs/final_model_comparison.csv")
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
names=[m.replace(" (lag=168h)","").replace(" (Zone 3A)","") for m in comp["model"]]
cols=[ORANGE,"#7030A0",BLUE,GREEN]
for k,(met,ttl) in enumerate([("RMSE","RMSE (kWh) ↓"),("MAPE_%","MAPE (%) ↓"),("R2","R² ↑")]):
    a=ax[k];v=comp[met].values;a.bar(range(len(v)),v,color=cols,alpha=.9,edgecolor="white")
    a.set_xticks(range(len(v)));a.set_xticklabels(names,rotation=20,ha="right",fontsize=9)
    a.set_title(ttl,fontweight="bold",color=NAVY)
    for i,val in enumerate(v): a.text(i,val,f"{val:.2f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
    if met=="R2": a.set_ylim(min(0,v.min())-.2,1.05)
fig.suptitle("Συγκριτική αξιολόγηση μοντέλων — Test Set",fontsize=14,fontweight="bold",color=NAVY)
fig.tight_layout();fig.savefig("ch4_fig_comparison.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ comparison")

# ════ image25: error analysis (hour/zone/season + residual hist) ════
te=pd.read_parquet("ch4_outputs/test_errors.parquet")
fig,ax=plt.subplots(2,2,figsize=(14,9))
he=te.groupby("hour")["pct_err"].mean();ax[0,0].bar(he.index,he.values,color=BLUE,alpha=.85);ax[0,0].set_title("Μέσο MAPE ανά ώρα ημέρας",fontweight="bold",color=NAVY);ax[0,0].set_xlabel("Ώρα");ax[0,0].set_ylabel("MAPE (%)")
zmape=S["zone_mape"];zk=["2A","3A","4A","5A","5B"];zv=[zmape[z] for z in zk];zc=[BLUE if z!="5B" else RED for z in zk]
ax[0,1].bar(zk,zv,color=zc,alpha=.85);ax[0,1].set_title("Μέσο MAPE ανά κλιματική ζώνη (5B = OOS)",fontweight="bold",color=NAVY);ax[0,1].set_ylabel("MAPE (%)")
for i,v in enumerate(zv): ax[0,1].text(i,v,f"{v:.1f}%",ha="center",va="bottom",fontsize=9,fontweight="bold")
sl={0:"Χειμώνας",1:"Άνοιξη",2:"Καλοκαίρι",3:"Φθινόπωρο"};te["sl"]=te["season"].map(sl)
se=te.groupby("sl")["pct_err"].mean();ax[1,0].bar(se.index,se.values,color=GREEN,alpha=.85);ax[1,0].set_title("Μέσο MAPE ανά εποχή",fontweight="bold",color=NAVY);ax[1,0].set_ylabel("MAPE (%)")
ax[1,1].hist(te["err"].clip(-3,3),bins=60,color=BLUE,alpha=.75,edgecolor="white");ax[1,1].axvline(0,color=RED,lw=1.5,ls="--")
ax[1,1].set_title(f"Κατανομή σφαλμάτων (mean={S['resid_mean']} kWh)",fontweight="bold",color=NAVY);ax[1,1].set_xlabel("Σφάλμα (kWh)")
fig.suptitle("Ανάλυση σφαλμάτων — XGBoost Unified",fontsize=14,fontweight="bold",color=NAVY)
fig.tight_layout();fig.savefig("ch4_fig_error.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ error")

# ════ image26: SHAP global importance (top 15) ════
sh=pd.read_csv("ch4_outputs/shap_feature_importance.csv").head(15).iloc[::-1]
CATC={"lag_rolling":BLUE,"energy_enduse":"#8a4b3a","building":GREEN,"interaction":"#e0a94b","temporal":ORANGE}
fig,ax=plt.subplots(figsize=(11,7))
ax.barh(range(len(sh)),sh["mean_abs_shap"],color=[CATC.get(c,"#888") for c in sh["category"]],alpha=.9)
ax.set_yticks(range(len(sh)));ax.set_yticklabels(sh["feature"],fontsize=10);ax.set_xlabel("Μέσο |SHAP| (επίδραση στην πρόβλεψη)")
ax.set_title("Σημασία χαρακτηριστικών (SHAP) — Top 15",fontweight="bold",color=NAVY)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v,label=k) for k,v in CATC.items()],fontsize=9,loc="lower right")
fig.tight_layout();fig.savefig("ch4_fig_shap.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ shap")

# ════ image27/28: SHAP cumulative + category breakdown ════
shall=pd.read_csv("ch4_outputs/shap_feature_importance.csv")
fig,ax=plt.subplots(1,2,figsize=(15,5))
cum=shall["mean_abs_shap"].cumsum()/shall["mean_abs_shap"].sum()*100
ax[0].plot(range(1,len(cum)+1),cum,color=BLUE,lw=2,marker="o",ms=3);ax[0].axhline(95,color=RED,ls="--",lw=1,label="95%")
n95=int((cum<95).sum())+1;ax[0].axvline(n95,color=GREEN,ls=":",lw=1.5,label=f"{n95} features → 95%")
ax[0].set_xlabel("Αριθμός χαρακτηριστικών");ax[0].set_ylabel("Αθροιστική σημασία (%)");ax[0].set_title("Αθροιστική σημασία χαρακτηριστικών",fontweight="bold",color=NAVY);ax[0].legend()
cat=shall.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=False)
catlbl={"lag_rolling":"Lag/Rolling","temporal":"Χρονικά","energy_enduse":"Ενέργεια","building":"Κτίριο","interaction":"Αλληλεπιδράσεις"}
ax[1].bar([catlbl.get(c,c) for c in cat.index],cat.values,color=[CATC.get(c,"#888") for c in cat.index],alpha=.9)
ax[1].set_ylabel("Άθροισμα |SHAP|");ax[1].set_title("Σημασία ανά κατηγορία χαρακτηριστικών",fontweight="bold",color=NAVY);ax[1].tick_params(axis="x",rotation=15)
fig.tight_layout();fig.savefig("ch4_fig_shap_cat.png",bbox_inches="tight",facecolor="white");plt.close();print("✓ shap cat")
print("CORE FIGURES DONE")
