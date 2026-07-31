# -*- coding: utf-8 -*-
"""
Κεφ. 4 — ML Pipeline (ΝΕΟ δείγμα: 5.000 κτίρια / 5 ζώνες).
Πιστή αναπαραγωγή της μεθοδολογίας των notebooks 07/08/09/11 (Αριστοτέλης Ζυγούρης)
στο merged_master.parquet. Μόνη ουσιαστική αλλαγή: zone_enc περιλαμβάνει 2A (Houston).
Παράγει: ch4_outputs/*.csv (Πίν 4.1-4.5), *.json, και figures.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json, pickle, os
from datetime import datetime
import xgboost as xgb, optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE=42; np.random.seed(RANDOM_STATE)
OUT="ch4_outputs"; os.makedirs(OUT,exist_ok=True)

TARGET_ENERGY="out.site_energy.total.energy_consumption.kwh"
TARGET_CO2="co2_total_kg"; COL_ZONE="zone"; COL_BTYPE="btype"; COL_DTYPE="dataset_type"
TRAIN_RATIO=0.70; VAL_RATIO=0.15; N_TRIALS=30
# ΝΕΟ: 5 ζώνες, 2A θερμότερη (θερμό→ψυχρό)
ZONE_ENC={"2A":0,"3A":1,"4A":2,"5A":3,"5B":4}

def compute_metrics(y_true,y_pred,name=""):
    y_t=np.array(y_true,float); y_p=np.array(y_pred,float)
    mask=~(np.isnan(y_t)|np.isnan(y_p)); y_t,y_p=y_t[mask],y_p[mask]
    rmse=np.sqrt(mean_squared_error(y_t,y_p)); mae=mean_absolute_error(y_t,y_p)
    mape=np.mean(np.abs((y_t-y_p)/(y_t+1e-8)))*100; r2=r2_score(y_t,y_p)
    return {"model":name,"RMSE":round(rmse,4),"MAE":round(mae,4),"MAPE_%":round(mape,2),"R2":round(r2,4),"N":int(mask.sum())}

# ── Feature engineering (πιστά από 08 cell 1) ──
df=pd.read_parquet("merged_master.parquet")
if not isinstance(df.index,pd.DatetimeIndex): df.index=pd.to_datetime(df.index)
df=df.sort_index()
idx=df.index
df["hour"]=idx.hour.astype(np.int8); df["day_of_week"]=idx.dayofweek.astype(np.int8)
df["month"]=idx.month.astype(np.int8)
df["season"]=df["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}).astype(np.int8)
df["is_weekend"]=(idx.dayofweek>=5).astype(np.int8); df["is_business"]=(idx.dayofweek<5).astype(np.int8)
df["hour_sin"]=np.sin(2*np.pi*df["hour"]/24); df["hour_cos"]=np.cos(2*np.pi*df["hour"]/24)
df["dow_sin"]=np.sin(2*np.pi*df["day_of_week"]/7); df["dow_cos"]=np.cos(2*np.pi*df["day_of_week"]/7)
df["month_sin"]=np.sin(2*np.pi*(df["month"]-1)/12); df["month_cos"]=np.cos(2*np.pi*(df["month"]-1)/12)
h=df["hour"]; df["is_evening_peak"]=h.between(18,21).astype(np.int8)
df["is_midday_peak"]=h.between(12,15).astype(np.int8); df["is_morning_peak"]=h.between(7,9).astype(np.int8)
df["zone_enc"]=df[COL_ZONE].map(ZONE_ENC).astype(np.int8)
df["is_res"]=(df[COL_DTYPE]=="residential").astype(np.int8); df["is_com"]=(df[COL_DTYPE]=="commercial").astype(np.int8)
df["is_val_zone"]=(df[COL_ZONE]=="5B").astype(np.int8)
gk=[COL_ZONE,COL_BTYPE]
for lag in [1,24,168]:
    df[f"energy_lag_{lag}h"]=df.groupby(gk)[TARGET_ENERGY].shift(lag).astype(np.float32)
df["energy_roll_mean_24h"]=df.groupby(gk)[TARGET_ENERGY].transform(lambda x:x.shift(1).rolling(24,min_periods=1).mean()).astype(np.float32)
df["energy_roll_std_24h"]=df.groupby(gk)[TARGET_ENERGY].transform(lambda x:x.shift(1).rolling(24,min_periods=1).std()).fillna(0).astype(np.float32)
df["intensity_x_is_res"]=(df["emission_intensity"]*df["is_res"]).astype(np.float32)
df["intensity_x_season"]=(df["emission_intensity"]*df["season"]).astype(np.float32)
df["zone_x_season"]=(df["zone_enc"]*df["season"]).astype(np.int8)
df["hour_x_weekend"]=(df["hour"]*df["is_weekend"]).astype(np.int16)
FEATURE_COLS=["hour","day_of_week","month","season","is_weekend","is_business",
 "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
 "is_evening_peak","is_midday_peak","is_morning_peak",
 "zone_enc","is_res","is_com","is_val_zone",
 "out.electricity.total.energy_consumption.kwh","out.natural_gas.total.energy_consumption.kwh","emission_intensity",
 "energy_lag_1h","energy_lag_24h","energy_lag_168h","energy_roll_mean_24h","energy_roll_std_24h",
 "intensity_x_is_res","intensity_x_season","zone_x_season","hour_x_weekend"]
dfc=df.dropna(subset=FEATURE_COLS+[TARGET_ENERGY,TARGET_CO2])
print(f"features={len(FEATURE_COLS)} | rows after dropna={len(dfc):,} | zones={sorted(dfc[COL_ZONE].unique())}")

# ── temporal split 70/15/15, 5B OOS ──
ut=dfc.index.unique().sort_values(); n=len(ut); n_tr=int(n*TRAIN_RATIO); n_val=int(n*VAL_RATIO)
t_tr=ut[n_tr]; t_va=ut[n_tr+n_val]
m_tr=dfc.index<t_tr; m_va=(dfc.index>=t_tr)&(dfc.index<t_va); m_te=dfc.index>=t_va
m5b=dfc[COL_ZONE]=="5B"; m_tr0=m_tr&~m5b
X_train=dfc.loc[m_tr0,FEATURE_COLS]; y_train=dfc.loc[m_tr0,TARGET_ENERGY]; y_train_co2=dfc.loc[m_tr0,TARGET_CO2]
X_val=dfc.loc[m_va&~m5b,FEATURE_COLS]; y_val=dfc.loc[m_va&~m5b,TARGET_ENERGY]
X_test=dfc.loc[m_te&~m5b,FEATURE_COLS]; y_test=dfc.loc[m_te&~m5b,TARGET_ENERGY]; y_test_co2=dfc.loc[m_te&~m5b,TARGET_CO2]
X_test_5b=dfc.loc[m_te&m5b,FEATURE_COLS]; y_test_5b=dfc.loc[m_te&m5b,TARGET_ENERGY]
split_info={"train_end":str(t_tr.date()),"val_end":str(t_va.date()),
 "n_train":len(X_train),"n_val":len(X_val),"n_test":len(X_test),"n_5b_oos":len(X_test_5b)}
print("split:",split_info)

# ── Optuna XGBoost ──
def objective(trial):
    p={"max_depth":trial.suggest_int("max_depth",3,10),
       "learning_rate":trial.suggest_float("learning_rate",0.01,0.3,log=True),
       "n_estimators":trial.suggest_int("n_estimators",100,800),
       "subsample":trial.suggest_float("subsample",0.6,1.0),
       "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
       "min_child_weight":trial.suggest_int("min_child_weight",1,10),
       "reg_alpha":trial.suggest_float("reg_alpha",0.0,1.0),
       "reg_lambda":trial.suggest_float("reg_lambda",1.0,5.0),
       "tree_method":"hist","random_state":RANDOM_STATE,"n_jobs":-1,"verbosity":0}
    mdl=xgb.XGBRegressor(**p); mdl.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
    return np.sqrt(mean_squared_error(y_val,mdl.predict(X_val)))
print(f"Optuna {N_TRIALS} trials...")
study=optuna.create_study(direction="minimize",sampler=TPESampler(seed=RANDOM_STATE))
study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
best=study.best_params; print("best val RMSE",round(study.best_value,4),"| params",best)
fp={**best,"tree_method":"hist","random_state":RANDOM_STATE,"n_jobs":-1,"verbosity":0}

# ── XGBoost models ──
# split res/com μέσω της στήλης is_res (robust σε κοινά timestamps)
mres_tr=(X_train["is_res"]==1).values
mres_va=(X_val["is_res"]==1).values
mres_te=(X_test["is_res"]==1).values
xgb_res=xgb.XGBRegressor(**fp); xgb_res.fit(X_train[mres_tr],y_train[mres_tr],eval_set=[(X_val[mres_va],y_val[mres_va])],verbose=False)
m_res=compute_metrics(y_test[mres_te],xgb_res.predict(X_test[mres_te]),"XGBoost Residential")
xgb_com=xgb.XGBRegressor(**fp); xgb_com.fit(X_train[~mres_tr],y_train[~mres_tr],eval_set=[(X_val[~mres_va],y_val[~mres_va])],verbose=False)
m_com=compute_metrics(y_test[~mres_te],xgb_com.predict(X_test[~mres_te]),"XGBoost Commercial")
xgb_uni=xgb.XGBRegressor(**fp); xgb_uni.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
pred_uni=xgb_uni.predict(X_test); m_uni=compute_metrics(y_test,pred_uni,"XGBoost Unified")
m_5b=compute_metrics(y_test_5b,xgb_uni.predict(X_test_5b),"XGBoost Zone 5B (OOS)")
xgb_co2=xgb.XGBRegressor(**fp); xgb_co2.fit(X_train,y_train_co2,eval_set=[(X_val,dfc.loc[m_va&~m5b,TARGET_CO2])],verbose=False)
m_co2=compute_metrics(y_test_co2,xgb_co2.predict(X_test),"XGBoost CO₂")
xgb_all=[m_res,m_com,m_uni,m_5b,m_co2]
pd.DataFrame(xgb_all).set_index("model").to_csv(f"{OUT}/xgb_results.csv",float_format="%.4f")
print("XGB:",{m["model"]:m["R2"] for m in xgb_all})

# ── Random Forest (09) ──
RF={"n_estimators":300,"max_depth":20,"max_features":"sqrt","min_samples_leaf":5,
    "oob_score":True,"random_state":RANDOM_STATE,"n_jobs":-1}
rf_uni=RandomForestRegressor(**RF); rf_uni.fit(X_train,y_train)
m_rfu=compute_metrics(y_test,rf_uni.predict(X_test),"RF Unified"); m_rfu["OOB_R2"]=round(rf_uni.oob_score_,4)
rf_res=RandomForestRegressor(**RF); rf_res.fit(X_train[mres_tr],y_train[mres_tr])
m_rfr=compute_metrics(y_test[mres_te],rf_res.predict(X_test[mres_te]),"RF Residential"); m_rfr["OOB_R2"]=round(rf_res.oob_score_,4)
rf_com=RandomForestRegressor(**RF); rf_com.fit(X_train[~mres_tr],y_train[~mres_tr])
m_rfc=compute_metrics(y_test[~mres_te],rf_com.predict(X_test[~mres_te]),"RF Commercial"); m_rfc["OOB_R2"]=round(rf_com.oob_score_,4)
m_rf5b=compute_metrics(y_test_5b,rf_uni.predict(X_test_5b),"RF Zone 5B (OOS)")
rf_co2=RandomForestRegressor(**RF); rf_co2.fit(X_train,y_train_co2)
m_rfco2=compute_metrics(y_test_co2,rf_co2.predict(X_test),"RF CO2"); m_rfco2["OOB_R2"]=round(rf_co2.oob_score_,4)
rf_all=[m_rfu,m_rfr,m_rfc,m_rf5b,m_rfco2]
pd.DataFrame(rf_all).set_index("model").to_csv(f"{OUT}/rf_results.csv",float_format="%.4f")
print("RF:",{m["model"]:m["R2"] for m in rf_all})

# ── Baselines: Naive Seasonal + Persistence (unified test) ──
te=dfc.loc[m_te&~m5b].copy()
mask_te0=(m_te&~m5b)
naive168=dfc.groupby(gk)[TARGET_ENERGY].shift(168)[mask_te0].values
naive1=dfc.groupby(gk)[TARGET_ENERGY].shift(1)[mask_te0].values
m_n168=compute_metrics(te[TARGET_ENERGY].values,naive168,"Naive Seasonal (lag=168h)")
m_n1=compute_metrics(te[TARGET_ENERGY].values,naive1,"Naive Persistence (lag=1h)")
print("Naive168:",m_n168["RMSE"],m_n168["R2"])

# ── Prophet (single-series baseline, Zone 3A residential) ──
try:
    from prophet import Prophet
    prim=dfc[(dfc[COL_ZONE]=="3A")&(dfc[COL_DTYPE]=="residential")][TARGET_ENERGY].sort_index()
    ptr=prim[prim.index<t_va]; pte=prim[prim.index>=t_va]
    pm=Prophet(yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=True,interval_width=0.95)
    pm.fit(pd.DataFrame({"ds":ptr.index,"y":ptr.values}))
    fc=pm.predict(pd.DataFrame({"ds":pte.index}))
    m_prophet=compute_metrics(pte.values,fc["yhat"].clip(lower=0).values,"Prophet (Zone 3A)")
except Exception as e:
    m_prophet={"model":"Prophet (Zone 3A)","RMSE":None,"R2":None,"N":0,"err":str(e)[:80]}
print("Prophet:",m_prophet.get("RMSE"),m_prophet.get("R2"))

# ── central comparison (Πίν 4.4) ──
central=[m_n168,m_prophet,m_uni,m_rfu]
pd.DataFrame(central).to_csv(f"{OUT}/final_model_comparison.csv",index=False,float_format="%.4f")
# xgb vs rf (Πίν 4.3 comparison)
cmp=[]
for lab,mx,mr in [("Residential",m_res,m_rfr),("Commercial",m_com,m_rfc),("Unified",m_uni,m_rfu),("CO2",m_co2,m_rfco2)]:
    cmp.append({"Model":lab,"XGB_RMSE":mx["RMSE"],"RF_RMSE":mr["RMSE"],"XGB_MAPE":mx["MAPE_%"],"RF_MAPE":mr["MAPE_%"],
                "XGB_R2":mx["R2"],"RF_R2":mr["R2"],"Winner":"XGBoost" if mx["RMSE"]<mr["RMSE"] else "RF"})
pd.DataFrame(cmp).to_csv(f"{OUT}/xgb_vs_rf_comparison.csv",index=False,float_format="%.4f")

# ── error analysis ανά ζώνη/ώρα/εποχή (για §4.4.4) ──
te["pred"]=pred_uni; te["err"]=te["pred"]-te[TARGET_ENERGY]; te["abs_err"]=te["err"].abs()
te["pct_err"]=te["abs_err"]/(te[TARGET_ENERGY]+1e-8)*100
zone_err=te.groupby(COL_ZONE)["pct_err"].mean().round(2).to_dict()
# add 5B OOS error
te5=dfc.loc[m_te&m5b].copy(); te5["pred"]=xgb_uni.predict(X_test_5b)
te5["pct_err"]=(te5["pred"]-te5[TARGET_ENERGY]).abs()/(te5[TARGET_ENERGY]+1e-8)*100
zone_err["5B"]=round(te5["pct_err"].mean(),2)
resid_mean=round(float(te["err"].mean()),2)
print("zone MAPE:",zone_err,"resid mean:",resid_mean)

# ── SHAP feature importance (11) ──
import shap
SHAP_N=min(2000,len(X_test))
Xs=X_test.sample(SHAP_N,random_state=RANDOM_STATE)
expl=shap.TreeExplainer(xgb_uni); sv=expl.shap_values(Xs)
mshap=pd.Series(np.abs(sv).mean(0),index=FEATURE_COLS).sort_values(ascending=False)
def catof(f):
    if any(k in f for k in ["electricity","natural_gas"]): return "energy_enduse"
    if "emission_intensity" in f and "_x_" not in f: return "energy_enduse"
    if any(k in f for k in ["lag","roll"]): return "lag_rolling"
    if "_x_" in f: return "interaction"
    if any(k in f for k in ["zone","is_res","is_com","is_val"]): return "building"
    return "temporal"
shap_df=pd.DataFrame({"feature":mshap.index,"mean_abs_shap":mshap.values,
    "rank":range(1,len(mshap)+1),"category":[catof(f) for f in mshap.index]})
shap_df.to_csv(f"{OUT}/shap_feature_importance.csv",index=False,float_format="%.6f")
print("SHAP top5:",mshap.head(5).round(3).to_dict())

# ── save everything ──
summary={"generated_at":datetime.now().isoformat(),"split":split_info,"best_params":best,
 "n_trials":N_TRIALS,"features":FEATURE_COLS,"n_features":len(FEATURE_COLS),
 "xgb":{m["model"]:m for m in xgb_all},"rf":{m["model"]:m for m in rf_all},
 "baselines":{m["model"]:m for m in [m_n168,m_n1,m_prophet]},
 "zone_mape":zone_err,"resid_mean":resid_mean,"shap_top10":mshap.head(10).round(4).to_dict()}
json.dump(summary,open(f"{OUT}/ch4_summary.json","w"),indent=2,ensure_ascii=False,default=str)
# save predictions for figures
np.savez(f"{OUT}/preds.npz",
 y_test=y_test.values,pred_uni=pred_uni,
 y_res=y_test[mres_te].values,pred_res=xgb_res.predict(X_test[mres_te]),
 y_com=y_test[~mres_te].values,pred_com=xgb_com.predict(X_test[~mres_te]),
 y_co2=y_test_co2.values,pred_co2=xgb_co2.predict(X_test),
 err=te["err"].values, trials=[t.value for t in study.trials if t.value is not None])
te[[COL_ZONE,"hour","season","pct_err","err"]].to_parquet(f"{OUT}/test_errors.parquet")
print("\n✓ ΟΛΟΚΛΗΡΩΘΗΚΕ. Αρχεία στο",OUT)
