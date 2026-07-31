# -*- coding: utf-8 -*-
"""
Κεφ. 4 — ML Pipeline (νέο δείγμα: 5.000 κτίρια, 5 ζώνες).
Ακολουθεί τη μεθοδολογία του project (Dataset_and_Model_Selection_Analysis,
Notebook_Pipeline_Map): Naive Seasonal + Prophet (baseline), XGBoost (Optuna),
Random Forest (validation). Μοντέλα: Residential / Commercial / Unified / CO2.
Χρονολογικός split (Ιαν–Σεπ train, Οκτ–15Νοε val, 15Νοε–Δεκ test)· 5B out-of-sample.
Features (§3.4): temporal, climate (temp/HDD/CDD από πραγματικές ζωνικές βαθμοημέρες),
energy (electricity/gas/emission_intensity), lag (1h/24h/168h), rolling, interactions.
"""
import pandas as pd, numpy as np, json, warnings
from scipy.optimize import brentq
warnings.filterwarnings("ignore")
np.random.seed(42)

TC=18.0
SITE="out.site_energy.total.energy_consumption.kwh"
ELEC="out.electricity.total.energy_consumption.kwh"
GAS="out.natural_gas.total.energy_consumption.kwh"
# Πραγματικές ASHRAE βαθμοημέρες ανά ζώνη (από §3.6 / eda_summary)
REAL_DD={"2A":(1400,2889),"3A":(2827,1810),"4A":(4654,1112),"5A":(6311,842),"5B":(6128,696)}
ZONE_ENC={"2A":0,"3A":1,"4A":2,"5A":3,"5B":4}

# ---------- 1) Συνθετική ωριαία θερμοκρασία ανά ζώνη (ERA5-consistent) ----------
def zone_temp_series(idx, zone):
    """Ωριαία θερμοκρασία 2018 που αναπαράγει τις πραγματικές HDD/CDD της ζώνης."""
    H,C=REAL_DD[zone]
    mu=TC-(H-C)/365.0                      # μέση ετήσια (ταυτότητα βαθμοημερών)
    doy=idx.dayofyear.values; hour=idx.hour.values
    # ετήσιος κύκλος: ψυχρότερα ~15 Ιαν, θερμότερα ~15 Ιουλ
    ann=-np.cos(2*np.pi*(doy-15)/365.0)
    diur=np.cos(2*np.pi*(hour-15)/24.0)     # αιχμή ~15:00
    def hdd_for_A(A,D=5.0):
        T=mu+A*ann+D*diur
        Td=pd.Series(T,index=idx).resample("D").mean()
        return np.maximum(0,TC-Td).sum()
    # calibrate ετήσιο πλάτος A ώστε να αναπαράγει το πραγματικό HDD
    try: A=brentq(lambda a: hdd_for_A(a)-H, 0.5, 30.0)
    except Exception: A=abs(mu-TC)+8.0
    T=mu+A*ann+5.0*diur
    return T

# ---------- 2) Feature engineering ----------
def build_features(df):
    df=df.sort_index().copy()
    out=[]
    for (z,t),g in df.groupby(["zone","dataset_type"]):
        g=g.sort_index().copy()
        idx=g.index
        g["temp"]=zone_temp_series(idx,z)
        g["hdd_h"]=np.maximum(0,TC-g["temp"]); g["cdd_h"]=np.maximum(0,g["temp"]-TC)
        g["temp_roll24"]=g["temp"].rolling(24,min_periods=1).mean()
        # temporal
        g["hour"]=idx.hour; g["dow"]=idx.dayofweek; g["month"]=idx.month; g["doy"]=idx.dayofyear
        g["is_weekend"]=(g["dow"]>=5).astype(int)
        g["hour_sin"]=np.sin(2*np.pi*g["hour"]/24); g["hour_cos"]=np.cos(2*np.pi*g["hour"]/24)
        g["doy_sin"]=np.sin(2*np.pi*g["doy"]/365); g["doy_cos"]=np.cos(2*np.pi*g["doy"]/365)
        g["dow_sin"]=np.sin(2*np.pi*g["dow"]/7); g["dow_cos"]=np.cos(2*np.pi*g["dow"]/7)
        g["season"]=(g["month"]%12//3)  # 0 χειμ,1 άνοιξη,2 καλοκ,3 φθιν
        # building
        g["is_residential"]=(t=="residential").astype(int)
        g["zone_enc"]=ZONE_ENC[z]; g["is_val_zone"]=(z=="5B").astype(int)
        # energy (documented §3.4.5)
        g["electricity"]=g[ELEC]; g["natural_gas"]=g[GAS]  # emission_intensity ήδη στήλη
        # lag/rolling του target (εντός group)
        y=g[SITE]
        g["lag_1h"]=y.shift(1); g["lag_24h"]=y.shift(24); g["lag_168h"]=y.shift(168)
        g["roll_mean_24h"]=y.shift(1).rolling(24,min_periods=1).mean()
        g["roll_std_24h"]=y.shift(1).rolling(24,min_periods=1).std()
        # interactions
        g["intensity_x_is_res"]=g["emission_intensity"]*g["is_residential"]
        g["temp_x_is_res"]=g["temp"]*g["is_residential"]
        g["hour_x_dow"]=g["hour"]*g["dow"]
        g["zone_x_season"]=g["zone_enc"]*4+g["season"]
        out.append(g)
    return pd.concat(out).sort_index()

FEATURES=["hour","dow","month","is_weekend","hour_sin","hour_cos","doy_sin","doy_cos",
 "dow_sin","dow_cos","season","temp","hdd_h","cdd_h","temp_roll24",
 "is_residential","zone_enc","is_val_zone","electricity","natural_gas","emission_intensity",
 "lag_1h","lag_24h","lag_168h","roll_mean_24h","roll_std_24h",
 "intensity_x_is_res","temp_x_is_res","hour_x_dow","zone_x_season"]

# ---------- 3) split ----------
def add_split(df):
    d=df.index
    split=np.where(df["zone"]=="5B","oos",
           np.where(d< "2018-10-01","train",
           np.where(d< "2018-11-15","val","test")))
    df=df.copy(); df["split"]=split; return df

# ---------- 4) metrics ----------
def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    rmse=float(np.sqrt(np.mean((y-p)**2))); mae=float(np.mean(np.abs(y-p)))
    m=y>np.nanpercentile(y,10)  # MAPE σε μη-αμελητέες τιμές
    mape=float(np.mean(np.abs((y[m]-p[m])/y[m]))*100)
    ss=np.sum((y-y.mean())**2); r2=float(1-np.sum((y-p)**2)/ss) if ss>0 else float("nan")
    return dict(RMSE=round(rmse,2),MAE=round(mae,2),MAPE=round(mape,2),R2=round(r2,4))

if __name__=="__main__":
    m=pd.read_parquet("merged_master.parquet")
    print("loaded",m.shape)
    fe=build_features(m); fe=add_split(fe)
    fe=fe.dropna(subset=["lag_168h","roll_std_24h"])  # πρώτες 168h κάθε group
    fe.to_parquet("ch4_features.parquet")
    print("features:",fe.shape,"| split counts:",fe["split"].value_counts().to_dict())
    # γρήγορος έλεγχος θερμοκρασιών ανά ζώνη
    chk=fe.groupby("zone").agg(temp_mean=("temp","mean"),
        hdd=("hdd_h",lambda s:s.sum()/ (len(s)/ (len(s)))),  # placeholder
        )
    print("temp mean ανά ζώνη:", fe.groupby("zone")["temp"].mean().round(1).to_dict())
