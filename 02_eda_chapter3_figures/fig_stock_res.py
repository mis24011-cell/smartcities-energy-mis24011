# -*- coding: utf-8 -*-
"""Σχ. 3.4 (νέο): Χαρακτηριστικά οικιστικών κτιρίων ανά ζώνη — τύπος & καύσιμο θέρμανσης."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
r=pd.read_parquet("resstock_metadata.parquet")
NAVY="#1f4e79"
S2Z={"TX":"2A","GA":"3A","MD":"4A","IL":"5A","CO":"5B"}; ZONES=["2A","3A","4A","5A","5B"]
r=r[r["in.state"].isin(S2Z)].copy(); r["zone"]=r["in.state"].map(S2Z)
BT="in.geometry_building_type_recs"; HF="in.heating_fuel"
# απλοποίηση τύπων
def simp_bt(x):
    x=str(x)
    if "Single" in x: return "Μονοκατοικία"
    if "Mobile" in x: return "Κινητή κατοικία"
    if "Multi" in x or "2 Unit" in x or "3 or 4" in x or "5 to 9" in x or "10 to" in x or "20 to" in x or "50 or" in x: return "Πολυκατοικία"
    return "Άλλο"
r["bt"]=r[BT].map(simp_bt)
def simp_hf(x):
    x=str(x)
    if "Natural Gas" in x: return "Φυσικό αέριο"
    if "Electric" in x: return "Ηλεκτρισμός"
    if "Fuel Oil" in x: return "Πετρέλαιο"
    if "Propane" in x: return "Προπάνιο"
    if "None" in x: return "Χωρίς"
    return "Άλλο"
r["hf"]=r[HF].map(simp_hf)

fig,ax=plt.subplots(1,2,figsize=(15,5.6))
fig.suptitle("Οι μονοκατοικίες κυριαρχούν παντού· το φυσικό αέριο για θέρμανση αυξάνεται στις ψυχρές ζώνες",
             fontsize=14.5,fontweight="bold",color=NAVY,x=0.012,ha="left",y=1.01)
# Panel 1: τύπος κτιρίου
BTC={"Μονοκατοικία":"#2e5077","Πολυκατοικία":"#7cb0d6","Κινητή κατοικία":"#e0a94b","Άλλο":"#bbb"}
bt_order=["Μονοκατοικία","Πολυκατοικία","Κινητή κατοικία","Άλλο"]
a=ax[0]; bottom=np.zeros(len(ZONES))
for b in bt_order:
    vals=[(r[(r.zone==z)&(r.bt==b)].shape[0]/max(r[r.zone==z].shape[0],1)*100) for z in ZONES]
    a.bar(ZONES,vals,bottom=bottom,label=b,color=BTC[b],edgecolor="white",lw=0.7); bottom+=np.array(vals)
a.set_title("Τύπος κτιρίου",fontsize=12.5,fontweight="bold"); a.set_ylabel("Μερίδιο (%)"); a.set_ylim(0,100)
a.legend(fontsize=9.5,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.26),ncol=4)
# Panel 2: καύσιμο θέρμανσης
HFC={"Φυσικό αέριο":"#7cb0d6","Ηλεκτρισμός":"#2e5077","Πετρέλαιο":"#e8897a","Προπάνιο":"#e0a94b","Χωρίς":"#ccc","Άλλο":"#999"}
hf_order=["Φυσικό αέριο","Ηλεκτρισμός","Πετρέλαιο","Προπάνιο","Χωρίς","Άλλο"]
a=ax[1]; bottom=np.zeros(len(ZONES))
for h in hf_order:
    vals=[(r[(r.zone==z)&(r.hf==h)].shape[0]/max(r[r.zone==z].shape[0],1)*100) for z in ZONES]
    if sum(vals)<0.5: continue
    a.bar(ZONES,vals,bottom=bottom,label=h,color=HFC[h],edgecolor="white",lw=0.7); bottom+=np.array(vals)
a.set_title("Καύσιμο θέρμανσης",fontsize=12.5,fontweight="bold"); a.set_ylabel("Μερίδιο (%)"); a.set_ylim(0,100)
a.legend(fontsize=9.5,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.26),ncol=5)
fig.tight_layout(rect=[0,0.06,1,0.96])
fig.text(0.012,0.005,"Πηγή: NREL ResStock 2024 R2 (baseline metadata) · 5 πολιτείες (TX/GA/MD/IL/CO) · ιδία επεξεργασία",fontsize=9,color="#8a97a5")
fig.savefig("fig_stock_res.png",bbox_inches="tight",facecolor="white"); print("✓ fig_stock_res.png")
for z in ZONES:
    d=r[r.zone==z]; gas=(d.hf=="Φυσικό αέριο").mean()*100
    print(f"  {z}: αέριο θέρμ. {gas:.0f}% · μονοκατοικίες {(d.bt=='Μονοκατοικία').mean()*100:.0f}%")
