# -*- coding: utf-8 -*-
"""Σχ. 3.11 (νέο): Χαρακτηριστικά εμπορικών κτιρίων ανά ζώνη — τύπος & καύσιμο θέρμανσης.
ΙΔΙΟΥ ΤΥΠΟΥ με το Σχ. 3.4 (οικιστικά): δύο panels stacked-bar, ίδιο στιλ/παλέτα."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})
c=pd.read_parquet("comstock_metadata.parquet")
NAVY="#1f4e79"
S2Z={"TX":"2A","GA":"3A","MD":"4A","IL":"5A","CO":"5B"}; ZONES=["2A","3A","4A","5A","5B"]
c=c[c["in.state"].astype(str).isin(S2Z)].copy(); c["zone"]=c["in.state"].astype(str).map(S2Z)
BT="in.comstock_building_type"; HF="in.heating_fuel"
# ── ομαδοποίηση τύπων κτιρίου ──
def grp(x):
    x=str(x).lower()
    if "office" in x: return "Γραφεία"
    if "retail" in x or "strip" in x: return "Λιανικό εμπόριο"
    if "restaurant" in x: return "Εστίαση"
    if "hotel" in x: return "Ξενοδοχεία"
    if "school" in x or "educat" in x: return "Εκπαίδευση"
    if "hospital" in x or "outpatient" in x or "health" in x: return "Υγεία"
    if "warehouse" in x: return "Αποθήκες"
    return "Άλλο"
c["bt"]=c[BT].map(grp)
# ── καύσιμο θέρμανσης (ίδια λογική με τα οικιστικά) ──
def simp_hf(x):
    x=str(x)
    if "NaturalGas" in x or "Natural Gas" in x: return "Φυσικό αέριο"
    if "Electric" in x: return "Ηλεκτρισμός"
    if "FuelOil" in x or "Fuel Oil" in x: return "Πετρέλαιο"
    if "Propane" in x: return "Προπάνιο"
    if "District" in x: return "Τηλεθέρμανση"
    if "None" in x: return "Χωρίς"
    return "Άλλο"
c["hf"]=c[HF].map(simp_hf)

fig,ax=plt.subplots(1,2,figsize=(15,5.6))
fig.suptitle("Γραφεία, λιανικό εμπόριο και αποθήκες κυριαρχούν· το φυσικό αέριο για θέρμανση αυξάνεται στις ψυχρές ζώνες",
             fontsize=14.5,fontweight="bold",color=NAVY,x=0.012,ha="left",y=1.01)
# ── Panel 1: τύπος κτιρίου ──
BTC={"Γραφεία":"#2e5077","Λιανικό εμπόριο":"#7cb0d6","Αποθήκες":"#8a6a3a","Εστίαση":"#c0435a",
     "Εκπαίδευση":"#3a7a6a","Υγεία":"#e0a94b","Ξενοδοχεία":"#9b8ec4","Άλλο":"#bbb"}
bt_order=["Γραφεία","Λιανικό εμπόριο","Αποθήκες","Εστίαση","Εκπαίδευση","Υγεία","Ξενοδοχεία","Άλλο"]
a=ax[0]; bottom=np.zeros(len(ZONES))
for b in bt_order:
    vals=[(c[(c.zone==z)&(c.bt==b)].shape[0]/max(c[c.zone==z].shape[0],1)*100) for z in ZONES]
    if sum(vals)<0.5: continue
    a.bar(ZONES,vals,bottom=bottom,label=b,color=BTC[b],edgecolor="white",lw=0.7); bottom+=np.array(vals)
a.set_title("Τύπος κτιρίου",fontsize=12.5,fontweight="bold"); a.set_ylabel("Μερίδιο (%)"); a.set_ylim(0,100)
a.legend(fontsize=9,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.30),ncol=4)
# ── Panel 2: καύσιμο θέρμανσης (ίδια παλέτα με τα οικιστικά) ──
HFC={"Φυσικό αέριο":"#7cb0d6","Ηλεκτρισμός":"#2e5077","Πετρέλαιο":"#e8897a","Προπάνιο":"#e0a94b",
     "Τηλεθέρμανση":"#3a7a6a","Χωρίς":"#ccc","Άλλο":"#999"}
hf_order=["Φυσικό αέριο","Ηλεκτρισμός","Πετρέλαιο","Προπάνιο","Τηλεθέρμανση","Χωρίς","Άλλο"]
a=ax[1]; bottom=np.zeros(len(ZONES))
for h in hf_order:
    vals=[(c[(c.zone==z)&(c.hf==h)].shape[0]/max(c[c.zone==z].shape[0],1)*100) for z in ZONES]
    if sum(vals)<0.5: continue
    a.bar(ZONES,vals,bottom=bottom,label=h,color=HFC[h],edgecolor="white",lw=0.7); bottom+=np.array(vals)
a.set_title("Καύσιμο θέρμανσης",fontsize=12.5,fontweight="bold"); a.set_ylabel("Μερίδιο (%)"); a.set_ylim(0,100)
a.legend(fontsize=9,frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.30),ncol=5)
fig.tight_layout(rect=[0,0.06,1,0.96])
fig.text(0.012,0.005,"Πηγή: NREL ComStock 2023 R2 (baseline metadata) · 5 πολιτείες (TX/GA/MD/IL/CO) · ιδία επεξεργασία",fontsize=9,color="#8a97a5")
fig.savefig("fig_stock_com.png",bbox_inches="tight",facecolor="white"); print("✓ fig_stock_com.png")
for z in ZONES:
    d=c[c.zone==z]
    print(f"  {z}: γραφεία {(d.bt=='Γραφεία').mean()*100:.0f}% · αέριο θέρμ. {(d.hf=='Φυσικό αέριο').mean()*100:.0f}%")
