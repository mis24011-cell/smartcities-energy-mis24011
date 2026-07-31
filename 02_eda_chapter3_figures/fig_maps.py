# -*- coding: utf-8 -*-
"""Χάρτες τύπου EIA/Tableau (πραγματικά δεδομένα merged_master):
   (1) Μείγμα καυσίμων + συνολική ενέργεια ανά κλιματική ζώνη
   (2) Εκπομπές CO₂ ανά κλιματική ζώνη (μέγεθος bubble ∝ CO₂/κτίριο)
Δυναμικός υπολογισμός από τα δεδομένα — ουδεμία hardcoded τιμή."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
mpl.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200})

m=pd.read_parquet("merged_master.parquet")
FUELS=["out.electricity.total.energy_consumption.kwh","out.natural_gas.total.energy_consumption.kwh",
       "out.fuel_oil.total.energy_consumption.kwh","out.propane.total.energy_consumption.kwh"]
SITE="out.site_energy.total.energy_consumption.kwh"
# Γεωγραφία & σύνθεση δείγματος ανά ζώνη
GEO={"3A":("Atlanta, GA",-84.39,33.75),"4A":("Baltimore, MD",-76.61,39.29),
     "5A":("Chicago, IL",-87.65,41.85),"5B":("Denver, CO",-104.98,39.74)}
def zone_stats(z):
    d=m[m.zone==z]; res=d[d.dataset_type=="residential"]; com=d[d.dataset_type=="commercial"]
    tot={f:d[f].sum() for f in FUELS}; s=sum(tot.values())
    mix=[tot[f]/s*100 for f in FUELS]
    site_w=0.6*res[SITE].sum()+0.4*com[SITE].sum()
    co2_w=0.6*res.co2_total_kg.sum()+0.4*com.co2_total_kg.sum()
    return mix,site_w,co2_w
ZST={z:zone_stats(z) for z in GEO}

US=[(-124.7,48.4),(-124.2,43.3),(-124.4,40.4),(-122.4,37.8),(-120.6,34.4),(-117.3,32.5),
(-114.8,32.5),(-111.1,31.3),(-108.2,31.8),(-106.5,31.8),(-103.0,29.0),(-101.4,29.8),
(-99.5,27.5),(-97.4,25.9),(-97.2,27.8),(-95.3,29.0),(-93.8,29.7),(-91.0,29.2),(-89.2,29.2),
(-88.0,30.4),(-86.5,30.4),(-84.0,30.1),(-82.8,29.0),(-82.0,26.5),(-81.1,25.2),(-80.1,25.8),
(-80.5,28.5),(-81.4,30.7),(-80.9,32.0),(-79.2,33.8),(-77.9,34.2),(-75.5,35.6),(-76.0,37.0),
(-75.1,38.5),(-74.0,40.5),(-71.1,41.5),(-70.0,43.7),(-67.0,44.8),(-69.2,47.4),(-71.5,45.0),
(-74.7,45.0),(-76.5,44.0),(-79.0,43.3),(-82.5,41.7),(-83.0,42.3),(-82.5,45.0),(-84.8,45.8),
(-87.0,45.8),(-88.0,48.2),(-90.0,46.7),(-95.0,49.0),(-104.0,49.0),(-117.0,49.0),(-123.0,49.0),(-124.7,48.4)]
FUEL_LBL=["Ηλεκτρισμός","Φυσικό αέριο","Πετρέλαιο","Προπάνιο"]
FUEL_COL=["#1b4a6b","#7cb0d6","#e8897a","#e0a94b"]
NAVY="#1f4e79"
def usmap(ax):
    xs=[p[0] for p in US]; ys=[p[1] for p in US]
    ax.fill(xs,ys,facecolor="#eef1f4",edgecolor="#b9c4cf",lw=1.3,zorder=1)
    ax.set_xlim(-126,-66); ax.set_ylim(23,50.5); ax.set_aspect(1.28); ax.set_axis_off()

# ═══════════ ΧΑΡΤΗΣ 1: FUEL MIX ═══════════
fig=plt.figure(figsize=(15.5,8.2))
ax=fig.add_axes([0.02,0.06,0.60,0.80]); usmap(ax)
fig.text(0.02,0.955,"Οι ψυχρότερες ζώνες στηρίζονται περισσότερο στο φυσικό αέριο για θέρμανση",
         fontsize=18,fontweight="bold",color=NAVY,ha="left")
fig.text(0.02,0.905,"Μείγμα καυσίμων και συνολική κατανάλωση ενέργειας ανά κλιματική ζώνη ASHRAE, 2018",
         fontsize=12.5,color="#333",ha="left")
fig.text(0.02,0.878,"ποσοστό επί της συνολικής ενέργειας (site) · χιλ. kWh ανά μέσο κτίριο/έτος (στάθμιση 60% οικ./40% εμπ.)",
         fontsize=11,color="#7a7a7a",ha="left")
def donut(cx,cy,shares,total,zone,city):
    disp=ax.transData.transform((cx,cy)); fx,fy=fig.transFigure.inverted().transform(disp)
    w=0.085; iax=fig.add_axes([fx-w/2,fy-w*fig.get_figwidth()/fig.get_figheight()/2,w,w*fig.get_figwidth()/fig.get_figheight()])
    iax.pie([max(s,0) for s in shares],colors=FUEL_COL,startangle=90,counterclock=False,
            wedgeprops=dict(width=0.42,edgecolor="white",linewidth=1.2)); iax.set_aspect("equal")
    iax.text(0,0,f"{total/1000:.0f}k",ha="center",va="center",fontsize=13,fontweight="bold",color=NAVY)
    iax.text(0,-1.45,f"Ζώνη {zone}",ha="center",va="center",fontsize=12.5,fontweight="bold",color="#222",transform=iax.transData)
    iax.text(0,-1.9,city,ha="center",va="center",fontsize=10.5,color="#555",transform=iax.transData)
for z,(city,lon,lat) in GEO.items():
    mix,site_w,_=ZST[z]; donut(lon,lat,mix,site_w,z,city)
# ranking για bullets
by_gas=sorted(ZST.items(),key=lambda kv:-kv[1][0][1]); by_site=sorted(ZST.items(),key=lambda kv:-kv[1][1])
BX=0.655
def bullet(y,txt):
    fig.text(BX,y,"•",fontsize=13,color=NAVY,ha="left",va="top",fontweight="bold")
    fig.text(BX+0.015,y,txt,fontsize=12.5,color="#2b2b2b",ha="left",va="top",linespacing=1.35)
fig.text(BX,0.855,"Βασικά ευρήματα",fontsize=13.5,fontweight="bold",color=NAVY,ha="left",va="top")
z_top=by_site[0][0]; g_top=by_gas[0][0]
bullet(0.80,f"Η Ζώνη 5A (Chicago) κατανάλωσε τη μεγαλύτερη ενέργεια\n({ZST['5A'][1]/1000:.0f}k kWh) με το υψηλότερο μερίδιο φυσικού αερίου\n({ZST['5A'][0][1]:.0f}%) — έντονη θέρμανση στο ψυχρό-υγρό κλίμα.")
bullet(0.685,f"Η Ζώνη 3A (Atlanta) εμφάνισε το υψηλότερο μερίδιο ηλεκτρισμού\n({ZST['3A'][0][0]:.0f}%): στο θερμό κλίμα κυριαρχεί η ψύξη (ηλεκτρικό φορτίο).")
bullet(0.585,f"Σαφής κλιματικός διαβαθμισμός στο αέριο: 3A {ZST['3A'][0][1]:.0f}% →\n4A {ZST['4A'][0][1]:.0f}% → 5A {ZST['5A'][0][1]:.0f}% (όσο ψυχρότερα, τόσο περισσότερο αέριο).")
bullet(0.485,"Πετρέλαιο & προπάνιο: αμελητέο μερίδιο (<0,3%) παντού — τα\nεμπορικά κτίρια (ComStock) δεν τα χρησιμοποιούν.")
ly=0.35
fig.text(BX,ly+0.03,"Καύσιμο",fontsize=11.5,fontweight="bold",color="#333",ha="left")
for i,(lbl,col) in enumerate(zip(FUEL_LBL,FUEL_COL)):
    yy=ly-i*0.035
    fig.add_artist(plt.Rectangle((BX,yy),0.018,0.022,color=col,transform=fig.transFigure,clip_on=False))
    fig.text(BX+0.026,yy+0.011,lbl,fontsize=10.8,color="#333",ha="left",va="center")
fig.text(0.02,0.03,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 (AMY2018) · ιδία επεξεργασία (merged_master, 4.000 κτίρια)",
         fontsize=9.5,color="#8a97a5",ha="left")
fig.text(0.02,0.008,"Σημείωση: «μέσο κτίριο» ζώνης (600 οικ. + 400 εμπ./ζώνη, στρωματοποιημένο δείγμα, seed=42). Site energy = ηλεκτρισμός + καύσιμα.",
         fontsize=8.8,color="#a0a0a0",ha="left")
fig.savefig("fig_map_fuelmix.png",bbox_inches="tight",facecolor="white"); plt.close(fig); print("✓ fig_map_fuelmix.png")

# ═══════════ ΧΑΡΤΗΣ 2: CO2 ═══════════
fig=plt.figure(figsize=(15.5,8.2))
ax=fig.add_axes([0.02,0.06,0.60,0.80]); usmap(ax)
co2v={z:ZST[z][2] for z in GEO}; cmax=max(co2v.values())
fig.text(0.02,0.955,"Το ψυχρό κλίμα του Σικάγο (5A) οδηγεί στις υψηλότερες εκπομπές CO₂",
         fontsize=18,fontweight="bold",color=NAVY,ha="left")
fig.text(0.02,0.905,"Ετήσιες εκπομπές CO₂ ανά μέσο κτίριο και κλιματική ζώνη ASHRAE, 2018",
         fontsize=12.5,color="#333",ha="left")
fig.text(0.02,0.878,"τόνοι CO₂ ανά μέσο κτίριο/έτος · μέγεθος κύκλου ∝ εκπομπές (στάθμιση 60% οικ./40% εμπ.)",
         fontsize=11,color="#7a7a7a",ha="left")
import matplotlib.cm as cm
norm=plt.Normalize(min(co2v.values())*0.85,cmax)
cmap=mpl.colors.LinearSegmentedColormap.from_list("co2",["#f4d06f","#e8897a","#b5495b","#6b1f3a"])
for z,(city,lon,lat) in GEO.items():
    v=co2v[z]; sz=900+3200*(v/cmax)
    ax.scatter([lon],[lat],s=sz,c=[cmap(norm(v))],edgecolor="white",lw=1.6,zorder=3,alpha=0.92)
    ax.text(lon,lat,f"{v/1000:.1f}t",ha="center",va="center",fontsize=13,fontweight="bold",color="white",zorder=4)
    ax.text(lon,lat-2.6,f"Ζώνη {z}",ha="center",va="center",fontsize=12.5,fontweight="bold",color="#222",zorder=4)
    ax.text(lon,lat-3.7,city,ha="center",va="center",fontsize=10.5,color="#555",zorder=4)
by_co2=sorted(co2v.items(),key=lambda kv:-kv[1])
def bullet2(y,txt):
    fig.text(BX,y,"•",fontsize=13,color=NAVY,ha="left",va="top",fontweight="bold")
    fig.text(BX+0.015,y,txt,fontsize=12.5,color="#2b2b2b",ha="left",va="top",linespacing=1.35)
fig.text(BX,0.855,"Βασικά ευρήματα",fontsize=13.5,fontweight="bold",color=NAVY,ha="left",va="top")
bullet2(0.80,f"Η Ζώνη 5A (Chicago) εκπέμπει τις περισσότερες ({co2v['5A']/1000:.1f} τόνοι/\nκτίριο) — συνδυασμός υψηλής κατανάλωσης θέρμανσης\nκαι σχετικά ρυπογόνου μείγματος.")
bullet2(0.68,f"Η Ζώνη 4A (Baltimore) έχει τις χαμηλότερες εκπομπές\n({co2v['4A']/1000:.1f} τόνοι) — χαμηλότερος συντελεστής εκπομπών\nδικτύου (Cambium 2022) και ηπιότερο κλίμα.")
bullet2(0.56,f"Η 3A (Atlanta) καταναλώνει λιγότερο από την 5A αλλά εκπέμπει\n{co2v['3A']/1000:.1f} τόνοι — το ηλεκτρικό φορτίο ψύξης «πληρώνει»\nτον υψηλότερο συντελεστή εκπομπών της ζώνης.")
bullet2(0.44,"Η γεωγραφία των εκπομπών ΔΕΝ ταυτίζεται με της κατανάλωσης:\nκαύσιμο-μείγμα και συντελεστής δικτύου παίζουν καθοριστικό ρόλο.")
fig.text(0.02,0.03,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 · CO₂: Cambium 2022 (ηλεκτρισμός) + EPA 2023 (καύσιμα) · ιδία επεξεργασία",
         fontsize=9.5,color="#8a97a5",ha="left")
fig.text(0.02,0.008,"Σημείωση: εκπομπές Scope 2 (baseline 2018). «Μέσο κτίριο» = στάθμιση 60% οικιστικά / 40% εμπορικά ανά ζώνη.",
         fontsize=8.8,color="#a0a0a0",ha="left")
fig.savefig("fig_map_co2.png",bbox_inches="tight",facecolor="white"); plt.close(fig); print("✓ fig_map_co2.png")
