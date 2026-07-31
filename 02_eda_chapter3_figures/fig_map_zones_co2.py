# -*- coding: utf-8 -*-
"""HERO CO2: υπόβαθρο κλιματικών ζωνών + bubbles εκπομπών CO2 ανά πόλη (merged_master)."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
mpl.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200})
NAVY="#1f4e79"
m=pd.read_parquet("merged_master.parquet")
SITE="out.site_energy.total.energy_consumption.kwh"
def co2z(z):
    d=m[m.zone==z]; res=d[d.dataset_type=="residential"]; com=d[d.dataset_type=="commercial"]
    return 0.6*res.co2_total_kg.sum()+0.4*com.co2_total_kg.sum()
US=[(-124.7,48.4),(-124.2,43.3),(-124.4,40.4),(-122.4,37.8),(-120.6,34.4),(-117.3,32.5),
(-114.8,32.5),(-111.1,31.3),(-108.2,31.8),(-106.5,31.8),(-103.0,29.0),(-101.4,29.8),
(-99.5,27.5),(-97.4,25.9),(-97.2,27.8),(-95.3,29.0),(-93.8,29.7),(-91.0,29.2),(-89.2,29.2),
(-88.0,30.4),(-86.5,30.4),(-84.0,30.1),(-82.8,29.0),(-82.0,26.5),(-81.1,25.2),(-80.1,25.8),
(-80.5,28.5),(-81.4,30.7),(-80.9,32.0),(-79.2,33.8),(-77.9,34.2),(-75.5,35.6),(-76.0,37.0),
(-75.1,38.5),(-74.0,40.5),(-71.1,41.5),(-70.0,43.7),(-67.0,44.8),(-69.2,47.4),(-71.5,45.0),
(-74.7,45.0),(-76.5,44.0),(-79.0,43.3),(-82.5,41.7),(-83.0,42.3),(-82.5,45.0),(-84.8,45.8),
(-87.0,45.8),(-88.0,48.2),(-90.0,46.7),(-95.0,49.0),(-104.0,49.0),(-117.0,49.0),(-123.0,49.0),(-124.7,48.4)]
def elev_bump(lon,lat):
    if -114<lon<-103 and lat>33: return 3.6
    if -122<lon<-114 and lat>37: return 1.8
    return 0.0
def zone_of(lon,lat):
    e=lat+elev_bump(lon,lat); return 2 if e<31 else 3 if e<35.5 else 4 if e<40.5 else 5 if e<44 else 6
ZCOL={2:"#c0435a",3:"#e08a4f",4:"#edc84e",5:"#7ab0d4",6:"#4574a8"}
lons=np.linspace(-125,-66,340); lats=np.linspace(24,49.5,190)
LON,LAT=np.meshgrid(lons,lats); Z=np.vectorize(zone_of)(LON,LAT)
cmap=mpl.colors.ListedColormap([ZCOL[z] for z in [2,3,4,5,6]])
norm=mpl.colors.BoundaryNorm([1.5,2.5,3.5,4.5,5.5,6.5],cmap.N)
fig=plt.figure(figsize=(16,11))
ax=fig.add_axes([0.02,0.20,0.96,0.66]); ax.set_axis_off()
ax.set_xlim(-126,-66); ax.set_ylim(23,50.5); ax.set_aspect(1.28)
mesh=ax.pcolormesh(LON,LAT,Z,cmap=cmap,norm=norm,shading="auto",zorder=1,alpha=0.5)
clip=PathPatch(Path(US),transform=ax.transData,facecolor="none",edgecolor="#5a6b7a",lw=1.8,zorder=3)
ax.add_patch(clip); mesh.set_clip_path(clip)
fig.text(0.02,0.955,"Εκπομπές CO₂ ανά πόλη — καύσιμο-μείγμα και δίκτυο, όχι μόνο το κλίμα",
         fontsize=20,fontweight="bold",color=NAVY,ha="left")
fig.text(0.02,0.917,"Ετήσιες εκπομπές CO₂ ανά μέσο κτίριο (Scope 2, baseline 2018) · μέγεθος κύκλου ∝ εκπομπές (στάθμιση 60% οικ./40% εμπ.)",
         fontsize=13,color="#333",ha="left")
CITY={"2A":("Houston, TX",-95.36,29.76),"3A":("Atlanta, GA",-84.39,33.75),"4A":("Baltimore, MD",-76.61,39.29),
      "5A":("Chicago, IL",-87.65,41.85),"5B":("Denver, CO",-104.98,39.74)}
co2v={z:co2z(z) for z in CITY}; cmax=max(co2v.values())
bub=mpl.colors.LinearSegmentedColormap.from_list("co2",["#f4d06f","#e8897a","#b5495b","#6b1f3a"])
bn=plt.Normalize(min(co2v.values())*0.8,cmax)
for z,(city,lon,lat) in CITY.items():
    v=co2v[z]; sz=1100+3400*(v/cmax)
    ax.scatter([lon],[lat],s=sz*1.25,c="white",zorder=4,alpha=0.9)
    ax.scatter([lon],[lat],s=sz,c=[bub(bn(v))],edgecolor="white",lw=2,zorder=5)
    ax.text(lon,lat,f"{v/1000:.1f}t",ha="center",va="center",fontsize=13.5,fontweight="bold",color="white",zorder=6)
    ax.text(lon,lat-3.0,f"Ζώνη {z}",ha="center",va="center",fontsize=12.5,fontweight="bold",color="#111",zorder=6,
            bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#888",lw=0.8))
    ax.text(lon,lat-4.3,city,ha="center",va="center",fontsize=11,color="#222",zorder=6,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.7))
# Υπόμνημα ζωνών+πόλεων
STUDY=[("2A","Houston, TX","Θερμό-Υγρό","#c0435a"),("3A","Atlanta, GA","Θερμό-Υγρό","#e08a4f"),
       ("4A","Baltimore, MD","Μεικτό-Υγρό","#edc84e"),("5A","Chicago, IL","Ψυχρό-Υγρό","#7ab0d4"),
       ("5B","Denver, CO","Ψυχρό-Ξηρό (ορ.)","#4574a8")]
fig.text(0.02,0.165,"Κλιματικές ζώνες — πόλη & κλιματικός χαρακτήρας:",fontsize=13.5,fontweight="bold",color=NAVY,ha="left")
for i,(z,city,char,col) in enumerate(STUDY):
    x=0.02+i*0.196
    fig.add_artist(plt.Rectangle((x,0.113),0.026,0.032,color=col,ec="#555",lw=0.8,transform=fig.transFigure,clip_on=False))
    fig.text(x+0.033,0.137,f"Ζώνη {z} — {city}",fontsize=11,fontweight="bold",color="#111",ha="left",va="center")
    fig.text(x+0.033,0.114,char,fontsize=9.3,color="#555",ha="left",va="center")
# Βασικά ευρήματα
by=sorted(co2v.items(),key=lambda kv:-kv[1])
fig.text(0.02,0.062,"Βασικά ευρήματα:",fontsize=13.5,fontweight="bold",color=NAVY,ha="left")
fig.text(0.02,0.038,f"• Μέγιστες εκπομπές: Ζώνη {by[0][0]} ({by[0][1]/1000:.1f}t) — υψηλή κατανάλωση θέρμανσης & ρυπογόνο μείγμα.   "
         f"• Ελάχιστες: Ζώνη {by[-1][0]} ({by[-1][1]/1000:.1f}t) — χαμηλός συντελεστής δικτύου (Cambium 2022) & ήπιο κλίμα.",
         fontsize=10.5,color="#222",ha="left")
fig.text(0.02,0.016,"• Η γεωγραφία των εκπομπών ΔΕΝ ταυτίζεται με της κατανάλωσης: το καύσιμο-μείγμα και ο συντελεστής εκπομπών του δικτύου παίζουν καθοριστικό ρόλο.",
         fontsize=10.5,color="#222",ha="left")
fig.text(0.02,-0.006,"Πηγή: NREL ResStock 2024 R2 + ComStock 2023 R2 · CO₂: Cambium 2022 (ηλεκτρισμός) + EPA 2023 (καύσιμα) · ιδία επεξεργασία (5.000 κτίρια).",
         fontsize=9,color="#8a97a5",ha="left")
fig.savefig("fig_map_zones_co2.png",bbox_inches="tight",facecolor="white"); print("✓ fig_map_zones_co2.png")
