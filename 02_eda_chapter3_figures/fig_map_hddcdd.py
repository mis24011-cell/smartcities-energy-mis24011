# -*- coding: utf-8 -*-
"""Σχ. 3.17 (νέο): HDD/CDD ανά ζώνη πάνω στον κλιματικό χάρτη-πρότυπο."""
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
mpl.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200})
NAVY="#1f4e79"
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
# HDD/CDD (base 18°C, τυπικές τιμές)
CITY={"2A":("Houston, TX",-95.36,29.76,1400,2889),"3A":("Atlanta, GA",-84.39,33.75,2827,1810),
      "4A":("Baltimore, MD",-76.61,39.29,4654,1112),"5A":("Chicago, IL",-87.65,41.85,6311,842),
      "5B":("Denver, CO",-104.98,39.74,6128,696)}
fig=plt.figure(figsize=(16,10.5))
ax=fig.add_axes([0.02,0.20,0.96,0.66]); ax.set_axis_off()
ax.set_xlim(-126,-66); ax.set_ylim(23,50.5); ax.set_aspect(1.28)
mesh=ax.pcolormesh(LON,LAT,Z,cmap=cmap,norm=norm,shading="auto",zorder=1,alpha=0.5)
clip=PathPatch(Path(US),transform=ax.transData,facecolor="none",edgecolor="#5a6b7a",lw=1.8,zorder=3)
ax.add_patch(clip); mesh.set_clip_path(clip)
fig.text(0.02,0.955,"Ανάγκες θέρμανσης (HDD) εκτοξεύονται στο βορρά· ανάγκες ψύξης (CDD) στο νότο",
         fontsize=20,fontweight="bold",color=NAVY,ha="left")
fig.text(0.02,0.917,"Βαθμοημέρες θέρμανσης (HDD) & ψύξης (CDD), βάση 18 °C, ανά αντιπροσωπευτική πόλη/ζώνη",
         fontsize=13,color="#333",ha="left")
for z,(city,lon,lat,hdd,cdd) in CITY.items():
    ax.scatter([lon],[lat],s=280,c="white",edgecolor="#333",lw=2,zorder=5)
    ax.annotate(f"Ζώνη {z} — {city.split(',')[0]}\nHDD {hdd:,}  ·  CDD {cdd:,}",xy=(lon,lat),xytext=(lon,lat+2.4),
                ha="center",va="bottom",fontsize=12.5,fontweight="bold",color="#111",zorder=6,
                bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="#333",lw=1.3))
# υπόμνημα
STUDY=[("2A","Houston, TX","#c0435a"),("3A","Atlanta, GA","#e08a4f"),("4A","Baltimore, MD","#edc84e"),
       ("5A","Chicago, IL","#7ab0d4"),("5B","Denver, CO","#4574a8")]
fig.text(0.02,0.145,"Κλιματικές ζώνες — πόλη:",fontsize=13.5,fontweight="bold",color=NAVY,ha="left")
for i,(z,city,col) in enumerate(STUDY):
    x=0.02+i*0.196
    fig.add_artist(plt.Rectangle((x,0.10),0.026,0.03,color=col,ec="#555",lw=0.8,transform=fig.transFigure,clip_on=False))
    fig.text(x+0.033,0.115,f"Ζώνη {z} — {city}",fontsize=11,fontweight="bold",color="#111",ha="left",va="center")
fig.text(0.02,0.05,"• HDD (θέρμανση): 1.400 (Houston) → 6.311 (Chicago) — τετραπλασιασμός από νότο σε βορρά.   "
         "• CDD (ψύξη): αντίστροφη τάση, μέγιστη στο Houston (2.889).",fontsize=11,color="#222",ha="left")
fig.text(0.02,0.02,"Πηγή: ASHRAE Standard 90.1 / IECC (τυπικές βαθμοημέρες, βάση 18 °C). Υπόβαθρο ζωνών: σχηματικό. Ιδία επεξεργασία.",
         fontsize=9,color="#8a97a5",ha="left")
fig.savefig("fig_map_hddcdd.png",bbox_inches="tight",facecolor="white"); print("✓ fig_map_hddcdd.png")
