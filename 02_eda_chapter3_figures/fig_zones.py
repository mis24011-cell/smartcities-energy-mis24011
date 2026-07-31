# -*- coding: utf-8 -*-
"""Χάρτης κλιματικών ζωνών ASHRAE/IECC — προσεγγιστικό χρωματιστό υπόβαθρο ΣΕ ΟΛΗ τη
χώρα (θερμό→ψυχρό) + τονισμένες οι 4 ζώνες/πόλεις της μελέτης. Δικό μας σχέδιο (offline)."""
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

# ── Προσεγγιστικός δείκτης ζώνης: «effective latitude» = γεωγρ. πλάτος + διόρθωση υψομέτρου (Δύση ψυχρότερη) ──
def elev_bump(lon,lat):
    if -114 < lon < -103 and lat > 33:  return 3.6   # Rockies / High Plains (π.χ. Denver 5B)
    if -122 < lon < -114 and lat > 37:  return 1.8   # Great Basin
    return 0.0
def zone_of(lon,lat):
    e = lat + elev_bump(lon,lat)
    if e < 31:    return 2
    if e < 35.5:  return 3
    if e < 40.5:  return 4
    if e < 44:    return 5
    return 6
ZCOL={2:"#c0435a",3:"#e08a4f",4:"#edc84e",5:"#7ab0d4",6:"#4574a8"}
ZLBL={2:"Ζώνη 2 — Πολύ θερμό",3:"Ζώνη 3 — Θερμό",4:"Ζώνη 4 — Μεικτό",
      5:"Ζώνη 5 — Ψυχρό",6:"Ζώνη 6 — Κρύο"}

# grid για χρωματισμό
lons=np.linspace(-125,-66,340); lats=np.linspace(24,49.5,190)
LON,LAT=np.meshgrid(lons,lats)
Z=np.vectorize(zone_of)(LON,LAT)
cmap=mpl.colors.ListedColormap([ZCOL[z] for z in [2,3,4,5,6]])
norm=mpl.colors.BoundaryNorm([1.5,2.5,3.5,4.5,5.5,6.5],cmap.N)

fig=plt.figure(figsize=(15,9.6))
ax=fig.add_axes([0.02,0.22,0.96,0.66]); ax.set_axis_off()
ax.set_xlim(-126,-66); ax.set_ylim(23,50.5); ax.set_aspect(1.28)
mesh=ax.pcolormesh(LON,LAT,Z,cmap=cmap,norm=norm,shading="auto",zorder=1,alpha=0.92)
clip=PathPatch(Path(US),transform=ax.transData,facecolor="none",edgecolor="#5a6b7a",lw=1.8,zorder=3)
ax.add_patch(clip); mesh.set_clip_path(clip)

fig.text(0.02,0.955,"Οι πέντε ζώνες της μελέτης στον εθνικό κλιματικό χάρτη ASHRAE/IECC",
         fontsize=19,fontweight="bold",color=NAVY,ha="left")
fig.text(0.02,0.912,"Προσεγγιστικό υπόβαθρο ζωνών (θερμό→ψυχρό) σε όλη τη χώρα · τονισμένες οι πόλεις 2A/3A/4A/5A (in-sample) και 5B (validation)",
         fontsize=12.5,color="#333",ha="left")

CITIES=[("2A","Houston, TX",-95.36,29.76),("3A","Atlanta, GA",-84.39,33.75),("4A","Baltimore, MD",-76.61,39.29),
        ("5A","Chicago, IL",-87.65,41.85),("5B","Denver, CO",-104.98,39.74)]
for z,city,lon,lat in CITIES:
    ax.scatter([lon],[lat],s=180,c="#111",edgecolor="white",lw=2.2,zorder=5)
    ax.scatter([lon],[lat],s=44,c="white",zorder=6)
    ax.annotate(f"Ζώνη {z}\n{city}",xy=(lon,lat),xytext=(lon,lat+2.2),ha="center",va="bottom",
                fontsize=13.5,fontweight="bold",color="#111",zorder=7,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#111",lw=1.4))

# Legend υποβάθρου (ζώνες)
fig.text(0.02,0.155,"Υπόβαθρο — κλιματικές ζώνες (προσεγγιστικά):",fontsize=13,fontweight="bold",color=NAVY,ha="left")
lx=0.03
for i,z in enumerate([2,3,4,5,6]):
    x=lx+i*0.185
    fig.add_artist(plt.Rectangle((x,0.108),0.03,0.03,color=ZCOL[z],transform=fig.transFigure,clip_on=False))
    fig.text(x+0.038,0.123,ZLBL[z],fontsize=11.5,color="#222",ha="left",va="center")
fig.text(0.02,0.055,"Οι πόλεις της μελέτης: 2A Houston (Θερμό-Υγρό) · 3A Atlanta (Θερμό-Υγρό) · 4A Baltimore (Μεικτό-Υγρό) · 5A Chicago (Ψυχρό-Υγρό) · 5B Denver (Ψυχρό-Ξηρό, ορεινό).",
         fontsize=11.5,color="#333",ha="left")
fig.text(0.02,0.02,"Σχηματική/προσεγγιστική απεικόνιση βάσει ASHRAE Standard 90.1 & IECC climate zones (διόρθωση υψομέτρου για τη Δύση). Σχέδιο: ιδία επεξεργασία.",
         fontsize=9.8,color="#8a97a5",ha="left")
fig.savefig("fig_zones.png",bbox_inches="tight",facecolor="white"); print("✓ fig_zones.png")
