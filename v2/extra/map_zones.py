# -*- coding: utf-8 -*-
"""Fig. 3.19 (β) — IECC / ASHRAE 169 climate zones of the contiguous US and the five sample cities.

Data: IECC climate-zone polygons (thermal zones 1–8) and moisture-regime polygons (A humid, B dry,
C marine) bundled in the open-source package eeweather 0.3.30 (OpenEEmeter, Apache-2.0,
resources/metadata.db), derived from the county-level IECC climate-zone assignment of DOE/PNNL.
Zone = thermal number × moisture regime (e.g. 5 ∩ B = 5B); zones 7–8 have no moisture regime. Validation: the five city coordinates
fall in 2A / 3A / 4A / 5A / 5B (asserted below), as in the NREL metadata of the sample (Ch.3 §3.1.4).
Same projection and frame as map_cities.py. Requires: matplotlib, basemap, shapely, eeweather (so that (α) and (β) align).
"""
import json
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, box, shape

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["hatch.linewidth"] = 0.6

# IECC polygons shipped with the open-source package eeweather (pip install eeweather)
# (only the data file is used: the package is located, not imported)
import importlib.util, os
_spec = importlib.util.find_spec("eeweather")
assert _spec, "pip install eeweather"
db = sqlite3.connect(os.path.join(os.path.dirname(_spec.origin), "resources", "metadata.db"))
CONUS = box(-125.5, 24.0, -66.5, 49.6)
ZONE = {z: shape(json.loads(g)).intersection(CONUS)
        for z, g in db.execute("select iecc_climate_zone, geometry from iecc_climate_zone_metadata")}
MOIST = {m: shape(json.loads(g)).intersection(CONUS)
         for m, g in db.execute("select iecc_moisture_regime, geometry from iecc_moisture_regime_metadata")}

CITIES = [("Houston", 29.760, -95.370, "2A"), ("Atlanta", 33.749, -84.388, "3A"),
          ("Baltimore", 39.290, -76.612, "4A"), ("Chicago", 41.878, -87.630, "5A"),
          ("Denver", 39.739, -104.990, "5B")]
for n, la, lo, exp in CITIES:                       # validation against the sample metadata
    p = Point(lo, la)
    z = next(k for k, g in ZONE.items() if g.contains(p))
    m = next(k for k, g in MOIST.items() if g.contains(p))
    assert z + m == exp, (n, z + m, exp)

# thermal zone colours: warm (1) → cold (7); diverging red–blue, light in the middle
COL = {"1": "#b2182b", "2": "#d6604d", "3": "#f4a582", "4": "#fbe3d4",
       "5": "#d1e5f0", "6": "#8cbfdd", "7": "#2f6fae", "8": "#1a3f7a"}
HATCH = {"A": None, "B": "////", "C": "...."}
NAME = {"1": "1 — πολύ θερμή", "2": "2 — θερμή", "3": "3 — θερμή-εύκρατη", "4": "4 — μεικτή",
        "5": "5 — ψυχρή", "6": "6 — ψυχρή", "7": "7 — πολύ ψυχρή"}

fig, ax = plt.subplots(figsize=(11, 7))
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection="lcc",
            lat_1=33, lat_2=45, lon_0=-95, resolution="l", ax=ax)
m.drawmapboundary(fill_color="#eef4f8")


def polys(geom):
    """flatten any (multi)polygon / geometry collection into a list of Polygons"""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [q for g in geom.geoms for q in polys(g)]
    return []


def ppath(pg):
    """matplotlib Path of a polygon with its holes, in map coordinates"""
    from matplotlib.path import Path as MPath
    verts, codes = [], []
    for ring in [pg.exterior, *pg.interiors]:
        x, y = m(*zip(*ring.coords))
        verts += list(zip(x, y))
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(x) - 2) + [MPath.CLOSEPOLY]
    return MPath(verts, codes)


# layer 1: thermal zone fill (zones 7–8 have no moisture regime in IECC, so fill by zone alone)
for z, zg in sorted(ZONE.items()):
    for pg in polys(zg.simplify(0.01)):
        ax.add_patch(PathPatch(ppath(pg), fc=COL[z], ec=COL[z], lw=0.3, zorder=2))
# layer 2: moisture regime as hatching (B dry, C marine; A humid = no hatching)
for mo, mg in MOIST.items():
    if HATCH[mo]:
        for pg in polys(mg.simplify(0.01)):
            ax.add_patch(PathPatch(ppath(pg), fc="none", ec="#555", lw=0, hatch=HATCH[mo], zorder=2.5))
m.drawcoastlines(linewidth=0.6, color="#555", zorder=3)
m.drawcountries(linewidth=0.8, color="#555", zorder=3)
m.drawstates(linewidth=0.4, color="#777", zorder=3)

offs = {"Houston": (-60, -38), "Atlanta": (62, -40), "Baltimore": (62, -12),
        "Chicago": (10, 48), "Denver": (-62, 40)}
for n, la, lo, z in CITIES:
    x, y = m(lo, la)
    oos = z == "5B"
    m.scatter(x, y, s=230 if oos else 120, marker="*" if oos else "o", color="#111",
              edgecolor="white", linewidth=1.2, zorder=6)
    ax.annotate(f"{n} — {z}", (x, y), xytext=offs[n], textcoords="offset points", ha="center",
                va="center", fontsize=10, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111", lw=1.0),
                arrowprops=dict(arrowstyle="-", color="#111", lw=1.0))

h1 = [Patch(fc=COL[k], ec="#777", label=v) for k, v in NAME.items()]
h2 = [Patch(fc="white", ec="#555", label="A — υγρή"), Patch(fc="white", ec="#555", hatch="////", label="B — ξηρή"),
      Patch(fc="white", ec="#555", hatch="....", label="C — θαλάσσια")]
l1 = ax.legend(handles=h1, title="Θερμική ζώνη", loc="lower left", fontsize=8.5, title_fontsize=9,
               frameon=True, framealpha=0.95)
ax.add_artist(l1)
ax.legend(handles=h2, title="Υγρασία", loc="lower right", fontsize=8.5, title_fontsize=9,
          frameon=True, framealpha=0.95)
ax.set_title("(β) Κλιματικές ζώνες IECC / ASHRAE 169 στις ΗΠΑ και οι πόλεις του δείγματος", fontsize=13)
plt.tight_layout()
from pathlib import Path
OUT = Path("outputs_v2/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / "fig_3_19b_climate_zones.png", dpi=300)
print("ok")
