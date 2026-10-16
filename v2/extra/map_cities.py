# -*- coding: utf-8 -*-
"""Fig. 3.19 (α) — the five sample cities on the US map. Requires: matplotlib, basemap
(pip install basemap). Run from the project folder; writes outputs_v2/figures/fig_3_19a_us_map_cities.png"""
# Fig. 3.19 — US map with the five sample cities and their ASHRAE/IECC climate zone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.rcParams["font.family"] = "DejaVu Sans"
cities = [  # name, lat, lon, zone, character (GR), county, colour
    ("Houston", 29.760, -95.370, "2A", "Θερμό-υγρό", "Harris County", "#d7301f"),
    ("Atlanta", 33.749, -84.388, "3A", "Θερμό-υγρό", "Fulton County", "#fc8d59"),
    ("Baltimore", 39.290, -76.612, "4A", "Μεικτό-υγρό", "Baltimore City", "#31a354"),
    ("Chicago", 41.878, -87.630, "5A", "Ψυχρό-υγρό", "Cook County", "#2b8cbe"),
    ("Denver", 39.739, -104.990, "5B", "Ψυχρό-ξηρό", "Denver County", "#756bb1"),
]
offs = {"Houston": (-170, -45), "Atlanta": (90, -60), "Baltimore": (95, 10),
        "Chicago": (20, 85), "Denver": (-120, 70)}
fig, ax = plt.subplots(figsize=(11, 7))
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection="lcc",
            lat_1=33, lat_2=45, lon_0=-95, resolution="l", ax=ax)
m.drawmapboundary(fill_color="#eef4f8")
m.fillcontinents(color="#f7f7f2", lake_color="#eef4f8")
m.drawcoastlines(linewidth=0.6, color="#666")
m.drawcountries(linewidth=0.8, color="#666")
m.drawstates(linewidth=0.4, color="#999")
for name, lat, lon, z, ch, county, col in cities:
    x, y = m(lon, lat)
    oos = z == "5B"
    m.scatter(x, y, s=260 if oos else 180, marker="*" if oos else "o", color=col,
              edgecolor="k", linewidth=1.0, zorder=5)
    dx, dy = offs[name]
    ax.annotate(f"{name} — {z}\n{ch}\n{county}" + ("\n(εκτός δείγματος)" if oos else ""),
                (x, y), xytext=(dx, dy), textcoords="offset points", ha="center", va="center",
                fontsize=10, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, lw=1.5),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.3))
ax.set_title("(α) Οι πέντε πόλεις του δείγματος και η κλιματική ζώνη ASHRAE/IECC τους", fontsize=13)
plt.tight_layout()
OUT = Path("outputs_v2/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT / "fig_3_19a_us_map_cities.png", dpi=300)
print("ok")
