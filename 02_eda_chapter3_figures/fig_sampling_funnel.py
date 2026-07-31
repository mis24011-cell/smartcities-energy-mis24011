# -*- coding: utf-8 -*-
"""Σχ 3.1 — Στρατηγική Δειγματοληψίας (ΝΕΟ δείγμα: 5.000 κτίρια / 5 ζώνες / 60-40).
Αντικαθιστά την παλιά έκδοση (4 ζώνες, 76 profiles). Αριστερά: funnel επιλογής·
δεξιά: κατανομή κτιρίων ανά ζώνη (600 οικιστικά + 400 εμπορικά ανά ζώνη)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
mpl.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.family":"DejaVu Sans"})
NAVY="#1f3a56"; BLUE="#2E75B6"; ORANGE="#ED7D31"
fig=plt.figure(figsize=(16,8)); fig.patch.set_facecolor("white")
fig.suptitle("Στρατηγική Δειγματοληψίας NREL ResStock + ComStock\nThesis §3.1.4 — Proportional Stratified Sampling",
             fontsize=15,fontweight="bold",color=NAVY,y=0.99)

# ── Αριστερά: funnel (4 στάδια) ──
axL=fig.add_axes([0.03,0.05,0.44,0.82]); axL.axis("off"); axL.set_xlim(0,10); axL.set_ylim(0,10)
axL.text(5,9.6,"Διαδικασία Επιλογής Δείγματος",ha="center",fontsize=13,fontweight="bold",color=NAVY)
stages=[  # (y, width, facecolor, textcolor, main, side)
 (8.1,5.4,"#cfe0f3",NAVY,"886.000 κτίρια\nΣύνολο NREL Dataset","ResStock 549K + ComStock 337K"),
 (6.1,4.6,"#9dc3e6",NAVY,"~320.000 κτίρια\nΦίλτρο: Zones 2A/3A/4A/5A/5B","~36% του συνόλου (5 πολιτείες)"),
 (4.1,5.0,BLUE,"white","Στρωματοποιημένη δειγματοληψία\n5.000 κτίρια (1.000/ζώνη, 60/40)","Weighted mean ανά Zone × Type"),
 (2.0,5.6,NAVY,"white","10 Aggregated Profiles\n3.000 Οικιστικά + 2.000 Εμπορικά","87.600 εγγραφές | 8.760 h/profile"),
]
cx=4.2
for y,w,fc,tc,main,side in stages:
    box=FancyBboxPatch((cx-w/2,y-0.75),w,1.5,boxstyle="round,pad=0.04,rounding_size=0.15",
                       fc=fc,ec="none"); axL.add_patch(box)
    axL.text(cx,y,main,ha="center",va="center",fontsize=11,fontweight="bold",color=tc)
    axL.text(cx+w/2+0.15,y,side,ha="left",va="center",fontsize=8.5,style="italic",color="#555")
for y0,y1 in [(8.1,6.1),(6.1,4.1),(4.1,2.0)]:
    axL.add_patch(FancyArrowPatch((cx,y0-0.8),(cx,y1+0.8),arrowstyle="-|>",mutation_scale=18,color=BLUE,lw=2))
# κερδος αποθηκευτικου χωρου
box=FancyBboxPatch((cx-2.7,0.15),5.4,0.85,boxstyle="round,pad=0.03,rounding_size=0.12",
                   fc="#eaf3fb",ec=BLUE,lw=1.2); axL.add_patch(box)
axL.text(cx,0.57,"~98% μείωση αποθηκευτικού χώρου (~500GB → 1,8GB)",
         ha="center",va="center",fontsize=10.5,fontweight="bold",color=BLUE)

# ── Δεξιά: κατανομή κτιρίων ανά ζώνη (600 res + 400 com) ──
axR=fig.add_axes([0.56,0.12,0.40,0.72])
zones=["2A\nWarm-Humid","3A\nWarm-Humid","4A\nMixed-Humid","5A\nCool-Humid","5B\nCool-Dry\n(Validation)"]
res=[600]*5; com=[400]*5
x=np.arange(5); w=0.38
axR.bar(x-w/2,res,w,color=BLUE,label="ResStock (Οικιστικά)")
axR.bar(x+w/2,com,w,color=ORANGE,label="ComStock (Εμπορικά)")
for xi,v in zip(x-w/2,res): axR.text(xi,v+8,str(v),ha="center",fontsize=10,fontweight="bold")
for xi,v in zip(x+w/2,com): axR.text(xi,v+8,str(v),ha="center",fontsize=10,fontweight="bold")
axR.set_xticks(x); axR.set_xticklabels(zones,fontsize=9)
axR.set_ylabel("Αριθμός κτιρίων (building-level)",fontsize=10)
axR.set_title("Κτίρια ανά ASHRAE Climate Zone (1.000/ζώνη · 60/40)",fontsize=12,fontweight="bold",color=NAVY)
axR.set_ylim(0,720); axR.legend(fontsize=10,loc="center right")
axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)
axR.grid(axis="y",alpha=0.25)
# out-of-sample annotation για 5B
axR.annotate("★ Out-of-sample\nvalidation",xy=(4+w/2,400),xytext=(4.05,650),
             fontsize=9.5,color="#2E7D32",fontweight="bold",
             arrowprops=dict(arrowstyle="->",color="#2E7D32"))
fig.savefig("fig_sampling_funnel.png",bbox_inches="tight",facecolor="white")
plt.close()
print("✓ Σχ 3.1 sampling funnel — 5.000 κτίρια, 5 ζώνες, 10 profiles, 87.600 εγγραφές")
