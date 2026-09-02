#!/usr/bin/env python3
"""Render the India coverage audit from the fixed counts in the diagnostic note."""

from pathlib import Path
import matplotlib.pyplot as plt

regions = ["North America", "Europe", "South Asia", "Broad India box"]
counts = [3719, 289, 21, 14]
shares = [46.5923, 3.6206, 0.2631, 0.1754]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
colors = ["#2b8cbe", "#7bccc4", "#fdae6b", "#d7301f"]
axes[0].barh(regions[::-1], counts[::-1], color=colors[::-1])
axes[0].set_xlabel("MMEarth SOC samples")
axes[0].set_title("Absolute coverage")
for i, value in enumerate(counts[::-1]):
    axes[0].text(value + 35, i, f"{value:,}", va="center", fontsize=9)
axes[0].set_xlim(0, 4100)

axes[1].barh(regions[::-1], shares[::-1], color=colors[::-1])
axes[1].set_xlabel("Share of all 7,982 samples (%)")
axes[1].set_title("Dataset share")
for i, value in enumerate(shares[::-1]):
    axes[1].text(value + .5, i, f"{value:.4f}%", va="center", fontsize=9)
axes[1].set_xlim(0, 52)

fig.suptitle("MMEarth SOC coverage audit: India is scarcely represented", fontsize=14, weight="bold")
fig.text(.5, .01, "Bounding-box counts; the broad India box is an upper bound, not a country-polygon count.",
         ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=[0, .05, 1, .93])
out = Path(__file__).resolve().parents[1] / "figures" / "india_coverage_audit.png"
fig.savefig(out, bbox_inches="tight")
print(out)
