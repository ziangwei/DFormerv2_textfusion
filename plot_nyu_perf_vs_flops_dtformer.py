#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a CVPR-style 'mIoU vs FLOPs' comparison plot for NYU Depth V2.
- Families are drawn as polylines connecting different model scales.
- Our DTFormer is highlighted with red star markers.
- No labels - add manually later.
"""

import matplotlib.pyplot as plt


# ========== Optional: try Times New Roman if available ==========
def maybe_use_times():
    try:
        from matplotlib import font_manager as fm
        fams = [f.name for f in fm.fontManager.ttflist]
        if "Times New Roman" in set(fams):
            plt.rcParams["font.family"] = "Times New Roman"
    except Exception:
        pass


plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
maybe_use_times()

# ========== Data (NYU Depth V2) - CORRECTED ==========
# Format: (FLOPs[G], mIoU)
families = {
    "Omnivore": [(59.8, 52.7), (109.3, 54.0)],
    "TokenFusion": [(55.2, 53.3), (94.4, 54.2)],
    "DFormer": [(25.6, 53.6), (41.9, 55.6), (65.7, 57.2)],
    "CMX": [(67.6, 54.4), (134.3, 56.3), (167.8, 56.9)],
    "CMNeXt": [(131.9, 56.9)],
    "MultiMAE": [(267.9, 56.0)],
    "GeminiFusion": [(138.2, 56.8), (256.1, 57.7)],
    "DFormerv2": [(33.9, 56.0), (67.2, 57.7), (124.1, 58.4)],

    # ---- Ours: DTFormer ----
    "DTFormer (Ours)": [(40.2, 57.8), (79.6, 58.3), (161.1, 58.6)],
}

# ========== Plot ==========
fig, ax = plt.subplots(figsize=(8, 6))

for fam, pts in families.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if fam == "DTFormer (Ours)":
        # Red color with star marker for ours
        ax.plot(xs, ys, marker="*", markersize=14, linewidth=1.5,
                label=fam, color='red', zorder=10)
    else:
        # All other methods with same line width
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=1.5,
                label=fam, alpha=0.7)

# Axes labels
ax.set_xlabel("FLOPs (G)", fontsize=13, fontweight='bold')
ax.set_ylabel("mIoU (%)", fontsize=13, fontweight='bold')

# Nice limits with margins
all_x = [x for pts in families.values() for x, _ in pts]
all_y = [y for pts in families.values() for _, y in pts]
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)
xr, yr = (xmax - xmin), (ymax - ymin)
ax.set_xlim(xmin - 0.08 * xr, xmax + 0.08 * xr)
ax.set_ylim(ymin - 0.08 * yr, ymax + 0.12 * yr)

# Grid for better readability
ax.grid(False)

# Minimal look (CVPR style)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.0)

# Legend (optional - uncomment if needed)
# ax.legend(frameon=False, ncol=2, fontsize=9, loc='lower right')

plt.tight_layout()

# Save both PNG and PDF
out_png = "nyu_perf_vs_flops_dtformer2.png"
out_pdf = "nyu_perf_vs_flops_dtformer2.pdf"
plt.savefig(out_png, dpi=400, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f"Saved to: {out_png} and {out_pdf}")

plt.show()