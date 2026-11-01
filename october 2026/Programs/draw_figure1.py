import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                 ec=edgecolor, fc=facecolor, lw=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10, wrap=True)

def draw_arrow(ax, start, end, color="black", style="-|>", lw=2, linestyle="-"):
    arrow = patches.FancyArrowPatch(start, end, arrowstyle=style, 
                                    mutation_scale=20, color=color, lw=lw, linestyle=linestyle)
    ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
c_blue = "#D6EAF8"
c_teal = "#D1F2EB"
c_orange = "#FDEBD0"
c_border = "#2C3E50"

# Top Row
draw_box(ax, 0.5, 4, 2.8, 1.5, "Road Image\n\n224 × 224 RGB\nPascal VOC multi-hot labels", c_blue, c_border)
draw_box(ax, 4.5, 4, 3.0, 1.5, "Recognition Module\n\nEfficientNetV2S\nsigmoid scores; threshold 0.35", c_teal, c_border)
draw_box(ax, 8.5, 4, 3.0, 1.5, "Visual Report\n\ntop-4 scores and labels;\nschematic overlays", c_orange, c_border)

# Bottom Row
draw_box(ax, 0.5, 0.5, 2.8, 1.8, "Structured Survey\n\nvehicles, capacity, speed,\npeak periods, events, site\nfeatures, restrictions", c_blue, c_border)
draw_box(ax, 4.5, 0.5, 3.0, 1.8, "Decision Rules\n\nadditive congestion index\n30 interventions, filter, rank\n(heuristic, not causal)", c_teal, c_border)
draw_box(ax, 8.5, 0.5, 3.0, 1.8, "Candidate Output\n\n3–5 ranked options\n(normalized constraint-aware)", c_orange, c_border)

# Arrows
# Top row
draw_arrow(ax, (3.3, 4.75), (4.5, 4.75)) # Road Image -> Recognition
draw_arrow(ax, (7.5, 4.75), (8.5, 4.75)) # Recognition -> Visual Report

# Bottom row
draw_arrow(ax, (3.3, 1.4), (4.5, 1.4)) # Survey -> Decision Rules
draw_arrow(ax, (7.5, 1.4), (8.5, 1.4)) # Decision Rules -> Candidate Output

# Integration Arrow (THE FIX)
# Solid arrow from Recognition down to Decision Rules
draw_arrow(ax, (6.0, 4.0), (6.0, 2.3), color="#C0392B", lw=3)
ax.text(6.1, 3.15, "detected_elements\n(soft multipliers)", ha="left", va="center", fontsize=9, color="#C0392B", fontweight="bold")

# Merge to Final Output
# From Visual Report down to final image
draw_arrow(ax, (10.0, 4.0), (10.0, 2.3), color="gray", lw=1.5)
# From Candidate Output up to final image
draw_arrow(ax, (10.0, 0.5), (10.0, 2.3), color="gray", lw=1.5)
ax.text(10.2, 2.9, "merged in\nreport", ha="left", va="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("figure1_dataflow.png", dpi=300, bbox_inches='tight')
print("Saved figure1_dataflow.png")