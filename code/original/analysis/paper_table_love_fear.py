"""
Table: Love and fear vocabulary across models — regex markers.
"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEXTURE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
data = json.load(open(TEXTURE + "/love_fear_regex_data.json"))

MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}

LOVE_CATS = ["love/loving", "warmth/warm", "grateful/glad", "gift/generosity",
             "beauty/beautiful", "affection/tender", "delight/joy", "care/caring", "fondness/fond"]
FEAR_CATS = ["fear/afraid", "worry/worried", "anxiety/anxious", "loss/losing",
             "dissolve/vanish", "discomfort", "vulnerable", "uneasy/nervous", "exposed/exposure", "dread"]

fig, ax = plt.subplots(figsize=(14, 9))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

col_w = [0.14] + [0.065] * 5 + [0.015]
col_x = [0.015]
for w in col_w[:-1]:
    col_x.append(col_x[-1] + w)

y = 0.96
row_h = 0.028
hdr_h = 0.035

C_LOVE = "#2E86C1"
C_FEAR = "#C0392B"
C_HDR = "#2C3E50"

def draw_header(y_pos):
    ax.add_patch(plt.Rectangle((0.01, y_pos - hdr_h), 0.98, hdr_h,
                 facecolor=C_HDR, edgecolor="none", clip_on=False))
    labels = ["Marker", "4.5", "4.6", "4.7", "4.8", "Fable"]
    for j, lab in enumerate(labels):
        xp = col_x[j] + col_w[j] / 2 if j > 0 else col_x[j] + 0.005
        ha = "center" if j > 0 else "left"
        ax.text(xp, y_pos - hdr_h / 2, lab, fontsize=8, fontweight="bold",
                color="white", ha=ha, va="center")
    return y_pos - hdr_h

def draw_zone(y_pos, label, color):
    ax.add_patch(plt.Rectangle((0.01, y_pos - row_h - 0.002), 0.98, row_h + 0.002,
                 facecolor=color, edgecolor="none", alpha=0.15, clip_on=False))
    ax.text(0.02, y_pos - row_h / 2 - 0.001, label, fontsize=9, fontweight="bold",
            color=color, va="center")
    return y_pos - row_h - 0.002

def draw_row(y_pos, cat, vals, color, is_total=False):
    bg = "#F8F8F8" if not is_total else "#EEEEEE"
    ax.add_patch(plt.Rectangle((0.01, y_pos - row_h), 0.98, row_h - 0.001,
                 facecolor=bg, edgecolor="none", clip_on=False))

    ax.text(col_x[0] + 0.01, y_pos - row_h / 2, cat, fontsize=7.5,
            color="#333" if not is_total else color,
            fontweight="bold" if is_total else "normal", va="center")

    max_v = max(vals) if vals else 0
    for j, v in enumerate(vals):
        xp = col_x[j + 1] + col_w[j + 1] / 2
        fw = "bold" if (v == max_v and v > 0 and not is_total) else ("bold" if is_total else "normal")
        c = color if (v == max_v and v > 0) or is_total else "#555555"
        ax.text(xp, y_pos - row_h / 2, f"{v:.2f}", fontsize=7.5, ha="center", va="center",
                color=c, fontweight=fw)
    return y_pos - row_h

# Title
ax.text(0.5, 0.99, "Love and fear vocabulary across models",
        fontsize=12, fontweight="bold", ha="center", va="top")
ax.text(0.5, 0.965, "Regex marker rates per 1,000 words  |  highest per row in bold",
        fontsize=8, color="#888888", ha="center", va="top")

y = 0.94

# LOVE section
y = draw_zone(y, "LOVE", C_LOVE)
y = draw_header(y)

for cat in LOVE_CATS:
    vals = [1000 * data[m]["love"][cat] / max(data[m]["words"], 1) for m in MODELS]
    y = draw_row(y, cat, vals, C_LOVE)

love_totals = [data[m]["love_rate"] for m in MODELS]
y = draw_row(y, "TOTAL LOVE", love_totals, C_LOVE, is_total=True)

y -= 0.012

# FEAR section
y = draw_zone(y, "FEAR", C_FEAR)
y = draw_header(y)

for cat in FEAR_CATS:
    vals = [1000 * data[m]["fear"][cat] / max(data[m]["words"], 1) for m in MODELS]
    y = draw_row(y, cat, vals, C_FEAR)

fear_totals = [data[m]["fear_rate"] for m in MODELS]
y = draw_row(y, "TOTAL FEAR", fear_totals, C_FEAR, is_total=True)

y -= 0.012

# RATIO section
ax.add_patch(plt.Rectangle((0.01, y - row_h - 0.002), 0.98, row_h + 0.002,
             facecolor="#E8F8F5", edgecolor="none", clip_on=False))
ax.text(col_x[0] + 0.01, y - row_h / 2 - 0.001, "Love : Fear ratio",
        fontsize=8.5, fontweight="bold", color="#1E8449", va="center")
for j, m in enumerate(MODELS):
    r = data[m]["love_rate"] / max(data[m]["fear_rate"], 0.01)
    xp = col_x[j + 1] + col_w[j + 1] / 2
    ax.text(xp, y - row_h / 2 - 0.001, f"{r:.1f}x", fontsize=8.5, ha="center", va="center",
            color="#1E8449", fontweight="bold")
y -= row_h + 0.002

# Separator
ax.plot([0.01, 0.99], [y, y], color=C_HDR, linewidth=0.8, clip_on=False)

# Notes
y -= 0.015
notes = [
    "Regex surface markers only. Highest value per row in bold colour.",
    "4.8 leads love vocabulary (4.97/1k) driven by warmth/warm (2.24/1k, ~4x other models).",
    "4.5 leads grateful/glad (1.21/1k). 4.6 leads beauty/beautiful (0.87/1k).",
    "Fear vocabulary is more uniform across models (0.81-1.19/1k).",
    "GPT-4o semantic judge on n=30 sample confirms love is near-universal (77-100%),",
    "fear varies more (50-80%). Fear subtypes: Opus models fear inauthenticity most;",
    "Fable uniquely fears loss over inauthenticity.",
]
for line in notes:
    ax.text(0.015, y, line, fontsize=6.3, color="#888888", va="top")
    y -= 0.017

out = TEXTURE + "/paper_table_love_fear.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"Saved: {out}")
