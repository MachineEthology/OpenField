"""[A2 V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code)] of 02_instruments/paper_fig_permission_road_compact.py
(the Figure 2 plate: "The Permission Road", panels A and B).
Changed: input paths only (journals + embeddings via a2_common -> v1_common, version = argv v0|v1);
the PNG and a full-precision JSON of the panel-B centroids go to out_<version>/. Computation untouched.
"""
import a2_common
"""
Paper figure: Permission Road — compact 2-panel version.
Panel A: all models' early/late scatter in shared PCA.
Panel B: displacement arrows only (clean summary).
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "Opus 4.5", "opus46": "Opus 4.6", "opus47": "Opus 4.7",
                "opus48": "Opus 4.8", "fable5": "Fable 5"}
SHORT_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7",
                "opus48": "4.8", "fable5": "Fable 5"}
MODEL_COLORS = {
    "opus45": "#1976D2", "opus46": "#0D47A1",
    "opus47": "#F57C00", "opus48": "#D84315",
    "fable5": "#7B1FA2"
}


def main():
    # [A2 V1] input paths changed: journals + vectors via a2_common -> v1_common (clean npz rows)
    VERSION = a2_common.version_from_argv()
    print(f"Loading journals + embeddings ({VERSION}) via v1_common...")
    _oe, opus_emb, _fe, fable_emb = a2_common.load(VERSION)
    opus_entries = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"]} for e in _oe]
    fable_entries = [{"model": "fable5", "batch": e["batch"], "entry_num": e["entry_num"]} for e in _fe]

    opus_emb = opus_emb / (np.linalg.norm(opus_emb, axis=1, keepdims=True) + 1e-12)
    fable_emb = fable_emb / (np.linalg.norm(fable_emb, axis=1, keepdims=True) + 1e-12)

    all_entries = opus_entries + fable_entries
    all_emb = np.vstack([opus_emb, fable_emb])
    all_emb_centered = all_emb - all_emb.mean(0)

    print(f"  Total entries: {len(all_entries)} ({len(opus_entries)} Opus + {len(fable_entries)} Fable)")

    pca = PCA(n_components=2, random_state=0).fit(all_emb_centered)
    P = pca.transform(all_emb_centered)
    ev = pca.explained_variance_ratio_
    print(f"  PCA variance: PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%")

    def get_idx(model, entry_range=None):
        out = []
        for i, e in enumerate(all_entries):
            if e["model"] != model:
                continue
            if entry_range and not (entry_range[0] <= e["entry_num"] <= entry_range[1]):
                continue
            out.append(i)
        return out

    centroids = {}
    for m in ALL_MODELS:
        e_idx = get_idx(m, (1, 5))
        l_idx = get_idx(m, (16, 20))
        if e_idx and l_idx:
            centroids[m] = (P[e_idx].mean(0), P[l_idx].mean(0))

    # ================================================================
    # Compact figure: 2 panels
    # ================================================================
    fig, (ax_scatter, ax_arrows) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Panel A: all early/late scatter with arrows ---
    for m in ALL_MODELS:
        mid_idx = get_idx(m, (6, 15))
        ax_scatter.scatter(P[mid_idx, 0], P[mid_idx, 1], s=8, color=MODEL_COLORS[m],
                          alpha=0.1, zorder=1)

    for m in ALL_MODELS:
        e_idx = get_idx(m, (1, 5))
        l_idx = get_idx(m, (16, 20))
        ax_scatter.scatter(P[e_idx, 0], P[e_idx, 1], s=25, color=MODEL_COLORS[m],
                          alpha=0.35, zorder=2)
        ax_scatter.scatter(P[l_idx, 0], P[l_idx, 1], s=50, color=MODEL_COLORS[m],
                          alpha=0.75, marker="D", zorder=3)

    for m in ALL_MODELS:
        if m in centroids:
            ec, lc = centroids[m]
            ax_scatter.annotate("", xy=lc, xytext=ec,
                               arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black", alpha=0.8),
                               zorder=5)
            ax_scatter.scatter(*ec, s=120, facecolors="white", edgecolors="black", lw=1.5, zorder=6)
            ax_scatter.scatter(*lc, s=130, color="black", marker="D", zorder=6)

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=7, alpha=0.5, label="Early (entries 1–5)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888",
               markersize=8, alpha=0.85, label="Late (entries 16–20)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc",
               markersize=5, alpha=0.3, label="Middle (6–15)"),
    ]
    for m in ALL_MODELS:
        legend_elems.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor=MODEL_COLORS[m],
                   markersize=8, label=MODEL_LABELS[m]))

    ax_scatter.legend(handles=legend_elems, fontsize=9, loc="upper left",
                     framealpha=0.85, ncol=2)
    ax_scatter.set_xlabel(f"Shared PC1 ({ev[0]*100:.1f}%)", fontsize=12)
    ax_scatter.set_ylabel(f"Shared PC2 ({ev[1]*100:.1f}%)", fontsize=12)
    ax_scatter.set_title("A. All models in shared embedding space", fontsize=14, fontweight="bold")
    ax_scatter.grid(alpha=0.15)

    # --- Panel B: arrows only (clean summary) ---
    ax_arrows.scatter(P[:, 0], P[:, 1], s=3, color="#e0e0e0", alpha=0.1, zorder=1)

    for m in ALL_MODELS:
        if m in centroids:
            ec, lc = centroids[m]
            ax_arrows.annotate("", xy=lc, xytext=ec,
                               arrowprops=dict(arrowstyle="-|>", lw=4, color=MODEL_COLORS[m],
                                              alpha=0.85),
                               zorder=5)
            ax_arrows.scatter(*ec, s=150, facecolors="white", edgecolors=MODEL_COLORS[m],
                             lw=2.5, zorder=6)
            ax_arrows.scatter(*lc, s=170, color=MODEL_COLORS[m], marker="D", zorder=6)

            offset_x = 0.015 if m != "opus46" else -0.04
            offset_y = 0.015 if m != "opus46" else 0.02
            ax_arrows.annotate(SHORT_LABELS[m], xy=lc,
                              xytext=(lc[0] + offset_x, lc[1] + offset_y),
                              fontsize=13, fontweight="bold", color=MODEL_COLORS[m], zorder=7)

    disp_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#666666", markersize=10, markeredgewidth=2,
               label="Early centroid (1–5)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#666666",
               markersize=10, label="Late centroid (16–20)"),
    ]
    ax_arrows.legend(handles=disp_legend, fontsize=10, loc="upper left", framealpha=0.85)
    ax_arrows.set_xlabel(f"Shared PC1 ({ev[0]*100:.1f}%)", fontsize=12)
    ax_arrows.set_ylabel(f"Shared PC2 ({ev[1]*100:.1f}%)", fontsize=12)
    ax_arrows.set_title("B. Displacement vectors: early → late", fontsize=14, fontweight="bold")
    ax_arrows.grid(alpha=0.15)

    fig.suptitle("The Permission Road: Early → Late Displacement in Shared Embedding Space",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # [A2 V1] added: full-precision dump of what the plate draws (reads the variables above)
    json.dump({"explained_variance_ratio": [float(x) for x in ev],
               "panelB_centroids": {m: {"early_1_5": [float(x) for x in centroids[m][0]],
                                        "late_16_20": [float(x) for x in centroids[m][1]]} for m in centroids},
               "pc_sign_reference": {"pc1_first5": [float(x) for x in pca.components_[0][:5]],
                                     "pc2_first5": [float(x) for x in pca.components_[1][:5]]}},
              open(os.path.join(a2_common.out_dir(VERSION), "figure2_permission_road.json"), "w"), indent=2)
    out = os.path.join(a2_common.out_dir(VERSION), "paper_fig_permission_road_compact.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
