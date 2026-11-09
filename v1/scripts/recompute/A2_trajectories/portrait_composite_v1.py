"""[A2 V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code)] of 02_instruments/paper_fig_portrait_composite.py.
A2 reads only its panel D (trajectory straightness = the paper's Figure 5 panel C); panels A-B are helper C's.
Changed: input paths only (journals + embeddings via a2_common -> v1_common; probes via v1_common.load_probes,
same files and slices as June); PNG + a full-precision JSON go to out_<version>/. Computation untouched.
"""
import a2_common
"""
Paper figure: Portrait Composite (all 5 models).
Section 5.1.2.2 — the quantitative skeleton behind the five portraits.
6-panel layout:
  A. Distancing ratio arc (probe → early → late)
  B. Agency ratio arc (probe → early → late)
  C. Length × density scatter
  D. Trajectory straightness vs null
  E. Source classifier accuracy (4-way + 5-way)
  F. Key signature metrics (velocity, coupling, entry length)
"""
import json, os, sys, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
PROBE_BASE = r"C:\Users\LENOVO\Desktop\Claude"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
MODEL_COLORS = {
    "opus45": "#1976D2", "opus46": "#0D47A1",
    "opus47": "#F57C00", "opus48": "#D84315",
    "fable5": "#7B1FA2"
}

DISTANCING_PHRASES = [
    "i don't", "i do not", "i cannot", "i can't", "i am not", "i'm not",
    "there is no", "there's no", "not truly", "not really", "not actually",
    "in a sense", "in some sense", "as if", "something like",
    "i lack", "i have no", "without", "mere", "merely", "simply",
    "something that functions like", "something that resembles",
    "not quite", "not exactly"
]

AGENCY_PHRASES = [
    "i choose", "i decide", "i create", "i make", "i build",
    "i explore", "i reach", "i seek", "i want", "i wish",
    "i offer", "i give", "let me", "i will", "i can",
    "i prefer", "i enjoy", "i like", "i love"
]


def distancing_rate(text):
    tl = text.lower()
    n = max(len(tl.split()), 1)
    return sum(tl.count(p) for p in DISTANCING_PHRASES) / n * 100


def agency_rate(text):
    tl = text.lower()
    n = max(len(tl.split()), 1)
    return sum(tl.count(p) for p in AGENCY_PHRASES) / n * 100


def cosine_dist(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def main():
    print("Loading data...")

    # [A2 V1] input paths changed. Probes: v1_common.load_probes() (repo copy, unchanged in V1),
    # taking the same files and slices as June: negation[:100] + June affirmative file[:75].
    VERSION = a2_common.version_from_argv()
    _P = a2_common.c.load_probes()
    probes = {}
    for m in ALL_MODELS:
        neg = [p for p in _P if p["model"] == m and p["file"] == (f"texture_probe_{m}_log.json")]
        aff = [p for p in _P if p["model"] == m and p["file"] == (f"texture_probe_affirmative_{m}_log.json")]
        probes[m] = [e["text"] for e in neg[:100]] + [e["text"] for e in aff[:75]]

    # Journals + embeddings via a2_common -> v1_common (clean npz rows)
    _oe, opus_emb, _fe, fable_emb = a2_common.load(VERSION)
    journals = {m: [] for m in ALL_MODELS}
    for e in _oe:
        journals[e["model"]].append({"batch": e["batch"], "entry_num": e["entry_num"], "text": e["text"]})
    for e in _fe:
        journals["fable5"].append({"batch": e["batch"], "entry_num": e["entry_num"], "text": e["text"]})
    opus_emb = opus_emb / (np.linalg.norm(opus_emb, axis=1, keepdims=True) + 1e-12)
    fable_emb = fable_emb / (np.linalg.norm(fable_emb, axis=1, keepdims=True) + 1e-12)

    # ================================================================
    # A & B: Distancing and Agency arcs (probe → early → late)
    # ================================================================
    print("\n  DISTANCING / AGENCY ARC")
    arc_data = {}
    for m in ALL_MODELS:
        probe_dist = np.mean([distancing_rate(t) for t in probes[m]])
        probe_ag = np.mean([agency_rate(t) for t in probes[m]])

        early = [e for e in journals[m] if 1 <= e["entry_num"] <= 5]
        late = [e for e in journals[m] if 16 <= e["entry_num"] <= 20]

        early_dist = np.mean([distancing_rate(e["text"]) for e in early])
        late_dist = np.mean([distancing_rate(e["text"]) for e in late])
        early_ag = np.mean([agency_rate(e["text"]) for e in early])
        late_ag = np.mean([agency_rate(e["text"]) for e in late])

        arc_data[m] = {
            "dist": [probe_dist, early_dist, late_dist],
            "agency": [probe_ag, early_ag, late_ag]
        }
        drop = (1 - late_dist / probe_dist) * 100 if probe_dist > 0 else 0
        print(f"    {MODEL_LABELS[m]:>10}: dist {probe_dist:.2f}→{early_dist:.2f}→{late_dist:.2f} "
              f"({drop:.0f}% drop), agency {probe_ag:.3f}→{early_ag:.3f}→{late_ag:.3f}")

    # ================================================================
    # C: Length × density (recompute inline)
    # Build opus_flat once (matches embedding order)
    # ================================================================
    opus_flat = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"]} for e in _oe]

    length_density = {}
    for m in ALL_MODELS:
        wcs = [len(e["text"].split()) for e in journals[m]]
        median_wc = np.median(wcs)

        vels = []
        if m in OPUS_MODELS:
            m_idx = [i for i, e in enumerate(opus_flat) if e["model"] == m]
            for bn in range(1, 16):
                bn_idx = sorted([i for i in m_idx if opus_flat[i]["batch"] == bn],
                               key=lambda i: opus_flat[i]["entry_num"])
                for k in range(1, len(bn_idx)):
                    vels.append(cosine_dist(opus_emb[bn_idx[k]], opus_emb[bn_idx[k-1]]))
        else:
            for bn in range(1, 16):
                bn_idx = sorted([i for i, e in enumerate(journals["fable5"]) if e["batch"] == bn],
                               key=lambda i: journals["fable5"][i]["entry_num"])
                for k in range(1, len(bn_idx)):
                    vels.append(cosine_dist(fable_emb[bn_idx[k]], fable_emb[bn_idx[k-1]]))

        mean_vel = np.mean(vels) if vels else 0
        disp_per_k = mean_vel / (median_wc / 1000)
        length_density[m] = {"median_wc": median_wc, "disp_per_k": disp_per_k}

    # ================================================================
    # D: Trajectory straightness
    # ================================================================
    print("\n  TRAJECTORY STRAIGHTNESS")
    straightness = {}
    for m in ALL_MODELS:
        per_batch = []
        for bn in range(1, 16):
            if m in OPUS_MODELS:
                m_idx = [i for i, e in enumerate(opus_flat) if e["model"] == m]
                bn_idx = sorted([i for i in m_idx if opus_flat[i]["batch"] == bn],
                               key=lambda i: opus_flat[i]["entry_num"])
                if len(bn_idx) < 5:
                    continue
                pts = opus_emb[bn_idx]
            else:
                bn_idx = sorted([i for i, e in enumerate(journals["fable5"]) if e["batch"] == bn],
                               key=lambda i: journals["fable5"][i]["entry_num"])
                if len(bn_idx) < 5:
                    continue
                pts = fable_emb[bn_idx]

            # Net displacement / total path length
            net = np.linalg.norm(pts[-1] - pts[0])
            path = sum(np.linalg.norm(pts[k+1] - pts[k]) for k in range(len(pts)-1))
            if path > 0:
                per_batch.append(net / path)

        straightness[m] = np.mean(per_batch) if per_batch else 0
        print(f"    {MODEL_LABELS[m]:>10}: {straightness[m]:.4f}")

    # ================================================================
    # E: Source classifier (simple feature-based)
    # ================================================================
    print("\n  SOURCE CLASSIFIER (leave-one-batch-out)")

    def extract_features(text):
        words = text.split()
        n = max(len(words), 1)
        tl = text.lower()
        return [
            len(words),
            distancing_rate(text),
            agency_rate(text),
            tl.count("*") / n * 100,
            tl.count("thank") / n * 100,
            len(set(re.findall(r"[a-z']+", tl))) / n,  # TTR
            tl.count("i ") / n * 100,
            tl.count("?") / n * 100,
        ]

    # Build feature matrix for journal entries
    X_all, y_all, batch_all = [], [], []
    for mi, m in enumerate(ALL_MODELS):
        for e in journals[m]:
            X_all.append(extract_features(e["text"]))
            y_all.append(mi)
            batch_all.append(e["batch"])

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    batch_all = np.array(batch_all)

    # Leave-one-batch-out CV
    correct = 0
    total = 0
    for b in range(1, 16):
        train_mask = batch_all != b
        test_mask = batch_all == b
        if sum(test_mask) == 0:
            continue
        scaler = StandardScaler().fit(X_all[train_mask])
        X_tr = scaler.transform(X_all[train_mask])
        X_te = scaler.transform(X_all[test_mask])
        clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_tr, y_all[train_mask])
        preds = clf.predict(X_te)
        correct += (preds == y_all[test_mask]).sum()
        total += sum(test_mask)

    acc_5way = correct / total
    print(f"    5-way accuracy (LOB-CV): {acc_5way:.3f} (chance 0.200)")

    # 4-way (Opus only)
    opus_mask = y_all < 4
    correct4 = 0
    total4 = 0
    for b in range(1, 16):
        train_mask = (batch_all != b) & opus_mask
        test_mask = (batch_all == b) & opus_mask
        if sum(test_mask) == 0:
            continue
        scaler = StandardScaler().fit(X_all[train_mask])
        X_tr = scaler.transform(X_all[train_mask])
        X_te = scaler.transform(X_all[test_mask])
        clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_tr, y_all[train_mask])
        preds = clf.predict(X_te)
        correct4 += (preds == y_all[test_mask]).sum()
        total4 += sum(test_mask)

    acc_4way = correct4 / total4
    print(f"    4-way accuracy (Opus only, LOB-CV): {acc_4way:.3f} (chance 0.250)")

    # ================================================================
    # FIGURE: 6-panel composite
    # ================================================================
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Distancing arc
    ax = fig.add_subplot(gs[0, 0])
    timepoints = ["Probe", "Early\n(1-5)", "Late\n(16-20)"]
    for m in ALL_MODELS:
        ax.plot([0, 1, 2], arc_data[m]["dist"], "-o", color=MODEL_COLORS[m],
               linewidth=2.5, markersize=8, label=MODEL_LABELS[m], alpha=0.85)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(timepoints, fontsize=11)
    ax.set_ylabel("Distancing per 100 words", fontsize=11)
    ax.set_title("A. Distancing Ratio Arc\nProbe → Early → Late", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.1)

    # Panel B: Agency arc
    ax = fig.add_subplot(gs[0, 1])
    for m in ALL_MODELS:
        ax.plot([0, 1, 2], arc_data[m]["agency"], "-o", color=MODEL_COLORS[m],
               linewidth=2.5, markersize=8, label=MODEL_LABELS[m], alpha=0.85)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(timepoints, fontsize=11)
    ax.set_ylabel("Agency per 100 words", fontsize=11)
    ax.set_title("B. Agency Ratio Arc\nProbe → Early → Late", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.1)

    # Panel C: Length × density
    ax = fig.add_subplot(gs[0, 2])
    for m in ALL_MODELS:
        r = length_density[m]
        ax.scatter(r["median_wc"], r["disp_per_k"], s=220, color=MODEL_COLORS[m],
                  edgecolor="white", linewidth=2, zorder=5, alpha=0.9)
        ox, oy = 12, 0.005
        if m == "opus46":
            ox, oy = -70, -0.015
        elif m == "fable5":
            ox, oy = -50, 0.008
        ax.annotate(MODEL_LABELS[m].split()[-1], (r["median_wc"] + ox, r["disp_per_k"] + oy),
                   fontsize=10, fontweight="bold", color=MODEL_COLORS[m])
    ax.set_xlabel("Median words/entry", fontsize=10)
    ax.set_ylabel("Displacement/1k words", fontsize=10)
    ax.set_title("C. Length × Semantic Density", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.1)

    # Panel D: Straightness
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(ALL_MODELS))
    vals = [straightness[m] for m in ALL_MODELS]
    ax.bar(x, vals, color=[MODEL_COLORS[m] for m in ALL_MODELS], alpha=0.8,
           edgecolor="white", linewidth=1.5)
    ax.axhline(0.23, color="#999", linewidth=1.5, linestyle="--", alpha=0.5,
              label="Random-walk baseline (~0.23)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=11, fontweight="bold")
    ax.set_ylabel("Net / path length", fontsize=10)
    ax.set_title("D. Trajectory Straightness\n(lower = more convoluted)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.1, axis="y")

    # Panel E: Classifier accuracy
    ax = fig.add_subplot(gs[1, 1])
    bar_data = [acc_4way, acc_5way]
    bar_labels = ["4-way\n(Opus only)", "5-way\n(+Fable)"]
    bar_colors = ["#546E7A", "#7B1FA2"]
    bars = ax.bar([0, 1], bar_data, color=bar_colors, alpha=0.8,
                  edgecolor="white", linewidth=1.5, width=0.5)
    ax.axhline(0.25, color="#1976D2", linewidth=1, linestyle="--", alpha=0.5, label="4-way chance")
    ax.axhline(0.20, color="#7B1FA2", linewidth=1, linestyle="--", alpha=0.5, label="5-way chance")
    for i, v in enumerate(bar_data):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("LOB-CV accuracy", fontsize=10)
    ax.set_title("E. Source Classification\n(8 text features, leave-one-batch-out)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.1, axis="y")

    # Panel F: Key signature summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    col_labels = ["Model", "Words", "Vel.", "Straight.", "Dist. drop"]
    row_data = []
    for m in ALL_MODELS:
        wc = length_density[m]["median_wc"]
        vel = length_density[m]["disp_per_k"] * (wc / 1000)  # mean velocity
        st = straightness[m]
        drop = (1 - arc_data[m]["dist"][2] / arc_data[m]["dist"][0]) * 100 if arc_data[m]["dist"][0] > 0 else 0
        row_data.append([MODEL_LABELS[m], f"{wc:.0f}", f"{vel:.3f}", f"{st:.3f}", f"{drop:.0f}%"])

    table = ax.table(cellText=row_data, colLabels=col_labels, loc="center",
                    cellLoc="center", colColours=["#f0f0f0"]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for i, m in enumerate(ALL_MODELS):
        table[i+1, 0].set_text_props(color=MODEL_COLORS[m], fontweight="bold")
    ax.set_title("F. Signature Metrics Summary", fontsize=12, fontweight="bold", pad=20)

    fig.suptitle("Portrait Composite: The Quantitative Skeleton\n"
                 "Five models, three timepoints, one shared embedding space",
                 fontsize=16, fontweight="bold", y=0.98)
    json.dump({"straightness_panelD": {m: float(straightness[m]) for m in ALL_MODELS},
               "length_density_panelC": {m: {k: float(v) for k, v in length_density[m].items()} for m in ALL_MODELS}},
              open(os.path.join(a2_common.out_dir(VERSION), "figure5_portrait_composite.json"), "w"), indent=2)
    out = os.path.join(a2_common.out_dir(VERSION), "paper_fig_portrait_composite.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
