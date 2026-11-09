"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/paper_fig_register_carryover.py
(the plate of Section 6.2: lag-1 residual cosine against a within-notebook shuffle null, and its decay by entry gap 1-8).
Changed: input paths only. Opus and Fable 5 entries + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json or fable5_embeddings_cache.json); the PNG goes to out_<version>/. One added block
dumps lag-1 / decay / arc results at full precision (reads the dicts computed above it; changes nothing).
Usage: python register_carryover_v1.py v0|v1
"""
import e_common
VERSION = e_common.version_from_argv()
"""
Paper figure: Register carry-over (§6.2).
4-panel composite:
  A. Lag-1 residual cosine vs permutation null (all 5 models)
  B. Decay curve: residual association by entry gap (1-5)
  C. Coupling across the arc (early vs late position)
  D. Local following vs notebook voice (carry-over vs handwriting accuracy)
"""
import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
MODEL_COLORS = {
    "opus45": "#1976D2", "opus46": "#0D47A1",
    "opus47": "#F57C00", "opus48": "#D84315",
    "fable5": "#7B1FA2"
}


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("Loading data...")

    # Load embeddings
    # [E V1] was journal_embeddings_cache_all.json + fable5_embeddings_cache.json; now e_common (clean npz rows)
    _oe, opus_emb = e_common.load_opus(VERSION, range(1, 16))
    _fe, fable_emb = e_common.load_fable(VERSION)
    opus_emb = opus_emb / (np.linalg.norm(opus_emb, axis=1, keepdims=True) + 1e-12)
    fable_emb = fable_emb / (np.linalg.norm(fable_emb, axis=1, keepdims=True) + 1e-12)

    # Build indices
    # [E V1] was: indices rebuilt from Texture\Batch_N and Journaling_data\Batch_N; now from the e_common rows (same order)
    opus_flat = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"]} for e in _oe]
    fable_flat = [{"batch": e["batch"], "entry_num": e["entry_num"]} for e in _fe]

    def get_model_emb(model):
        if model in OPUS_MODELS:
            return opus_emb, opus_flat, lambda e: e["model"] == model
        else:
            return fable_emb, fable_flat, lambda e: True

    # ================================================================
    # Compute residuals: subtract per-position mean
    # ================================================================
    def get_residuals(model):
        emb, flat, filt = get_model_emb(model)
        m_idx = [i for i, e in enumerate(flat) if filt(e)]

        # Position means
        pos_means = {}
        for p in range(1, 21):
            p_idx = [i for i in m_idx if flat[i]["entry_num"] == p]
            if p_idx:
                pos_means[p] = emb[p_idx].mean(0)

        # Per-batch residual sequences
        batches = {}
        for i in m_idx:
            b = flat[i]["batch"]
            p = flat[i]["entry_num"]
            if p not in pos_means:
                continue
            resid = emb[i] - pos_means[p]
            if b not in batches:
                batches[b] = []
            batches[b].append((p, resid))

        # Sort each batch by entry number
        for b in batches:
            batches[b].sort(key=lambda x: x[0])

        return batches

    # ================================================================
    # A. Lag-1 residual cosine + permutation null
    # ================================================================
    print("\n  LAG-1 RESIDUAL COSINE")
    N_PERM = 500
    np.random.seed(42)

    lag1_results = {}
    for m in ALL_MODELS:
        batches = get_residuals(m)

        # Observed lag-1
        obs_cosines = []
        for b, entries in batches.items():
            for k in range(1, len(entries)):
                obs_cosines.append(cos_sim(entries[k-1][1], entries[k][1]))
        obs_mean = np.mean(obs_cosines)

        # Permutation null: shuffle within each batch
        null_means = []
        for _ in range(N_PERM):
            perm_cosines = []
            for b, entries in batches.items():
                resids = [e[1] for e in entries]
                np.random.shuffle(resids)
                for k in range(1, len(resids)):
                    perm_cosines.append(cos_sim(resids[k-1], resids[k]))
            null_means.append(np.mean(perm_cosines))

        null_mean = np.mean(null_means)
        null_std = np.std(null_means)
        excess = obs_mean - null_mean

        lag1_results[m] = {"observed": obs_mean, "null_mean": null_mean,
                           "null_std": null_std, "excess": excess}
        print(f"    {MODEL_LABELS[m]:>10}: observed={obs_mean:.4f}, null={null_mean:.4f}±{null_std:.4f}, "
              f"excess=+{excess:.3f}")

    # ================================================================
    # B. Decay by gap (1-8)
    # ================================================================
    print("\n  DECAY BY GAP")
    MAX_GAP = 8
    decay_results = {m: {} for m in ALL_MODELS}

    for m in ALL_MODELS:
        batches = get_residuals(m)
        for gap in range(1, MAX_GAP + 1):
            cosines = []
            for b, entries in batches.items():
                for k in range(gap, len(entries)):
                    cosines.append(cos_sim(entries[k-gap][1], entries[k][1]))
            if cosines:
                decay_results[m][gap] = np.mean(cosines)

        gaps = sorted(decay_results[m].keys())
        vals = [decay_results[m][g] for g in gaps]
        print(f"    {MODEL_LABELS[m]:>10}: " + " ".join(f"g{g}={v:.3f}" for g, v in zip(gaps, vals)))

    # ================================================================
    # C. Coupling across the arc (early vs late position)
    # ================================================================
    print("\n  COUPLING ACROSS ARC")
    arc_results = {}

    for m in ALL_MODELS:
        batches = get_residuals(m)
        early_cos = []
        late_cos = []

        for b, entries in batches.items():
            for k in range(1, len(entries)):
                pos = entries[k][0]
                c = cos_sim(entries[k-1][1], entries[k][1])
                if pos <= 10:
                    early_cos.append(c)
                else:
                    late_cos.append(c)

        early_m = np.mean(early_cos) if early_cos else 0
        late_m = np.mean(late_cos) if late_cos else 0
        arc_results[m] = {"early": early_m, "late": late_m}
        print(f"    {MODEL_LABELS[m]:>10}: early(1-10)={early_m:.4f}, late(11-20)={late_m:.4f}, "
              f"Δ={late_m - early_m:+.4f}")

    # ================================================================
    # FIGURE: 4-panel composite
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    # Panel A: Lag-1 excess over null
    ax = axes[0, 0]
    x = np.arange(len(ALL_MODELS))
    excess_vals = [lag1_results[m]["excess"] for m in ALL_MODELS]
    null_stds = [lag1_results[m]["null_std"] * 2 for m in ALL_MODELS]  # 2σ

    bars = ax.bar(x, excess_vals, color=[MODEL_COLORS[m] for m in ALL_MODELS],
                  alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.errorbar(x, [0]*len(x), yerr=null_stds, fmt="none", ecolor="#999",
               capsize=5, linewidth=1.5, label="±2σ null")
    for i, v in enumerate(excess_vals):
        ax.text(i, v + 0.005, f"+{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=12, fontweight="bold")
    ax.set_ylabel("Excess lag-1 residual cosine\n(observed − permutation null)", fontsize=10)
    ax.set_title("A. Register Carry-Over: Lag-1 Excess\nAll models positive; degree, not kind",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.1, axis="y")

    # Panel B: Decay curves
    ax = axes[0, 1]
    for m in ALL_MODELS:
        gaps = sorted(decay_results[m].keys())
        vals = [decay_results[m][g] for g in gaps]
        ax.plot(gaps, vals, "-o", color=MODEL_COLORS[m], linewidth=2.5,
               markersize=7, label=MODEL_LABELS[m], alpha=0.85)

    ax.axhline(0, color="#999", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Entry gap", fontsize=11)
    ax.set_ylabel("Mean residual cosine", fontsize=11)
    ax.set_title("B. Decay by Entry Gap\nHow fast does carry-over fade?",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.1)

    # Panel C: Coupling across arc
    ax = axes[1, 0]
    width = 0.32
    early_vals = [arc_results[m]["early"] for m in ALL_MODELS]
    late_vals = [arc_results[m]["late"] for m in ALL_MODELS]

    ax.bar(x - width/2, early_vals, width, color=[MODEL_COLORS[m] for m in ALL_MODELS],
           alpha=0.45, edgecolor="white", linewidth=1.5, label="Early (pos 1-10)")
    ax.bar(x + width/2, late_vals, width, color=[MODEL_COLORS[m] for m in ALL_MODELS],
           alpha=0.9, edgecolor="white", linewidth=1.5, label="Late (pos 11-20)")

    for i in range(len(ALL_MODELS)):
        delta = late_vals[i] - early_vals[i]
        y_top = max(early_vals[i], late_vals[i])
        ax.text(i, y_top + 0.003, f"Δ{delta:+.3f}",
               ha="center", fontsize=9, fontweight="bold",
               color="#D32F2F" if delta < -0.005 else "#388E3C" if delta > 0.005 else "#666")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean lag-1 residual cosine", fontsize=10)
    ax.set_title("C. Coupling Across the Arc\nDoes carry-over loosen or tighten?",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.1, axis="y")

    # Panel D: Local following vs notebook voice
    ax = axes[1, 1]
    # Handwriting accuracies from earlier computation
    handwriting = {"opus45": 0.773, "opus46": 0.457, "opus47": 0.716,
                   "opus48": 0.577, "fable5": 0.862}

    for m in ALL_MODELS:
        ax.scatter(lag1_results[m]["excess"], handwriting[m],
                  s=220, color=MODEL_COLORS[m], edgecolor="white",
                  linewidth=2, zorder=5, alpha=0.9)
        ox, oy = 0.008, 0.01
        if m == "opus46":
            ox, oy = 0.008, -0.03
        elif m == "opus48":
            ox, oy = -0.04, -0.025
        ax.annotate(MODEL_LABELS[m], (lag1_results[m]["excess"] + ox, handwriting[m] + oy),
                   fontsize=11, fontweight="bold", color=MODEL_COLORS[m])

    ax.set_xlabel("Lag-1 carry-over (local following)", fontsize=11)
    ax.set_ylabel("Batch handwriting accuracy\n(notebook-level voice)", fontsize=11)
    ax.set_title("D. Two Kinds of Persistence\nLocal following ≠ notebook commitment",
                fontsize=12, fontweight="bold")
    ax.grid(alpha=0.15)

    fig.suptitle("Register Carry-Over: Degree, Not Kind\n"
                 "All models carry register forward; they differ in how far and how locally",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(e_common.out_dir(VERSION), "paper_fig_register_carryover.png")  # [E V1] was TEXTURE_BASE
    json.dump({"lag1": {m: {k: float(v) for k, v in d.items()} for m, d in lag1_results.items()},
               "decay_gap_1_8": {m: {int(g): float(v) for g, v in d.items()} for m, d in decay_results.items()},
               "arc": {m: {k: float(v) for k, v in d.items()} for m, d in arc_results.items()}},
              open(os.path.join(e_common.out_dir(VERSION), "register_carryover.json"), "w"), indent=1)  # [E V1] added
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
