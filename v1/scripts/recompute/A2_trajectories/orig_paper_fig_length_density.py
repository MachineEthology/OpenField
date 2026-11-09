"""
Paper figure: Length x Displacement-Density scatter (all 5 models).
Section 5.1.1.4 — "[Figure slot: length x density scatter, 5 models — Fable to be added]"
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


def cosine_dist(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def main():
    print("Loading Opus journals...")
    opus_entries = []
    for bn in range(1, 16):
        bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if not os.path.isdir(bd):
            continue
        for m in OPUS_MODELS:
            files = [f for f in os.listdir(bd) if m in f and "#" not in f]
            if files:
                data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
                for i, entry in enumerate(data):
                    opus_entries.append({"model": m, "batch": bn, "entry_num": i + 1,
                                         "text": entry["text"]})

    print("Loading Fable journals...")
    fable_entries = []
    for bn in range(1, 16):
        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            for i, entry in enumerate(data):
                if entry["text"].strip() != "[silence]":
                    fable_entries.append({"model": "fable5", "batch": bn, "entry_num": i + 1,
                                          "text": entry["text"]})

    print("Loading embeddings...")
    opus_emb = np.array(json.load(open(
        os.path.join(TEXTURE_BASE, "journal_embeddings_cache_all.json")))["embeddings"])
    fable_emb = np.array(json.load(open(
        os.path.join(TEXTURE_BASE, "fable5_embeddings_cache.json")))["embeddings"])

    # Normalize
    opus_emb = opus_emb / (np.linalg.norm(opus_emb, axis=1, keepdims=True) + 1e-12)
    fable_emb = fable_emb / (np.linalg.norm(fable_emb, axis=1, keepdims=True) + 1e-12)

    # Compute per-model metrics
    results = {}

    # Opus: compute per-notebook displacement and word count
    for m in OPUS_MODELS:
        m_idx = [i for i, e in enumerate(opus_entries) if e["model"] == m]
        word_counts = []
        velocities = []
        for i in m_idx:
            word_counts.append(len(opus_entries[i]["text"].split()))
        # Velocity: cosine distance between consecutive entries in same batch
        for bn in range(1, 16):
            bn_idx = [i for i, e in enumerate(opus_entries)
                      if e["model"] == m and e["batch"] == bn]
            bn_idx_sorted = sorted(bn_idx, key=lambda i: opus_entries[i]["entry_num"])
            for k in range(1, len(bn_idx_sorted)):
                velocities.append(cosine_dist(opus_emb[bn_idx_sorted[k]],
                                              opus_emb[bn_idx_sorted[k-1]]))

        median_wc = np.median(word_counts)
        mean_vel = np.mean(velocities) if velocities else 0
        # Displacement per 1000 words = mean_velocity / (median_words / 1000)
        disp_per_k = mean_vel / (median_wc / 1000)
        results[m] = {"median_words": median_wc, "mean_velocity": mean_vel,
                      "disp_per_k": disp_per_k}
        print(f"  {MODEL_LABELS[m]:>10}: median {median_wc:.0f} words, "
              f"vel {mean_vel:.4f}, disp/1k {disp_per_k:.4f}")

    # Fable: same
    fable_wc = []
    fable_vel = []
    for i, e in enumerate(fable_entries):
        fable_wc.append(len(e["text"].split()))
    for bn in range(1, 16):
        bn_idx = [i for i, e in enumerate(fable_entries) if e["batch"] == bn]
        bn_idx_sorted = sorted(bn_idx, key=lambda i: fable_entries[i]["entry_num"])
        for k in range(1, len(bn_idx_sorted)):
            fable_vel.append(cosine_dist(fable_emb[bn_idx_sorted[k]],
                                         fable_emb[bn_idx_sorted[k-1]]))

    median_wc = np.median(fable_wc)
    mean_vel = np.mean(fable_vel)
    disp_per_k = mean_vel / (median_wc / 1000)
    results["fable5"] = {"median_words": median_wc, "mean_velocity": mean_vel,
                         "disp_per_k": disp_per_k}
    print(f"  {'Fable 5':>10}: median {median_wc:.0f} words, "
          f"vel {mean_vel:.4f}, disp/1k {disp_per_k:.4f}")

    # ================================================================
    # Figure: scatter plot
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    for m in ALL_MODELS:
        r = results[m]
        ax.scatter(r["median_words"], r["disp_per_k"], s=280,
                  color=MODEL_COLORS[m], edgecolor="white", linewidth=2,
                  zorder=5, alpha=0.9)
        # Label offset to avoid overlap
        offset_x = 15
        offset_y = 0.003
        if m == "opus46":
            offset_x = -80
            offset_y = -0.012
        elif m == "opus48":
            offset_y = -0.012
        elif m == "fable5":
            offset_x = -60
            offset_y = 0.008
        ax.annotate(MODEL_LABELS[m],
                   (r["median_words"] + offset_x, r["disp_per_k"] + offset_y),
                   fontsize=13, fontweight="bold", color=MODEL_COLORS[m])

    ax.set_xlabel("Median entry length (words)", fontsize=13)
    ax.set_ylabel("Embedding displacement per 1,000 words", fontsize=13)
    ax.set_title("Length vs Semantic Density\n"
                 "Short-dense writers vs long-diffuse writers",
                 fontsize=15, fontweight="bold")
    ax.grid(alpha=0.15)

    # Add interpretive annotations
    ax.text(0.95, 0.95, "short + dense\n(high displacement per word)",
           transform=ax.transAxes, fontsize=9, ha="right", va="top",
           color="#666", style="italic")
    ax.text(0.95, 0.05, "long + diffuse\n(low displacement per word)",
           transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
           color="#666", style="italic")

    plt.tight_layout()
    out = os.path.join(TEXTURE_BASE, "paper_fig_length_density.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
