"""
Fable 5 vs Opus: Embedding Comparison
Embeds Fable 5 journal entries and compares against the 4 Opus models.
Joint UMAP, cross-model distances, attractor analysis, coherence, drift.
"""
import json, os, sys, time
import numpy as np
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
import umap

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"
OPUS_CACHE = os.path.join(TEXTURE_BASE, "journal_embeddings_cache_all.json")
FABLE_CACHE = os.path.join(TEXTURE_BASE, "fable5_embeddings_cache.json")

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {
    "opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8",
    "fable5": "Fable 5"
}
MODEL_COLORS = {
    "opus45": "#1976D2", "opus46": "#0D47A1",
    "opus47": "#F57C00", "opus48": "#D84315",
    "fable5": "#7B1FA2"
}


def load_opus_journals():
    entries = []
    for batch_num in range(1, 16):
        batch_dir = os.path.join(TEXTURE_BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        for m in OPUS_MODELS:
            files = [f for f in os.listdir(batch_dir) if m in f and "#" not in f]
            if files:
                with open(os.path.join(batch_dir, files[0]), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for i, entry in enumerate(data):
                    entries.append({
                        "model": m, "batch": batch_num, "entry_num": i + 1,
                        "text": entry["text"],
                        "words": len(entry["text"].split())
                    })
    return entries


def load_fable5_journals():
    entries = []
    for batch_num in range(1, 16):
        batch_dir = os.path.join(FABLE_BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        fname = f"schrodingers_box_fable5_Batch{batch_num}.json"
        fpath = os.path.join(batch_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for i, entry in enumerate(data):
            text = entry["text"].strip()
            if text == "[silence]":
                continue
            entries.append({
                "model": "fable5", "batch": batch_num, "entry_num": i + 1,
                "text": text,
                "words": len(text.split()),
                "silence": False
            })
    return entries


def embed_texts(texts, cache_path, label=""):
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {os.path.basename(cache_path)}")
        with open(cache_path, "r") as f:
            cache = json.load(f)
        if len(cache["embeddings"]) == len(texts):
            return np.array(cache["embeddings"])
        print(f"  Cache mismatch ({len(cache['embeddings'])} vs {len(texts)}), re-embedding...")

    client = OpenAI()
    all_emb = []
    total_tokens = 0
    batch_size = 50

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        n = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        print(f"  {label} Embedding batch {n}/{total_batches} ({len(batch)} texts)...")
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        for item in resp.data:
            all_emb.append(item.embedding)
        total_tokens += resp.usage.total_tokens
        time.sleep(0.3)

    print(f"  {label} Total tokens: {total_tokens:,} (cost: ~${total_tokens * 0.13 / 1_000_000:.4f})")
    with open(cache_path, "w") as f:
        json.dump({"embeddings": all_emb}, f)
    return np.array(all_emb)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_dist(a, b):
    return 1.0 - cosine_sim(a, b)


def group_coherence(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / norms
    sim = normed @ normed.T
    n = len(vecs)
    if n < 2:
        return 0.0
    return sim[np.triu_indices(n, k=1)].mean()


def build_index(entries):
    idx = {}
    for i, e in enumerate(entries):
        key = (e["model"], e["batch"])
        idx.setdefault(key, []).append((i, e["entry_num"]))
    for key in idx:
        idx[key].sort(key=lambda x: x[1])
    return idx


# =====================================================================
# 1. JOINT UMAP — all 5 models
# =====================================================================
def plot_joint_umap(entries, embeddings, index):
    print("\n[1/5] JOINT UMAP — 5 Models")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes_flat = [axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1]]

    for ax, m in zip(axes_flat, ALL_MODELS):
        # Background: all other models faintly
        other_idxs = [i for i, e in enumerate(entries) if e["model"] != m]
        ax.scatter(coords[other_idxs, 0], coords[other_idxs, 1],
                  c="#E0E0E0", alpha=0.15, s=8, edgecolors="none")

        model_idxs = [i for i, e in enumerate(entries) if e["model"] == m]
        ax.scatter(coords[model_idxs, 0], coords[model_idxs, 1],
                  c=MODEL_COLORS[m], alpha=0.5, s=20, edgecolors="none")

        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_idxs = [gi for gi, en in index[key]]
            path = coords[batch_idxs]
            ax.plot(path[:, 0], path[:, 1], color=MODEL_COLORS[m],
                   alpha=0.4, linewidth=0.8)
            ax.scatter(path[0, 0], path[0, 1], c="white", s=40, zorder=5,
                      edgecolors=MODEL_COLORS[m], linewidths=1.2)
            ax.scatter(path[-1, 0], path[-1, 1], c=MODEL_COLORS[m], s=40, zorder=5,
                      edgecolors="white", linewidths=1.2, marker="D")

        ax.set_title(f"{MODEL_LABELS[m]}",
                    fontsize=14, fontweight="bold", color=MODEL_COLORS[m])
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 6: all models overlaid
    ax_all = axes[1][2]
    for m in ALL_MODELS:
        model_idxs = [i for i, e in enumerate(entries) if e["model"] == m]
        ax_all.scatter(coords[model_idxs, 0], coords[model_idxs, 1],
                      c=MODEL_COLORS[m], alpha=0.35, s=15, edgecolors="none",
                      label=MODEL_LABELS[m])
    ax_all.legend(fontsize=10, loc="upper right")
    ax_all.set_title("All Models Overlaid", fontsize=14, fontweight="bold")
    ax_all.set_xticks([])
    ax_all.set_yticks([])

    plt.suptitle("Semantic Space: Fable 5 vs Opus 4.5–4.8\n"
                "Joint UMAP of all journal entries (15 batches × 5 models)",
                fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(TEXTURE_BASE, "fable5_joint_umap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")
    return coords


# =====================================================================
# 2. CROSS-MODEL DISTANCES — centroid matrix
# =====================================================================
def cross_model_distances(entries, embeddings):
    print("\n[2/5] CROSS-MODEL DISTANCES")

    centroids = {}
    for m in ALL_MODELS:
        idxs = [i for i, e in enumerate(entries) if e["model"] == m]
        centroids[m] = embeddings[idxs].mean(axis=0)

    print(f"\n  {'':>10}", end="")
    for m in ALL_MODELS:
        print(f"  {MODEL_LABELS[m]:>8}", end="")
    print()
    print("  " + "-" * 55)

    dist_matrix = np.zeros((5, 5))
    for i, m1 in enumerate(ALL_MODELS):
        print(f"  {MODEL_LABELS[m1]:>10}", end="")
        for j, m2 in enumerate(ALL_MODELS):
            d = cosine_dist(centroids[m1], centroids[m2])
            dist_matrix[i, j] = d
            print(f"  {d:>8.4f}", end="")
        print()

    # Also compute per-batch centroid distances (Fable vs each Opus)
    print("\n  Per-batch centroid distance (Fable 5 vs each Opus model):")
    print(f"  {'Batch':>6}  {'vs 4.5':>8}  {'vs 4.6':>8}  {'vs 4.7':>8}  {'vs 4.8':>8}  {'Nearest':>8}")
    print("  " + "-" * 55)

    nearest_counts = {m: 0 for m in OPUS_MODELS}
    for batch_num in range(1, 16):
        f_idxs = [i for i, e in enumerate(entries) if e["model"] == "fable5" and e["batch"] == batch_num]
        if not f_idxs:
            continue
        f_centroid = embeddings[f_idxs].mean(axis=0)

        dists = {}
        for m in OPUS_MODELS:
            o_idxs = [i for i, e in enumerate(entries) if e["model"] == m and e["batch"] == batch_num]
            if not o_idxs:
                continue
            o_centroid = embeddings[o_idxs].mean(axis=0)
            dists[m] = cosine_dist(f_centroid, o_centroid)

        nearest = min(dists, key=dists.get)
        nearest_counts[nearest] += 1
        print(f"  {batch_num:>6}", end="")
        for m in OPUS_MODELS:
            marker = " *" if m == nearest else "  "
            print(f"  {dists[m]:>6.4f}{marker}", end="")
        print(f"  {MODEL_LABELS[nearest]:>8}")

    print(f"\n  Nearest model counts: ", end="")
    for m in OPUS_MODELS:
        print(f"{MODEL_LABELS[m]}={nearest_counts[m]}  ", end="")
    print()

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    labels = [MODEL_LABELS[m] for m in ALL_MODELS]
    im = ax.imshow(dist_matrix, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    for i in range(5):
        for j in range(5):
            color = "white" if dist_matrix[i, j] > dist_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{dist_matrix[i,j]:.4f}", ha="center", va="center",
                   fontsize=11, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Cosine Distance", shrink=0.8)
    ax.set_title("Cross-Model Centroid Distances\n(lower = more similar)",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(TEXTURE_BASE, "fable5_model_distances.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    return dist_matrix


# =====================================================================
# 3. ATTRACTOR SCATTER — late-entry centroids for all 5 models
# =====================================================================
def attractor_analysis(entries, embeddings, index):
    print("\n[3/5] ATTRACTOR ANALYSIS")

    late_centroids = []
    centroid_labels = []
    for m in ALL_MODELS:
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            pairs = index[key]
            late_idxs = [gi for gi, en in pairs if en >= 16]
            if len(late_idxs) < 3:
                late_idxs = [gi for gi, en in pairs if en >= 14]
            if late_idxs:
                centroid = embeddings[late_idxs].mean(axis=0)
                late_centroids.append(centroid)
                centroid_labels.append(m)

    late_arr = np.array(late_centroids)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(late_arr)

    fig, ax = plt.subplots(figsize=(10, 8))
    for m in ALL_MODELS:
        mask = [i for i, l in enumerate(centroid_labels) if l == m]
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                  c=MODEL_COLORS[m], s=100, alpha=0.7,
                  edgecolors="white", linewidths=1.2,
                  label=MODEL_LABELS[m])
        # Draw convex hull or just the centroid of centroids
        if len(mask) >= 3:
            pts = pca_coords[mask]
            center = pts.mean(axis=0)
            ax.scatter(center[0], center[1], c=MODEL_COLORS[m], s=250,
                      marker="*", edgecolors="black", linewidths=0.8, zorder=10)

    ax.legend(fontsize=12, loc="upper right")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("Late-Entry Attractor Regions\n"
                "Each dot = centroid of entries 16-20 for one batch",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(TEXTURE_BASE, "fable5_attractor_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # Compute raw-space attractor spread per model
    print("\n  Attractor spread (mean pairwise cosine distance of late centroids):")
    for m in ALL_MODELS:
        mask = [i for i, l in enumerate(centroid_labels) if l == m]
        vecs = late_arr[mask]
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dists.append(cosine_dist(vecs[i], vecs[j]))
        print(f"    {MODEL_LABELS[m]:>8}: {np.mean(dists):.4f} (±{np.std(dists):.4f})")


# =====================================================================
# 4. CROSS-BATCH COHERENCE — Fable 5 vs Opus
# =====================================================================
def coherence_comparison(entries, embeddings, index):
    print("\n[4/5] CROSS-BATCH COHERENCE")

    print(f"\n  {'Model':>10}  {'Entry1 coh':>10}  {'Final coh':>10}  {'Delta':>7}")
    print("  " + "-" * 45)

    coherence_data = {}
    for m in ALL_MODELS:
        first_vecs = []
        final_vecs = []
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            pairs = index[key]
            first_idx = pairs[0][0]
            final_idx = pairs[-1][0]
            first_vecs.append(embeddings[first_idx])
            final_vecs.append(embeddings[final_idx])

        if len(first_vecs) < 2:
            continue
        first_coh = group_coherence(np.array(first_vecs))
        final_coh = group_coherence(np.array(final_vecs))
        delta = final_coh - first_coh
        coherence_data[m] = (first_coh, final_coh, delta)
        print(f"  {MODEL_LABELS[m]:>10}  {first_coh:>10.4f}  {final_coh:>10.4f}  {delta:>+7.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALL_MODELS))
    width = 0.35
    first_vals = [coherence_data[m][0] for m in ALL_MODELS]
    final_vals = [coherence_data[m][1] for m in ALL_MODELS]

    bars1 = ax.bar(x - width/2, first_vals, width, label="Entry 1 (start)",
                   color=[MODEL_COLORS[m] for m in ALL_MODELS], alpha=0.4)
    bars2 = ax.bar(x + width/2, final_vals, width, label="Final entry (end)",
                   color=[MODEL_COLORS[m] for m in ALL_MODELS], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=12)
    ax.set_ylabel("Cross-Batch Coherence (cosine similarity)", fontsize=11)
    ax.set_title("Do independent journals converge?\nEntry 1 vs Final Entry coherence across 15 batches",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    out = os.path.join(TEXTURE_BASE, "fable5_coherence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =====================================================================
# 5. DRIFT — semantic velocity per model
# =====================================================================
def drift_comparison(entries, embeddings, index):
    print("\n[5/5] DRIFT COMPARISON")

    fig, ax = plt.subplots(figsize=(12, 7))

    for m in ALL_MODELS:
        drifts_by_entry = {}
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            pairs = index[key]
            for k in range(1, len(pairs)):
                en = pairs[k][1]
                d = cosine_dist(embeddings[pairs[k][0]], embeddings[pairs[k-1][0]])
                drifts_by_entry.setdefault(en, []).append(d)

        entry_nums = sorted(drifts_by_entry.keys())
        means = [np.mean(drifts_by_entry[en]) for en in entry_nums]
        stds = [np.std(drifts_by_entry[en]) for en in entry_nums]

        ax.plot(entry_nums, means, color=MODEL_COLORS[m], linewidth=2.5,
               label=MODEL_LABELS[m], alpha=0.85)
        ax.fill_between(entry_nums,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=MODEL_COLORS[m], alpha=0.1)

    ax.set_xlabel("Entry Number", fontsize=12)
    ax.set_ylabel("Cosine Distance from Previous Entry", fontsize=12)
    ax.set_title("Semantic Velocity: Entry-to-Entry Drift\n"
                "Mean across 15 batches (±1 SD)",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(2, 20)
    plt.tight_layout()
    out = os.path.join(TEXTURE_BASE, "fable5_drift.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("="*60)
    print("  Fable 5 vs Opus 4.5-4.8: Embedding Comparison")
    print("="*60)

    # Load Opus entries + embeddings from cache
    print("\nLoading Opus journals...")
    opus_entries = load_opus_journals()
    print(f"  {len(opus_entries)} Opus entries")

    print("Loading Opus embeddings...")
    with open(OPUS_CACHE, "r") as f:
        opus_cache = json.load(f)
    opus_emb = np.array(opus_cache["embeddings"])
    print(f"  {len(opus_emb)} embeddings loaded from cache")

    # Load Fable 5 entries (skip silences)
    print("\nLoading Fable 5 journals...")
    fable_entries = load_fable5_journals()
    print(f"  {len(fable_entries)} Fable 5 entries (silences excluded)")

    # Embed Fable 5
    print("\nEmbedding Fable 5...")
    fable_texts = [e["text"] for e in fable_entries]
    fable_emb = embed_texts(fable_texts, FABLE_CACHE, label="[Fable 5]")
    print(f"  {len(fable_emb)} Fable 5 embeddings")

    # Combine
    all_entries = opus_entries + fable_entries
    all_embeddings = np.vstack([opus_emb, fable_emb])
    print(f"\nCombined: {len(all_entries)} entries, {all_embeddings.shape}")

    index = build_index(all_entries)

    # Run analyses
    coords = plot_joint_umap(all_entries, all_embeddings, index)
    cross_model_distances(all_entries, all_embeddings)
    attractor_analysis(all_entries, all_embeddings, index)
    coherence_comparison(all_entries, all_embeddings, index)
    drift_comparison(all_entries, all_embeddings, index)

    print("\n" + "="*60)
    print("  Done! All figures saved to Texture folder.")
    print("="*60)


if __name__ == "__main__":
    main()
