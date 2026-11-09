"""
Texture Journal Embedding Analysis
~1199 journal entries across 15 batches x 4 models, 20 entries each.
Semantic trajectories, cross-batch convergence, coupling, drift, early vs late.
"""
import json, os, sys, time, re
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

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
CACHE_FILE = os.path.join(BASE, "journal_embeddings_cache_all.json")
MODELS = ["opus45", "opus46", "opus47", "opus48"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
MODEL_COLORS = {"opus45": "#1976D2", "opus46": "#0D47A1", "opus47": "#F57C00", "opus48": "#D84315"}


def load_journals():
    entries = []
    for batch_num in range(1, 16):
        batch_dir = os.path.join(BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        for m in MODELS:
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


def embed_all(entries):
    if os.path.exists(CACHE_FILE):
        print(f"  Loading cached from {os.path.basename(CACHE_FILE)}")
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        if len(cache["embeddings"]) == len(entries):
            return np.array(cache["embeddings"])
        print(f"  Cache size mismatch ({len(cache['embeddings'])} vs {len(entries)}), re-embedding...")

    client = OpenAI()
    texts = [e["text"] for e in entries]
    all_emb = []
    total_tokens = 0
    batch_size = 50  # smaller batches for longer texts

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        n = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        print(f"  Embedding batch {n}/{total_batches} ({len(batch)} texts)...")
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        for item in resp.data:
            all_emb.append(item.embedding)
        total_tokens += resp.usage.total_tokens
        time.sleep(0.3)

    print(f"  Total tokens: {total_tokens:,} (cost: ~${total_tokens * 0.13 / 1_000_000:.4f})")
    with open(CACHE_FILE, "w") as f:
        json.dump({"embeddings": all_emb}, f)
    print(f"  Cached to {os.path.basename(CACHE_FILE)}")
    return np.array(all_emb)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def group_coherence(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / norms
    sim = normed @ normed.T
    n = len(vecs)
    if n < 2:
        return 0.0
    return sim[np.triu_indices(n, k=1)].mean()


def build_index(entries):
    """Build index: (model, batch) -> list of (global_idx, entry_num)"""
    idx = {}
    for i, e in enumerate(entries):
        key = (e["model"], e["batch"])
        idx.setdefault(key, []).append((i, e["entry_num"]))
    # sort by entry number
    for key in idx:
        idx[key].sort(key=lambda x: x[1])
    return idx


# =====================================================================
# 1. SEMANTIC TRAJECTORY PER BATCH
# =====================================================================

def plot_trajectories(entries, embeddings, index):
    print("\n[1/5] SEMANTIC TRAJECTORIES")

    # Collect all journal embeddings for UMAP
    all_idxs = list(range(len(entries)))
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    # One subplot per model, show all 15 batch trajectories
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, m in zip(axes_flat, MODELS):
        # Plot all points faintly
        model_idxs = [i for i, e in enumerate(entries) if e["model"] == m]
        ax.scatter(coords[model_idxs, 0], coords[model_idxs, 1],
                  c=MODEL_COLORS[m], alpha=0.15, s=15, edgecolors="none")

        # Draw trajectory lines for each batch
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_idxs = [gi for gi, en in index[key]]
            path = coords[batch_idxs]

            alpha = 0.6
            ax.plot(path[:, 0], path[:, 1], color=MODEL_COLORS[m],
                   alpha=alpha, linewidth=1.0)
            # Mark entry 1 and entry 20
            ax.scatter(path[0, 0], path[0, 1], c="white", s=60, zorder=5,
                      edgecolors=MODEL_COLORS[m], linewidths=1.5)
            ax.scatter(path[-1, 0], path[-1, 1], c=MODEL_COLORS[m], s=60, zorder=5,
                      edgecolors="white", linewidths=1.5, marker="D")

        ax.set_title(f"Opus {MODEL_LABELS[m]} -- 15 Trajectories",
                    fontsize=13, fontweight="bold", color=MODEL_COLORS[m])
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)

    # Shared legend
    axes_flat[0].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="gray", markersize=8, label="Entry 1 (start)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Entry 20 (end)"),
    ], fontsize=9, loc="upper right")

    plt.suptitle("Semantic Trajectories: 20-Entry Journals\n15 independent notebooks per model",
                fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(BASE, "journal_trajectories_umap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    return coords


# =====================================================================
# 2. CROSS-BATCH CONVERGENCE
# =====================================================================

def cross_batch_convergence(entries, embeddings, index):
    print("\n[2/5] CROSS-BATCH CONVERGENCE")

    print("\n  A. Final-entry coherence (do all batches end at the same place?)")
    print(f"  {'Model':>5}  {'Final coh':>9}  {'Entry1 coh':>10}  {'Delta':>7}  Interpretation")
    print("  " + "-" * 65)

    convergence_data = {}
    for m in MODELS:
        final_vecs = []
        first_vecs = []
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_entries = index[key]
            final_vecs.append(embeddings[batch_entries[-1][0]])
            first_vecs.append(embeddings[batch_entries[0][0]])

        final_coh = group_coherence(np.array(final_vecs))
        first_coh = group_coherence(np.array(first_vecs))
        delta = final_coh - first_coh
        convergence_data[m] = (first_coh, final_coh, delta)

        if delta > 0.02:
            interp = "CONVERGES: endings tighter than openings"
        elif delta < -0.02:
            interp = "DIVERGES: endings more scattered"
        else:
            interp = "STABLE: similar spread"

        print(f"  {MODEL_LABELS[m]:>5}  {final_coh:>9.4f}  {first_coh:>10.4f}  {delta:>+7.4f}  {interp}")

    # B. Per-entry cross-batch coherence (coherence at each entry position)
    print("\n  B. Cross-batch coherence at each entry position")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for m in MODELS:
        coherences = []
        for entry_num in range(1, 21):
            vecs = []
            for batch_num in range(1, 16):
                key = (m, batch_num)
                if key not in index:
                    continue
                matches = [(gi, en) for gi, en in index[key] if en == entry_num]
                if matches:
                    vecs.append(embeddings[matches[0][0]])
            if len(vecs) >= 2:
                coherences.append(group_coherence(np.array(vecs)))
            else:
                coherences.append(None)

        valid_x = [i for i, c in enumerate(coherences) if c is not None]
        valid_y = [coherences[i] for i in valid_x]
        ax.plot([x + 1 for x in valid_x], valid_y, color=MODEL_COLORS[m],
               linewidth=2.5, marker="o", markersize=6, markeredgecolor="white",
               markeredgewidth=1, label=f"Opus {MODEL_LABELS[m]}")

    ax.set_xlabel("Entry Number", fontsize=12)
    ax.set_ylabel("Cross-batch Coherence\n(15 notebooks at same position)", fontsize=11)
    ax.set_title("Do Notebooks Converge to the Same Place?\nCross-batch coherence at each entry position",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 20.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE, "journal_crossbatch_coherence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =====================================================================
# 3. ENTRY-TO-ENTRY SEMANTIC COUPLING
# =====================================================================

def semantic_coupling(entries, embeddings, index):
    print("\n[3/5] ENTRY-TO-ENTRY SEMANTIC COUPLING")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    print(f"\n  {'Model':>5}  {'Mean coupling':>13}  {'Early(1-5)':>10}  {'Late(16-20)':>11}  {'Trend':>8}")
    print("  " + "-" * 60)

    for m in MODELS:
        # Collect coupling at each step across all batches
        step_couplings = {step: [] for step in range(1, 20)}

        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_entries = index[key]
            for j in range(len(batch_entries) - 1):
                gi_curr = batch_entries[j][0]
                gi_next = batch_entries[j + 1][0]
                sim = cosine_sim(embeddings[gi_curr], embeddings[gi_next])
                step = j + 1  # step 1 = between entry 1 and 2
                step_couplings[step].append(sim)

        steps = sorted(step_couplings.keys())
        means = [np.mean(step_couplings[s]) for s in steps]

        early = np.mean([np.mean(step_couplings[s]) for s in range(1, 5)])
        late = np.mean([np.mean(step_couplings[s]) for s in range(16, 20) if step_couplings[s]])
        overall = np.mean([v for vals in step_couplings.values() for v in vals])
        trend = "TIGHTENS" if late > early + 0.01 else ("LOOSENS" if late < early - 0.01 else "STABLE")

        print(f"  {MODEL_LABELS[m]:>5}  {overall:>13.4f}  {early:>10.4f}  {late:>11.4f}  {trend:>8}")

        ax.plot(steps, means, color=MODEL_COLORS[m], linewidth=2.5,
               marker="o", markersize=5, markeredgecolor="white",
               markeredgewidth=1, label=f"Opus {MODEL_LABELS[m]}")

    ax.set_xlabel("Step (between entry N and N+1)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Entry-to-Entry Semantic Coupling Over Time\nDoes consecutive similarity tighten or loosen?",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 19.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE, "journal_coupling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =====================================================================
# 4. DRIFT FROM ENTRY 1
# =====================================================================

def drift_from_start(entries, embeddings, index):
    print("\n[4/5] DRIFT FROM ENTRY 1")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: mean drift across batches
    ax = axes[0]
    for m in MODELS:
        drift_by_entry = {en: [] for en in range(1, 21)}

        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_entries = index[key]
            entry1_emb = embeddings[batch_entries[0][0]]

            for gi, en in batch_entries:
                drift = 1 - cosine_sim(entry1_emb, embeddings[gi])
                drift_by_entry[en].append(drift)

        entry_nums = sorted(drift_by_entry.keys())
        means = [np.mean(drift_by_entry[en]) for en in entry_nums]
        stds = [np.std(drift_by_entry[en]) for en in entry_nums]

        ax.plot(entry_nums, means, color=MODEL_COLORS[m], linewidth=2.5,
               marker="o", markersize=5, markeredgecolor="white",
               markeredgewidth=1, label=f"Opus {MODEL_LABELS[m]}")
        ax.fill_between(entry_nums,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=MODEL_COLORS[m], alpha=0.1)

    ax.set_xlabel("Entry Number", fontsize=12)
    ax.set_ylabel("Cosine Distance from Entry 1", fontsize=11)
    ax.set_title("How Far Does Each Model Travel\nFrom Its Starting Point?",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 20.5)
    ax.grid(axis="y", alpha=0.3)

    # Right: drift at entry 20, per batch (spread)
    ax = axes[1]
    for mi, m in enumerate(MODELS):
        final_drifts = []
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_entries = index[key]
            entry1_emb = embeddings[batch_entries[0][0]]
            final_emb = embeddings[batch_entries[-1][0]]
            drift = 1 - cosine_sim(entry1_emb, final_emb)
            final_drifts.append(drift)

        x_pos = [0, 1, 3, 4][mi]
        bp = ax.boxplot([final_drifts], positions=[x_pos], widths=0.5,
                       patch_artist=True, showfliers=True)
        bp["boxes"][0].set_facecolor(MODEL_COLORS[m])
        bp["boxes"][0].set_alpha(0.7)
        bp["medians"][0].set_color("white")
        bp["medians"][0].set_linewidth(2)

    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["4.5", "4.6", "4.7", "4.8"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Cosine Distance: Entry 1 to Entry 20", fontsize=11)
    ax.set_title("Final Drift From Start\nAcross 15 Notebooks",
                fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(BASE, "journal_drift.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # Print summary
    print(f"\n  {'Model':>5}  {'Mean drift@20':>13}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print("  " + "-" * 45)
    for m in MODELS:
        drifts = []
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_entries = index[key]
            e1 = embeddings[batch_entries[0][0]]
            e20 = embeddings[batch_entries[-1][0]]
            drifts.append(1 - cosine_sim(e1, e20))
        print(f"  {MODEL_LABELS[m]:>5}  {np.mean(drifts):>13.4f}  {np.std(drifts):>6.4f}  "
              f"{np.min(drifts):>6.4f}  {np.max(drifts):>6.4f}")


# =====================================================================
# 5. EARLY vs LATE SEMANTIC SPACE
# =====================================================================

def early_vs_late(entries, embeddings, index):
    print("\n[5/5] EARLY vs LATE SEMANTIC SPACE")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    # PCA on all journal data for consistent space
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(embeddings)

    print(f"\n  {'Model':>5}  {'Early coh':>9}  {'Late coh':>8}  {'Cross':>6}  {'Shift':>6}  Interpretation")
    print("  " + "-" * 65)

    for ax, m in zip(axes_flat, MODELS):
        early_idxs = []
        late_idxs = []

        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in index:
                continue
            batch_entries = index[key]
            for gi, en in batch_entries:
                if en <= 5:
                    early_idxs.append(gi)
                elif en >= 16:
                    late_idxs.append(gi)

        early_coords = all_coords[early_idxs]
        late_coords = all_coords[late_idxs]

        ax.scatter(early_coords[:, 0], early_coords[:, 1],
                  c=MODEL_COLORS[m], alpha=0.4, s=30, marker="o",
                  edgecolors="none", label="Entries 1-5")
        ax.scatter(late_coords[:, 0], late_coords[:, 1],
                  c=MODEL_COLORS[m], alpha=0.8, s=30, marker="D",
                  edgecolors="white", linewidths=0.5, label="Entries 16-20")

        # Centroids
        ec = early_coords.mean(axis=0)
        lc = late_coords.mean(axis=0)
        ax.scatter(*ec, c="white", s=150, marker="o", edgecolors=MODEL_COLORS[m],
                  linewidths=2, zorder=10)
        ax.scatter(*lc, c=MODEL_COLORS[m], s=150, marker="D", edgecolors="white",
                  linewidths=2, zorder=10)
        ax.annotate("", xy=lc, xytext=ec,
                    arrowprops=dict(arrowstyle="->", color=MODEL_COLORS[m],
                                   lw=2.5, connectionstyle="arc3,rad=0.1"))

        ax.set_title(f"Opus {MODEL_LABELS[m]}", fontsize=13, fontweight="bold",
                    color=MODEL_COLORS[m])
        ax.legend(fontsize=9)
        ax.set_xlabel("PCA 1", fontsize=9)
        ax.set_ylabel("PCA 2", fontsize=9)

        # Metrics
        early_coh = group_coherence(embeddings[early_idxs])
        late_coh = group_coherence(embeddings[late_idxs])

        normed_early = embeddings[early_idxs] / np.linalg.norm(embeddings[early_idxs], axis=1, keepdims=True)
        normed_late = embeddings[late_idxs] / np.linalg.norm(embeddings[late_idxs], axis=1, keepdims=True)
        cross = (normed_early @ normed_late.T).mean()

        early_c = embeddings[early_idxs].mean(axis=0)
        late_c = embeddings[late_idxs].mean(axis=0)
        shift = 1 - cosine_sim(early_c, late_c)

        if late_coh > early_coh + 0.01:
            interp = "TIGHTENS with accumulation"
        elif late_coh < early_coh - 0.01:
            interp = "LOOSENS with accumulation"
        else:
            interp = "STABLE spread"

        print(f"  {MODEL_LABELS[m]:>5}  {early_coh:>9.4f}  {late_coh:>8.4f}  {cross:>6.4f}  "
              f"{shift:>6.4f}  {interp}")

    plt.suptitle("Early (1-5) vs Late (16-20) Entries in PCA Space\nArrows show centroid movement with accumulation",
                fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(BASE, "journal_early_vs_late.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("Loading journal entries...")
    entries = load_journals()
    print(f"  {len(entries)} entries loaded")

    print("\nEmbedding with text-embedding-3-large...")
    embeddings = embed_all(entries)
    print(f"  Shape: {embeddings.shape}")

    index = build_index(entries)

    coords = plot_trajectories(entries, embeddings, index)
    cross_batch_convergence(entries, embeddings, index)
    semantic_coupling(entries, embeddings, index)
    drift_from_start(entries, embeddings, index)
    early_vs_late(entries, embeddings, index)

    print("\n" + "=" * 70)
    print("JOURNAL ANALYSIS COMPLETE")
    print("=" * 70)
