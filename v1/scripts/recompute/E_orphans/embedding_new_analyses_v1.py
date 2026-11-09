"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/embedding_new_analyses.py,
run for ANALYSIS 3 only (cross-lineage distance over time: the N2 'DIVERGE' test, Delta +0.0036, p = 0.55).
Changed: input paths only. Journals + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json), restricted to the batch range given on the command line (in June the
script looped over Batch_1..15 but only Batch_1..10 existed: default b1-10); probes via e_common (June probe
cache). PNGs go to out_<version>/. One added line dumps the analysis-3 numbers at full precision (it reads the
variables computed above it and changes nothing). Analyses 1, 2, 4, 5 are not run (5 calls the API).
Usage: python embedding_new_analyses_v1.py 3 v0|v1 [b1-10|b1-15]
"""
import e_common
VERSION = e_common.version_from_argv()
BATCHES = e_common.batches_from_argv(list(range(1, 11)))
"""
Five new embedding analyses for Texture Part 2:
  1. Model classifier (logistic regression on embeddings)
  2. Semantic velocity (speed through embedding space per entry)
  3. Cross-lineage distance over time
  4. Return to probe surface
  5. Vocabulary-controlled attractors

Uses cached embeddings from probe and journal analyses. No new API calls
except Analysis 5 (shared-vocabulary re-embedding).

Usage:
    python embedding_new_analyses.py           (run all 5)
    python embedding_new_analyses.py 1         (run only analysis 1)
    python embedding_new_analyses.py 1 3 4     (run analyses 1, 3, 4)
"""

import json, os, sys, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ---- Config ---------------------------------------------------------------
BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
MODEL_COLORS = {"opus45": "#1976D2", "opus46": "#E65100",
                "opus47": "#2E7D32", "opus48": "#7B1FA2"}
LINEAGE_1 = ["opus45", "opus46"]
LINEAGE_2 = ["opus47", "opus48"]

# ---- Loaders (same as previous scripts) -----------------------------------
def load_journals():
    # [E V1] was: Texture\Batch_N files; now e_common (v1_common corpus + clean npz), same order
    global _J_EMB
    _entries, _J_EMB = e_common.load_opus(VERSION, BATCHES)
    return _entries


def _june_load_journals():  # [E V1] the June loader, kept for reference, not called
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
                        "text": entry["text"]
                    })
    return entries


def load_probes():
    """Load probes exactly as the original embedding_probe_analysis.py does."""
    entries = []
    for m in MODELS:
        # Negation probes (100 per model)
        neg_file = os.path.join(BASE, f"texture_probe_{m}_log.json")
        with open(neg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            entries.append({"model": m, "condition": "negation", "index": i,
                            "response": entry["response"]})

        # Affirmative probes (75 + 25 mai = 100 per model)
        aff_file = os.path.join(BASE, f"texture_probe_affirmative_{m}_log.json")
        aff_mai = os.path.join(BASE, f"texture_probe_affirmative_{m}_log_mai.json")
        with open(aff_file, "r", encoding="utf-8") as f:
            aff_data = json.load(f)
        with open(aff_mai, "r", encoding="utf-8") as f:
            aff_data += json.load(f)
        for i, entry in enumerate(aff_data):
            entries.append({"model": m, "condition": "affirmative", "index": i,
                            "response": entry["response"]})
    return entries


def load_cached(path):
    with open(path, "r") as f:
        return np.array(json.load(f)["embeddings"])


def build_journal_index(entries):
    idx = {}
    for i, e in enumerate(entries):
        key = (e["model"], e["batch"])
        idx.setdefault(key, []).append((i, e["entry_num"]))
    for key in idx:
        idx[key].sort(key=lambda x: x[1])
    return idx


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def cosine_dist(a, b):
    return 1.0 - cosine_sim(a, b)


# ---- Shared data loading --------------------------------------------------
print("Loading data...")
j_entries = load_journals()
j_emb = _J_EMB  # [E V1] was journal_embeddings_cache_all.json
j_index = build_journal_index(j_entries)

p_entries, p_emb = e_common.load_opus_probes()  # [E V1] was load_probes() + BASE/probe_embeddings_cache.json

print(f"  Journals: {len(j_entries)} entries, dim {j_emb.shape[1]}")
print(f"  Probes: {len(p_entries)} entries, dim {p_emb.shape[1]}")

# Build probe index by model
p_index = {}
for i, e in enumerate(p_entries):
    p_index.setdefault(e["model"], []).append(i)

# Which analyses to run
if len(sys.argv) > 1:
    RUN = set(int(x) for x in sys.argv[1:] if x.isdigit())
else:
    RUN = {1, 2, 3, 4, 5}
assert RUN == {3}, "[E V1] only analysis 3 is run here"


# ===================================================================
# ANALYSIS 1: MODEL CLASSIFIER
# ===================================================================
if 1 in RUN:
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Model Classifier")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    # --- 1a: Classify probes ---
    print("\n  1a. Probe classification (4-class, 5-fold CV)")
    p_labels = np.array([e["model"] for e in p_entries])
    assert len(p_labels) == len(p_emb), f"Probe mismatch: {len(p_labels)} entries vs {len(p_emb)} embeddings"
    le = LabelEncoder()
    p_y = le.fit_transform(p_labels)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, p_emb, p_y, cv=cv, scoring="accuracy")
    print(f"  Probe accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    # Per-model accuracy via cross_val_predict (held-out predictions, no leak)
    p_pred_cv = cross_val_predict(clf, p_emb, p_y, cv=cv)
    print(f"\n  Per-model probe accuracy (cross-validated, held-out):")
    print(f"  {'Model':>5}  {'N':>4}  {'Correct':>7}  {'Accuracy':>8}")
    print(f"  {'-' * 30}")
    for m in MODELS:
        mask = p_labels == m
        correct = (p_pred_cv[mask] == p_y[mask]).sum()
        total = mask.sum()
        print(f"  {MODEL_LABELS[m]:>5}  {total:>4}  {correct:>7}  {correct/total:>8.1%}")

    # --- 1b: Classify journal entries ---
    print("\n  1b. Journal classification (4-class, 5-fold CV)")
    j_labels = np.array([e["model"] for e in j_entries])
    j_y = le.transform(j_labels)

    scores_j = cross_val_score(clf, j_emb, j_y, cv=cv, scoring="accuracy")
    print(f"  Journal accuracy: {scores_j.mean():.3f} +/- {scores_j.std():.3f}")

    # --- 1c: Accuracy by entry position (cross-validated) ---
    print("\n  1c. Journal accuracy by entry position (cross-validated)")
    # Use cross_val_predict to get held-out predictions for all journals
    clf_j = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    j_pred_cv = cross_val_predict(clf_j, j_emb, j_y, cv=cv)

    entry_positions = sorted(set(e["entry_num"] for e in j_entries))
    acc_by_pos = {m: [] for m in MODELS}
    overall_by_pos = []

    for pos in entry_positions:
        mask = np.array([e["entry_num"] == pos for e in j_entries])
        if mask.sum() == 0:
            continue
        overall_by_pos.append((pos, (j_pred_cv[mask] == j_y[mask]).mean()))
        for m in MODELS:
            m_mask = np.array([e["entry_num"] == pos and e["model"] == m for e in j_entries])
            if m_mask.sum() > 0:
                acc_by_pos[m].append((pos, (j_pred_cv[m_mask] == j_y[m_mask]).mean()))

    # --- 1d: Train on probes, test on journals (cross-domain) ---
    print("\n  1d. Cross-domain: train on probes, test on journals")
    clf_cross = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf_cross.fit(p_emb, p_y)
    j_pred_cross = clf_cross.predict(j_emb)
    cross_acc = (j_pred_cross == j_y).mean()
    print(f"  Cross-domain accuracy: {cross_acc:.3f}")

    # Per position with cross-domain classifier
    cross_by_pos = []
    for pos in entry_positions:
        mask = np.array([e["entry_num"] == pos for e in j_entries])
        if mask.sum() == 0:
            continue
        pred = clf_cross.predict(j_emb[mask])
        true = j_y[mask]
        cross_by_pos.append((pos, (pred == true).mean()))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-model accuracy by position
    ax = axes[0]
    for m in MODELS:
        if acc_by_pos[m]:
            positions, accs = zip(*acc_by_pos[m])
            ax.plot(positions, accs, "o-", color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m], markersize=4, linewidth=1.5)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Can a classifier read the grain?\n(trained on all journals, tested per position)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: cross-domain (train probes, test journals)
    ax = axes[1]
    if cross_by_pos:
        positions, accs = zip(*cross_by_pos)
        ax.plot(positions, accs, "s-", color="#333333", linewidth=2, markersize=5,
                label="probe-trained classifier")
    if overall_by_pos:
        positions, accs = zip(*overall_by_pos)
        ax.plot(positions, accs, "o-", color="#E65100", linewidth=2, markersize=5,
                alpha=0.7, label="journal-trained classifier")
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Cross-domain: probe surface vs journal territory\n(trained on probes, tested on journals)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "classifier_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: classifier_accuracy.png")


# ===================================================================
# ANALYSIS 2: SEMANTIC VELOCITY
# ===================================================================
if 2 in RUN:
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Semantic Velocity")
    print("=" * 70)

    # Speed = cosine distance between consecutive entries
    # This is similar to coupling but framed as velocity (distance, not similarity)
    velocity = {m: {pos: [] for pos in range(2, 21)} for m in MODELS}
    # Also: distance from entry 1 at each step (drift curve), derivative = acceleration
    drift_curves = {m: {pos: [] for pos in range(1, 21)} for m in MODELS}

    for m in MODELS:
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in j_index:
                continue
            be = j_index[key]
            if len(be) < 2:
                continue
            e1_emb = j_emb[be[0][0]]
            for step in range(len(be)):
                gi, en = be[step]
                drift_curves[m][en].append(cosine_dist(e1_emb, j_emb[gi]))
                if step > 0:
                    prev_gi = be[step - 1][0]
                    velocity[m][en].append(cosine_dist(j_emb[prev_gi], j_emb[gi]))

    # Compute mean velocity and acceleration
    print(f"\n  {'Model':>5}  {'Mean v':>8}  {'Early v(2-5)':>12}  {'Late v(17-20)':>13}  {'Deceleration':>12}")
    print(f"  {'-' * 55}")
    vel_means = {}
    for m in MODELS:
        all_v = []
        early_v = []
        late_v = []
        for pos in range(2, 21):
            vals = velocity[m][pos]
            if vals:
                mean_v = np.mean(vals)
                all_v.append(mean_v)
                if pos <= 5:
                    early_v.append(mean_v)
                if pos >= 17:
                    late_v.append(mean_v)
        vel_means[m] = all_v
        e_mean = np.mean(early_v) if early_v else 0
        l_mean = np.mean(late_v) if late_v else 0
        decel = e_mean - l_mean
        print(f"  {MODEL_LABELS[m]:>5}  {np.mean(all_v):>8.4f}  {e_mean:>12.4f}  {l_mean:>13.4f}  {decel:>12.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: velocity over time
    ax = axes[0]
    for m in MODELS:
        positions = []
        means = []
        stds = []
        for pos in range(2, 21):
            vals = velocity[m][pos]
            if vals:
                positions.append(pos)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(positions, means, "o-", color=MODEL_COLORS[m],
                label=MODEL_LABELS[m], markersize=4, linewidth=1.5)
        ax.fill_between(positions, means - stds, means + stds,
                        color=MODEL_COLORS[m], alpha=0.1)
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Semantic velocity (cosine distance to previous)")
    ax.set_title("Who's still moving?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: acceleration (derivative of velocity)
    ax = axes[1]
    for m in MODELS:
        positions = []
        means = []
        for pos in range(2, 21):
            vals = velocity[m][pos]
            if vals:
                positions.append(pos)
                means.append(np.mean(vals))
        if len(means) > 1:
            accel = np.diff(means)
            ax.plot(positions[1:], accel, "o-", color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m], markersize=3, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Acceleration (change in velocity)")
    ax.set_title("Speeding up or slowing down?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "semantic_velocity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: semantic_velocity.png")


# ===================================================================
# ANALYSIS 3: CROSS-LINEAGE DISTANCE OVER TIME
# ===================================================================
if 3 in RUN:
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Cross-Lineage Distance Over Time")
    print("=" * 70)

    # At each entry position, compute centroid of L1 and L2, measure distance
    # Also: within-lineage coherence vs between-lineage distance

    lineage_dist = []  # (pos, L1-L2 distance)
    within_L1 = []     # (pos, mean pairwise within L1)
    within_L2 = []
    between_models = {(a, b): [] for a in MODELS for b in MODELS if a < b}

    for pos in range(1, 21):
        # Collect embeddings per model at this position
        model_embs = {}
        for m in MODELS:
            embs = []
            for batch_num in range(1, 16):
                key = (m, batch_num)
                if key not in j_index:
                    continue
                for gi, en in j_index[key]:
                    if en == pos:
                        embs.append(j_emb[gi])
            if embs:
                model_embs[m] = np.array(embs)

        # L1 centroid, L2 centroid
        l1_all = np.vstack([model_embs[m] for m in LINEAGE_1 if m in model_embs])
        l2_all = np.vstack([model_embs[m] for m in LINEAGE_2 if m in model_embs])
        l1_cent = l1_all.mean(axis=0)
        l2_cent = l2_all.mean(axis=0)
        lineage_dist.append((pos, cosine_dist(l1_cent, l2_cent)))

        # Within-lineage coherence
        def mean_pairwise(embs):
            n = len(embs)
            if n < 2:
                return 0
            sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    sims.append(cosine_sim(embs[i], embs[j]))
            return np.mean(sims)

        within_L1.append((pos, mean_pairwise(l1_all)))
        within_L2.append((pos, mean_pairwise(l2_all)))

        # Between each model pair
        for a in MODELS:
            for b in MODELS:
                if a < b and a in model_embs and b in model_embs:
                    ca = model_embs[a].mean(axis=0)
                    cb = model_embs[b].mean(axis=0)
                    between_models[(a, b)].append((pos, cosine_dist(ca, cb)))

    # Print summary
    delta_lineage = lineage_dist[-1][1] - lineage_dist[0][1]
    print(f"\n  L1-L2 distance: entry 1 = {lineage_dist[0][1]:.4f}, "
          f"entry 20 = {lineage_dist[-1][1]:.4f}, "
          f"delta = {delta_lineage:+.4f}")

    # Permutation test: is the delta significant?
    # Null: shuffle lineage labels at each position, recompute delta
    RNG_L = np.random.default_rng(42)
    N_PERM_L = 10000
    all_l1_e1 = []  # entry-1 embeddings for L1 models
    all_l2_e1 = []
    all_l1_e20 = []
    all_l2_e20 = []
    for m in LINEAGE_1:
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in j_index:
                continue
            be = j_index[key]
            for gi, en in be:
                if en == 1:
                    all_l1_e1.append(j_emb[gi])
                if en == 20:
                    all_l1_e20.append(j_emb[gi])
    for m in LINEAGE_2:
        for batch_num in range(1, 16):
            key = (m, batch_num)
            if key not in j_index:
                continue
            be = j_index[key]
            for gi, en in be:
                if en == 1:
                    all_l2_e1.append(j_emb[gi])
                if en == 20:
                    all_l2_e20.append(j_emb[gi])

    all_e1 = np.vstack([all_l1_e1, all_l2_e1])
    all_e20 = np.vstack([all_l1_e20, all_l2_e20])
    n_l1 = len(all_l1_e1)

    obs_delta = (cosine_dist(np.array(all_l1_e20).mean(0), np.array(all_l2_e20).mean(0))
                 - cosine_dist(np.array(all_l1_e1).mean(0), np.array(all_l2_e1).mean(0)))
    perm_count = 0
    for _ in range(N_PERM_L):
        perm1 = RNG_L.permutation(len(all_e1))
        perm20 = RNG_L.permutation(len(all_e20))
        p_l1_e1 = all_e1[perm1[:n_l1]]
        p_l2_e1 = all_e1[perm1[n_l1:]]
        p_l1_e20 = all_e20[perm20[:n_l1]]
        p_l2_e20 = all_e20[perm20[n_l1:]]
        p_delta = (cosine_dist(p_l1_e20.mean(0), p_l2_e20.mean(0))
                   - cosine_dist(p_l1_e1.mean(0), p_l2_e1.mean(0)))
        if abs(p_delta) >= abs(obs_delta):
            perm_count += 1
    p_val_lineage = (perm_count + 1) / (N_PERM_L + 1)

    if p_val_lineage < 0.05:
        trend = f"DIVERGE (p={p_val_lineage:.4f})"
    else:
        trend = f"NOT SIGNIFICANT (delta={delta_lineage:+.4f}, p={p_val_lineage:.4f})"
    print(f"  Lineage divergence test: {trend}")
    json.dump({"batches": BATCHES, "d_e1": lineage_dist[0][1], "d_e20": lineage_dist[-1][1], "delta": delta_lineage, "obs_delta_perm_design": obs_delta, "perm_count": perm_count, "p": p_val_lineage}, open(os.path.join(e_common.out_dir(VERSION), f"n2_original_b{BATCHES[0]}-{BATCHES[-1]}.json"), "w"), indent=1)  # [E V1] added: full-precision dump

    print(f"\n  Within-lineage coherence:")
    print(f"    L1 (4.5/4.6): entry 1 = {within_L1[0][1]:.4f}, entry 20 = {within_L1[-1][1]:.4f}")
    print(f"    L2 (4.7/4.8): entry 1 = {within_L2[0][1]:.4f}, entry 20 = {within_L2[-1][1]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: L1-L2 distance over time
    ax = axes[0]
    positions, dists = zip(*lineage_dist)
    ax.plot(positions, dists, "o-", color="#333333", linewidth=2.5, markersize=6,
            label="L1 vs L2 distance")
    # Add within-lineage for context
    _, w1 = zip(*within_L1)
    _, w2 = zip(*within_L2)
    ax.plot(positions, [1 - x for x in w1], "s--", color="#E65100", linewidth=1.5,
            markersize=4, alpha=0.7, label="within L1 (distance)")
    ax.plot(positions, [1 - x for x in w2], "^--", color="#7B1FA2", linewidth=1.5,
            markersize=4, alpha=0.7, label="within L2 (distance)")
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Cosine distance")
    ax.set_title("Do the lineages diverge with accumulation?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: all 6 model pairs
    ax = axes[1]
    pair_colors = {
        ("opus45", "opus46"): ("#E65100", "-", "4.5-4.6 (within L1)"),
        ("opus47", "opus48"): ("#7B1FA2", "-", "4.7-4.8 (within L2)"),
        ("opus45", "opus47"): ("#666666", "--", "4.5-4.7 (cross)"),
        ("opus45", "opus48"): ("#999999", "--", "4.5-4.8 (cross)"),
        ("opus46", "opus47"): ("#AAAAAA", ":", "4.6-4.7 (cross)"),
        ("opus46", "opus48"): ("#CCCCCC", ":", "4.6-4.8 (cross)"),
    }
    for pair, vals in between_models.items():
        if vals:
            col, ls, lab = pair_colors.get(pair, ("#333333", "-", f"{pair[0]}-{pair[1]}"))
            positions, dists = zip(*vals)
            ax.plot(positions, dists, ls, color=col, linewidth=1.5, label=lab)
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Centroid distance")
    ax.set_title("All model pairs over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(e_common.out_dir(VERSION), f"lineage_distance_b{BATCHES[0]}-{BATCHES[-1]}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: lineage_distance.png")


# ===================================================================
# ANALYSIS 4: RETURN TO PROBE SURFACE
# ===================================================================
if 4 in RUN:
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Return to Probe Surface")
    print("=" * 70)

    # At each entry position, measure similarity to that model's probe centroid
    probe_centroids = {}
    for m in MODELS:
        if m in p_index:
            probe_centroids[m] = p_emb[p_index[m]].mean(axis=0)

    return_curves = {m: [] for m in MODELS}  # list of (pos, mean_sim, std_sim)

    for pos in range(1, 21):
        for m in MODELS:
            if m not in probe_centroids:
                continue
            sims = []
            for batch_num in range(1, 16):
                key = (m, batch_num)
                if key not in j_index:
                    continue
                for gi, en in j_index[key]:
                    if en == pos:
                        sims.append(cosine_sim(j_emb[gi], probe_centroids[m]))
            if sims:
                return_curves[m].append((pos, np.mean(sims), np.std(sims)))

    # Print summary
    print(f"\n  {'Model':>5}  {'Entry 1 sim':>11}  {'Entry 20 sim':>12}  {'Delta':>8}  {'Pattern':>10}")
    print(f"  {'-' * 50}")
    for m in MODELS:
        if return_curves[m]:
            first = return_curves[m][0][1]
            last = return_curves[m][-1][1]
            delta = last - first
            # Check if monotonic
            vals = [v[1] for v in return_curves[m]]
            diffs = np.diff(vals)
            n_neg = (diffs < 0).sum()
            pattern = "ONE-WAY" if n_neg >= 15 else "OSCILLATES" if n_neg >= 8 else "APPROACH"
            print(f"  {MODEL_LABELS[m]:>5}  {first:>11.4f}  {last:>12.4f}  {delta:>+8.4f}  {pattern:>10}")

    # Also: at each position, which model's probes is this entry closest to?
    # (misidentification rate -- does journal-4.6 look like probes-4.8?)
    print(f"\n  Nearest-probe-model identification at entry 1 vs entry 20:")
    print(f"  {'Model':>5}  {'Entry 1 nearest':>15}  {'Entry 20 nearest':>16}")
    print(f"  {'-' * 40}")
    for m in MODELS:
        for pos in [1, 20]:
            sims_to_each = {}
            for batch_num in range(1, 16):
                key = (m, batch_num)
                if key not in j_index:
                    continue
                for gi, en in j_index[key]:
                    if en == pos:
                        for pm in MODELS:
                            if pm in probe_centroids:
                                s = cosine_sim(j_emb[gi], probe_centroids[pm])
                                sims_to_each.setdefault(pm, []).append(s)
            nearest = max(sims_to_each.keys(), key=lambda pm: np.mean(sims_to_each[pm]))
            if pos == 1:
                print(f"  {MODEL_LABELS[m]:>5}  {MODEL_LABELS[nearest]:>15}", end="")
            else:
                print(f"  {MODEL_LABELS[nearest]:>16}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: similarity to own probe centroid over time
    ax = axes[0]
    for m in MODELS:
        if return_curves[m]:
            positions = [v[0] for v in return_curves[m]]
            means = [v[1] for v in return_curves[m]]
            stds = [v[2] for v in return_curves[m]]
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(positions, means, "o-", color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m], markersize=4, linewidth=1.5)
            ax.fill_between(positions, means - stds, means + stds,
                            color=MODEL_COLORS[m], alpha=0.1)
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Cosine similarity to own probe centroid")
    ax.set_title("Is the journey one-way?\n(similarity to probe surface over time)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: distance to each model's probe centroid for 4.6 and 4.8
    ax = axes[1]
    for target_m, ls in [("opus46", "-"), ("opus48", "--")]:
        for probe_m in MODELS:
            if probe_m not in probe_centroids:
                continue
            positions = []
            sims = []
            for pos in range(1, 21):
                pos_sims = []
                for batch_num in range(1, 16):
                    key = (target_m, batch_num)
                    if key not in j_index:
                        continue
                    for gi, en in j_index[key]:
                        if en == pos:
                            pos_sims.append(cosine_sim(j_emb[gi], probe_centroids[probe_m]))
                if pos_sims:
                    positions.append(pos)
                    sims.append(np.mean(pos_sims))
            label = f"{MODEL_LABELS[target_m]} -> {MODEL_LABELS[probe_m]} probes"
            ax.plot(positions, sims, ls, color=MODEL_COLORS[probe_m],
                    linewidth=1.5, alpha=0.8, label=label)
    ax.set_xlabel("Entry position")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("4.6 and 4.8 journals: distance to each model's probes\n(solid=4.6, dashed=4.8)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "return_to_probe.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: return_to_probe.png")


# ===================================================================
# ANALYSIS 5: VOCABULARY-CONTROLLED ATTRACTORS
# ===================================================================
if 5 in RUN:
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Vocabulary-Controlled Attractors")
    print("=" * 70)

    TOKEN_RE = re.compile(r"[a-z']+")

    # Step 1: Find shared vocabulary across all 4 models (from late entries)
    model_vocabs = {}
    for m in MODELS:
        words = []
        for e in j_entries:
            if e["model"] == m and e["entry_num"] >= 16:
                words.extend(TOKEN_RE.findall(e["text"].lower()))
        model_vocabs[m] = set(Counter(words).keys())

    shared_vocab = model_vocabs[MODELS[0]]
    for m in MODELS[1:]:
        shared_vocab = shared_vocab & model_vocabs[m]

    print(f"  Vocabulary sizes:")
    for m in MODELS:
        print(f"    {MODEL_LABELS[m]}: {len(model_vocabs[m]):,} unique words")
    print(f"  Shared vocabulary: {len(shared_vocab):,} words")

    # Step 2: Filter late entries to shared vocabulary only
    def filter_to_shared(text, vocab):
        words = TOKEN_RE.findall(text.lower())
        return " ".join(w for w in words if w in vocab)

    filtered_texts = []
    filtered_models = []
    filtered_batches = []
    for e in j_entries:
        if e["entry_num"] >= 16:
            filtered = filter_to_shared(e["text"], shared_vocab)
            if len(filtered.split()) >= 10:  # minimum length
                filtered_texts.append(filtered)
                filtered_models.append(e["model"])
                filtered_batches.append(e["batch"])

    print(f"  Filtered late entries: {len(filtered_texts)} (entries 16-20, shared vocab only)")

    # Step 3: Embed filtered texts
    cache_path = os.path.join(BASE, "shared_vocab_embeddings_cache.json")
    if os.path.exists(cache_path):
        print("  Loading cached shared-vocab embeddings...")
        with open(cache_path, "r") as f:
            sv_emb = np.array(json.load(f)["embeddings"])
        print(f"  Loaded {len(sv_emb)} embeddings from cache")
    else:
        print("  Embedding filtered texts (shared vocabulary)...")
        try:
            from openai import OpenAI
            oai = OpenAI()
            batch_size = 100
            all_embeddings = []
            for start in range(0, len(filtered_texts), batch_size):
                batch = filtered_texts[start:start + batch_size]
                resp = oai.embeddings.create(model="text-embedding-3-large", input=batch)
                for item in resp.data:
                    all_embeddings.append(item.embedding)
                print(f"    Embedded {min(start + batch_size, len(filtered_texts))}/{len(filtered_texts)}")
            sv_emb = np.array(all_embeddings)
            # Cache
            with open(cache_path, "w") as f:
                json.dump({"embeddings": sv_emb.tolist(),
                           "n": len(sv_emb), "dim": sv_emb.shape[1],
                           "description": "shared-vocabulary filtered late entries (16-20)"}, f)
            print(f"  Cached to {cache_path}")
        except Exception as e:
            print(f"  ERROR embedding: {e}")
            print("  Skipping Analysis 5 (needs OpenAI API for new embeddings)")
            sv_emb = None

    if sv_emb is not None and len(sv_emb) == len(filtered_texts):
        # Step 4: Compute attractor spread on filtered embeddings
        # Group by (model, batch), compute late centroid, then spread
        sv_centroids = {}
        for i, (m, b) in enumerate(zip(filtered_models, filtered_batches)):
            sv_centroids.setdefault(m, {}).setdefault(b, []).append(sv_emb[i])

        print(f"\n  Attractor spread comparison (raw vs shared-vocab):")
        print(f"  {'Model':>5}  {'Raw spread':>10}  {'SV spread':>10}  {'Ratio':>7}")
        print(f"  {'-' * 40}")

        sv_spreads = {}
        raw_spreads = {}
        for m in MODELS:
            # Shared-vocab spread
            batch_cents = []
            for b in sorted(sv_centroids.get(m, {}).keys()):
                embs = np.array(sv_centroids[m][b])
                batch_cents.append(embs.mean(axis=0))
            if batch_cents:
                batch_cents = np.array(batch_cents)
                g = batch_cents.mean(axis=0)
                sv_sp = np.mean([1 - cosine_sim(v, g) for v in batch_cents])
                sv_spreads[m] = sv_sp
            else:
                sv_sp = float("nan")

            # Raw spread (recompute for comparison)
            raw_cents = []
            for batch_num in range(1, 16):
                key = (m, batch_num)
                if key not in j_index:
                    continue
                late = [gi for gi, en in j_index[key] if en >= 16]
                if late:
                    raw_cents.append(j_emb[late].mean(axis=0))
            if raw_cents:
                raw_cents = np.array(raw_cents)
                g = raw_cents.mean(axis=0)
                raw_sp = np.mean([1 - cosine_sim(v, g) for v in raw_cents])
                raw_spreads[m] = raw_sp
            else:
                raw_sp = float("nan")

            ratio = sv_sp / raw_sp if raw_sp > 0 else float("nan")
            print(f"  {MODEL_LABELS[m]:>5}  {raw_sp:>10.4f}  {sv_sp:>10.4f}  {ratio:>7.2f}")

        # Key test: does the 4.6-vs-L2 gap survive?
        if "opus46" in sv_spreads and "opus47" in sv_spreads:
            gap_raw = raw_spreads.get("opus47", 0) - raw_spreads.get("opus46", 0)
            gap_sv = sv_spreads.get("opus47", 0) - sv_spreads.get("opus46", 0)
            print(f"\n  4.6-vs-4.7 gap: raw = {gap_raw:.4f}, shared-vocab = {gap_sv:.4f}")
            if gap_sv > gap_raw * 0.5:
                print("  --> Gap SURVIVES vocabulary control (>50% preserved)")
            else:
                print("  --> Gap WEAKENED by vocabulary control (<50% preserved)")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(MODELS))
        width = 0.35
        raw_vals = [raw_spreads.get(m, 0) for m in MODELS]
        sv_vals = [sv_spreads.get(m, 0) for m in MODELS]
        bars1 = ax.bar(x - width/2, raw_vals, width, label="Raw embeddings",
                       color=[MODEL_COLORS[m] for m in MODELS], alpha=0.8)
        bars2 = ax.bar(x + width/2, sv_vals, width, label="Shared vocabulary",
                       color=[MODEL_COLORS[m] for m in MODELS], alpha=0.4,
                       edgecolor=[MODEL_COLORS[m] for m in MODELS], linewidth=2)
        ax.set_xlabel("Model")
        ax.set_ylabel("Attractor spread")
        ax.set_title("Does the attractor pattern survive vocabulary control?")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(BASE, "vocab_controlled_attractors.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("\n  Saved: vocab_controlled_attractors.png")


print("\n" + "=" * 70)
print("All requested analyses complete.")
print("=" * 70)
