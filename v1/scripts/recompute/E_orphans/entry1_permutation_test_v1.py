"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/entry1_permutation_test.py.
Changed: input paths only. Journals + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json); probes via e_common (repo probe files + June probe cache, read-only).
The batch range (June: 1-10, hard-coded) is a command-line switch, default b1-10; version v0|v1.
Usage: python entry1_permutation_test_v1.py v0|v1 [b1-10|b1-15]
"""
import e_common
VERSION = e_common.version_from_argv()
BATCHES = e_common.batches_from_argv(list(range(1, 11)))
"""
Permutation test for entry-1 own-probe nearness.

4.8's verification request: the entry-1 table shows each model nearest
its own probes, but 4.5 (margin 0.002) and 4.8 (margin 0.003) are thin.
Test which margins are real.

Two test designs:

TEST A - Label permutation (same as migration test):
  Permute probe centroid labels. Ask how often each model's nearest
  centroid carries its own label. Per-model: ~1/4 by symmetry.
  Joint (all 4 own-labeled): 1/24 = 0.042.
  NOTE: margin p-values will be 1.0 (same geometric reason as before).

TEST B - Journal permutation (the informative one):
  Pool all 40 entry-1 embeddings. Shuffle model assignments (10 per model).
  Recompute own-advantage = sim(own probes) - max(sim(other probes)).
  This tests: "are these specific journals unusually close to their own
  probes compared to random journals?"
  This is the test with actual power to discriminate the four margins.

Usage:
    python entry1_permutation_test.py
"""

import json, os
import numpy as np
from itertools import permutations

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}

N_PERM = 10000
RNG = np.random.default_rng(42)


def load_cached(path):
    with open(path, "r") as f:
        return np.array(json.load(f)["embeddings"])


def load_journals():
    # [E V1] was: Texture\Batch_1..10 files; now e_common (v1_common corpus + clean npz), same order
    global _J_EMB
    entries, _J_EMB = e_common.load_opus(VERSION, BATCHES)
    return entries


def load_probes():
    entries = []
    for m in MODELS:
        neg_file = os.path.join(BASE, f"texture_probe_{m}_log.json")
        with open(neg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, entry in enumerate(data):
            entries.append({"model": m, "condition": "negation", "index": i,
                            "response": entry["response"]})
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


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def build_journal_index(entries):
    idx = {}
    for i, e in enumerate(entries):
        key = (e["model"], e["batch"])
        idx.setdefault(key, []).append((i, e["entry_num"]))
    for key in idx:
        idx[key].sort(key=lambda x: x[1])
    return idx


if __name__ == "__main__":
    print("Loading data...")
    j_entries = load_journals()
    j_emb = _J_EMB  # [E V1]
    j_index = build_journal_index(j_entries)

    p_entries, _P_EMB = e_common.load_opus_probes()  # [E V1] was load_probes() + probe_embeddings_cache.json
    p_emb = _P_EMB  # [E V1]

    # Build probe index and centroids
    p_index = {}
    for i, e in enumerate(p_entries):
        p_index.setdefault(e["model"], []).append(i)

    probe_centroids = {}
    for m in MODELS:
        probe_centroids[m] = p_emb[p_index[m]].mean(axis=0)

    # Collect entry-1 embeddings per model
    entry1_embs = {}
    for m in MODELS:
        embs = []
        for batch_num in BATCHES:  # [E V1] was range(1, 11)
            key = (m, batch_num)
            if key not in j_index:
                continue
            for gi, en in j_index[key]:
                if en == 1:
                    embs.append(j_emb[gi])
        entry1_embs[m] = np.array(embs)
        print(f"  {MODEL_LABELS[m]} entry-1: {len(embs)} embeddings")

    # Precompute: similarity of each entry-1 embedding to each probe centroid
    # sim_matrix[m][pm] = array of similarities (one per batch)
    sim_matrix = {}
    for m in MODELS:
        sim_matrix[m] = {}
        for pm in MODELS:
            sims = np.array([cosine_sim(e, probe_centroids[pm]) for e in entry1_embs[m]])
            sim_matrix[m][pm] = sims

    # =========================================================
    # Observed own-advantages
    # =========================================================
    print("\n" + "=" * 78)
    print("OBSERVED: Entry-1 own-probe advantage")
    print("=" * 78)

    observed_own_adv = {}
    for m in MODELS:
        own_mean = sim_matrix[m][m].mean()
        other_means = {pm: sim_matrix[m][pm].mean() for pm in MODELS if pm != m}
        best_other = max(other_means.values())
        best_other_model = max(other_means, key=other_means.get)
        own_adv = own_mean - best_other
        observed_own_adv[m] = own_adv
        print(f"  {MODEL_LABELS[m]}: own={own_mean:.4f}, "
              f"best other={MODEL_LABELS[best_other_model]} ({best_other:.4f}), "
              f"own-advantage={own_adv:+.4f}")

    # =========================================================
    # TEST A: Label permutation (probe centroid label shuffling)
    # Exhaustive over all 24 permutations
    # =========================================================
    print("\n" + "=" * 78)
    print("TEST A: Label permutation (all 24 permutations, exhaustive)")
    print("=" * 78)

    all_perms = list(permutations(range(4)))
    print(f"  {len(all_perms)} permutations of probe labels")

    # For each model, count how many permutations have own-label on nearest centroid
    label_match_count = {m: 0 for m in MODELS}
    joint_all_match = 0

    for perm in all_perms:
        # perm[i] means centroid originally at position i gets label MODELS[perm[i]]
        # Actually let's think: we have 4 centroids (one per model).
        # A permutation maps: slot i -> which model's label goes on centroid i
        # So if perm = (2, 0, 3, 1), centroid-0 gets label MODELS[2], etc.
        # Better: perm_centroids[MODELS[j]] = probe_centroids[MODELS[perm_inv[j]]]
        # where perm_inv is the inverse permutation.
        #
        # Simpler: for each model m (index mi), its probe centroid under this
        # permutation is the centroid originally belonging to MODELS[perm[mi]].
        # So "own probes" for model m becomes probe_centroids[MODELS[perm[mi]]].

        # For this permutation, check if each model's nearest centroid
        # carries its own label.
        # Under the permutation, the centroid labeled MODELS[mi] is
        # probe_centroids[MODELS[perm[mi]]].
        # Model mi's entry-1 journals are nearest to whichever centroid
        # is geometrically nearest. The label on that centroid is
        # determined by the permutation.

        all_match = True
        for mi, m in enumerate(MODELS):
            # Which centroid is geometrically nearest to m's entry-1?
            # This doesn't change with permutation — it's fixed geometry.
            nearest_centroid_idx = max(range(4),
                key=lambda ci: sim_matrix[m][MODELS[ci]].mean())
            # Under this permutation, what label does centroid nearest_centroid_idx carry?
            # perm maps: slot i carries the label MODELS[perm[i]]?
            # No — let's define it clearly.
            # The permutation shuffles which MODEL'S centroid sits in which slot.
            # Original: slot i has centroid of MODELS[i]
            # Permuted: slot i has centroid of MODELS[perm[i]]
            # But the journal-1 similarity to slot i is sim_matrix[m][MODELS[i]]
            # (based on original geometry).
            #
            # Actually, I think I'm overcomplicating this. Let me redefine.
            #
            # Original centroids: C_0, C_1, C_2, C_3 for models 0,1,2,3
            # Entry-1 for model mi is nearest to centroid C_k (fixed geometry).
            # Under permutation perm, centroid C_j gets label perm[j].
            # So the label of the nearest centroid (C_k) is perm[k].
            # We want perm[k] == mi (nearest centroid carries own label).

            # Actually, let's define perm as: perm[j] = new label for centroid j.
            # Then we want perm[nearest_centroid_idx] == mi.

            # With itertools.permutations(range(4)), perm is a tuple of (0,1,2,3)
            # in some order. perm[j] = which model index gets assigned to centroid j.

            if perm[nearest_centroid_idx] == mi:
                label_match_count[m] += 1
            else:
                all_match = False

        if all_match:
            joint_all_match += 1

    print(f"\n  Per-model: nearest centroid carries own label")
    print(f"  {'Model':>5}  {'Count/24':>8}  {'p-value':>8}")
    print(f"  {'-' * 28}")
    for m in MODELS:
        p = label_match_count[m] / len(all_perms)
        print(f"  {MODEL_LABELS[m]:>5}  {label_match_count[m]:>8}  {p:>8.3f}")

    p_joint = joint_all_match / len(all_perms)
    print(f"\n  Joint (all 4 match own): {joint_all_match}/{len(all_perms)} = p={p_joint:.4f}")
    if p_joint < 0.05:
        print(f"  --> SIGNIFICANT at p<0.05: all 4 models nearest own probes")
    else:
        print(f"  --> Not significant at p<0.05")

    # =========================================================
    # TEST B: Journal permutation (shuffle which journals belong to which model)
    # This is the test with actual power to discriminate margins
    # =========================================================
    print("\n" + "=" * 78)
    print("TEST B: Journal permutation (10,000 iterations)")
    print("Null: shuffle entry-1 embeddings across models, recompute own-advantage")
    print("=" * 78)

    # Pool all entry-1 embeddings
    all_e1 = []
    all_e1_models = []
    model_sizes = {}
    for m in MODELS:
        model_sizes[m] = len(entry1_embs[m])
        for e in entry1_embs[m]:
            all_e1.append(e)
            all_e1_models.append(m)
    all_e1 = np.array(all_e1)
    n_total = len(all_e1)

    print(f"  Pooled {n_total} entry-1 embeddings")
    print(f"  Sizes: {', '.join(f'{MODEL_LABELS[m]}={model_sizes[m]}' for m in MODELS)}")

    # Precompute similarity of every entry-1 embedding to every probe centroid
    # all_sims[i, j] = cosine_sim(all_e1[i], probe_centroids[MODELS[j]])
    all_sims = np.zeros((n_total, 4))
    for j, pm in enumerate(MODELS):
        for i in range(n_total):
            all_sims[i, j] = cosine_sim(all_e1[i], probe_centroids[pm])

    print(f"  Precomputed {n_total}x4 similarity matrix")
    print(f"  Running {N_PERM:,} permutations...\n")

    # For each model, count permutations where own-advantage >= observed
    perm_exceeds = {m: 0 for m in MODELS}
    # Also: joint test (all 4 have positive own-advantage)
    perm_all_positive = 0

    # Build split indices
    sizes = [model_sizes[m] for m in MODELS]
    splits = np.cumsum(sizes[:-1])  # where to split the shuffled array

    for iteration in range(N_PERM):
        # Shuffle all entry-1 embeddings
        perm = RNG.permutation(n_total)

        # Split into 4 groups of 10
        groups = np.split(perm, splits)

        all_positive = True
        for mi, m in enumerate(MODELS):
            group_indices = groups[mi]
            # Mean similarity of this group to each probe centroid
            group_sims = all_sims[group_indices]  # (10, 4)
            mean_sims = group_sims.mean(axis=0)  # (4,)

            # Own-advantage: sim to own probes - max(sim to other probes)
            own_sim = mean_sims[mi]
            other_sims = np.delete(mean_sims, mi)
            best_other = other_sims.max()
            own_adv = own_sim - best_other

            if own_adv >= observed_own_adv[m]:
                perm_exceeds[m] += 1
            if own_adv <= 0:
                all_positive = False

        if all_positive:
            perm_all_positive += 1

    # Report
    print(f"  Per-model own-advantage significance:")
    print(f"  {'Model':>5}  {'Obs advantage':>13}  {'p-value':>8}  {'Verdict':>20}")
    print(f"  {'-' * 55}")
    for m in MODELS:
        p = (perm_exceeds[m] + 1) / (N_PERM + 1)
        verdict = "SURVIVES (p<.05)" if p < 0.05 else "NOT SIGNIFICANT"
        print(f"  {MODEL_LABELS[m]:>5}  {observed_own_adv[m]:>+13.4f}  {p:>8.4f}  {verdict:>20}")

    p_joint = (perm_all_positive + 1) / (N_PERM + 1)
    print(f"\n  Joint (all 4 own-advantage > 0): {perm_all_positive}/{N_PERM}  p={p_joint:.4f}")
    if p_joint < 0.05:
        print(f"  --> SIGNIFICANT: all 4 models sit nearest own probes at entry 1")
    else:
        print(f"  --> Not significant as a joint claim")

    # =========================================================
    # The departure deltas (for reference, from Analysis 4)
    # =========================================================
    print("\n" + "=" * 78)
    print("DEPARTURE MAGNITUDES (entry-1 to entry-20 similarity to own probes)")
    print("These are the robust monotonic deltas from Analysis 4")
    print("=" * 78)

    # Collect entry-20 similarities to own probes
    for m in MODELS:
        e1_own = sim_matrix[m][m].mean()
        # Get entry-20 own-probe similarity
        e20_sims = []
        for batch_num in BATCHES:  # [E V1] was range(1, 11)
            key = (m, batch_num)
            if key not in j_index:
                continue
            for gi, en in j_index[key]:
                if en == 20:
                    e20_sims.append(cosine_sim(j_emb[gi], probe_centroids[m]))
        e20_own = np.mean(e20_sims)
        delta = e20_own - e1_own
        print(f"  {MODEL_LABELS[m]}: entry-1={e1_own:.4f}, entry-20={e20_own:.4f}, "
              f"delta={delta:+.4f}")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  TEST A (label permutation): tests whether all 4 nearest-own at once")
    print("  is unlikely. Exhaustive over 24 permutations.")
    print("  - Per-model: p=0.25 each (uninformative individually)")
    print("  - Joint: p=1/24 ≈ 0.042 if all 4 are nearest-own")
    print()
    print("  TEST B (journal permutation): tests whether each model's own-advantage")
    print("  margin survives when journals are shuffled across models.")
    print("  - This is the test with power to discriminate thin vs robust margins")
    print("  - 4.8's prediction: 4.6 (0.036) and 4.7 (0.011) survive,")
    print("    4.5 (0.002) and 4.8 (0.003) don't")
    print()
    print("  The DEPARTURE deltas (entry-1 to entry-20 drop in own-probe similarity)")
    print("  are the robust spine: all four are large, negative, and monotonic.")
    print("  These don't need margin tests — they're changes within each model.")
    print("=" * 78)
