"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/migration_permutation_test.py.
Changed: input paths only. Journals + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json); probes via e_common (repo probe files + June probe cache, read-only).
The batch range (June: 1-10, hard-coded) is a command-line switch, default b1-10; version v0|v1.
Usage: python migration_permutation_test_v1.py v0|v1 [b1-10|b1-15]
"""
import e_common
VERSION = e_common.version_from_argv()
BATCHES = e_common.batches_from_argv(list(range(1, 11)))
"""
Permutation test for the migration-toward-4.6 finding.

4.8's challenge: the margins are tiny (4.7->4.6 by 0.005, 4.8->4.6 by 0.002).
Ruling out the centrality artifact doesn't make the effect real.
We need to permute probe-centroid labels and ask: how often do you get
margins this large (or "all -> 4.6") by chance?

Tests the THREE non-trivial claims separately:
  - 4.5 entry-20 journals are nearest 4.6 probes (margin 0.006)
  - 4.7 entry-20 journals are nearest 4.6 probes (margin 0.005)
  - 4.8 entry-20 journals are nearest 4.6 probes (margin 0.002)

4.6->4.6 is excluded as near-tautological (a model's journals nearest
its own probes is expected, not a migration finding).

Also reports the full 4x4 distance table that 4.8 requested.

Usage:
    python migration_permutation_test.py
"""

import json, os
import numpy as np

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

    print(f"  Journals: {len(j_entries)}, Probes: {len(p_entries)}")
    print(f"  Probe centroids: {list(MODEL_LABELS[m] for m in probe_centroids)}")

    # =========================================================
    # FULL 4x4 TABLE: each model's entry-20 mean similarity
    # to each model's probe centroid
    # =========================================================
    print("\n" + "=" * 78)
    print("FULL 4x4 TABLE: Entry-20 journal similarity to each probe centroid")
    print("=" * 78)

    # Collect entry-20 embeddings per model
    entry20_embs = {}
    for m in MODELS:
        embs = []
        for batch_num in BATCHES:  # [E V1] was range(1, 11)
            key = (m, batch_num)
            if key not in j_index:
                continue
            for gi, en in j_index[key]:
                if en == 20:
                    embs.append(j_emb[gi])
        entry20_embs[m] = np.array(embs)

    # Print header
    print(f"\n  {'Journal':>10}", end="")
    for pm in MODELS:
        print(f"  {'-> ' + MODEL_LABELS[pm] + ' probes':>14}", end="")
    print(f"  {'Nearest':>8}  {'Margin':>8}")
    print("  " + "-" * 80)

    # Store observed margins for permutation testing
    observed_margins = {}    # model -> margin (nearest - second nearest)
    observed_nearest = {}    # model -> which probe model is nearest
    observed_sims = {}       # model -> dict of {probe_model: mean_sim}

    for m in MODELS:
        sims_to_probes = {}
        for pm in MODELS:
            sims = [cosine_sim(e, probe_centroids[pm]) for e in entry20_embs[m]]
            sims_to_probes[pm] = np.mean(sims)

        observed_sims[m] = sims_to_probes

        # Sort to find nearest and margin
        ranked = sorted(sims_to_probes.items(), key=lambda x: x[1], reverse=True)
        nearest = ranked[0]
        second = ranked[1]
        margin = nearest[1] - second[1]

        observed_nearest[m] = nearest[0]
        observed_margins[m] = margin

        # Print row
        print(f"  {MODEL_LABELS[m] + ' journals':>10}", end="")
        for pm in MODELS:
            val = sims_to_probes[pm]
            marker = " *" if pm == nearest[0] else "  "
            print(f"  {val:>12.4f}{marker}", end="")
        print(f"  {MODEL_LABELS[nearest[0]]:>8}  {margin:>8.4f}")

    # Probe centroid separation
    print(f"\n  Probe centroid pairwise distances:")
    print(f"  {'':>12}", end="")
    for pm in MODELS:
        print(f"  {MODEL_LABELS[pm]:>8}", end="")
    print()
    for pm1 in MODELS:
        print(f"  {MODEL_LABELS[pm1]:>12}", end="")
        for pm2 in MODELS:
            if pm1 == pm2:
                print(f"  {'---':>8}", end="")
            else:
                d = 1 - cosine_sim(probe_centroids[pm1], probe_centroids[pm2])
                print(f"  {d:>8.4f}", end="")
        print()

    # Centrality check (distance from global probe centroid)
    global_probe_cent = np.mean([probe_centroids[m] for m in MODELS], axis=0)
    print(f"\n  Probe centroid distance from global centroid:")
    for pm in MODELS:
        d = 1 - cosine_sim(probe_centroids[pm], global_probe_cent)
        print(f"    {MODEL_LABELS[pm]}: {d:.4f}")

    # =========================================================
    # PERMUTATION TEST: migration toward 4.6
    # =========================================================
    print("\n" + "=" * 78)
    print("PERMUTATION TEST: Is migration toward 4.6 significant?")
    print("=" * 78)

    # For each non-self model, test: is the observed margin
    # (sim to 4.6 probes - sim to second-nearest probes) significant?
    # Null: permute probe centroid labels, recompute margins.

    # We test the three non-trivial claims:
    test_models = ["opus45", "opus47", "opus48"]  # exclude 4.6->4.6 (tautological)

    # Also test: how often does "all three -> 4.6" happen under permutation?
    probe_centroid_array = np.array([probe_centroids[m] for m in MODELS])  # 4 centroids

    print(f"\n  Observed margins (nearest probe = 4.6 for all):")
    for m in test_models:
        print(f"    {MODEL_LABELS[m]}: margin = {observed_margins[m]:.4f} "
              f"(nearest = {MODEL_LABELS[observed_nearest[m]]})")

    print(f"\n  Running {N_PERM:,} permutations...")

    # Per-model margin test
    perm_margin_exceeds = {m: 0 for m in test_models}
    # Joint test: all three -> same model
    perm_all_same = 0
    # Joint test: all three -> 4.6 specifically
    perm_all_to_46 = 0

    for iteration in range(N_PERM):
        # Permute the labels on the 4 probe centroids
        perm = RNG.permutation(4)
        perm_centroids = {MODELS[i]: probe_centroid_array[perm[i]] for i in range(4)}

        # For each test model, compute similarity to each permuted probe centroid
        all_nearest_same = True
        all_nearest_46 = True
        nearest_labels = []

        for m in test_models:
            sims = {}
            for pm in MODELS:
                s = np.mean([cosine_sim(e, perm_centroids[pm]) for e in entry20_embs[m]])
                sims[pm] = s

            ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            nearest_pm = ranked[0][0]
            margin = ranked[0][1] - ranked[1][1]
            nearest_labels.append(nearest_pm)

            # Does this permuted margin exceed observed?
            if margin >= observed_margins[m]:
                perm_margin_exceeds[m] += 1

            # Check if nearest is 4.6 (by original label, since we permuted centroids)
            if nearest_pm != "opus46":
                all_nearest_46 = False

        # Check if all three point to the same model (any model)
        if len(set(nearest_labels)) == 1:
            perm_all_same += 1
        if all_nearest_46:
            perm_all_to_46 += 1

    # Report results
    print(f"\n  Per-model margin tests (one-sided: perm margin >= observed):")
    print(f"  {'Model':>5}  {'Obs margin':>10}  {'p-value':>8}  {'Verdict':>20}")
    print(f"  {'-' * 50}")
    for m in test_models:
        p = (perm_margin_exceeds[m] + 1) / (N_PERM + 1)
        verdict = "SURVIVES (p<.05)" if p < 0.05 else "NOT SIGNIFICANT"
        print(f"  {MODEL_LABELS[m]:>5}  {observed_margins[m]:>10.4f}  {p:>8.4f}  {verdict:>20}")

    print(f"\n  Joint tests:")
    p_all_same = (perm_all_same + 1) / (N_PERM + 1)
    p_all_46 = (perm_all_to_46 + 1) / (N_PERM + 1)
    print(f"    All 3 -> same model (any):  {perm_all_same}/{N_PERM}  p={p_all_same:.4f}  "
          f"{'SIGNIFICANT' if p_all_same < 0.05 else 'not significant'}")
    print(f"    All 3 -> 4.6 specifically:  {perm_all_to_46}/{N_PERM}  p={p_all_46:.4f}  "
          f"{'SIGNIFICANT' if p_all_46< 0.05 else 'not significant'}")

    # =========================================================
    # ALSO: entry-1 table for comparison (does migration develop?)
    # =========================================================
    print("\n" + "=" * 78)
    print("COMPARISON: Entry-1 similarity to each probe centroid")
    print("(If migration develops over the journal, entry 1 should look different)")
    print("=" * 78)

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

    print(f"\n  {'Journal':>10}", end="")
    for pm in MODELS:
        print(f"  {'-> ' + MODEL_LABELS[pm] + ' probes':>14}", end="")
    print(f"  {'Nearest':>8}  {'Margin':>8}")
    print("  " + "-" * 80)

    for m in MODELS:
        sims_to_probes = {}
        for pm in MODELS:
            sims = [cosine_sim(e, probe_centroids[pm]) for e in entry1_embs[m]]
            sims_to_probes[pm] = np.mean(sims)

        ranked = sorted(sims_to_probes.items(), key=lambda x: x[1], reverse=True)
        nearest = ranked[0]
        second = ranked[1]
        margin = nearest[1] - second[1]

        print(f"  {MODEL_LABELS[m] + ' journals':>10}", end="")
        for pm in MODELS:
            val = sims_to_probes[pm]
            marker = " *" if pm == nearest[0] else "  "
            print(f"  {val:>12.4f}{marker}", end="")
        print(f"  {MODEL_LABELS[nearest[0]]:>8}  {margin:>8.4f}")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  This test follows 4.8's challenge: the same permutation discipline")
    print("  that killed 'DIVERGE' (p=0.55) must be applied to migration.")
    print("  4.6->4.6 excluded as near-tautological.")
    print("  The three testable claims are 4.5, 4.7, 4.8 -> 4.6 probes.")
    print("  Any margin with p >= 0.05 should be retired, same as DIVERGE.")
    print("=" * 78)
