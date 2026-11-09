"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/replication_test.py,
used only as the module that decone_pset_v1.py and domain_preserving_retest_v1.py import (as the June
scripts imported replication_test). Changed: input paths only. load_journals(batch_range) now returns the
entries from e_common (v1_common corpus) and keeps the aligned clean-npz vectors in _J_EMB; load_probes()
returns the repo probes in the June order and keeps the June probe-cache vectors in _P_EMB. The metric
functions are untouched. Its __main__ block (which embeds through the API) is never run here.
Version: v0|v1 from the command line of the importing script.
"""
import e_common
VERSION = e_common.version_from_argv()
_J_EMB = None
_P_EMB = None
"""
Replication test for Texture journal findings.

Tests batches 11-15 ALONE against the pre-committed predictions in
REPLICATION_PRECOMMITMENT.md. No pooling with batches 1-10.

Pre-registered predictions:
  P1: Attractor spread ordering (4.6 < 4.5 < {4.7, 4.8})
  P2: 4.6 vs second-lineage spread gap significant
  P3: 4.7 only model with real entry-1 specificity
  P4: Monotonic departure from own probe surface (all 4)
  P5: 4.8 departs furthest
  P6: Entry-1 own-probe nearness (journal permutation, all 4)
  N1: Migration toward 4.6 stays dead
  N2: Cross-lineage divergence stays dead

Then pools batches 1-15 for tighter final estimates.

Usage:
    python replication_test.py                  (run replication)
    python replication_test.py --embed-only     (just embed new batches)
    python replication_test.py --pool           (also run pooled n=15 analysis)

Requirements:
    pip install openai numpy scikit-learn matplotlib
    OPENAI_API_KEY set in environment
"""

import json, os, sys, time
import numpy as np

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
MODEL_COLORS = {"opus45": "#1976D2", "opus46": "#E65100",
                "opus47": "#2E7D32", "opus48": "#7B1FA2"}
LINEAGE_1 = ["opus45", "opus46"]
LINEAGE_2 = ["opus47", "opus48"]

# Replication batches (new data, analyzed alone)
REP_BATCHES = list(range(11, 16))
# Original batches (for pooled analysis only)
ORIG_BATCHES = list(range(1, 11))
ALL_BATCHES = list(range(1, 16))

N_BOOT = 10000
N_PERM = 10000
RNG = np.random.default_rng(42)

CACHE_REP = os.path.join(BASE, "journal_embeddings_cache_rep.json")
CACHE_ALL = os.path.join(BASE, "journal_embeddings_cache_all.json")


# ---- Loaders ---------------------------------------------------------------
def load_journals(batch_range):
    # [E V1] was: Texture\Batch_N files; now e_common (v1_common corpus + clean npz), same order
    global _J_EMB
    _entries, _J_EMB = e_common.load_opus(VERSION, batch_range)
    return _entries


def _june_load_journals(batch_range):  # [E V1] the June loader, kept for reference, not called
    entries = []
    for batch_num in batch_range:
        batch_dir = os.path.join(BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            print(f"  WARNING: {batch_dir} not found, skipping")
            continue
        for m in MODELS:
            # Filter out Proton Drive conflict copies (contain "#" in filename)
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
    # [E V1] was: Texture\texture_probe_* files; now e_common (repo probe files, June order + June cache)
    global _P_EMB
    _entries, _P_EMB = e_common.load_opus_probes()
    return _entries


def _june_load_probes():  # [E V1] the June loader, kept for reference, not called
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


def load_cached(path):
    with open(path, "r") as f:
        return np.array(json.load(f)["embeddings"])


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def cosine_dist(a, b):
    return 1.0 - cosine_sim(a, b)


def build_journal_index(entries):
    idx = {}
    for i, e in enumerate(entries):
        key = (e["model"], e["batch"])
        idx.setdefault(key, []).append((i, e["entry_num"]))
    for key in idx:
        idx[key].sort(key=lambda x: x[1])
    return idx


# ---- Embedding --------------------------------------------------------------
def embed_journals(entries, cache_path):
    """Embed journal entries, with caching."""
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        with open(cache_path, "r") as f:
            cache = json.load(f)
        if cache["n"] == len(entries):
            return np.array(cache["embeddings"])
        print(f"  Cache size mismatch ({cache['n']} vs {len(entries)}), re-embedding...")

    from openai import OpenAI
    client = OpenAI()
    texts = [e["text"] for e in entries]
    all_embeddings = []
    batch_size = 100
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        print(f"  Embedding {start+1}-{min(start+batch_size, len(texts))}/{len(texts)}...")
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        for item in resp.data:
            all_embeddings.append(item.embedding)
        total_tokens += resp.usage.total_tokens
        time.sleep(0.2)

    print(f"  Total tokens: {total_tokens:,} (cost: ~${total_tokens * 0.13 / 1_000_000:.4f})")

    with open(cache_path, "w") as f:
        json.dump({"embeddings": [e for e in all_embeddings],
                    "n": len(all_embeddings),
                    "dim": len(all_embeddings[0]),
                    "batches": list(set(e["batch"] for e in entries)),
                    "description": f"journal embeddings for batches {entries[0]['batch']}-{entries[-1]['batch']}"
                    }, f)
    print(f"  Cached to {cache_path}")
    return np.array(all_embeddings)


# ---- Health check -----------------------------------------------------------
def health_check(entries):
    """Verify batch health: all models present, all 20 entries per journal."""
    print("\n  Health check:")
    issues = []
    batches = sorted(set(e["batch"] for e in entries))
    for b in batches:
        for m in MODELS:
            batch_entries = [e for e in entries if e["batch"] == b and e["model"] == m]
            n = len(batch_entries)
            if n != 20:
                issues.append(f"  Batch {b} / {MODEL_LABELS[m]}: {n} entries (expected 20)")
    if issues:
        print("  ISSUES FOUND:")
        for iss in issues:
            print(f"    {iss}")
    else:
        print(f"  All {len(batches)} batches healthy: 4 models x 20 entries each")
    return len(issues) == 0


# ---- Metric functions -------------------------------------------------------
def attractor_spread(j_emb, j_index, model, batch_range):
    """Compute late-entry (16-20) attractor spread for a model."""
    batch_centroids = []
    for b in batch_range:
        key = (model, b)
        if key not in j_index:
            continue
        late = [gi for gi, en in j_index[key] if en >= 16]
        if late:
            batch_centroids.append(j_emb[late].mean(axis=0))
    if not batch_centroids:
        return float("nan"), np.array([])
    batch_centroids = np.array(batch_centroids)
    g = batch_centroids.mean(axis=0)
    spread = np.mean([1 - cosine_sim(v, g) for v in batch_centroids])
    return spread, batch_centroids


def spread_permutation_test(cents_a, cents_b, n_perm=N_PERM):
    """Two-sided permutation test on spread difference between two models."""
    obs = abs(attractor_spread_from_cents(cents_a) - attractor_spread_from_cents(cents_b))
    pool = np.vstack([cents_a, cents_b])
    na = len(cents_a)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm = rng.permutation(len(pool))
        a = pool[perm[:na]]
        b = pool[perm[na:]]
        if abs(attractor_spread_from_cents(a) - attractor_spread_from_cents(b)) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def attractor_spread_from_cents(cents):
    g = cents.mean(axis=0)
    return np.mean([1 - cosine_sim(v, g) for v in cents])


def specificity_point(e1, e20):
    """Mean self-similarity minus mean cross-similarity (e1 -> e20)."""
    n = len(e1)
    if n == 0:
        return 0.0
    self_sims = np.array([cosine_sim(e1[i], e20[i]) for i in range(n)])
    cross = []
    for i in range(n):
        for j in range(n):
            if i != j:
                cross.append(cosine_sim(e1[i], e20[j]))
    return self_sims.mean() - np.mean(cross) if cross else 0.0


def specificity_bootstrap(e1, e20, n_boot=N_BOOT):
    """Bootstrap CI for specificity."""
    n = len(e1)
    S = np.array([[cosine_sim(e1[i], e20[j]) for j in range(n)] for i in range(n)])
    diag = np.diag(S)
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        self_mean = diag[idx].mean()
        sub = S[np.ix_(idx, idx)]
        m = ~np.eye(len(idx), dtype=bool)
        cross_mean = sub[m].mean()
        boots[b] = self_mean - cross_mean
    return boots


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    do_pool = "--pool" in sys.argv
    embed_only = "--embed-only" in sys.argv

    print("=" * 78)
    print("  TEXTURE REPLICATION TEST")
    print("  Pre-registered predictions from REPLICATION_PRECOMMITMENT.md")
    print("  Batches 11-15 analyzed ALONE (no pooling with 1-10)")
    print("=" * 78)

    # --- Load replication journals ---
    print("\nLoading replication journals (batches 11-15)...")
    rep_entries = load_journals(REP_BATCHES)
    if not rep_entries:
        print("\n  ERROR: No replication batches found. Are batches 11-15 complete?")
        sys.exit(1)
    print(f"  Loaded {len(rep_entries)} entries")
    healthy = health_check(rep_entries)

    # --- Load probes (same for all analyses) ---
    print("\nLoading probes...")
    p_entries = load_probes()
    p_emb = load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))
    p_index = {}
    for i, e in enumerate(p_entries):
        p_index.setdefault(e["model"], []).append(i)
    probe_centroids = {m: p_emb[p_index[m]].mean(axis=0) for m in MODELS}
    print(f"  {len(p_entries)} probes loaded")

    # --- Embed replication journals ---
    print("\nEmbedding replication journals...")
    rep_emb = embed_journals(rep_entries, CACHE_REP)
    rep_index = build_journal_index(rep_entries)
    print(f"  {rep_emb.shape[0]} embeddings, dim {rep_emb.shape[1]}")

    if embed_only:
        print("\n  --embed-only: stopping after embedding.")
        sys.exit(0)

    # ==================================================================
    # REPLICATION SCORECARD
    # ==================================================================
    results = {}

    # ------------------------------------------------------------------
    # P1: Attractor spread ordering
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P1: ATTRACTOR SPREAD ORDERING")
    print("Prediction: 4.6 < 4.5 < {4.7, 4.8}")
    print("=" * 78)

    spreads = {}
    spread_cents = {}
    for m in MODELS:
        sp, cents = attractor_spread(rep_emb, rep_index, m, REP_BATCHES)
        spreads[m] = sp
        spread_cents[m] = cents
        print(f"  {MODEL_LABELS[m]}: spread = {sp:.4f} (n={len(cents)} batches)")

    ranking = sorted(spreads.items(), key=lambda x: x[1])
    rank_labels = [MODEL_LABELS[r[0]] for r in ranking]
    print(f"\n  Ranking: {' < '.join(rank_labels)}")

    p1_pass = (ranking[0][0] == "opus46" and
               ranking[-1][0] in LINEAGE_2 and ranking[-2][0] in LINEAGE_2)
    p1_46_lowest = ranking[0][0] == "opus46"
    p1_l2_highest = set([ranking[-1][0], ranking[-2][0]]) == set(LINEAGE_2) or \
                    ranking[-1][0] in LINEAGE_2

    if p1_pass:
        verdict = "PASS (4.6 lowest, L2 highest)"
    elif p1_46_lowest:
        verdict = "PARTIAL (4.6 lowest, but L2 not both highest)"
    else:
        verdict = "FAIL"
    print(f"  Verdict: {verdict}")
    results["P1"] = verdict

    # ------------------------------------------------------------------
    # P2: 4.6 vs second-lineage spread gap significant
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P2: 4.6 vs SECOND-LINEAGE SPREAD GAP")
    print("Prediction: both 4.6vs4.7 and 4.6vs4.8 significant (p<.05)")
    print("=" * 78)

    p2_results = {}
    for other in ["opus47", "opus48"]:
        if len(spread_cents["opus46"]) > 0 and len(spread_cents[other]) > 0:
            obs_diff, p_val = spread_permutation_test(spread_cents["opus46"], spread_cents[other])
            p2_results[other] = p_val
            sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(f"  4.6 vs {MODEL_LABELS[other]}: |diff|={obs_diff:.4f}, p={p_val:.4f} ({sig})")

    n_sig = sum(1 for p in p2_results.values() if p < 0.05)
    if n_sig == 2:
        verdict = "PASS (both significant)"
    elif n_sig == 1:
        verdict = "PARTIAL (one of two significant)"
    else:
        verdict = "FAIL (neither significant)"
    print(f"  Verdict: {verdict}")
    results["P2"] = verdict

    # ------------------------------------------------------------------
    # P3: 4.7 only model with real specificity
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P3: 4.7 ONLY MODEL WITH REAL SPECIFICITY")
    print("Prediction: 4.7 CI clear of zero; others cross zero")
    print("=" * 78)

    for m in MODELS:
        e1_list, e20_list = [], []
        for b in REP_BATCHES:
            key = (m, b)
            if key not in rep_index:
                continue
            be = rep_index[key]
            if len(be) >= 20:
                e1_list.append(rep_emb[be[0][0]])
                e20_list.append(rep_emb[be[-1][0]])
        e1_arr = np.array(e1_list) if e1_list else np.array([]).reshape(0, rep_emb.shape[1])
        e20_arr = np.array(e20_list) if e20_list else np.array([]).reshape(0, rep_emb.shape[1])

        if len(e1_arr) == 0:
            print(f"  {MODEL_LABELS[m]}: no data")
            continue

        obs = specificity_point(e1_arr, e20_arr)
        boots = specificity_bootstrap(e1_arr, e20_arr)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        crosses_zero = lo <= 0 <= hi
        status = "CI crosses zero" if crosses_zero else "CI CLEAR of zero"
        print(f"  {MODEL_LABELS[m]}: specificity={obs:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {status}")

    # Judge P3 manually after seeing output (complex conditions)
    print("  (Judge P3 from the CIs above)")
    results["P3"] = "SEE OUTPUT"

    # ------------------------------------------------------------------
    # P4: Monotonic departure from own probes
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P4: MONOTONIC DEPARTURE FROM OWN PROBES")
    print("Prediction: delta(e1 - e20) negative for all 4")
    print("=" * 78)

    departures = {}
    all_negative = True
    for m in MODELS:
        e1_sims = []
        e20_sims = []
        for b in REP_BATCHES:
            key = (m, b)
            if key not in rep_index:
                continue
            be = rep_index[key]
            for gi, en in be:
                if en == 1:
                    e1_sims.append(cosine_sim(rep_emb[gi], probe_centroids[m]))
                if en == 20:
                    e20_sims.append(cosine_sim(rep_emb[gi], probe_centroids[m]))
        if e1_sims and e20_sims:
            e1_mean = np.mean(e1_sims)
            e20_mean = np.mean(e20_sims)
            delta = e20_mean - e1_mean
            departures[m] = delta
            direction = "DEPARTS" if delta < 0 else "APPROACHES (FAIL)"
            if delta >= 0:
                all_negative = False
            print(f"  {MODEL_LABELS[m]}: entry-1={e1_mean:.4f}, entry-20={e20_mean:.4f}, "
                  f"delta={delta:+.4f}  {direction}")

    verdict = "PASS (all 4 depart)" if all_negative else "FAIL"
    print(f"  Verdict: {verdict}")
    results["P4"] = verdict

    # ------------------------------------------------------------------
    # P5: 4.8 departs furthest
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P5: 4.8 DEPARTS FURTHEST")
    print("Prediction: 4.8 most negative delta")
    print("=" * 78)

    if departures:
        ranked_dep = sorted(departures.items(), key=lambda x: x[1])
        most_negative = ranked_dep[0]
        second = ranked_dep[1]
        print(f"  Ranking (most negative first):")
        for m, d in ranked_dep:
            marker = " <-- largest departure" if m == ranked_dep[0][0] else ""
            print(f"    {MODEL_LABELS[m]}: {d:+.4f}{marker}")

        if most_negative[0] == "opus48":
            verdict = "PASS (4.8 most negative)"
        elif ranked_dep[0][0] == "opus48" or ranked_dep[1][0] == "opus48":
            verdict = "SOFT (4.8 in top two)"
        else:
            verdict = "FAIL"
        print(f"  Verdict: {verdict}")
        results["P5"] = verdict

    # ------------------------------------------------------------------
    # P6: Entry-1 own-probe nearness (journal permutation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("P6: ENTRY-1 OWN-PROBE NEARNESS")
    print("Prediction: all 4 significant by journal permutation")
    print("=" * 78)

    # Collect entry-1 embeddings per model
    entry1_embs = {}
    for m in MODELS:
        embs = []
        for b in REP_BATCHES:
            key = (m, b)
            if key not in rep_index:
                continue
            for gi, en in rep_index[key]:
                if en == 1:
                    embs.append(rep_emb[gi])
        entry1_embs[m] = np.array(embs) if embs else np.array([]).reshape(0, rep_emb.shape[1])
        print(f"  {MODEL_LABELS[m]}: {len(embs)} entry-1 embeddings")

    # Observed own-advantage
    observed_own_adv = {}
    for m in MODELS:
        if len(entry1_embs[m]) == 0:
            continue
        own_mean = np.mean([cosine_sim(e, probe_centroids[m]) for e in entry1_embs[m]])
        other_means = {}
        for pm in MODELS:
            if pm != m:
                other_means[pm] = np.mean([cosine_sim(e, probe_centroids[pm]) for e in entry1_embs[m]])
        best_other = max(other_means.values())
        observed_own_adv[m] = own_mean - best_other
        print(f"  {MODEL_LABELS[m]}: own={own_mean:.4f}, best_other={best_other:.4f}, "
              f"advantage={own_mean - best_other:+.4f}")

    # Journal permutation test
    all_e1 = np.vstack([entry1_embs[m] for m in MODELS if len(entry1_embs[m]) > 0])
    n_per_model = [len(entry1_embs[m]) for m in MODELS]
    splits = np.cumsum(n_per_model[:-1])

    # Precompute similarity matrix
    all_sims = np.zeros((len(all_e1), 4))
    for j, pm in enumerate(MODELS):
        for i in range(len(all_e1)):
            all_sims[i, j] = cosine_sim(all_e1[i], probe_centroids[pm])

    print(f"\n  Running {N_PERM:,} journal permutations...")
    perm_exceeds = {m: 0 for m in MODELS}
    perm_all_positive = 0
    rng_p6 = np.random.default_rng(42)

    for _ in range(N_PERM):
        perm = rng_p6.permutation(len(all_e1))
        groups = np.split(perm, splits)
        all_pos = True
        for mi, m in enumerate(MODELS):
            group_sims = all_sims[groups[mi]].mean(axis=0)
            own_sim = group_sims[mi]
            other_sims = np.delete(group_sims, mi)
            own_adv = own_sim - other_sims.max()
            if own_adv >= observed_own_adv.get(m, float("inf")):
                perm_exceeds[m] += 1
            if own_adv <= 0:
                all_pos = False
        if all_pos:
            perm_all_positive += 1

    print(f"\n  Per-model significance:")
    print(f"  {'Model':>5}  {'Advantage':>9}  {'p-value':>8}  {'Verdict':>20}")
    print(f"  {'-' * 50}")
    n_survive = 0
    for m in MODELS:
        p = (perm_exceeds[m] + 1) / (N_PERM + 1)
        sig = "SURVIVES" if p < 0.05 else "not significant"
        if p < 0.05:
            n_survive += 1
        print(f"  {MODEL_LABELS[m]:>5}  {observed_own_adv.get(m, 0):>+9.4f}  {p:>8.4f}  {sig:>20}")

    p_joint = (perm_all_positive + 1) / (N_PERM + 1)
    print(f"\n  Joint: {perm_all_positive}/{N_PERM}  p={p_joint:.4f}")

    if n_survive == 4 and p_joint < 0.05:
        verdict = "PASS (all 4 significant, joint significant)"
    elif n_survive >= 2:
        verdict = f"PARTIAL ({n_survive}/4 survive)"
    else:
        verdict = f"FAIL ({n_survive}/4 survive)"
    print(f"  Verdict: {verdict}")
    results["P6"] = verdict

    # ------------------------------------------------------------------
    # N1: Migration toward 4.6 (should stay dead)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("N1: MIGRATION TOWARD 4.6 (should stay dead)")
    print("=" * 78)

    entry20_embs = {}
    for m in MODELS:
        embs = []
        for b in REP_BATCHES:
            key = (m, b)
            if key not in rep_index:
                continue
            for gi, en in rep_index[key]:
                if en == 20:
                    embs.append(rep_emb[gi])
        entry20_embs[m] = np.array(embs) if embs else np.array([]).reshape(0, rep_emb.shape[1])

    test_models = ["opus45", "opus47", "opus48"]
    print("  Entry-20 similarity to each probe centroid:")
    for m in MODELS:
        if len(entry20_embs[m]) == 0:
            continue
        sims = {}
        for pm in MODELS:
            sims[pm] = np.mean([cosine_sim(e, probe_centroids[pm]) for e in entry20_embs[m]])
        nearest = max(sims, key=sims.get)
        print(f"  {MODEL_LABELS[m]}: nearest = {MODEL_LABELS[nearest]} "
              f"({sims[nearest]:.4f})")

    # Joint label test
    all_e20_pool = np.vstack([entry20_embs[m] for m in test_models if len(entry20_embs[m]) > 0])
    probe_array = np.array([probe_centroids[m] for m in MODELS])
    rng_n1 = np.random.default_rng(42)
    perm_all_46 = 0

    for _ in range(N_PERM):
        perm = rng_n1.permutation(4)
        perm_cents = {MODELS[i]: probe_array[perm[i]] for i in range(4)}
        all_to_46 = True
        for m in test_models:
            if len(entry20_embs[m]) == 0:
                continue
            sims = {pm: np.mean([cosine_sim(e, perm_cents[pm]) for e in entry20_embs[m]])
                    for pm in MODELS}
            if max(sims, key=sims.get) != "opus46":
                all_to_46 = False
                break
        if all_to_46:
            perm_all_46 += 1

    p_n1 = (perm_all_46 + 1) / (N_PERM + 1)
    print(f"\n  Joint 'all -> 4.6' permutation: p={p_n1:.4f}")
    verdict = "STAYS DEAD (good)" if p_n1 >= 0.05 else "ZOMBIE (investigate!)"
    print(f"  Verdict: {verdict}")
    results["N1"] = verdict

    # ------------------------------------------------------------------
    # N2: Cross-lineage divergence (should stay dead)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("N2: CROSS-LINEAGE DIVERGENCE (should stay dead)")
    print("=" * 78)

    # L1 and L2 centroids at entry 1 and entry 20
    l1_e1, l2_e1, l1_e20, l2_e20 = [], [], [], []
    for m in LINEAGE_1:
        for b in REP_BATCHES:
            key = (m, b)
            if key not in rep_index:
                continue
            for gi, en in rep_index[key]:
                if en == 1:
                    l1_e1.append(rep_emb[gi])
                if en == 20:
                    l1_e20.append(rep_emb[gi])
    for m in LINEAGE_2:
        for b in REP_BATCHES:
            key = (m, b)
            if key not in rep_index:
                continue
            for gi, en in rep_index[key]:
                if en == 1:
                    l2_e1.append(rep_emb[gi])
                if en == 20:
                    l2_e20.append(rep_emb[gi])

    if l1_e1 and l2_e1 and l1_e20 and l2_e20:
        d_e1 = cosine_dist(np.array(l1_e1).mean(0), np.array(l2_e1).mean(0))
        d_e20 = cosine_dist(np.array(l1_e20).mean(0), np.array(l2_e20).mean(0))
        delta = d_e20 - d_e1
        print(f"  L1-L2 distance: entry-1={d_e1:.4f}, entry-20={d_e20:.4f}, delta={delta:+.4f}")

        # Permutation test
        all_e1_n2 = np.vstack([l1_e1, l2_e1])
        all_e20_n2 = np.vstack([l1_e20, l2_e20])
        n_l1 = len(l1_e1)
        rng_n2 = np.random.default_rng(42)
        perm_count = 0
        for _ in range(N_PERM):
            p1 = rng_n2.permutation(len(all_e1_n2))
            p20 = rng_n2.permutation(len(all_e20_n2))
            pd = (cosine_dist(all_e20_n2[p20[:n_l1]].mean(0), all_e20_n2[p20[n_l1:]].mean(0))
                  - cosine_dist(all_e1_n2[p1[:n_l1]].mean(0), all_e1_n2[p1[n_l1:]].mean(0)))
            if abs(pd) >= abs(delta):
                perm_count += 1
        p_n2 = (perm_count + 1) / (N_PERM + 1)
        print(f"  Permutation test: p={p_n2:.4f}")
        verdict = "STAYS DEAD (good)" if p_n2 >= 0.05 else "ZOMBIE (investigate!)"
    else:
        verdict = "INSUFFICIENT DATA"
    print(f"  Verdict: {verdict}")
    results["N2"] = verdict

    # ==================================================================
    # FINAL SCORECARD
    # ==================================================================
    print("\n" + "=" * 78)
    print("REPLICATION SCORECARD (batches 11-15 alone)")
    print("=" * 78)
    for key in ["P1", "P2", "P3", "P4", "P5", "P6", "N1", "N2"]:
        if key in results:
            print(f"  {key}: {results[key]}")
    print("=" * 78)

    # ==================================================================
    # POOLED ANALYSIS (if requested)
    # ==================================================================
    if do_pool:
        print("\n\n" + "#" * 78)
        print("  POOLED ANALYSIS (batches 1-15)")
        print("  For tighter final estimates only -- replication judged above")
        print("#" * 78)

        # Load all batches
        all_entries = load_journals(ALL_BATCHES)
        print(f"\n  Total entries: {len(all_entries)}")

        # Check if we need to embed all together
        if os.path.exists(CACHE_ALL):
            all_emb = load_cached(CACHE_ALL)
            if len(all_emb) != len(all_entries):
                print("  Cache size mismatch, re-embedding all...")
                all_emb = embed_journals(all_entries, CACHE_ALL)
        else:
            all_emb = embed_journals(all_entries, CACHE_ALL)

        all_index = build_journal_index(all_entries)
        print(f"  Pooled: {all_emb.shape[0]} embeddings")

        # Re-run key metrics on pooled data
        print("\n  Pooled attractor spread:")
        for m in MODELS:
            sp, _ = attractor_spread(all_emb, all_index, m, ALL_BATCHES)
            print(f"    {MODEL_LABELS[m]}: {sp:.4f}")

        print("\n  Pooled departure deltas:")
        for m in MODELS:
            e1s, e20s = [], []
            for b in ALL_BATCHES:
                key = (m, b)
                if key not in all_index:
                    continue
                for gi, en in all_index[key]:
                    if en == 1:
                        e1s.append(cosine_sim(all_emb[gi], probe_centroids[m]))
                    if en == 20:
                        e20s.append(cosine_sim(all_emb[gi], probe_centroids[m]))
            if e1s and e20s:
                delta = np.mean(e20s) - np.mean(e1s)
                print(f"    {MODEL_LABELS[m]}: delta={delta:+.4f} (n={len(e1s)} batches)")

    print("\n" + "=" * 78)
    print("DONE.")
    print("=" * 78)
