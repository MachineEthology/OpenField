"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/decone_pset.py.
Changed: input paths only. Imports replication_test_v1 (the module copy with e_common loaders) instead of
replication_test; journal vectors = the clean npz (never journal_embeddings_cache_all.json); probe vectors =
the June probe cache. Usage: python decone_pset_v1.py v0|v1
"""
"""
P-set on principled de-coned geometry  (#2 follow-up)
=====================================================

Re-runs the pre-registered P-set (P1-P6, N1-N2) under a single principled
de-coning, side by side with raw, for batches 11-15 (replication) and 1-15
(pooled). Reuses the metric functions from replication_test.py verbatim so the
tests are identical -- only the geometry the embeddings live in changes.

PRINCIPLED DE-CONING (one choice, declared up front, not tuned to an outcome):
  fit on the POOLED journal+probe corpus:  x' = (x - mu) - ((x-mu).v1) v1
  i.e. mean-center + remove the single dominant direction. Chosen because:
   - the raw space is severely anisotropic (random-pair cosine 0.706);
   - it is defined IDENTICALLY for journals and probes (the cross-domain
     predictions P3-P6/N1-N2 require one shared space -- a per-domain length
     residualization is not well defined for probes);
   - it is the mildest standard de-coning (all-but-the-top-1).

CAVEAT recorded in advance: the SPREAD metric (P1/P2) is already known to be
unstable across de-coning choices (whitening K-sweep + length residualization).
We report its de-coned value but do NOT treat any single choice as settling P1.
The spread-stability table at the end makes that instability explicit.

Cache-only (uses journal_embeddings_cache_all.json + probe cache). No API.
"""
import os
import numpy as np
import replication_test_v1 as rt   # [E V1] was replication_test

BASE = rt.BASE
MODELS = rt.MODELS
LAB = rt.MODEL_LABELS
L1, L2 = rt.LINEAGE_1, rt.LINEAGE_2
cos = rt.cosine_sim
cdist = rt.cosine_dist
N_PERM = 10000
N_BOOT = 10000

# ---- load everything from caches -------------------------------------------
all_entries = rt.load_journals(range(1, 16))
all_emb_raw = rt._J_EMB  # [E V1] was journal_embeddings_cache_all.json
assert len(all_entries) == len(all_emb_raw)
p_entries = rt.load_probes()
p_emb_raw = rt._P_EMB  # [E V1] was BASE/probe_embeddings_cache.json (same file, now in 03_experiments)

p_index = {}
for i, e in enumerate(p_entries):
    p_index.setdefault(e["model"], []).append(i)


def decone(journal, probe, k=1):
    """Fit mean + top-k directions on pooled corpus; remove from both."""
    pooled = np.vstack([journal, probe])
    mu = pooled.mean(axis=0)
    Jc, Pc = journal - mu, probe - mu
    if k == 0:
        return Jc, Pc
    U, S, Vt = np.linalg.svd(np.vstack([Jc, Pc]), full_matrices=False)
    top = Vt[:k]
    return Jc - (Jc @ top.T) @ top, Pc - (Pc @ top.T) @ top


# ---- the P-set as a function of geometry + batch set -----------------------
def run_pset(j_emb, p_centroids, batches, label):
    idx = rt.build_journal_index([e for e in all_entries])  # full index; filter by batch in metrics
    res = {}
    bl = list(batches)

    # P1 spread ordering
    spreads, cents = {}, {}
    for m in MODELS:
        sp, c = rt.attractor_spread(j_emb, idx, m, bl)
        spreads[m], cents[m] = sp, c
    order = sorted(MODELS, key=lambda m: spreads[m])
    p1 = (order[0] == "opus46") and (set(order[-2:]) == set(L2))
    res["P1"] = ("PASS" if p1 else ("PARTIAL(4.6 low)" if order[0] == "opus46" else "FAIL"),
                 " < ".join(LAB[m] for m in order))

    # P2 spread gap permutation (4.6 vs each L2)
    p2p = {}
    for other in L2:
        if len(cents["opus46"]) and len(cents[other]):
            _, pv = rt.spread_permutation_test(cents["opus46"], cents[other], n_perm=N_PERM)
            p2p[other] = pv
    nsig = sum(1 for v in p2p.values() if v < 0.05)
    res["P2"] = (["FAIL", "PARTIAL", "PASS"][nsig], "  ".join(f"4.6v{LAB[o]} p={p2p[o]:.3f}" for o in L2))

    # collect e1/e20 per model
    def collect(pos_set):
        out = {m: [] for m in MODELS}
        for m in MODELS:
            for b in bl:
                key = (m, b)
                if key not in idx:
                    continue
                for gi, en in idx[key]:
                    if en in pos_set:
                        out[m].append((en, gi))
        return out

    e1 = {m: [] for m in MODELS}
    e20 = {m: [] for m in MODELS}
    for m in MODELS:
        for b in bl:
            key = (m, b)
            if key not in idx or len(idx[key]) < 20:
                continue
            e1[m].append(j_emb[idx[key][0][0]])
            e20[m].append(j_emb[idx[key][-1][0]])
        e1[m] = np.array(e1[m]); e20[m] = np.array(e20[m])

    # P3 specificity (4.7 CI clear, others cross zero)
    p3lines, p3_ok = [], {}
    for m in MODELS:
        if len(e1[m]) < 3:
            p3lines.append(f"{LAB[m]}:n/a"); continue
        boots = rt.specificity_bootstrap(e1[m], e20[m], n_boot=N_BOOT)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        clear = not (lo <= 0 <= hi)
        p3_ok[m] = clear
        p3lines.append(f"{LAB[m]}:[{lo:+.3f},{hi:+.3f}]{'*' if clear else ''}")
    p3pass = p3_ok.get("opus47", False) and not any(p3_ok.get(m, False) for m in ["opus45", "opus46", "opus48"])
    res["P3"] = ("PASS" if p3pass else ("4.7+others" if p3_ok.get("opus47") else "4.7 not clear"),
                 "  ".join(p3lines))

    # P4 departure all negative + P5 4.8 furthest
    dep = {}
    for m in MODELS:
        if len(e1[m]) and len(e20[m]):
            d = np.mean([cos(v, p_centroids[m]) for v in e20[m]]) - \
                np.mean([cos(v, p_centroids[m]) for v in e1[m]])
            dep[m] = d
    res["P4"] = ("PASS" if all(v < 0 for v in dep.values()) else "FAIL",
                 "  ".join(f"{LAB[m]}={dep[m]:+.3f}" for m in MODELS if m in dep))
    if dep:
        dep_order = sorted(dep, key=lambda m: dep[m])
        res["P5"] = ("PASS" if dep_order[0] == "opus48"
                     else ("SOFT" if "opus48" in dep_order[:2] else "FAIL"),
                     "furthest=" + LAB[dep_order[0]])

    # P6 entry-1 own-probe nearness (journal permutation)
    own_adv = {}
    for m in MODELS:
        if not len(e1[m]):
            continue
        own = np.mean([cos(v, p_centroids[m]) for v in e1[m]])
        best_other = max(np.mean([cos(v, p_centroids[pm]) for v in e1[m]])
                         for pm in MODELS if pm != m)
        own_adv[m] = own - best_other
    all_e1 = np.vstack([e1[m] for m in MODELS if len(e1[m])])
    sizes = [len(e1[m]) for m in MODELS]
    splits = np.cumsum(sizes[:-1])
    sims = np.zeros((len(all_e1), 4))
    for j, pm in enumerate(MODELS):
        sims[:, j] = [cos(v, p_centroids[pm]) for v in all_e1]
    rng = np.random.default_rng(42)
    exc = {m: 0 for m in MODELS}; joint = 0
    for _ in range(N_PERM):
        groups = np.split(rng.permutation(len(all_e1)), splits)
        allpos = True
        for mi, m in enumerate(MODELS):
            gs = sims[groups[mi]].mean(axis=0)
            adv = gs[mi] - np.delete(gs, mi).max()
            if adv >= own_adv.get(m, np.inf):
                exc[m] += 1
            if adv <= 0:
                allpos = False
        if allpos:
            joint += 1
    nsurv = sum(1 for m in MODELS if (exc[m] + 1) / (N_PERM + 1) < 0.05)
    pj = (joint + 1) / (N_PERM + 1)
    res["P6"] = ("PASS" if nsurv == 4 and pj < 0.05 else f"{nsurv}/4 surv",
                 "  ".join(f"{LAB[m]} p={(exc[m]+1)/(N_PERM+1):.3f}" for m in MODELS) + f"  joint={pj:.3f}")

    # N1 migration toward 4.6 (joint label perm over 4.5/4.7/4.8 -> should stay dead)
    test_models = ["opus45", "opus47", "opus48"]
    parr = np.array([p_centroids[m] for m in MODELS])
    rng = np.random.default_rng(42)
    all46 = 0
    for _ in range(N_PERM):
        perm = rng.permutation(4)
        pc = {MODELS[i]: parr[perm[i]] for i in range(4)}
        ok = True
        for m in test_models:
            if not len(e20[m]):
                continue
            s = {pm: np.mean([cos(v, pc[pm]) for v in e20[m]]) for pm in MODELS}
            if max(s, key=s.get) != "opus46":
                ok = False; break
        if ok:
            all46 += 1
    pn1 = (all46 + 1) / (N_PERM + 1)
    res["N1"] = ("STAYS DEAD" if pn1 >= 0.05 else "ZOMBIE!", f"p={pn1:.3f}")

    # N2 cross-lineage divergence (should stay dead)
    l1e1 = np.vstack([e1[m] for m in L1 if len(e1[m])])
    l2e1 = np.vstack([e1[m] for m in L2 if len(e1[m])])
    l1e20 = np.vstack([e20[m] for m in L1 if len(e20[m])])
    l2e20 = np.vstack([e20[m] for m in L2 if len(e20[m])])
    d1 = cdist(l1e1.mean(0), l2e1.mean(0))
    d20 = cdist(l1e20.mean(0), l2e20.mean(0))
    delta = d20 - d1
    ae1 = np.vstack([l1e1, l2e1]); ae20 = np.vstack([l1e20, l2e20]); nl1 = len(l1e1)
    rng = np.random.default_rng(42)
    cnt = 0
    for _ in range(N_PERM):
        p1p = rng.permutation(len(ae1)); p20 = rng.permutation(len(ae20))
        pd = (cdist(ae20[p20[:nl1]].mean(0), ae20[p20[nl1:]].mean(0))
              - cdist(ae1[p1p[:nl1]].mean(0), ae1[p1p[nl1:]].mean(0)))
        if abs(pd) >= abs(delta):
            cnt += 1
    pn2 = (cnt + 1) / (N_PERM + 1)
    res["N2"] = ("STAYS DEAD" if pn2 >= 0.05 else "ZOMBIE!", f"delta={delta:+.4f} p={pn2:.3f}")
    return res


def probe_cents(p_emb):
    return {m: p_emb[p_index[m]].mean(axis=0) for m in MODELS}


# ---- run: raw vs de-coned, for 11-15 and 1-15 ------------------------------
Jdc, Pdc = decone(all_emb_raw, p_emb_raw, k=1)
geometries = {
    "RAW": (all_emb_raw, probe_cents(p_emb_raw)),
    "DECONE(k=1)": (Jdc, probe_cents(Pdc)),
}
batchsets = {"replication 11-15": range(11, 16), "pooled 1-15": range(1, 16)}

for bname, bset in batchsets.items():
    print("\n" + "#" * 80)
    print(f"#  BATCH SET: {bname}")
    print("#" * 80)
    out = {}
    for gname, (je, pc) in geometries.items():
        out[gname] = run_pset(je, pc, bset, gname)
    keys = ["P1", "P2", "P3", "P4", "P5", "P6", "N1", "N2"]
    for k in keys:
        print(f"\n  {k}")
        for gname in geometries:
            v, detail = out[gname][k]
            print(f"    {gname:>12}: {v:<16} {detail}")

# ---- spread-stability table (P1 across de-coning choices, pooled) ----------
print("\n" + "#" * 80)
print("#  SPREAD STABILITY (P1) across de-coning choices  [pooled 1-15]")
print("#" * 80)
idx = rt.build_journal_index(all_entries)


def spread_order(E):
    sp = {m: rt.attractor_spread(E, idx, m, range(1, 16))[0] for m in MODELS}
    return sorted(MODELS, key=lambda m: sp[m]), sp


variants = {"raw": all_emb_raw}
for k in [0, 1, 2, 3]:
    Jk, _ = decone(all_emb_raw, p_emb_raw, k=k)
    variants[f"mean+top{k}" if k else "mean-center"] = Jk
print(f"\n  {'variant':>14}  {'order (low->high)':>26}  {'4.6 lowest?':>11}  {'P1?':>4}")
print("  " + "-" * 64)
for name, E in variants.items():
    order, sp = spread_order(E)
    p1 = order[0] == "opus46" and set(order[-2:]) == set(L2)
    print(f"  {name:>14}  {' < '.join(LAB[m] for m in order):>26}  "
          f"{'YES' if order[0]=='opus46' else 'no':>11}  {'YES' if p1 else 'no':>4}")
print("\n  Confirms: P1 spread ordering is choice-dependent. Lead with coupling.")
