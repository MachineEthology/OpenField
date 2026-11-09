"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/domain_preserving_retest.py.
Changed: input paths only. Imports replication_test_v1 (the module copy with e_common loaders) instead of
replication_test; journal vectors = the clean npz (never journal_embeddings_cache_all.json); probe vectors =
the June probe cache. Usage: python domain_preserving_retest_v1.py v0|v1
"""
"""
Domain-preserving length control for the cross-domain predictions (#2b)
=======================================================================

decone_pset.py showed pooled-PC1 removal is invalid for cross-domain tests
(PC1 = the journal-vs-probe domain axis, r=0.974). The right control removes the
LENGTH gradient WITHIN each domain separately, preserving the domain axis:

    J' = residual(J on [1, log-words_J]) + mean(J)     # journals, own length
    P' = residual(P on [1, log-words_P]) + mean(P)     # probes,   own length

This keeps journals centered at the journal mean and probes at the probe mean
(domain structure intact) but removes the within-domain length-linear component.
We do NOT residualize on arc position -- that would remove the e1->e20 departure
by construction. The question is purely: is the departure driven by length, or
by length-independent content?

Re-tests P4 (all depart), P5 (4.8 furthest), P6 (entry-1 own-probe nearness),
pooled 1-15, raw vs length-residualized. Plus a direct diagnostic: per-batch
correlation between departure magnitude and length growth (e20 minus e1 words).

Cache-only, no API.
"""
import os, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import replication_test_v1 as rt  # [E V1] was replication_test

BASE = rt.BASE
MODELS = rt.MODELS
LAB = rt.MODEL_LABELS
L1, L2 = rt.LINEAGE_1, rt.LINEAGE_2
cos = rt.cosine_sim
N_PERM = 10000

# ---- load -----------------------------------------------------------------
J = rt.load_journals(range(1, 16))
JE = rt._J_EMB  # [E V1] was journal_embeddings_cache_all.json
P = rt.load_probes()
PE = rt._P_EMB  # [E V1] was BASE/probe_embeddings_cache.json (same file, now in 03_experiments)
idx = rt.build_journal_index(J)

# word counts (journals from text; probes need their response text)
jwc = np.array([len(e["text"].split()) for e in J], dtype=float)
pwc = np.array([len(e["response"].split()) for e in P], dtype=float)
p_index = {}
for i, e in enumerate(P):
    p_index.setdefault(e["model"], []).append(i)


def residualize_domain(Y, cov):
    """OLS-residualize each dim on [1, cov], then add the domain mean back."""
    X = np.column_stack([np.ones(len(Y)), cov])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return (Y - X @ beta) + Y.mean(axis=0)


JEr = residualize_domain(JE, np.log(jwc + 1))
PEr = residualize_domain(PE, np.log(pwc + 1))

# sanity: domain still separated? (a journal should still look like a journal)
dom_raw = cos(JE.mean(0), PE.mean(0))
dom_res = cos(JEr.mean(0), PEr.mean(0))
print(f"Domain preserved check (cos journal-mean vs probe-mean): "
      f"raw={dom_raw:.3f}  residualized={dom_res:.3f}  (should stay similar)")


def probe_cents(pe):
    return {m: pe[p_index[m]].mean(axis=0) for m in MODELS}


def e1_e20(emb, batches):
    e1 = {m: [] for m in MODELS}
    e20 = {m: [] for m in MODELS}
    for m in MODELS:
        for b in batches:
            key = (m, b)
            if key not in idx or len(idx[key]) < 20:
                continue
            e1[m].append(emb[idx[key][0][0]])
            e20[m].append(emb[idx[key][-1][0]])
        e1[m] = np.array(e1[m]); e20[m] = np.array(e20[m])
    return e1, e20


def departure(emb, pc, batches):
    e1, e20 = e1_e20(emb, batches)
    dep = {}
    for m in MODELS:
        if len(e1[m]) and len(e20[m]):
            dep[m] = (np.mean([cos(v, pc[m]) for v in e20[m]])
                      - np.mean([cos(v, pc[m]) for v in e1[m]]))
    return dep


def p6(emb, pc, batches):
    e1, _ = e1_e20(emb, batches)
    own_adv = {}
    for m in MODELS:
        if not len(e1[m]):
            continue
        own = np.mean([cos(v, pc[m]) for v in e1[m]])
        best = max(np.mean([cos(v, pc[pm]) for v in e1[m]]) for pm in MODELS if pm != m)
        own_adv[m] = own - best
    all_e1 = np.vstack([e1[m] for m in MODELS if len(e1[m])])
    sizes = [len(e1[m]) for m in MODELS]
    splits = np.cumsum(sizes[:-1])
    sims = np.zeros((len(all_e1), 4))
    for j, pm in enumerate(MODELS):
        sims[:, j] = [cos(v, pc[pm]) for v in all_e1]
    rng = np.random.default_rng(42)
    exc = {m: 0 for m in MODELS}; joint = 0
    for _ in range(N_PERM):
        groups = np.split(rng.permutation(len(all_e1)), splits)
        allpos = True
        for mi, m in enumerate(MODELS):
            gs = sims[groups[mi]].mean(0)
            adv = gs[mi] - np.delete(gs, mi).max()
            if adv >= own_adv.get(m, np.inf):
                exc[m] += 1
            if adv <= 0:
                allpos = False
        if allpos:
            joint += 1
    return {m: (exc[m] + 1) / (N_PERM + 1) for m in MODELS}, (joint + 1) / (N_PERM + 1), own_adv


BATCHES = list(range(1, 16))
print("\n" + "=" * 74)
print("P4 / P5  Departure from own probe surface  (pooled 1-15)")
print("=" * 74)
for name, emb, pc in [("raw", JE, probe_cents(PE)),
                      ("len-residualized (domain-preserving)", JEr, probe_cents(PEr))]:
    dep = departure(emb, pc, BATCHES)
    order = sorted(dep, key=lambda m: dep[m])
    allneg = all(v < 0 for v in dep.values())
    print(f"\n  {name}")
    print("    " + "  ".join(f"{LAB[m]}={dep[m]:+.4f}" for m in MODELS))
    print(f"    P4 (all depart): {'PASS' if allneg else 'FAIL'}   "
          f"P5 (4.8 furthest): {'PASS' if order[0]=='opus48' else ('SOFT' if 'opus48' in order[:2] else 'FAIL')}"
          f"  [furthest={LAB[order[0]]}]")

print("\n" + "=" * 74)
print("Diagnostic: is departure just length growth? per-batch correlation")
print("=" * 74)
print("  corr(per-batch departure Δcos, per-batch length growth Δwords)  [raw geometry]")
for m in MODELS:
    deps, grows = [], []
    for b in BATCHES:
        key = (m, b)
        if key not in idx or len(idx[key]) < 20:
            continue
        gi1, gi20 = idx[key][0][0], idx[key][-1][0]
        deps.append(cos(JE[gi20], probe_cents(PE)[m]) - cos(JE[gi1], probe_cents(PE)[m]))
        grows.append(jwc[gi20] - jwc[gi1])
    r = np.corrcoef(deps, grows)[0, 1] if len(deps) > 2 else np.nan
    print(f"    {LAB[m]}: r = {r:+.3f}  (n={len(deps)} batches)")

print("\n" + "=" * 74)
print("P6  Entry-1 own-probe nearness  (pooled 1-15)")
print("=" * 74)
for name, emb, pc in [("raw", JE, probe_cents(PE)),
                      ("len-residualized (domain-preserving)", JEr, probe_cents(PEr))]:
    pv, pj, adv = p6(emb, pc, BATCHES)
    nsurv = sum(1 for m in MODELS if pv[m] < 0.05)
    print(f"\n  {name}")
    print("    " + "  ".join(f"{LAB[m]} adv={adv.get(m,0):+.3f} p={pv[m]:.3f}" for m in MODELS))
    print(f"    joint p={pj:.4f}   {nsurv}/4 survive   "
          f"{'PASS' if nsurv==4 and pj<0.05 else 'partial'}")

print("\n" + "=" * 74)
print("Reading: if departure SURVIVES within-domain length control, it has a")
print("real length-independent component. If it collapses AND the per-batch")
print("corr is high, the departure was substantially length growth. P6 should")
print("survive either way (it does not depend on length).")
print("=" * 74)
