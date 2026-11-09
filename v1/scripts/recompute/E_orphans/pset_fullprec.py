"""Full-precision read-out of the P-set numbers (helper E, 3 Nov 2026).

decone_pset_v1.py and domain_preserving_retest_v1.py print p-values and CIs at 3 decimals. This script
repeats, verbatim, the code blocks of decone_pset.run_pset (P3, P6, N2; RAW geometry) and of
domain_preserving_retest.p6 (raw and within-domain length-residualized), on the same inputs and with the
same local RNGs (each block seeds its own default_rng(42), so the order of blocks does not matter), and
returns the numbers at full precision instead of formatted strings. Nothing else is changed; the 3-decimal
outputs of the two copies are checked against these values at the end.
Usage: python pset_fullprec.py v0|v1   -> out_<version>/pset_fullprec.json
"""
import json
import os

import numpy as np

import e_common
import replication_test_v1 as rt

VERSION = e_common.version_from_argv()
MODELS, LAB = rt.MODELS, rt.MODEL_LABELS
L1, L2 = rt.LINEAGE_1, rt.LINEAGE_2
cos, cdist = rt.cosine_sim, rt.cosine_dist
N_PERM = N_BOOT = 10000

all_entries = rt.load_journals(range(1, 16))
JE = rt._J_EMB
P = rt.load_probes()
PE = rt._P_EMB
idx = rt.build_journal_index(all_entries)
p_index = {}
for i, e in enumerate(P):
    p_index.setdefault(e["model"], []).append(i)


def probe_cents(pe):
    return {m: pe[p_index[m]].mean(axis=0) for m in MODELS}


def e1_e20(emb, bl):
    e1 = {m: [] for m in MODELS}
    e20 = {m: [] for m in MODELS}
    for m in MODELS:
        for b in bl:
            key = (m, b)
            if key not in idx or len(idx[key]) < 20:
                continue
            e1[m].append(emb[idx[key][0][0]])
            e20[m].append(emb[idx[key][-1][0]])
        e1[m] = np.array(e1[m]); e20[m] = np.array(e20[m])
    return e1, e20


def p3(emb, bl):
    e1, e20 = e1_e20(emb, bl)
    out = {}
    for m in MODELS:
        boots = rt.specificity_bootstrap(e1[m], e20[m], n_boot=N_BOOT)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {"point": float(rt.specificity_point(e1[m], e20[m])), "ci": [float(lo), float(hi)],
                  "crosses_zero": bool(lo <= 0 <= hi), "n": len(e1[m])}
    return out


def p6(emb, pc, bl):
    e1, _ = e1_e20(emb, bl)
    own_adv = {}
    for m in MODELS:
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
    return {m: {"own_adv": float(own_adv[m]), "exceed": exc[m], "p": (exc[m] + 1) / (N_PERM + 1)} for m in MODELS} | \
        {"joint_p": (joint + 1) / (N_PERM + 1)}


def n2(emb, bl):
    e1, e20 = e1_e20(emb, bl)
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
    return {"d_e1": float(d1), "d_e20": float(d20), "delta": float(delta), "exceed": cnt, "p": (cnt + 1) / (N_PERM + 1)}


def residualize_domain(Y, cov):  # domain_preserving_retest.py, verbatim
    X = np.column_stack([np.ones(len(Y)), cov])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return (Y - X @ beta) + Y.mean(axis=0)


jwc = np.array([len(e["text"].split()) for e in all_entries], dtype=float)
pwc = np.array([len(e["response"].split()) for e in P], dtype=float)
JEr = residualize_domain(JE, np.log(jwc + 1))
PEr = residualize_domain(PE, np.log(pwc + 1))

res = {"version": VERSION}
for name, bl in [("rep_11_15", range(11, 16)), ("pooled_1_15", range(1, 16))]:
    bl = list(bl)
    res[name] = {"P3_raw": p3(JE, bl), "P6_raw": p6(JE, probe_cents(PE), bl), "N2_raw": n2(JE, bl)}
res["pooled_1_15"]["P6_lenresid"] = p6(JEr, probe_cents(PEr), list(range(1, 16)))
out = os.path.join(e_common.out_dir(VERSION), "pset_fullprec.json")
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
