"""REBUILD (3 Nov 2026, Claude Opus 5.5 (Code), helper A2): Fable 5's lag-1 residual cosine (+0.347, §6.2.2).

The 20 June compute pass (04_analysis/Compute_Results_2026-06-20.md, item 5) ported analysis 1 of
embedding_keystone_geometry.py to Fable 5: "computed on the 289 text entries, the 11 [silence] events
excluded, positions renumbered over text entries". That port script was not found (searched
02_instruments, 04_analysis, 03_experiments, the repository's code/, Desktop). This file applies the
code of analysis 1 VERBATIM (mean trajectory per position, unit residuals, consecutive pairs, other-batch
permutation null, 2000 perms, RNG seed 42) to Fable 5's entries, with:
  variant "renumbered": entry_num = rank among the notebook's text entries (the port as described);
  variant "file_pos":   entry_num = position in the file, silences counted (pairs across a silence drop).
It counts as a rebuild only if a variant hits the printed +0.347 on v0 inputs.
Inputs via a2_common -> v1_common (version = argv v0|v1); output out_<version>/fable_carryover.json.
"""
import json, os
import numpy as np
import a2_common

VERSION = a2_common.version_from_argv()
_oe, _ov, FE_entries, FE = a2_common.load(VERSION)
ALL_BATCHES = list(range(1, 16))


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def analysis1(J, JE, m="fable5", N_PERM=2000, seed=42):
    RNG = np.random.default_rng(seed)
    JI = {}
    for i, e in enumerate(J):
        JI.setdefault((e["model"], e["batch"]), []).append((i, e["entry_num"]))
    for k in JI:
        JI[k].sort(key=lambda x: x[1])

    def batch_traj(m):
        for b in ALL_BATCHES:
            key = (m, b)
            if key in JI and len(JI[key]) >= 2:
                yield b, JI[key]

    # ---- verbatim from analysis 1 ----
    mean_traj = {}
    for pos in range(1, 21):
        idxs = [gi for b, be in batch_traj(m) for gi, en in be if en == pos]
        if idxs:
            mean_traj[(m, pos)] = JE[idxs].mean(axis=0)
    Rn = np.zeros_like(JE)
    for i, e in enumerate(J):
        key = (e["model"], e["entry_num"])
        if key in mean_traj:
            r = JE[i] - mean_traj[key]
            n = np.linalg.norm(r)
            Rn[i] = r / n if n > 1e-12 else r
    iN, iNp1, pos_next, batch_of = [], [], [], []
    for b, be in batch_traj(m):
        for k in range(len(be) - 1):
            gi, en = be[k]; gj, en2 = be[k + 1]
            if en2 == en + 1:
                iN.append(gi); iNp1.append(gj); pos_next.append(en2); batch_of.append(b)
    iN = np.array(iN); iNp1 = np.array(iNp1)
    pos_next = np.array(pos_next); batch_of = np.array(batch_of)
    A = Rn[iN]
    obs_per_pair = np.sum(A * Rn[iNp1], axis=1)
    obs_cos = obs_per_pair.mean()
    per_batch = {}
    for b in np.unique(batch_of):
        per_batch[int(b)] = obs_per_pair[batch_of == b].mean()
    n_pos = sum(1 for v in per_batch.values() if v > 0)
    pos_pool = {}
    for i, e in enumerate(J):
        if e["model"] == m:
            pos_pool.setdefault(e["entry_num"], {}).setdefault(e["batch"], []).append(i)
    cand_lists = []
    for p_next, b in zip(pos_next, batch_of):
        pool = [idx for bb, idxs in pos_pool.get(p_next, {}).items() if bb != b for idx in idxs]
        cand_lists.append(np.array(pool))
    null_means = np.empty(N_PERM)
    for t in range(N_PERM):
        picks = np.array([cl[RNG.integers(len(cl))] for cl in cand_lists])
        null_means[t] = np.sum(A * Rn[picks], axis=1).mean()
    p = (np.sum(null_means >= obs_cos) + 1) / (N_PERM + 1)
    gaps = []
    for gap in range(1, 6):
        sims = []
        for b, be in batch_traj(m):
            for k in range(len(be) - gap):
                gi, en = be[k]; gj, en2 = be[k + gap]
                if en2 == en + gap:
                    sims.append(float(np.dot(Rn[gi], Rn[gj])))
        gaps.append(float(np.mean(sims)) if sims else float("nan"))
    return {"obs": float(obs_cos), "p": float(p), "n_pos": n_pos, "n_batch": len(per_batch),
            "n_pairs": int(len(iN)), "null_mean": float(null_means.mean()), "decay_gap1_5": gaps}


res = {}
# variant renumbered
J = []
for b in ALL_BATCHES:
    k = 0
    for e in FE_entries:
        if e["batch"] == b:
            k += 1
            J.append({"model": "fable5", "batch": b, "entry_num": k})
order = [i for b in ALL_BATCHES for i, e in enumerate(FE_entries) if e["batch"] == b]
res["renumbered"] = analysis1(J, FE[order])
J2 = [{"model": "fable5", "batch": e["batch"], "entry_num": e["entry_num"]} for e in FE_entries]
res["file_pos"] = analysis1(J2, FE.copy())
for k, v in res.items():
    print(VERSION, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "decay_gap1_5"})
json.dump(res, open(os.path.join(a2_common.out_dir(VERSION), "fable_carryover.json"), "w"), indent=2)
