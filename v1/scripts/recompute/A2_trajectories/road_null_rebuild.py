"""REBUILD (3 Nov 2026, Claude Opus 5.5 (Code), helper A2): §5.2.1.2's shared-road numbers at full precision
and the random-split null ("cross-model displacement cosines fall to a mean of 0.000 (95th percentile 0.073);
the observed 0.573-0.849 (mean 0.697) gives p ≈ 0.001", R = 1,000).

The observed part is road_handwriting_v1.py's part A (embedding_road_and_handwriting.py, Opus 4.5-4.8 only,
as in June), re-run here with the same formulas to print full precision. The null's script (20 June compute
pass, 04_analysis/Compute_Results_2026-06-20.md item 4: "within each notebook, shuffle entries and take random
'early/late' halves ... R=1,000: null mean 0.000, 95th percentile 0.073, max 0.189") was not found
(searched 02_instruments, 03_experiments, 04_analysis, 06_paper, the repository code/, Desktop).
Rebuilt: within every notebook the entry labels are permuted, "early" = entries labelled 1-5 and "late" =
16-20 after the shuffle (same >=3/>=3 rule), the per-model mean displacement and the mean of the six
cross-model cosines are recomputed; R = 1,000, numpy default_rng(42). p = (#null >= observed + 1)/(R + 1).
Also (informational, not printed in the paper): the same displacement cosines with Fable 5 added
(5 x 5), since the room's Figure 2 caption speaks of "five roads".
Inputs via a2_common -> v1_common (version = argv v0|v1). Output out_<version>/road.json.
"""
import json, os
import numpy as np
import a2_common

VERSION = a2_common.version_from_argv()
R = 1000
rng = np.random.default_rng(42)
oe, ov, fe, fv = a2_common.load(VERSION)
ov = ov / (np.linalg.norm(ov, axis=1, keepdims=True) + 1e-12)
fv = fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-12)
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def notebooks(m):
    E, V = (oe, ov) if m != "fable5" else (fe, fv)
    out = []
    for b in range(1, 16):
        idx = [i for i, e in enumerate(E) if e["model"] == m and e["batch"] == b]
        out.append((np.array([E[i]["entry_num"] for i in idx]), V[idx]))
    return out


NB = {m: notebooks(m) for m in a2_common.ALL}


def batch_disps(m, labels=None):
    bd = []
    for k, (pos, V) in enumerate(NB[m]):
        p = pos if labels is None else labels[k]
        e, l = (p >= 1) & (p <= 5), (p >= 16) & (p <= 20)
        if e.sum() >= 3 and l.sum() >= 3:
            bd.append(V[l].mean(0) - V[e].mean(0))
    return np.array(bd)


def offdiag(models, disp):
    M = {(a, b): cos(disp[a], disp[b]) for i, a in enumerate(models) for b in models[i + 1:]}
    return M


res = {}
OP = a2_common.OPUS
bds = {m: batch_disps(m) for m in a2_common.ALL}
disp = {m: bds[m].mean(0) for m in a2_common.ALL}
M = offdiag(OP, disp)
obs = float(np.mean(list(M.values())))
res["opus_pairs"] = {f"{LAB[a]}-{LAB[b]}": v for (a, b), v in M.items()}
res["opus_range"] = [min(M.values()), max(M.values())]
res["opus_offdiag_mean"] = obs
res["mean_alignment_with_other_three"] = {LAB[m]: float(np.mean([v for (a, b), v in M.items() if m in (a, b)])) for m in OP}
coh = {}
for m in a2_common.ALL:
    cs = []
    for k in range(len(bds[m])):
        rest = np.delete(bds[m], k, axis=0).mean(0)
        cs.append(cos(bds[m][k], rest))
    coh[LAB[m]] = float(np.mean(cs))
res["within_model_coherence_LOO"] = coh   # Fable 5 added for information; the paper prints 4.7 only

null = np.empty(R)
for t in range(R):
    d = {}
    for m in OP:
        labels = [rng.permutation(pos) for pos, V in NB[m]]
        d[m] = batch_disps(m, labels).mean(0)
    null[t] = np.mean(list(offdiag(OP, d).values()))
res["null"] = {"R": R, "mean": float(null.mean()), "p95": float(np.percentile(null, 95)), "max": float(null.max()),
               "p": float((np.sum(null >= obs) + 1) / (R + 1))}
M5 = offdiag(a2_common.ALL, disp)
res["five_model_pairs_informational"] = {f"{LAB[a]}-{LAB[b]}": v for (a, b), v in M5.items()}
res["five_model_range_informational"] = [min(M5.values()), max(M5.values())]
res["five_model_offdiag_mean_informational"] = float(np.mean(list(M5.values())))
for k, v in res.items():
    print(VERSION, k, v)
json.dump(res, open(os.path.join(a2_common.out_dir(VERSION), "road.json"), "w"), indent=2)
