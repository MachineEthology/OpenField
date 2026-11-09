"""REBUILD (3 Nov 2026, Claude Opus 5.5 (Code), helper A2): the straightness permutation null of §5.2.2.3 /
Figure 5 panel C, printed "0.053 [95% CI: 0.039, 0.068]" ("we shuffle entry order within each batch 2,000 times").

The producer of review_desktop/trajectory_permutation_results.json ("Monte Carlo permutation (entry-order
shuffle, 2000 perms)") was not found: searched 02_instruments, 03_experiments, 04_analysis, 06_paper, the
repository code/, Desktop (grep "trajectory_permutation", "straightness"). That June JSON is used here only as
a pointer to the summary's shape (per-model null mean + CI of the per-permutation mean; a "grand_null").

Method, as the paper states it, on paper_fig_portrait_composite.py's own straightness (panel D):
straightness of a notebook = ||e_last - e_first|| / sum ||e_k+1 - e_k|| on unit-normed vectors, entries in
file order (Fable 5: text entries only); per model, mean over its 15 notebooks. Null: shuffle the order of
the entries within every notebook, recompute; 2,000 permutations, numpy default_rng(42).
Summaries reported:
  per_model: null mean, 2.5/97.5 percentiles of the per-permutation model means;
  grand_pooled_notebooks: mean and 2.5/97.5 percentiles over every permuted NOTEBOOK value, pooled over the
    five models (the definition whose width matches the printed [0.039, 0.068]);
  grand_model_means: the same over the per-permutation MODEL means pooled (shown for comparison).
Inputs via a2_common -> v1_common (version = argv v0|v1). Output out_<version>/straightness_null.json.
"""
import json, os
import numpy as np
import a2_common

VERSION = a2_common.version_from_argv()
N_PERM = 2000
RNG = np.random.default_rng(42)
oe, ov, fe, fv = a2_common.load(VERSION)
ov = ov / (np.linalg.norm(ov, axis=1, keepdims=True) + 1e-12)
fv = fv / (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-12)


def straight(pts):
    net = np.linalg.norm(pts[-1] - pts[0])
    path = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return net / path


notebooks = {}
for m in a2_common.ALL:
    E, V = (oe, ov) if m != "fable5" else (fe, fv)
    nbs = []
    for b in range(1, 16):
        idx = sorted([i for i, e in enumerate(E) if e["model"] == m and e["batch"] == b], key=lambda i: E[i]["entry_num"])
        if len(idx) >= 5:
            nbs.append(V[idx])
    notebooks[m] = nbs

res = {"per_model": {}}
pooled_nb, pooled_means = [], []
for m in a2_common.ALL:
    obs = float(np.mean([straight(p) for p in notebooks[m]]))
    null_means = np.empty(N_PERM)
    for t in range(N_PERM):
        vals = []
        for p in notebooks[m]:
            vals.append(straight(p[RNG.permutation(len(p))]))
        pooled_nb.extend(vals)
        null_means[t] = np.mean(vals)
    pooled_means.extend(null_means.tolist())
    res["per_model"][m] = {"observed": obs, "null_mean": float(null_means.mean()),
                           "null_ci95": [float(x) for x in np.percentile(null_means, [2.5, 97.5])],
                           "p_null_ge_observed": float((np.sum(null_means >= obs) + 1) / (N_PERM + 1)),
                           "n_notebooks": len(notebooks[m])}
    print(VERSION, m, {k: v for k, v in res["per_model"][m].items()})
pooled_nb = np.array(pooled_nb); pooled_means = np.array(pooled_means)
res["grand_pooled_notebooks"] = {"mean": float(pooled_nb.mean()),
                                 "ci95": [float(x) for x in np.percentile(pooled_nb, [2.5, 97.5])]}
res["grand_model_means"] = {"mean": float(pooled_means.mean()),
                            "ci95": [float(x) for x in np.percentile(pooled_means, [2.5, 97.5])]}
print(VERSION, "grand (pooled notebooks):", res["grand_pooled_notebooks"])
print(VERSION, "grand (model means):", res["grand_model_means"])
json.dump(res, open(os.path.join(a2_common.out_dir(VERSION), "straightness_null.json"), "w"), indent=2)
