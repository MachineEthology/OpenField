# COPY of the repo's code/reproduce/topology_chain_axis.py (Desktop/the-open-field), for the V1 recompute (A1).
# Changed, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper):
#   - INPUT: the vectors are read from v1_common.embeddings_path(version), version = "v0" | "v1" (argv[1]),
#     instead of the repo's data/embeddings/journal_embeddings_paper_rows.npz (v0 is that same file).
#   - OUTPUT ONLY (computation untouched): the full-precision values are also written to
#     out/topology_chain_axis_<version>.json, plus two read-offs of the same matrix used by the ledger
#     (the sorted pair list is already printed by the original; nearest Opus to Fable 5 is read from it).
"""Section 5.2.2.3 from the packaged paper-era vectors: the centroid chain, the 4.6-Fable 5 pair, the PCA
of the five centroids (the 'dominant axis' = PC1), and the five self-axis drifts.

Space: raw text-embedding-3-large journal vectors (3,072-d), plain means, NOT mean-centred, no length
control (this is what the June scripts did, although section 3.1 says geometry is mean-centred).
Rows: 1,488 entries exactly as analysed (Opus 4.6 'Batch 8' is the duplicate of Batch 6; Opus 4.7
Batch 1 has 19 entries). Self-axis drift = mean projection at entry 20 minus at entry 1.
"""
import itertools
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

LABEL = c.LABEL
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

z = np.load(c.embeddings_path(VERSION))
V, model, pos = z["vectors"].astype(np.float64), z["model"], z["pos"]
ORDER = ["opus46", "opus45", "opus47", "opus48", "fable5"]
C = {m: V[model == m].mean(0) for m in ORDER}


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


print(f"== version {VERSION}: {len(V)} rows")
print("chain as printed:  " + "  ".join(f"{LABEL[a]} ({cos(C[a], C[b]):.3f})" for a, b in zip(ORDER, ORDER[1:])) + f"  {LABEL[ORDER[-1]]}")
print("all pairs, highest first:")
for a, b in sorted(itertools.combinations(ORDER, 2), key=lambda p: -cos(C[p[0]], C[p[1]])):
    print(f"   {LABEL[a]:>7} - {LABEL[b]:<7} {cos(C[a], C[b]):.4f}")

pca = PCA(n_components=2).fit(np.array([C[m] for m in ORDER]))
pc = pca.transform(np.array([C[m] for m in ORDER]))
sign = -1.0 if pc[0, 0] > 0 else 1.0          # orient: 4.6 at the 'thinking about thinking' (negative) end
print(f"\nPCA of the five centroids: PC1 {100 * pca.explained_variance_ratio_[0]:.1f}% + PC2 {100 * pca.explained_variance_ratio_[1]:.1f}% "
      f"= {100 * pca.explained_variance_ratio_.sum():.1f}%")
print("PC1 coordinates (the dominant axis; never printed in the paper): "
      + ", ".join(f"{LABEL[m]} {sign * pc[i, 0]:+.3f}" for i, m in enumerate(ORDER)))

print("\nself-axis drift (E20 - E1):")
drift = {}
for m in ORDER:
    others = V[model != m].mean(0)
    axis = C[m] - others
    axis /= np.linalg.norm(axis)
    proj = V[model == m] @ axis
    p = pos[model == m]
    drift[m] = float(proj[p == 20].mean() - proj[p == 1].mean())
    print(f"   {LABEL[m]:>7}: {proj[p == 20].mean() - proj[p == 1].mean():+.3f}")

# ---- output only: full precision for the ledger ----
pairs = {f"{LABEL[a]}-{LABEL[b]}": cos(C[a], C[b]) for a, b in itertools.combinations(ORDER, 2)}
pc1 = {LABEL[m]: float(sign * pc[i, 0]) for i, m in enumerate(ORDER)}
res = {
    "version": VERSION, "n_rows": int(len(V)), "n_per_model": {LABEL[m]: int((model == m).sum()) for m in ORDER},
    "chain_neighbour_cosines_in_printed_order": {f"{LABEL[a]}-{LABEL[b]}": cos(C[a], C[b]) for a, b in zip(ORDER, ORDER[1:])},
    "all_pairs": pairs,
    "pairs_sorted_high_to_low": sorted(pairs.items(), key=lambda kv: -kv[1]),
    "pca_evr": [float(x) for x in pca.explained_variance_ratio_],
    "pca_evr_sum": float(pca.explained_variance_ratio_.sum()),
    "pc1_coords_oriented_46_negative": pc1,
    "pc2_coords": {LABEL[m]: float(pc[i, 1]) for i, m in enumerate(ORDER)},
    "pc1_order_low_to_high": [k for k, _ in sorted(pc1.items(), key=lambda kv: kv[1])],
    "self_axis_drift_e20_minus_e1": {LABEL[m]: drift[m] for m in ORDER},
    "fable5_cosine_to_each_opus": {LABEL[m]: cos(C["fable5"], C[m]) for m in ORDER if m != "fable5"},
    "strongest_link_per_model": {LABEL[m]: max(((LABEL[o], cos(C[m], C[o])) for o in ORDER if o != m), key=lambda t: t[1]) for m in ORDER},
}
json.dump(res, open(os.path.join(OUT, f"topology_chain_axis_{VERSION}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nwritten out/topology_chain_axis_{VERSION}.json")
