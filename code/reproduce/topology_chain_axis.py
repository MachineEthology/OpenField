"""Section 5.2.2.3 from the packaged paper-era vectors: the centroid chain, the 4.6-Fable 5 pair, the PCA
of the five centroids (the 'dominant axis' = PC1), and the five self-axis drifts.

Space: raw text-embedding-3-large journal vectors (3,072-d), plain means, NOT mean-centred, no length
control (this is what the June scripts did, although section 3.1 says geometry is mean-centred).
Rows: 1,488 entries exactly as analysed (Opus 4.6 'Batch 8' is the duplicate of Batch 6; Opus 4.7
Batch 1 has 19 entries). Self-axis drift = mean projection at entry 20 minus at entry 1.
"""
import itertools
import os

import numpy as np
from sklearn.decomposition import PCA

from common import DATA, LABEL

z = np.load(os.path.join(DATA, "embeddings", "journal_embeddings_paper_rows.npz"))
V, model, pos = z["vectors"].astype(np.float64), z["model"], z["pos"]
ORDER = ["opus46", "opus45", "opus47", "opus48", "fable5"]
C = {m: V[model == m].mean(0) for m in ORDER}


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


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
for m in ORDER:
    others = V[model != m].mean(0)
    axis = C[m] - others
    axis /= np.linalg.norm(axis)
    proj = V[model == m] @ axis
    p = pos[model == m]
    print(f"   {LABEL[m]:>7}: {proj[p == 20].mean() - proj[p == 1].mean():+.3f}")
