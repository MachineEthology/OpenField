"""[A2 V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code)] of 02_instruments/embedding_road_and_handwriting.py.
Changed: input paths only (loader replaced by a2_common -> v1_common, version = argv v0|v1);
the PNG goes to this folder's out_<version>/. Computation untouched (Opus 4.5-4.8 only, as in June).
"""
import a2_common
"""
embedding_road_and_handwriting.py — Fable 5, 2026-06-10.

Three analyses on the existing journal embedding cache (n=15 batches, 1199
entries, text-embedding-3-large 3072-dim). Predictions registered in-session
BEFORE running:
  A. Displacement alignment ("one road or four?"):
     early->late displacement vectors strongly aligned across models
     (cos > 0.5); 4.6 the LEAST aligned of the four.
  B. Batch handwriting (nearest-centroid batch ID, adjacents excluded):
     4.8 highest accuracy, 4.6 lowest, all > chance (1/15).
  C. Shared-space PCA arrows (one PCA pooled over all models) — figure only.

Loaders match embedding_new_analyses.py conventions exactly.
"""
import json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["opus45", "opus46", "opus47", "opus48"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
rng = np.random.default_rng(42)

# [A2 V1] input paths changed: journals + vectors via a2_common -> v1_common (clean npz rows),
# Opus rows in the June order (batch -> model -> entry). Only the Opus part is used, as in June.
VERSION = a2_common.version_from_argv()
print(f"Loading ({VERSION}) via v1_common...")
_oe, _ov, _fe, _fv = a2_common.load(VERSION)
entries = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"]} for e in _oe]
emb = _ov.copy()
assert len(entries) == emb.shape[0], (len(entries), emb.shape)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
print(f"  {len(entries)} entries, dim {emb.shape[1]}")

model_idx = {m: [i for i, e in enumerate(entries) if e["model"] == m] for m in MODELS}
def sel(m, batch=None, pos=None):
    out = []
    for i in model_idx[m]:
        e = entries[i]
        if batch is not None and e["batch"] != batch: continue
        if pos is not None and not (pos[0] <= e["entry_num"] <= pos[1]): continue
        out.append(i)
    return out

batches_of = {m: sorted(set(entries[i]["batch"] for i in model_idx[m])) for m in MODELS}

# =====================================================================
# A. ONE ROAD OR FOUR? — early->late displacement alignment
# =====================================================================
print("\n=== A. Displacement alignment (early 1-5 -> late 16-20) ===")
disp = {}
batch_disps = {}
for m in MODELS:
    bd = []
    for b in batches_of[m]:
        e_idx, l_idx = sel(m, b, (1, 5)), sel(m, b, (16, 20))
        if len(e_idx) >= 3 and len(l_idx) >= 3:
            bd.append(emb[l_idx].mean(0) - emb[e_idx].mean(0))
    batch_disps[m] = np.array(bd)
    disp[m] = batch_disps[m].mean(0)
    print(f"  {LAB[m]}: {len(bd)} batch displacements, mean |d| = "
          f"{np.linalg.norm(disp[m]):.4f}")

def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("\nCross-model displacement cosine matrix:")
print("      " + "".join(f"{LAB[m]:>8}" for m in MODELS))
M = np.zeros((4, 4))
for i, a in enumerate(MODELS):
    row = ""
    for j, b in enumerate(MODELS):
        M[i, j] = cos(disp[a], disp[b])
        row += f"{M[i,j]:8.3f}"
    print(f"  {LAB[a]} {row}")

mean_align = {m: np.mean([M[i, j] for j in range(4) if j != i])
              for i, m in enumerate(MODELS)}
print("\nMean alignment with the other three:")
for m in MODELS:
    print(f"  {LAB[m]}: {mean_align[m]:.3f}")

# reference: within-model coherence (batch displacement vs own model mean, LOO)
print("\nWithin-model road coherence (batch disp vs own-model mean, leave-one-out):")
for m in MODELS:
    cs = []
    bd = batch_disps[m]
    for k in range(len(bd)):
        rest = np.delete(bd, k, axis=0).mean(0)
        cs.append(cos(bd[k], rest))
    print(f"  {LAB[m]}: {np.mean(cs):.3f}")

# bootstrap CI on pairwise alignments (resample batches)
print("\nBootstrap 95% CI (1000x, resampling batches) on pairwise alignment:")
for i, a in enumerate(MODELS):
    for j in range(i + 1, 4):
        b = MODELS[j]
        vals = []
        for _ in range(1000):
            da = batch_disps[a][rng.integers(0, len(batch_disps[a]), len(batch_disps[a]))].mean(0)
            db = batch_disps[b][rng.integers(0, len(batch_disps[b]), len(batch_disps[b]))].mean(0)
            vals.append(cos(da, db))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {LAB[a]}-{LAB[b]}: {M[i,j]:.3f}  [{lo:.3f}, {hi:.3f}]")

# =====================================================================
# B. BATCH HANDWRITING — nearest-centroid batch ID
# =====================================================================
print("\n=== B. Batch handwriting (nearest-centroid, +/-1 adjacents excluded) ===")

def position_means(m):
    pm = {}
    for p in range(1, 21):
        idx = sel(m, None, (p, p))
        if idx:
            pm[p] = emb[idx].mean(0)
    return pm

for variant in ["raw", "residual"]:
    print(f"\n  [{variant}]")
    for m in MODELS:
        pm = position_means(m) if variant == "residual" else None
        vecs, meta = {}, {}
        for b in batches_of[m]:
            idx = sel(m, b)
            v = emb[idx].copy()
            if variant == "residual":
                for k, i in enumerate(idx):
                    p = entries[i]["entry_num"]
                    if p in pm: v[k] = v[k] - pm[p]
            vecs[b] = v
            meta[b] = [entries[i]["entry_num"] for i in idx]
        correct = total = 0
        for b in batches_of[m]:
            for k in range(len(vecs[b])):
                p = meta[b][k]
                own_mask = [abs(meta[b][q] - p) > 1 for q in range(len(meta[b]))]
                if sum(own_mask) < 3: continue
                own_c = vecs[b][own_mask].mean(0)
                best, best_s = None, -2
                for b2 in batches_of[m]:
                    c = own_c if b2 == b else vecs[b2].mean(0)
                    s = cos(vecs[b][k], c)
                    if s > best_s: best_s, best = s, b2
                correct += (best == b); total += 1
        nb = len(batches_of[m])
        print(f"    {LAB[m]}: {correct/total:.3f}  (chance {1/nb:.3f}, n={total})")

# =====================================================================
# C. SHARED-SPACE PCA FIGURE
# =====================================================================
print("\n=== C. Shared-space PCA figure ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = emb - emb.mean(0)
pca = PCA(n_components=2, random_state=0).fit(X)
P = pca.transform(X)
print(f"  explained variance: {pca.explained_variance_ratio_[:2]}")

COL = {"opus45": "#1f77b4", "opus46": "#d95f02", "opus47": "#2a7e43", "opus48": "#7b2d8b"}
fig, ax = plt.subplots(figsize=(11, 8))
for m in MODELS:
    e_idx, l_idx = sel(m, None, (1, 5)), sel(m, None, (16, 20))
    ax.scatter(P[e_idx, 0], P[e_idx, 1], s=14, alpha=0.25, color=COL[m])
    ax.scatter(P[l_idx, 0], P[l_idx, 1], s=22, alpha=0.55, color=COL[m], marker="D")
    ec, lc = P[e_idx].mean(0), P[l_idx].mean(0)
    ax.annotate("", xy=lc, xytext=ec,
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=COL[m]))
    ax.scatter(*ec, s=130, facecolors="none", edgecolors=COL[m], lw=2.2, zorder=5)
    ax.scatter(*lc, s=150, color=COL[m], marker="D", zorder=5)
    ax.text(lc[0], lc[1] + 0.012, LAB[m], color=COL[m], fontsize=13,
            fontweight="bold", ha="center")
ax.set_xlabel(f"shared PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"shared PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Early (o, light) vs late (◆) — ONE shared PCA, all four models comparable\n"
             "circles = early centroid (1-5), diamonds = late centroid (16-20)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(a2_common.out_dir(VERSION), "shared_space_arrows.png"), dpi=150)
print("  saved shared_space_arrows.png")
