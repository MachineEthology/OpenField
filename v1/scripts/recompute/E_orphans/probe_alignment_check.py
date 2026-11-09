"""Sanity check of the probe-cache alignment assumed by e_common.load_opus_probes() (helper E, 3 Nov 2026).

The June probe cache (03_experiments/probe_embeddings_cache.json) has 800 vectors and no row keys.
If rows follow the June load order (per model: negation 100, affirmative June 75, affirmative May 25),
then (a) leave-one-out nearest-centroid by model and by condition should be far above chance, and
(b) the per-row word count should correlate with the vector's position along the within-block length axis
the same way under the assumed order and worse under a shifted order. Read-only; no calls.
"""
import numpy as np
import e_common as e

P, E = e.load_opus_probes()
En = E / np.linalg.norm(E, axis=1, keepdims=True)
models = np.array([p["model"] for p in P])
conds = np.array([p["condition"] for p in P])
wc = np.array([len(p["response"].split()) for p in P], float)


def loo_acc(labels, X):
    labs = sorted(set(labels))
    S = {l: X[labels == l].sum(0) for l in labs}
    N = {l: (labels == l).sum() for l in labs}
    ok = 0
    for i in range(len(X)):
        best, bl = -9, None
        for l in labs:
            cvec = (S[l] - X[i]) / (N[l] - 1) if labels[i] == l else S[l] / N[l]
            s = X[i] @ cvec / np.linalg.norm(cvec)
            if s > best:
                best, bl = s, l
        ok += bl == labels[i]
    return ok / len(X)


def len_corr(X, w):
    # correlation between log word count and the first principal axis, per model block, mean |r|
    rs = []
    for m in e.OPUS:
        idx = models == m
        Xc = X[idx] - X[idx].mean(0)
        u = np.linalg.svd(Xc, full_matrices=False)[2][0]
        rs.append(abs(np.corrcoef(Xc @ u, np.log(w[idx] + 1))[0, 1]))
    return float(np.mean(rs))


print(f"LOO nearest-centroid accuracy, model (chance 0.25): {loo_acc(models, En):.3f}")
for m in e.OPUS:
    idx = models == m
    print(f"  {m}: LOO accuracy, condition within model (chance 0.5): {loo_acc(conds[idx], En[idx]):.3f}")
print(f"|r| log-words vs PC1 within model, assumed order: {len_corr(En, wc):.3f}")
rng = np.random.default_rng(0)
wc_shuf = wc.copy()
for m in e.OPUS:
    idx = np.where(models == m)[0]
    wc_shuf[idx] = wc[rng.permutation(idx)]
print(f"|r| log-words vs PC1 within model, word counts shuffled within model: {len_corr(En, wc_shuf):.3f}")
# direct per-row check: correlation of log word count with the vector's projection on the model's
# own length direction (fit on half the rows, tested on the other half)
r_list = []
for m in e.OPUS:
    idx = np.where(models == m)[0]
    a, b = idx[::2], idx[1::2]
    Xa = En[a] - En[a].mean(0)
    wa = np.log(wc[a] + 1) - np.log(wc[a] + 1).mean()
    d = Xa.T @ wa
    d /= np.linalg.norm(d)
    r_list.append(np.corrcoef((En[b] - En[a].mean(0)) @ d, np.log(wc[b] + 1))[0, 1])
print("held-out r(log words, length direction) per model:", [round(float(r), 3) for r in r_list])
