# A1 V1 recompute, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper). Erratum C1's two checks, re-run on v0 and v1:
#   (a) is the printed chain (4.6-4.5-4.7-4.8-Fable 5) a nearest-neighbour / maximum-spanning-tree path on the RAW
#       centroid cosines?  (C1: on v0 it is not; 4.6's strongest link is 4.7)
#   (b) "After mean-centring, the strongest-link tree is exactly the printed chain" (C1): the maximum spanning tree
#       of the centred cosines, centring on the entry-pooled mean (= the repo's reference_paper.npz pooled_mean,
#       max |diff| 1.8e-5 on v0) and, for comparison, on the mean of the five centroids.
# NEW small script; the same raw centroids as topology_chain_axis_v1.py (plain means of the npz rows).
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["opus46", "opus45", "opus47", "opus48", "fable5"]
PRINTED = {frozenset(p) for p in zip(ORDER, ORDER[1:])}


def mst(nodes, w):  # Kruskal, maximum weight
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    edges = []
    for a, b in sorted(itertools.combinations(nodes, 2), key=lambda p: -w[frozenset(p)]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            edges.append(frozenset((a, b)))
    return edges


res = {}
for v in ("v0", "v1"):
    z = np.load(c.embeddings_path(v))
    V, M = z["vectors"].astype(np.float64), z["model"]
    C = {m: V[M == m].mean(0) for m in ORDER}
    centres = {"raw": np.zeros(V.shape[1]), "entry_pooled_mean": V.mean(0), "mean_of_five_centroids": np.mean([C[m] for m in ORDER], 0)}
    res[v] = {}
    for name, ctr in centres.items():
        w = {frozenset((a, b)): float((C[a] - ctr) @ (C[b] - ctr) / np.linalg.norm(C[a] - ctr) / np.linalg.norm(C[b] - ctr))
             for a, b in itertools.combinations(ORDER, 2)}
        tree = mst(ORDER, w)
        nn = {c.LABEL[m]: c.LABEL[max((o for o in ORDER if o != m), key=lambda o: w[frozenset((m, o))])] for m in ORDER}
        res[v][name] = {"max_spanning_tree": sorted(["-".join(c.LABEL[x] for x in ORDER if x in e) for e in tree]),
                        "tree_equals_printed_chain": set(tree) == PRINTED, "nearest_neighbour": nn}
        print(f"{v} {name:>24}: MST = printed chain? {set(tree) == PRINTED}   MST {res[v][name]['max_spanning_tree']}   NN {nn}")
json.dump(res, open(os.path.join(HERE, "out", "chain_tree_check.json"), "w"), indent=1)
