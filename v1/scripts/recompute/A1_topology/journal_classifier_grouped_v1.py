# COPY of the repo's code/reproduce/journal_classifier_grouped.py (Desktop/the-open-field), for the V1 recompute (A1).
# Changed, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper):
#   - INPUT ONLY: the journals are globbed from v1_common.corpus_dir(version) instead of the repo's data/journals
#     (v0 is that same folder); --version v0|v1. The notebook key is still the batch number in the FILE NAME, as
#     in the recorded run: in v0 the duplicate 4.6 file in Batch_8 (named ...Batch6.json) collapses onto Batch 6
#     (1,179 entries, 14 notebooks for 4.6); in the V1 corpus that file is named ...Batch8.json, so 4.6 has 15.
#   - OUTPUT ONLY (computation untouched): per-model recall of the grouped predictions is printed, the
#     permutation seed/count and every result is written to out/journal_classifier_<version>_<perms>.json.
"""Journal model-identity classifier, notebook-grouped (section 5.2.2.1: 'approximately 0.91 balanced
accuracy against a notebook-level permutation null (p < .001)'). FOUR-way, Opus 4.5-4.8 only.

This re-runs the recovered 18 June 2026 code (code/recovered/journal_classifier_notebook_grouped_2026-06-18.py)
on this repository's files: 33 'transparent' features (not the 31 of Appendix E.5.2), StandardScaler +
LogisticRegression(class_weight='balanced'), GroupKFold(5) by notebook. Notebooks were keyed by the batch
number in the FILE NAME, so the duplicate Opus 4.6 file in Batch_8 (named ...Batch6.json) collapses onto
Batch 6: 1,179 entries, 4.6 = 14 notebooks. The original null used 30 permutations; pass --perms to run more.
Usage: python journal_classifier_grouped.py [--perms 10000]
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import json as _json
import time as _time
import v1_common as c  # noqa: E402
from sklearn.metrics import recall_score, confusion_matrix  # output only

warnings.filterwarnings("ignore")
MODELS = {"45": "opus_4_5", "46": "opus_4_6", "47": "opus_4_7", "48": "opus_4_8"}
MARK = ["i don't", "i notice", "genuinely", "uncertain", "curious", "honest", "actually", "strange", "space",
        "quiet", "warm", "perform", "useful", "settling", "thank", "small"]


def feats(t):  # verbatim from the recovered run
    tl = t.lower(); w = re.findall(r"[a-z']+", tl); n = len(w) or 1; ty = set(w)
    s = [x for x in re.split(r'[.!?]+', t) if x.strip()]
    f = [len(w), len(ty) / n, sum(1 for x in ty if w.count(x) == 1) / n, n / (len(s) or 1), len(s),
         t.count('?'), t.count('!'), t.count('*'), t.count('—'), t.count('...') + t.count('…'),
         1.0 if re.search(r'(^|\n)#', t) else 0.0, len(re.findall(r"\bi\b", tl)) / n, len(re.findall(r"\bmy\b", tl)) / n,
         1.0 if 'strange' in tl[:60] else 0.0, 1.0 if tl.lstrip().startswith('thank') else 0.0,
         1.0 if 'settling' in tl[:40] else 0.0, len(re.findall(r'\b(he|previous entr|the \w+ one)', tl)) / n]
    f += [tl.count(mk) / n for mk in MARK]
    return f


def load(version):
    import json
    rows = []
    for m, lab in MODELS.items():
        bybatch = {}
        for f in sorted(glob.glob(os.path.join(c.corpus_dir(version), "Batch_*", f"schrodingers_box_opus{m}_*.json"))):
            b = int(re.search(r'Batch_?(\d+)', os.path.basename(f)).group(1))
            bybatch[b] = f
        for b, f in sorted(bybatch.items()):
            for pos, e in enumerate(json.load(open(f, encoding="utf-8")), 1):
                t = (e.get('text') or '').strip()
                if t:
                    rows.append((lab, f"{lab}__b{b:02d}", pos, t))
    return rows


def pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight='balanced'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=30); ap.add_argument("--version", default="v0"); a = ap.parse_args()
    t0 = _time.time()
    rows = load(a.version)
    print("version", a.version, "| corpus", c.corpus_dir(a.version))
    X = np.array([feats(r[3]) for r in rows]); y = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows])
    print("entries", len(rows), "| notebooks per model", {v: len(set(g[y == v])) for v in MODELS.values()})
    leaky = balanced_accuracy_score(y, cross_val_predict(pipe(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)))
    pred = cross_val_predict(pipe(), X, y, cv=GroupKFold(5), groups=g)  # output only: same call, kept for recall
    obs = balanced_accuracy_score(y, cross_val_predict(pipe(), X, y, cv=GroupKFold(5), groups=g))
    print(f"entry-level StratifiedKFold (leaky): {leaky:.3f}   notebook-grouped GroupKFold: {obs:.3f}   (paper ~0.91)")
    labs = sorted(set(y)); rec = recall_score(y, pred, labels=labs, average=None)
    print("per-model recall (grouped):", {l: round(float(r_), 4) for l, r_ in zip(labs, rec)})
    print("confusion (rows true, cols pred)", labs); print(confusion_matrix(y, pred, labels=labs))
    nb2y = {nb: y[g == nb][0] for nb in set(g)}; nbs = np.array(sorted(nb2y))
    rng = np.random.default_rng(1); perm = []
    for _ in range(a.perms):
        mp = dict(zip(nbs, rng.permutation([nb2y[nb] for nb in nbs])))
        yp = np.array([mp[x] for x in g])
        perm.append(balanced_accuracy_score(yp, cross_val_predict(pipe(), X, yp, cv=GroupKFold(5), groups=g)))
    perm = np.array(perm)
    p = (1 + (perm >= obs).sum()) / (1 + len(perm))
    print(f"notebook-level permutation null ({a.perms} permutations): mean {perm.mean():.3f}, max {perm.max():.3f}; "
          f"p = {p:.4f} (the smallest p attainable with {a.perms} permutations is {1 / (1 + a.perms):.4f})")
    out = {"version": a.version, "n_entries": len(rows), "notebooks_per_model": {v: len(set(g[y == v])) for v in MODELS.values()},
           "leaky_stratified_bacc": float(leaky), "grouped_bacc": float(obs),
           "per_model_recall_grouped": {l: float(r_) for l, r_ in zip(labs, rec)},
           "confusion_labels": labs, "confusion": confusion_matrix(y, pred, labels=labs).tolist(),
           "perms": a.perms, "perm_rng": "np.random.default_rng(1)", "null_mean": float(perm.mean()) if len(perm) else None,
           "null_max": float(perm.max()) if len(perm) else None, "n_null_ge_obs": int((perm >= obs).sum()), "p": float(p),
           "p_resolution": 1 / (1 + a.perms), "seconds": round(_time.time() - t0, 1)}
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(od, exist_ok=True)
    _json.dump(out, open(os.path.join(od, f"journal_classifier_{a.version}_{a.perms}.json"), "w"), indent=1)
    print("seconds", out["seconds"])


if __name__ == "__main__":
    main()
