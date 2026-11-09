"""A2 diagnostics for Figure 13 (3 Nov 2026, Claude Opus 5.5 (Code)). Not a producer.
Re-implements handwriting_v1.py's nearest-centroid rule (identical formulas) to report, per model and
variant: per-notebook hits, where misses went, and the margin of each decision. Used to (a) explain the
4.5 raw 0.527 vs printed 0.523 and (b) measure what 4.6's duplicated notebook (B8 = byte copy of B6)
did to 4.6's v0 accuracy. Writes figure13_diagnostics.json (both versions) in this folder.
"""
import json, os
import numpy as np
import a2_common

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run(version, model, variant):
    oe, ov, fe, fv = a2_common.load(version)
    E, V = (oe, ov) if model != "fable5" else (fe, fv)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    m_idx = [i for i, e in enumerate(E) if e["model"] == model]
    batches = sorted(set(E[i]["batch"] for i in m_idx))
    pos_means = {}
    if variant == "residual":
        for p in range(1, 21):
            p_idx = [i for i in m_idx if E[i]["entry_num"] == p]
            if p_idx:
                pos_means[p] = V[p_idx].mean(0)
    vecs, meta = {}, {}
    for b in batches:
        b_idx = [i for i in m_idx if E[i]["batch"] == b]
        v = V[b_idx].copy()
        if variant == "residual":
            for k, i in enumerate(b_idx):
                v[k] = v[k] - pos_means[E[i]["entry_num"]]
        vecs[b] = v; meta[b] = [E[i]["entry_num"] for i in b_idx]
    per_batch, decisions = {}, []
    for b in batches:
        hit = n = 0
        for k in range(len(vecs[b])):
            p = meta[b][k]
            own_mask = [abs(meta[b][q] - p) > 1 for q in range(len(meta[b]))]
            if sum(own_mask) < 3:
                continue
            own_c = vecs[b][own_mask].mean(0)
            scores = {b2: cos(vecs[b][k], own_c if b2 == b else vecs[b2].mean(0)) for b2 in batches}
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            best = ranked[0][0]
            decisions.append({"batch": b, "pos": p, "assigned": int(best), "correct": best == b,
                              "margin_own_minus_best_other": scores[b] - max(s for b2, s in scores.items() if b2 != b)})
            hit += best == b; n += 1
        per_batch[int(b)] = [hit, n]
    tot = sum(h for h, n in per_batch.values()); N = sum(n for h, n in per_batch.values())
    return {"accuracy": tot / N, "hits": tot, "n": N, "per_batch": per_batch, "decisions": decisions}

out = {}
for version in ("v0", "v1"):
    out[version] = {}
    for model in a2_common.ALL:
        for variant in ("raw", "residual"):
            out[version][f"{model}_{variant}"] = run(version, model, variant)

# 4.5 raw: the closest calls
for version in ("v0", "v1"):
    d = out[version]["opus45_raw"]["decisions"]
    close = sorted(d, key=lambda x: abs(x["margin_own_minus_best_other"]))[:3]
    print(version, "4.5 raw", out[version]["opus45_raw"]["hits"], "/", out[version]["opus45_raw"]["n"],
          "closest calls:", [(x["batch"], x["pos"], round(x["margin_own_minus_best_other"], 6)) for x in close])
# 4.6 duplicate
for version in ("v0", "v1"):
    for variant in ("raw", "residual"):
        r = out[version][f"opus46_{variant}"]
        pb = r["per_batch"]
        to_twin = sum(1 for x in r["decisions"] if (x["batch"], x["assigned"]) in ((6, 8), (8, 6)))
        others = [pb[b] for b in pb if b not in (6, 8)]
        print(version, "4.6", variant, "acc %.4f" % r["accuracy"], "B6", pb[6], "B8", pb[8],
              "B10", pb[10], "misses sent to the twin (6<->8):", to_twin,
              "other 13 notebooks: %d/%d" % (sum(h for h, n in others), sum(n for h, n in others)))
for version in ("v0", "v1"):
    for k in out[version]:
        for x in out[version][k]["decisions"]:
            x["correct"] = bool(x["correct"])
json.dump(out, open(os.path.join(a2_common.HERE, "figure13_diagnostics.json"), "w"), indent=1)
