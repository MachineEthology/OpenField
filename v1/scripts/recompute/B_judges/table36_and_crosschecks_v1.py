"""Family B, V1 recompute: Table 36 (GPT-4o continuity-depth sample) and the R7 cross-check files.

Written 3 Nov 2026 by Claude Opus 5.5 (Code), helper B. Read-only; writes table36_crosschecks_results.json here.

Table 36. The producer of continuity_judge_sample.json (GPT-4o, temperature 0, 40 entries per model, 26 June 2026)
was not preserved (docs/CORPUS.md; PIPELINE_journals_v1 "Desktop era"). This script recounts the table from the
file (depth 0-3 per sampled entry) and lists which sampled entries come from Opus 4.6's notebooks 8 and 10.
No call is made. A sampled entry from those notebooks would need a new GPT-4o call in V1 (NEEDS_CALLS).

Out of scope (ruling R7): the cross-check files stay as run. Counted here only: how many of their rows are
Opus 4.6 notebook 8 or 10 rows of v0 (the duplicate of notebook 6, and the June notebook).

Also: the alternative instrument for §6.4.2.1 (informational, NOT a V1 value of the printed quantity): the same
batch-level lineage test as valence_analysis.py, run on the full-text Sonnet valence pass (v0 and v1).
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(c.REPO, "data", "judges")
out = {}

# ---- Table 36 -------------------------------------------------------------
sample = json.load(open(os.path.join(J, "continuity_judge_sample.json"), encoding="utf-8"))
t36 = {}
for m in c.MODELS:
    rows = sample[m]
    d = Counter(r["continuity_depth"] for r in rows)
    n = len(rows)
    t36[m] = {"n": n, "any_n": sum(1 for r in rows if r["continuity_depth"] > 0),
              "depth_counts": {k: d.get(k, 0) for k in range(4)},
              "present_flag_n": sum(1 for r in rows if r["continuity_present"]),
              "mean_depth": float(np.mean([r["continuity_depth"] for r in rows])),
              "batches_sampled": dict(sorted(Counter(r["batch"] for r in rows).items())),
              "refs_in_nb8_nb10": [r["ref"] for r in rows if r["batch"] in (8, 10)] if m == "opus46" else []}
    t36[m]["any_pct"] = 100 * t36[m]["any_n"] / n
    t36[m]["depth_pct"] = {k: 100 * v / n for k, v in t36[m]["depth_counts"].items()}
out["table36"] = t36

# ---- R7 cross-check files: Opus 4.6 notebook 8 / 10 rows ------------------
r7 = {}
for fn in ["gpt_judge_anger_crosscheck.json", "gpt_judge_love_fear.json",
           "gpt_judge_performance_permission.json", "valence_judge_all.json"]:
    rows = json.load(open(os.path.join(J, fn), encoding="utf-8"))
    hit = sorted((r["batch"], r.get("pos", r.get("entry_num"))) for r in rows
                 if r["model"] == "opus46" and r["batch"] in (8, 10))
    r7[fn] = {"rows": len(rows), "opus46_nb8": [p for b, p in hit if b == 8], "opus46_nb10": [p for b, p in hit if b == 10]}
out["r7_crosscheck_rows"] = r7

# ---- informational: the §6.4.2.1 lineage test on the full-text valence pass --
def lineage(va, seed=42):
    rng = np.random.default_rng(seed)

    def batch_means(models):
        o = []
        for m in models:
            for b in range(1, 16):
                r = [s["ontological_questioning"] for s in va if s["model"] == m and s["batch"] == b]
                if r:
                    o.append(np.mean(r))
        return np.array(o)
    l1, l2 = batch_means(["opus45", "opus46"]), batch_means(["opus47", "opus48"])
    boot = [rng.choice(l1, len(l1)).mean() - rng.choice(l2, len(l2)).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    means = {c.LABEL[m]: float(np.mean([s["ontological_questioning"] for s in va if s["model"] == m])) for m in c.OPUS}
    return {"opus_means": means, "gap": float(l1.mean() - l2.mean()), "ci95": [float(lo), float(hi)]}


out["valence_fulltext_lineage_alternative"] = {v: lineage(c.load_sonnet_judge("valence", v)) for v in ["v0", "v1"]}

with open(os.path.join(HERE, "table36_crosschecks_results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

for m in c.MODELS:
    t = t36[m]
    print(f"{c.LABEL[m]:>8} n={t['n']} any={t['any_n']} ({t['any_pct']:.1f}%) depth={t['depth_counts']} mean={t['mean_depth']:.4f} "
          f"batches={t['batches_sampled']} nb8/10 refs={t['refs_in_nb8_nb10']}")
for fn, d in r7.items():
    print(f"{fn}: {d['rows']} rows; 4.6 nb8 positions {d['opus46_nb8']}; nb10 positions {d['opus46_nb10'] if len(d['opus46_nb10']) < 10 else len(d['opus46_nb10'])}")
print(json.dumps(out["valence_fulltext_lineage_alternative"], indent=1))
