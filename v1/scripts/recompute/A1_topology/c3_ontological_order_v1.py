# A1 V1 recompute, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper). Erratum C3 / plan L1.13.
# NEW small script (no producer survives for the October check, which ERRATA C3 and ANSWERS §1 report in words):
# the per-model mean of `ontological_questioning` in the Sonnet 4.6 full-text valence pass
# (v1_common.load_sonnet_judge("valence", version)), plain mean over the rows present (no rows dropped, no
# weighting), and the order it gives, set beside the PC1 order from out/topology_chain_axis_<version>.json.
# Note: these are NOT §6.4.2.1's printed Opus means (4.6 +0.86, 4.5 +0.80, 4.7 +0.69, 4.8 +0.69); the full-text
# pass gives 0.78 / 0.68 / 0.55 / 0.54 on v0, so §6.4.2.1 rests on another pass (not A1's family; left to B).
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for v in ("v0", "v1"):
    rows = c.load_sonnet_judge("valence", v)
    by = {m: [r["ontological_questioning"] for r in rows if r["model"] == m and r.get("ontological_questioning") is not None]
          for m in c.MODELS}
    nnull = sum(1 for r in rows if r.get("ontological_questioning") is None)
    means = {c.LABEL[m]: float(np.mean(x)) for m, x in by.items()}
    ns = {c.LABEL[m]: len(x) for m, x in by.items()}
    judge_order = [k for k, _ in sorted(means.items(), key=lambda kv: -kv[1])]  # most questioning first
    t = json.load(open(os.path.join(HERE, "out", f"topology_chain_axis_{v}.json"), encoding="utf-8"))
    pc1_order = t["pc1_order_low_to_high"]                                        # 'thinking about thinking' end first
    res[v] = {"n_rows": len(rows), "n_null": nnull, "n_per_model": ns, "means": means,
              "judge_order_high_to_low": judge_order, "pc1_order_low_to_high": pc1_order,
              "same_order": judge_order == pc1_order}
    print(v, ns, "null", nnull)
    print("   means:", {k: round(x, 4) for k, x in means.items()})
    print("   judge order (high -> low):", " > ".join(judge_order))
    print("   PC1 order (introspective -> world-facing):", " < ".join(pc1_order), "| same:", judge_order == pc1_order)
json.dump(res, open(os.path.join(HERE, "out", "c3_ontological_order.json"), "w", encoding="utf-8"), indent=1)
