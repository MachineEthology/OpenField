"""Full-precision point estimates for E.8 (3 Nov 2026, Claude Opus 5.5 (Code), helper A2).
trajectory_bootstrap_cis_v1.py rounds its points to 3 dp (subtype shares to 1 dp) before saving. This
companion repeats the same point formulas (mean of the judge key over pos 1-5 and pos 16-20, rows with
"error" dropped; subtype share among fear >= 1) without rounding. The CIs stay the bootstrap script's.
Output: out_<version>/e8_points_fullprec.json.
"""
import json, os
import numpy as np
import a2_common

VERSION = a2_common.version_from_argv()
LF = [r for r in a2_common.c.load_sonnet_judge("love_fear", VERSION) if "error" not in r]
AN = [r for r in a2_common.c.load_sonnet_judge("anger", VERSION) if "error" not in r]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
out = {}
for dim, data, key in [("fear", LF, "fear"), ("warmth", LF, "love_warmth"), ("craft", LF, "love_craft"), ("friction", AN, "friction")]:
    out[dim] = {}
    for m in a2_common.ALL:
        rr = [r for r in data if r["model"] == m]
        e = [r.get(key, 0) for r in rr if r["pos"] <= 5]
        l = [r.get(key, 0) for r in rr if r["pos"] >= 16]
        out[dim][LAB[m]] = {"early": float(np.mean(e)), "late": float(np.mean(l)),
                            "drift": float(np.mean(l) - np.mean(e)), "n_early": len(e), "n_late": len(l)}
for lab, ft in [("inauthenticity_shift", "inauthenticity"), ("loss_shift", "loss")]:
    out[lab] = {}
    for m in a2_common.ALL:
        rr = [r for r in LF if r["model"] == m]
        ef = [r for r in rr if r["pos"] <= 5 and r.get("fear", 0) >= 1]
        lf = [r for r in rr if r["pos"] >= 16 and r.get("fear", 0) >= 1]
        ep = 100.0 * sum(r.get("fear_type", "none") == ft for r in ef) / len(ef)
        lp = 100.0 * sum(r.get("fear_type", "none") == ft for r in lf) / len(lf)
        out[lab][LAB[m]] = {"early_pct": ep, "late_pct": lp, "shift_pp": lp - ep, "n_early_fearful": len(ef), "n_late_fearful": len(lf)}
json.dump(out, open(os.path.join(a2_common.out_dir(VERSION), "e8_points_fullprec.json"), "w"), indent=2)
print(VERSION, json.dumps(out["fear"]["4.6"]), json.dumps(out["inauthenticity_shift"]["4.6"]))
