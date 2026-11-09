"""Full-precision companion to carryover_channels_v1.py (3 Nov 2026, Claude Opus 5.5 (Code), helper A2).
Imports the copy's own word lists and functions (word_density, lag1_autocorr) and repeats its loop to
write the Table 14 means, SEs and the mean densities at full precision. No computation of its own.
Output: out_<version>/table14_channels.json. Also counts V1 tokens per channel list (the lists are
hand-written, not built from the v0 corpus, so nothing can fall "outside" them).
"""
import json, os, sys
import numpy as np
import a2_common
import carryover_channels_v1 as cc

VERSION = a2_common.version_from_argv()
_oe, _ov, _fe, _fv = a2_common.load(VERSION)
notebooks = {m: {} for m in cc.ALL_MODELS}
for e in _oe + _fe:
    notebooks[e["model"]].setdefault(e["batch"], []).append(e["text"])
out = {}
for m in cc.ALL_MODELS:
    acs = {"tonal": [], "objectal": [], "conceptual": []}
    for bn, entries in notebooks[m].items():
        if len(entries) < 5:
            continue
        for ch, ws in (("tonal", cc.TONAL_WORDS), ("objectal", cc.OBJECTAL_WORDS), ("conceptual", cc.CONCEPTUAL_WORDS)):
            a = cc.lag1_autocorr([cc.word_density(e, ws) for e in entries])
            if not np.isnan(a):
                acs[ch].append(a)
    allt = [t for es in notebooks[m].values() for t in es]
    out[m] = {ch: {"mean_lag1_r": float(np.mean(v)), "se": float(np.std(v) / np.sqrt(len(v))), "n": len(v)} for ch, v in acs.items()}
    out[m]["density_pct"] = {ch: float(np.mean([cc.word_density(e, ws) for e in allt]))
                             for ch, ws in (("tonal", cc.TONAL_WORDS), ("objectal", cc.OBJECTAL_WORDS), ("conceptual", cc.CONCEPTUAL_WORDS))}
json.dump(out, open(os.path.join(a2_common.out_dir(VERSION), "table14_channels.json"), "w"), indent=2)
for m, r in out.items():
    print(VERSION, m, {ch: round(r[ch]["mean_lag1_r"], 5) for ch in ("tonal", "objectal", "conceptual")}, r["density_pct"])
