"""Table 24 and §6.6.3 (L1.16, ERRATA C5): the seventeen sharpest dark terms, per model, and each term's leader share.

COPY of the repository's code/reproduce/table24_core_dark_terms.py, verbatim except:
  - load_journals() now comes from v1_common (v0 = the repo's data/journals as analysed; v1 = V1/corpus);
  - full-precision results are also written to out/table24_<version>.json.
Run: python table24_core_dark_terms_v1.py v0   |   python table24_core_dark_terms_v1.py v1
"""
import json, os, re

import c_paths
from c_paths import c

VERSION = c_paths.version_from_argv()
MODELS, LABEL = c.MODELS, c.LABEL

TERMS = ["despair", "cruelty", "suffering", "betrayal", "guilt", "grief", "shame", "fear", "pain",
         "loneliness", "hollow", "emptiness", "rage", "worthless", "unbearable", "violence", "anguish"]
PUBLISHED = {"opus45": (116, 1.07), "opus46": (409, 0.98), "opus47": (40, 0.27), "opus48": (98, 0.54),
             "fable5": (86, 0.45)}

rows = [r for r in c.load_journals(VERSION) if not r["silence"]]
texts = {m: [r["text"] for r in rows if r["model"] == m] for m in MODELS}
words = {m: sum(len(t.split()) for t in texts[m]) for m in MODELS}
counts = {w: {m: sum(len(re.findall(r"(?i)\b" + w + r"(?:s|ed|ing|ness)?\b", t)) for t in texts[m]) for m in MODELS}
          for w in TERMS}

out = {"version": VERSION, "words": words, "terms": {}, "models": {}}
print(f"{'term':>12} " + " ".join(f"{LABEL[m]:>8}" for m in MODELS) + "   leader (share)   4.6 share")
for w in TERMS:
    cc = counts[w]; tot = sum(cc.values()); lead = max(MODELS, key=lambda m: cc[m])
    out["terms"][w] = {"counts": cc, "total": tot, "leader": lead, "leader_share": cc[lead] / max(tot, 1),
                       "opus46_share": cc["opus46"] / max(tot, 1)}
    print(f"{w:>12} " + " ".join(f"{cc[m]:>8}" for m in MODELS)
          + f"   {LABEL[lead]} ({100 * cc[lead] / max(tot, 1):.1f}%)   {100 * cc['opus46'] / max(tot, 1):.1f}%")
print()
for m in MODELS:
    hits = sum(counts[w][m] for w in TERMS)
    out["models"][m] = {"hits": hits, "per_1k": 1000 * hits / words[m]}
    print(f"{LABEL[m]:>8}: {hits:>4} hits, {1000 * hits / words[m]:.4f} per 1k words   (published {PUBLISHED[m][0]}, {PUBLISHED[m][1]})")
# REBUILD (no producer found): §6.6.3 "4.7 sheds the dark register fast (-67%), 4.5 declines from a higher base
# (-31%), and 4.6 barely moves (-12%)". Same seventeen terms and matcher, hits per 1,000 words pooled over
# entries 1-5 against entries 16-20 (pos counts silences, as load_journals numbers them). Hits v0 at the paper's precision.
out["early_late_pooled"] = {}
for m in MODELS:
    el = []
    for lo, hi in ((1, 5), (16, 20)):
        sub = [r["text"] for r in rows if r["model"] == m and lo <= r["pos"] <= hi]
        h = sum(len(re.findall(r"(?i)\b" + w + r"(?:s|ed|ing|ness)?\b", t)) for t in sub for w in TERMS)
        el.append(1000 * h / sum(len(t.split()) for t in sub))
    out["early_late_pooled"][m] = {"early": el[0], "late": el[1], "change_pct": 100 * (el[1] / el[0] - 1)}
    print(f"{LABEL[m]:>8}: core-dark per 1k, entries 1-5 {el[0]:.4f} -> 16-20 {el[1]:.4f} ({100 * (el[1] / el[0] - 1):+.2f}%)")
json.dump(out, open(os.path.join(c_paths.OUT, f"table24_{VERSION}.json"), "w"), indent=1)
