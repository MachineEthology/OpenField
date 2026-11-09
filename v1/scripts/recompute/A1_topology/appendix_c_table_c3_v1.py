# A1 V1 recompute, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper). Appendix C, Table C3 (= the E.4 corpus-size
# table): total words and entries per model. No producer script survives; ERRATA A8 states the recipe that
# reproduces the printed totals exactly, and this script follows it word for word:
#   [a-zA-Z']+ word counts over Opus Batches 1-10 "as copied on the author's desktop" (4.6: Batches 1-9 with the
#   REAL Batch 8), and all 15 Fable 5 batches (silences excluded).
# The real 4.6 Batch 8 is read from the V1 corpus (byte-identical to the repo's supplementary copy); every other
# A8 file is read from v0 (the V1 corpus differs from v0 only in 4.6 Batches 8 and 10, neither of which the
# 4.6 rows of the A8 recipe read except the real Batch 8).
# Then the same counter on the full corpora (all 15 batches, every text entry), v0 and v1, which is what the plan
# (Layer 3, step 4) asks for in V1. Case-sensitivity: the pattern is applied to the raw text (A-Z and a-z both in it).
import json
import os
import re
import sys

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = re.compile(r"[a-zA-Z']+")
PRINTED = {"4.5": (69395, 300), "4.6": (252211, 300), "4.7": (97276, 300), "4.8": (122526, 300), "Fable 5": (189153, 289)}


def count(rows):
    rows = [r for r in rows if not r["silence"]]
    return sum(len(W.findall(r["text"])) for r in rows), len(rows)


v0 = c.load_journals("v0")
v1 = c.load_journals("v1")
a8 = {}
for m in c.MODELS:
    if m == "fable5":
        rows = [r for r in v0 if r["model"] == m]
    elif m == "opus46":
        rows = [r for r in v1 if r["model"] == m and r["batch"] <= 9]   # 1-9 with the real Batch 8
    else:
        rows = [r for r in v0 if r["model"] == m and r["batch"] <= 10]
    a8[c.LABEL[m]] = count(rows)
full = {v: {c.LABEL[m]: count([r for r in rows if r["model"] == m]) for m in c.MODELS} for v, rows in (("v0", v0), ("v1", v1))}
res = {"printed": PRINTED, "a8_recipe_v0": a8, "full_corpus": full,
       "a8_matches_printed_words": {k: a8[k][0] == PRINTED[k][0] for k in PRINTED}}
json.dump(res, open(os.path.join(HERE, "out", "appendix_c_table_c3.json"), "w"), indent=1)
for k in PRINTED:
    print(f"{k:>8}  printed {PRINTED[k][0]:>7,} ({PRINTED[k][1]})   A8 recipe {a8[k][0]:>7,} ({a8[k][1]})   "
          f"full v0 {full['v0'][k][0]:>7,} ({full['v0'][k][1]})   full v1 {full['v1'][k][0]:>7,} ({full['v1'][k][1]})")
