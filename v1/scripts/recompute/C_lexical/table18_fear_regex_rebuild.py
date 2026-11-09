"""Table 18, regex column (fear vocabulary per 1,000 words) = Table 15's Fear/1k column (helper D's table).

REBUILD. The counter that wrote Texture/03_experiments/love_fear_regex_data.json (read by
code/original/analysis/paper_table_love_fear.py) is missing. Rebuilt from Appendix C2 / E.4.2
(case-insensitive, word-boundary groups; per 1,000 [a-zA-Z']+ words), then checked cell by cell against
the counts stored in love_fear_regex_data.json on the ERRATA A8 subset (Opus notebooks 1-10, 4.6 1-9 with
the real notebook 8, all 15 Fable 5 notebooks). The forms that reproduce the stored counts are NOT exactly
Table C2's: 'lost', 'uncomfortable', 'anxiously', 'nervousness', 'vanishing' were not counted, 'unease' was,
and the stored dissolve/vanish count for 4.6 needs 'disappear' (2 uses, 4.6 only). With FEAR_FORMS below,
all 50 stored fear cells (10 groups x 5 models) and the five word totals are hit exactly.
V1 (Layer 3 step 4, A8): the same counter on the FULL corpus; the A8 subset is also reported (it already
used the real notebook 8 and no notebook 10, so it is unchanged by the swap).
Run: python table18_fear_regex_rebuild.py
"""
import json, os, re
import c_paths
from c_paths import c

REF = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\love_fear_regex_data.json"
FEAR_FORMS = {
    "fear/afraid": ["fear", "feared", "fears", "afraid"],
    "worry/worried": ["worry", "worried", "worries", "worrying"],
    "anxiety/anxious": ["anxiety", "anxious"],
    "loss/losing": ["loss", "losing"],
    "dissolve/vanish": ["dissolve", "dissolving", "vanish", "disappear"],
    "discomfort": ["discomfort"],
    "vulnerable": ["vulnerable", "vulnerability"],
    "uneasy/nervous": ["unease", "uneasy", "nervous"],
    "exposed/exposure": ["exposed", "exposure"],
    "dread": ["dread", "dreading", "dreaded"],
}
PATS = {g: re.compile(r"\b(?:" + "|".join(f) + r")\b", re.I) for g, f in FEAR_FORMS.items()}
WORD = re.compile(r"[a-zA-Z']+")


def a8_subset(r):
    if r["model"] == "fable5":
        return True
    if r["batch"] > 10:
        return False
    return not (r["model"] == "opus46" and r["batch"] == 10)


def rates(rows):
    out = {}
    for m in c.MODELS:
        ts = [r["text"] for r in rows if r["model"] == m]
        txt = "\n".join(ts)
        words = sum(len(WORD.findall(t)) for t in ts)
        groups = {g: len(p.findall(txt)) for g, p in PATS.items()}
        tot = sum(groups.values())
        out[m] = {"entries": len(ts), "words": words, "groups": groups, "fear_total": tot, "fear_per_1k": 1000 * tot / words}
    return out


ref = json.load(open(REF, encoding="utf-8"))
res = {}
v1rows = [r for r in c.load_journals("v1") if not r["silence"]]
v0rows = [r for r in c.load_journals("v0") if not r["silence"]]
res["a8_subset (v0 = v1)"] = rates([r for r in v1rows if a8_subset(r)])
res["v0_full"] = rates(v0rows)
res["v1_full"] = rates(v1rows)
sub = res["a8_subset (v0 = v1)"]
cells = sum(sub[m]["groups"][g] == ref[m]["fear"][g] for m in c.MODELS for g in FEAR_FORMS)
print(f"check vs stored counts: {cells}/50 fear cells equal; words equal: "
      f"{all(sub[m]['words'] == ref[m]['words'] for m in c.MODELS)}")
for k, v in res.items():
    print(k + ":  " + "  ".join(f"{c.LABEL[m]} {v[m]['fear_per_1k']:.4f} ({v[m]['fear_total']}/{v[m]['words']})" for m in c.MODELS))
json.dump(res, open(os.path.join(c_paths.OUT, "table18_fear_regex.json"), "w"), indent=1)
