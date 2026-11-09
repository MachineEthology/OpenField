"""Figure 11 / §6.6.2 as printed: stark and dark densities per 1,000 words over ONE pooled lexicon.

REBUILD. paper_fig_stark_dark.py, as it stands on disk, counts each model's OWN dark-cluster words and does not
hit the plate (see stark_dark_v1.py). The plate is hit exactly when the lexicon is pooled: the union over the five
models of the words that script's pipeline puts in the "Darkness / Negative" cluster on the v0 corpus, judged by
stark_vs_dark_judge.json - 261 'dark' types (the paper's "261 words") and 172 'stark' types - and every model's
journal tokens ([a-z']+, lower-cased, as tokenize()) are counted against it, per 1,000 str.split() words.
The lexicon is read from out/stark_dark.json (written by stark_dark_v1.py, key "v0") and kept FIXED for V1.
DIAGNOSTIC extras: the same counts for early (1-5) and late (16-20) entries (§6.6.3's decline sentence).
Run: python fig11_union_lexicon_v1.py
"""
import json, os, re
import numpy as np
import c_paths
from stark_dark_v1 import load_journals, tokenize, ALL_MODELS, MODEL_LABELS

lex = json.load(open(os.path.join(c_paths.OUT, "stark_dark.json"), encoding="utf-8"))["v0"]
DARK, STARK = set(lex["union_dark_types"]), set(lex["union_stark_types"])
assert len(DARK) == 261 and len(STARK) == 172, (len(DARK), len(STARK))


def count(ents):
    toks = [t for e in ents for t in tokenize(e["text"])]
    words = sum(len(e["text"].split()) for e in ents)
    s, d = sum(t in STARK for t in toks), sum(t in DARK for t in toks)
    return {"words": words, "stark_n": s, "dark_n": d, "stark_per_1k": 1000 * s / words, "dark_per_1k": 1000 * d / words,
            "ratio": s / d if d else None}


out = {}
for version in ("v0", "v1"):
    J = load_journals(version)
    out[version] = {}
    for m in ALL_MODELS:
        ents = [e for e in J if e["model"] == m]
        r = count(ents)
        E, L = count([e for e in ents if e["pos"] <= 5]), count([e for e in ents if e["pos"] >= 16])
        em = lambda sub, k: float(np.mean([1000 * sum(t in (DARK if k == 'd' else STARK) for t in tokenize(e['text'])) / max(len(e['text'].split()), 1) for e in sub]))
        Ee, Le = [e for e in ents if e["pos"] <= 5], [e for e in ents if e["pos"] >= 16]
        r["early_late_dark_pooled"] = (E["dark_per_1k"], L["dark_per_1k"], 100 * (L["dark_per_1k"] / E["dark_per_1k"] - 1))
        r["early_late_dark_entry_mean"] = (em(Ee, 'd'), em(Le, 'd'), 100 * (em(Le, 'd') / em(Ee, 'd') - 1))
        out[version][m] = r
        print(f"[{version}] {MODEL_LABELS[m]:>8}: stark {r['stark_per_1k']:.4f}  dark {r['dark_per_1k']:.4f}  ratio {r['ratio']:.4f}   "
              f"dark early->late pooled {r['early_late_dark_pooled'][2]:+.1f}%  entry-mean {r['early_late_dark_entry_mean'][2]:+.1f}%")
# V1 tokens of 4.6 that the fixed lexicon cannot classify: journal-only types without an embedding (from stark_dark_v1)
unc = json.load(open(os.path.join(c_paths.OUT, "stark_dark.json"), encoding="utf-8"))["v1_new_missing_embedding_opus46"]
out["v1_opus46_types_outside_classification"] = {"types": len(unc), "tokens": sum(unc.values())}
json.dump(out, open(os.path.join(c_paths.OUT, "fig11_union_lexicon.json"), "w"), indent=1)
