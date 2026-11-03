"""Table 24: the seventeen 'sharpest' dark terms (intensity measure, section 6.6.3).

Same method as the recovered original (code/recovered/table24_core_dark_terms_2026-06-22.py):
case-insensitive whole-word match of each term plus an optional suffix s/ed/ing/ness, over all journal
text entries as analysed; per 1,000 words with words = str.split(). The original list had 18 words;
'hatred' scored 0 for every model and was dropped, leaving seventeen. The two the paper does not name
are 'rage' and 'unbearable'.
"""
import re

from common import LABEL, MODELS, load_journals

TERMS = ["despair", "cruelty", "suffering", "betrayal", "guilt", "grief", "shame", "fear", "pain",
         "loneliness", "hollow", "emptiness", "rage", "worthless", "unbearable", "violence", "anguish"]
PUBLISHED = {"opus45": (116, 1.07), "opus46": (409, 0.98), "opus47": (40, 0.27), "opus48": (98, 0.54),
             "fable5": (86, 0.45)}

rows = [r for r in load_journals() if not r["silence"]]
texts = {m: [r["text"] for r in rows if r["model"] == m] for m in MODELS}
words = {m: sum(len(t.split()) for t in texts[m]) for m in MODELS}
counts = {w: {m: sum(len(re.findall(r"(?i)\b" + w + r"(?:s|ed|ing|ness)?\b", t)) for t in texts[m]) for m in MODELS}
          for w in TERMS}

print(f"{'term':>12} " + " ".join(f"{LABEL[m]:>8}" for m in MODELS) + "   leader (share)")
for w in TERMS:
    c = counts[w]; tot = sum(c.values()); lead = max(MODELS, key=lambda m: c[m])
    print(f"{w:>12} " + " ".join(f"{c[m]:>8}" for m in MODELS) + f"   {LABEL[lead]} ({100 * c[lead] / max(tot, 1):.0f}%)")
print()
for m in MODELS:
    hits = sum(counts[w][m] for w in TERMS)
    print(f"{LABEL[m]:>8}: {hits:>4} hits, {1000 * hits / words[m]:.2f} per 1k words   (published {PUBLISHED[m][0]}, {PUBLISHED[m][1]})")
