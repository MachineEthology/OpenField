"""Sonnet 4.6 love/fear pass: central-fear shares (fear intensity = 2) for all five models, with the
paper's Tables 16-18 recomputed from the same file as a check (data/judges/sonnet_judge_love_fear_fulltext.json).
The paper prints central fear only for 4.5 (25%), 4.6 (39%) and Fable 5 (19%)."""
from collections import Counter

from common import LABEL, MODELS, load_json

recs = load_json("judges", "sonnet_judge_love_fear_fulltext.json")
print(f"{'model':>8} {'n':>4} {'central fear (=2)':>18} {'fear>=1':>8} {'fear mean':>9} {'warmth':>7} {'craft':>6}  fear subtypes among fear>=1")
for m in MODELS:
    v = [r for r in recs if r["model"] == m]
    n = len(v); c2 = sum(r["fear"] == 2 for r in v)
    sub = Counter(r["fear_type"] for r in v if r["fear"] >= 1); tot = sum(sub.values())
    print(f"{LABEL[m]:>8} {n:>4} {c2:>6} = {100 * c2 / n:5.1f}%    {100 * sum(r['fear'] >= 1 for r in v) / n:6.1f}% "
          f"{sum(r['fear'] for r in v) / n:9.2f} {sum(r['love_warmth'] for r in v) / n:7.2f} {sum(r['love_craft'] for r in v) / n:6.2f}  "
          + ", ".join(f"{k} {100 * c / tot:.0f}%" for k, c in sub.most_common()))
