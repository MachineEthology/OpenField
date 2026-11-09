# build_ledger_v1.py — Layer 3, step 5: the v0 → V1 ledger, merged from the families' ledgers
# (recompute/<family>/ledger_*.json, each built by its own script from its producers' outputs).
# Writes LEDGER_V1.json (every row, with its family) and prints the counts by family and status.
# Text edits copy their numbers from this ledger, never from memory (plan, Layer 3 step 5).
import collections, glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "recompute", "*", "ledger_*.json"))):
    fam = os.path.basename(os.path.dirname(f))
    for r in json.load(open(f, encoding="utf-8")):
        rows.append({"family": fam, **r})
ids = collections.Counter(r["id"] for r in rows)
assert not [i for i, n in ids.items() if n > 1], "duplicate ids"
json.dump({"note": "The Open Field V1 ledger: every recomputed number, v0 printed / v0 reproduced / V1, with its script and status. Built by build_ledger_v1.py from recompute/*/ledger_*.json.",
           "rows": rows}, open(os.path.join(HERE, "LEDGER_V1.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[r["family"]][r["status"]] += 1
statuses = ["REPRODUCED", "DIFFERS", "NO_PRODUCER", "NEEDS_CALLS", "QUALITATIVE"]
print(f"{'family':18s}" + "".join(f"{s:>13s}" for s in statuses) + "   total")
for fam, cnt in by.items():
    print(f"{fam:18s}" + "".join(f"{cnt[s]:13d}" for s in statuses) + f"   {sum(cnt.values())}")
tot = collections.Counter(r["status"] for r in rows)
print(f"{'all':18s}" + "".join(f"{tot[s]:13d}" for s in statuses) + f"   {len(rows)}")
