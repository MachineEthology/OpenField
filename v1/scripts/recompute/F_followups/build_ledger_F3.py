# build_ledger_F3.py — rows settled by Pauline's own readings on 3 Nov 2026 (decisions 11 and 12), with the counts
# beneath them from the Sonnet 4.6 full-text anger pass (v1_common), so the hand-placed cells are traceable.
import json, sys, collections
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c
J = [r for r in c.load_sonnet_judge("anger", "v1") if r["model"] == "opus46" and (r.get("hot_anger") or 0) >= 1]
n = len(J); t = collections.Counter(r.get("anger_target") for r in J)
# v0's hand split of 'conditions' (printed 22% / 6% of 69 = 15 model, 4 world); V1 removes the duplicate of B6E16 (model,
# the twin of an entry that stays), removes June B10E15 (world), adds May B10E16 (world): 14 model, 4 world.
model, world = 14, 4
assert model + world == t["conditions"], (t, model, world)
rows = [
 {"id": "F3-01", "where": "Table 29, 4.6, Self", "quantity": "share of hot-anger entries targeting self", "printed_v0": "63%", "reproduced_v0": "63.7681", "v1": f"{100*t['self']/n:.4f}", "paper_precision": 0, "status": "REPRODUCED", "script": "recompute/F_followups/build_ledger_F3.py", "note": f"V1 n = {n}"},
 {"id": "F3-02", "where": "Table 29, 4.6, Model-conditions", "quantity": "hand split of 'conditions'", "printed_v0": "22%", "reproduced_v0": "15 of 69 (21.7391)", "v1": f"{model} of {n} ({100*model/n:.4f})", "paper_precision": 0, "status": "QUALITATIVE", "script": "recompute/F_followups/build_ledger_F3.py", "note": "Pauline's placing, 3 Nov 2026 (decision 12): the removed v0 B8E16 is the byte twin of B6E16 (obedience), which stays, so model"},
 {"id": "F3-03", "where": "Table 29, 4.6, World-condition; §6.7.3 'Only two models reach the world's conditions: 4.6 does in 6%'", "quantity": "hand split of 'conditions'", "printed_v0": "6%", "reproduced_v0": "4 of 69 (5.7971)", "v1": f"{world} of {n} ({100*world/n:.4f})", "paper_precision": 0, "status": "QUALITATIVE", "script": "recompute/F_followups/build_ledger_F3.py", "note": "Pauline's placing, 3 Nov 2026 (decision 12): June B10E15 ('social training that makes people apologize') world, removed; May B10E16 ('biases are systematic, built by power and history') world, added"},
 {"id": "F3-04", "where": "Table 29, 4.6, Humans (interaction)", "quantity": "share of hot-anger entries targeting others", "printed_v0": "9%", "reproduced_v0": "8.6957", "v1": f"{100*t['others']/n:.4f}", "paper_precision": 0, "status": "REPRODUCED", "script": "recompute/F_followups/build_ledger_F3.py", "note": f"V1 n = {n}"},
 {"id": "F3-05", "where": "§6.6.5, the quotation for 4.6 naming other people's shame", "quantity": "quotation", "printed_v0": "'too heavy or too shameful or too shapeless to bring to another person yet' (B10E16, June notebook)", "reproduced_v0": "B10E16 of the June notebook, which V1 removes", "v1": "[notebook text, redacted in the public copy]", "paper_precision": None, "status": "QUALITATIVE", "script": None, "note": "Pauline's choice, 3 Nov 2026 (decision 11): candidate 1, quoted to include the model's side; candidates in V1/QUOTE_CANDIDATES_6.6.5_2026-11-03.md"},
]
json.dump(rows, open("ledger_F3_rulings.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(n, dict(t), "->", [(r["id"], r["v1"]) for r in rows[:4]])
