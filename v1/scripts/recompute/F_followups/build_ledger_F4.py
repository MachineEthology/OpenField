# build_ledger_F4.py — Table 15 as finally ruled (decision 10, revised by Pauline on 3 Nov 2026: "Yes for (b)"):
#   Fear column: the counter that reproduces v0's stored fear counts exactly (C_lexical/table18_fear_regex_rebuild.py);
#                Appendix C2's printed fear list is corrected to the forms it counts.
#   Love column: the forms Appendix C2 prints (F_followups/table15_c2_forms_v1.py), with a note.
#   Both on the full corpus (decision 9). Love:Fear is their ratio. Table 18's regex column is the Fear column.
# Values are read from the two producers' saved outputs. These rows supersede the V1 values of F-01..F-15 and C-14..C-18.
import json
love = json.load(open("out/table15_c2_forms.json", encoding="utf-8"))
fear = json.load(open("../C_lexical/out/table18_fear_regex.json", encoding="utf-8"))
PR = {"opus45": ("4.5", 3.03, 1.14, "2.7x"), "opus46": ("4.6", 2.73, 1.19, "2.3x"), "opus47": ("4.7", 2.27, 0.85, "2.7x"),
      "opus48": ("4.8", 4.97, 1.12, "4.4x"), "fable5": ("Fable 5", 1.92, 0.81, "2.4x")}
rows, n = [], 0
for m, (lab, pl, pf, pr) in PR.items():
    L0, L1 = love["full_v0"][m]["love_per_1k"], love["full_v1"][m]["love_per_1k"]
    F0, F1 = fear["v0_full"][m]["fear_per_1k"], fear["v1_full"][m]["fear_per_1k"]
    Fs = fear["a8_subset (v0 = v1)"][m]["fear_per_1k"]
    for q, p, v0, v1, st, note in (
        ("Love/1k", f"{pl:.2f}", L0, L1, "DIFFERS", "C2 forms, full corpus; v0's love counter is lost and its forms cannot be resolved (care/caring)"),
        ("Fear/1k (= Table 18 regex)", f"{pf:.2f}", F0, F1, "REPRODUCED", f"stored-forms counter, full corpus; on the A8 subset it gives {Fs:.4f} = the printed value; C2's fear list is corrected to these forms"),
        ("Love:Fear", pr, L0 / F0, L1 / F1, "DIFFERS", "ratio of the two columns above")):
        n += 1
        rows.append({"id": f"F4-{n:02d}", "where": f"Table 15, {lab}, {q}", "quantity": q, "printed_v0": p,
                     "reproduced_v0": f"{v0:.4f} (full v0)", "v1": f"{v1:.4f}", "paper_precision": 2 if "/1k" in q else 1,
                     "status": st, "script": "recompute/F_followups/build_ledger_F4.py", "note": "decision 10 (b), ruled 3 Nov 2026; " + note})
json.dump(rows, open("ledger_F4_table15.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for r in rows: print(r["where"], "|", r["printed_v0"], "->", r["v1"])
