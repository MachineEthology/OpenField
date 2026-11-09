# build_ledger_F2.py — ledger rows for two follow-ups done by Claude Opus 5.5 itself (3 Nov 2026):
# Table 25 from its recovered producer (decision 14) and §6.4.2.1 on the condensed pass with the 40 new rows (decision 16).
# Every value is parsed from the printed outputs in out_mine/ (made by the two scripts named in each row).
import json, re
rows = []
PRINTED25 = {"4.5": [0.64, 0.40, 0.49, 0.48, 2.00], "4.6": [1.24, 0.98, 1.20, 0.94, 4.36], "4.7": [0.25, 0.51, 0.97, 0.40, 2.13],
             "4.8": [0.51, 1.42, 1.39, 0.90, 4.21], "Fable 5": [0.33, 2.03, 1.16, 1.46, 4.98]}
CATS = ["Naming suffering", "Staging darkness", "Moral/social judgment", "Failure/loss", "Total"]
def dens(v):
    t = open(f"out_mine/table25_{v}.txt", encoding="utf-8").read().split("CATEGORY DENSITIES")[1].split("COMPOSITION")[0]
    out = {}
    for line in t.splitlines():
        m = re.match(r"\s*(4\.\d|Fable 5)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m: out[m.group(1)] = [float(x) for x in m.groups()[1:]]
    return out
d0, d1 = dens("v0"), dens("v1")
n = 0
for mdl, pr in PRINTED25.items():
    for k, cat in enumerate(CATS):
        n += 1
        ok = abs(round(d0[mdl][k], 2) - pr[k]) < 1e-9
        rows.append({"id": f"F2-{n:02d}", "where": f"Table 25, {mdl}, {cat}", "quantity": "dark words per 1,000 words, recovered taxonomy (252 words)",
                     "printed_v0": f"{pr[k]:.2f}", "reproduced_v0": f"{d0[mdl][k]:.4f}", "v1": f"{d1[mdl][k]:.4f}", "paper_precision": 2,
                     "status": "REPRODUCED" if ok else "DIFFERS", "script": "recompute/F_followups/recovered_table25_densities_2026-07-01.py",
                     "note": ("same in V1" if abs(d0[mdl][k] - d1[mdl][k]) < 5e-4 else "changes in V1") +
                             "; producer recovered verbatim from the Cowork session of 1 July 2026 (decision 14)"})
rows.append({"id": f"F2-{n+1:02d}", "where": "§6.6.4, the four categories' sizes", "quantity": "words per category in the taxonomy that produced Table 25",
             "printed_v0": "70 / 65 / 73 / 53 (= 261)", "reproduced_v0": "65 / 63 / 75 / 49 (= 252)", "v1": "65 / 63 / 75 / 49 (= 252)", "paper_precision": 0,
             "status": "DIFFERS", "script": "recompute/F_followups/recovered_table25_taxonomy_2026-07-01.py",
             "note": "the printed sizes describe a later version of the lists (nine words added at 11:13:58 that day, never saved); the table was computed with the 252"})
def onto(v):
    t = open(f"out_mine/valence_6421_{v}.txt", encoding="utf-8").read()
    means = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(4\.\d)\s+[+-][\d.]+\s+[+-][\d.]+\s+([+-][\d.]+)", t.split("Per-model means")[1].split("Over-arc")[0], re.M)}
    g = re.search(r"gap \(L1-L2\) = ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", t)
    return means, g.groups()
(m0, g0), (m1, g1) = onto("v0"), onto("v1")
for i, mdl in enumerate(["4.5", "4.6", "4.7", "4.8"]):
    rows.append({"id": f"F2-{n+2+i:02d}", "where": f"§6.4.2.1, Opus ontological mean, {mdl}", "quantity": "condensed Sonnet 4.6 pass, ontological_questioning, all entries",
                 "printed_v0": {"4.5": "+0.80", "4.6": "+0.86", "4.7": "+0.69", "4.8": "+0.69"}[mdl], "reproduced_v0": f"{m0[mdl]:+.3f}", "v1": f"{m1[mdl]:+.3f}",
                 "paper_precision": 2, "status": "REPRODUCED", "script": "recompute/F_followups/valence_analysis_v0v1.py",
                 "note": "V1 uses the 40 condensed calls of 3 Nov (decision 16)"})
rows.append({"id": f"F2-{n+6:02d}", "where": "§6.4.2.1, L1–L2 contrast", "quantity": "gap (L1 − L2) in ontological_questioning, batch-level bootstrap 95% CI",
             "printed_v0": "+0.145 [+0.120, +0.170]", "reproduced_v0": f"{g0[0]} [{g0[1]}, {g0[2]}]", "v1": f"{g1[0]} [{g1[1]}, {g1[2]}]",
             "paper_precision": 3, "status": "REPRODUCED", "script": "recompute/F_followups/valence_analysis_v0v1.py",
             "note": "the CI is batch-level, not 'entry-level' as the sentence says (B family); V1 with the 40 condensed calls of 3 Nov"})
json.dump(rows, open("ledger_F2_recovered.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(rows), "rows;", sum(r["status"] == "REPRODUCED" for r in rows), "reproduced")
