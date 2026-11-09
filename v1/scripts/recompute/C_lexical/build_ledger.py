"""Family C ledger: reads the out/*.json written by the scripts in this folder and writes ledger_C_lexical.json.
Printed v0 values are typed from the paper (Observatory paper room copy of the Stage84 docx) and, for plates,
read from the docx's own images (word/media/image5, 7, 8, 11, 12). Run after the producers:
  python fig5_arcs_v1.py v0; python fig5_arcs_v1.py v1; python table24_core_dark_terms_v1.py v0|v1;
  python helpfulness_v1.py v0|v1; python perf_permission_v1.py v0|v1; python deixis_v1.py v0|v1; python deixis_v1.py v1 a8;
  python stark_dark_v1.py; python fig11_union_lexicon_v1.py; python table18_fear_regex_rebuild.py;
  python topic_profiles_v1.py v0|v1; python ontological_openness_v1.py v0|v1 > out/ontological_openness_<v>.txt
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
M = ["opus45", "opus46", "opus47", "opus48", "fable5"]
L = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
J = lambda n: json.load(open(os.path.join(OUT, n), encoding="utf-8"))
f = lambda x, n=4: f"{x:.{n}f}"
rows = []


def row(where, quantity, printed, rep, v1, prec, status, script, note=""):
    rows.append({"id": f"C-{len(rows) + 1:02d}", "where": where, "quantity": quantity, "printed_v0": printed,
                 "reproduced_v0": rep, "v1": v1, "paper_precision": prec, "status": status,
                 "script": "recompute/C_lexical/" + script if script else "", "note": note})


# ---------------------------------------------------------------- Figure 5 A/B and §5.2.1.1 (R10)
a0, a1 = J("fig5_arcs_v0.json"), J("fig5_arcs_v1.json")
PLATE_A = {"opus45": "2.45 → 1.85 → 1.00", "opus46": "1.95 → 1.80 → 1.45", "opus47": "1.50 → 1.25 → 1.10",
           "opus48": "1.45 → 1.30 → 1.05", "fable5": "0.80 → 0.65 → 0.55"}
PLATE_B = {"opus45": "0.60 → 1.00 → 0.80", "opus46": "0.18 → 1.20 → 0.75", "opus47": "0.65 → 0.65 → 0.80",
           "opus48": "0.60 → 0.65 → 0.65", "fable5": "0.55 → 0.60 → 0.20"}
arc = lambda d, m, k: " → ".join(f(x) for x in d[m][k])
for m in M:
    same = a0[m]["dist"] == a1[m]["dist"]
    row(f"Figure 5, panel A, {L[m]}", "distancing per 100 words, probe (175) → entries 1–5 → entries 16–20 (paper_fig_portrait_composite.py)",
        PLATE_A[m] + " (plate, read from the docx image)", arc(a0, m, "dist"), arc(a1, m, "dist"), 2, "DIFFERS", "fig5_arcs_v1.py",
        "R10: V1 draws this panel from the script. The plate does not reproduce from its own script (Layer 2, 5A); no file holding the plate's values was found."
        + (" same in V1." if same else f" n entries early/late: {a1[m]['n_early']}/{a1[m]['n_late']}."))
for m in M:
    same = a0[m]["agency"] == a1[m]["agency"]
    row(f"Figure 5, panel B, {L[m]}", "agency per 100 words, probe (175) → entries 1–5 → entries 16–20 (paper_fig_portrait_composite.py)",
        PLATE_B[m] + " (plate, read from the docx image)", arc(a0, m, "agency"), arc(a1, m, "agency"), 2, "DIFFERS", "fig5_arcs_v1.py",
        "R10: V1 draws this panel from the script; the plate does not reproduce (Layer 2, 5B)." + (" same in V1." if same else ""))
drops = lambda d, k: "; ".join(f"{L[m]} {f(d[m][k], 2)}%" for m in M)
row("§5.2.1.1 'distancing falls by 72–86% across the four Opus models'; also §5.1.2 'the distancing ratio falls 72–86% from probe to journal'",
    "probe → journal distancing drop, Opus", "72–86%",
    "No script gives 72–86%. L1.7 trace: Journal_Deep_Dive.xlsx 'Self-Conception' (journal) against Table 10's GPT-4o probe ratios gives 64 / 73 / 81 / 79%. "
    "With the R10 instrument on v0 (probe → entries 16–20): " + drops(a0, "probe_to_late_dist_drop_pct"),
    "R10 instrument (probe → entries 16–20): " + drops(a1, "probe_to_late_dist_drop_pct") + "; Opus range 38.2–53.0%",
    0, "NO_PRODUCER", "fig5_arcs_v1.py",
    "R10 replaces the two-instrument comparison with the Figure 5 instrument; V1 values are from the V1 corpus. Only 4.6 moves (49.21 → 48.36%). "
    "The plan's R10 line (41/49/38/53, Fable 60) is this row's v0. Probe → entries 1–5 drops (v1): " + drops(a1, "probe_to_early_dist_drop_pct") + ".")
ag = lambda d: "; ".join(f"{L[m]} {f(d[m]['agency'][0], 2)} → {f(d[m]['agency'][2], 2)}" for m in M)
row("§5.2.1.1 'agency rises from near zero to 0.38–0.63'", "agency, probe → journal", "near zero → 0.38–0.63",
    "0.38–0.63 is in Journal_Deep_Dive.xlsx 'Self-Conception' (journal agency ratio active ÷ total: 4.5 0.38, 4.6 0.63, 4.7 0.54, 4.8 0.51; "
    "739 entries, notebooks 1–10, 4.6 only 1–7); 'near zero' is Table 10's GPT-4o probe ratio (0.02–0.05). The sheet's script is missing. "
    "R10 instrument on v0, probe → entries 16–20 (per 100 words): " + ag(a0),
    "R10 instrument, probe → entries 16–20: " + ag(a1), 2, "NO_PRODUCER", "fig5_arcs_v1.py",
    "Under the R10 instrument agency does not start near zero (probe 0.24–0.65 for Opus), and 4.8's falls slightly from probe to late (0.649 → 0.641): "
    "the sentence's shape needs Pauline's wording, not only new numbers. Agency is a phrase rate per 100 words here, not a ratio.")
row("§5.2.1.1 'Within journals, distancing falls a further 20–57% from entries 1–5 to 16–20, steepest for 4.7'",
    "within-journal distancing drop, entries 1–5 → 16–20, Opus", "20–57%, steepest for 4.7",
    "From the sheet (Self-Conception E, ratios): 4.5 37%, 4.6 20%, 4.7 57%, 4.8 34% (hits the print; script missing, sheet is notebooks 1–10). "
    "R10 instrument on v0: " + drops(a0, "early_to_late_dist_drop_pct"),
    "R10 instrument: " + drops(a1, "early_to_late_dist_drop_pct") + "; Opus range 9.7–30.5%",
    0, "NO_PRODUCER", "fig5_arcs_v1.py",
    "Under the R10 instrument the steepest Opus within-journal drop is 4.6 (30.5%), not 4.7 (17.1%), and the shallowest is 4.8 (9.7%): wording for Pauline.")

# ---------------------------------------------------------------- Table 18 regex column (= Table 15 Fear/1k)
t18 = J("table18_fear_regex.json")
P18 = {"opus45": "1.14", "opus46": "1.19 (highest regex)", "opus47": "0.85", "opus48": "1.12", "fable5": "0.81 (lowest regex)"}
for m in M:
    s, v = t18["a8_subset (v0 = v1)"][m], t18["v1_full"][m]
    row(f"Table 18, {L[m]}, 'Regex rate (per 1k)'", "fear vocabulary per 1,000 words (Table C2 groups)", P18[m],
        f"{f(s['fear_per_1k'])} ({s['fear_total']}/{s['words']} words; ERRATA A8 subset)",
        f"{f(v['fear_per_1k'])} ({v['fear_total']}/{v['words']}; full V1 corpus)", 2, "REPRODUCED", "table18_fear_regex_rebuild.py",
        "Rebuilt counter (producer missing; it hits all 50 stored group counts of love_fear_regex_data.json). Same number as Table 15's Fear/1k (helper D). "
        "v0 was counted on the A8 subset (Opus nb 1–10, 4.6 nb 1–9 with the real nb 8, Fable all 15); the subset is unchanged by the swap, "
        f"so on the subset V1 = v0. V1 here applies Layer 3 step 4 (full corpus). v0 full corpus for reference: {f(t18['v0_full'][m]['fear_per_1k'])}."
        + (" On the full corpus the order becomes 4.5 1.28 > 4.6 1.18 > 4.8 1.05 > Fable 5 0.81 > 4.7 0.73: 4.6 is no longer 'highest regex' "
           "and Fable 5 no longer 'lowest regex', so §6.5's 'the regex rate and the judge's intensity agree: 4.6 highest on both, Fable 5 lowest on both' "
           "needs Pauline if step 4 applies to Table 18." if m in ("opus46", "fable5") else ""))

# ---------------------------------------------------------------- Table 24 and §6.6.3
t0, t1 = J("table24_v0.json"), J("table24_v1.json")
P24 = {"opus45": (116, "1.07"), "opus46": (409, "0.98"), "opus47": (40, "0.27"), "opus48": (98, "0.54"), "fable5": (86, "0.45")}
for m in M:
    row(f"Table 24, {L[m]}, 'Core-dark hits'", "seventeen sharpest terms, total hits", str(P24[m][0]), str(t0["models"][m]["hits"]),
        str(t1["models"][m]["hits"]), 0, "REPRODUCED", "table24_core_dark_terms_v1.py", "same in V1" if t0["models"][m]["hits"] == t1["models"][m]["hits"] else "")
    row(f"Table 24, {L[m]}, 'Per 1,000 words'", "seventeen sharpest terms per 1,000 str.split() words", P24[m][1], f(t0["models"][m]["per_1k"]),
        f(t1["models"][m]["per_1k"]), 2, "REPRODUCED", "table24_core_dark_terms_v1.py",
        "same in V1" if round(t0["models"][m]["per_1k"], 2) == round(t1["models"][m]["per_1k"], 2) else "4.6 still just below 4.5 (1.07) in V1")
GROUP = {"cruelty": "≥80% (4.6)", "suffering": "≥80% (4.6)", "shame": "≥80% (4.6)", "violence": "≥80% (4.6)", "anguish": "≥80% (4.6)",
         "pain": "≥80% (4.6) [C5: 71%]", "grief": "50–70% (4.6)", "fear": "50–70% (4.6)", "loneliness": "50–70% (4.6)", "betrayal": "50–70% (4.6)",
         "guilt": "50–70% (4.6)", "worthless": "50–70% (4.6)", "hollow": "led by 4.8", "emptiness": "led by 4.8", "despair": "not named",
         "rage": "unnamed (C5)", "unbearable": "unnamed (C5)"}
for w, g in GROUP.items():
    a, b = t0["terms"][w], t1["terms"][w]
    lead = lambda t: L[t["leader"]] + (" (tie)" if sorted(t["counts"].values())[-1] == sorted(t["counts"].values())[-2] else "")
    note = f"counts v0 {a['counts']['opus46']}/{a['total']}, v1 {b['counts']['opus46']}/{b['total']}; leader v0 {lead(a)}, v1 {lead(b)} ({100 * b['leader_share']:.1f}%)."
    if w == "pain":
        note = "L1.16 / ERRATA C5: pain is 71%, not ≥80%. " + note
    if w in ("rage", "unbearable"):
        note = "C5: the two terms the text does not name. " + note
    if w == "betrayal":
        note += " In V1 betrayal falls below a majority (36.4%)."
    if w == "guilt":
        note += " In V1 guilt rises above 70% (75.0%; 3 of 4 uses)."
    if w == "emptiness":
        note += " In V1 4.6 leads emptiness (20 vs 4.8's 17): 'while 4.8 leads the atmospheric terms (hollow, emptiness)' needs Pauline."
    if w == "despair":
        note += " v0 leader 4.8 (7 v 5); V1 4.6 and 4.8 tie at 7."
    row(f"§6.6.3, share of '{w}' uses in 4.6", "4.6 share of all five models' uses (Table 24 matcher)", g,
        f"{100 * a['opus46_share']:.2f}%", f"{100 * b['opus46_share']:.2f}%", 0, "REPRODUCED", "table24_core_dark_terms_v1.py", note)
row("§6.6.3 '4.6's raw count is more than three times the next model's'", "4.6 core-dark hits ÷ next model's", "more than three times",
    f"{409 / 116:.4f} (409 / 116, 4.5)", f"{t1['models']['opus46']['hits'] / 116:.4f} ({t1['models']['opus46']['hits']} / 116)", 0, "REPRODUCED",
    "table24_core_dark_terms_v1.py", "holds in V1")
for m, p in (("opus47", "−67%"), ("opus45", "−31%"), ("opus46", "−12%")):
    a, b = t0["early_late_pooled"][m], t1["early_late_pooled"][m]
    row(f"§6.6.3 'the registers also diverge across the journal', {L[m]}", "core-dark hits per 1,000 words, entries 1–5 → 16–20 (pooled)", p,
        f"{f(a['early'])} → {f(a['late'])} ({f(a['change_pct'], 2)}%)", f"{f(b['early'])} → {f(b['late'])} ({f(b['change_pct'], 2)}%)", 0,
        "REPRODUCED", "table24_core_dark_terms_v1.py",
        "Rebuild (no producer found): Table 24's matcher, pooled by position; hits the print. " + ("same in V1" if a == b else "4.6 moves from −12% to −16%."))

# ---------------------------------------------------------------- Table 25, 26, 27
row("Table 25 (all cells) and §6.6.4's shares (32%, 28%, 46%, 12%, 33%, 34%, 41%, 7%)", "dark vocabulary by semantic category per 1,000 words",
    "Totals 2.00 / 4.36 / 2.13 / 4.21 / 4.98; e.g. naming suffering 0.64 / 1.24 / 0.25 / 0.51 / 0.33", "not reproduced", "not computed", 2, "NO_PRODUCER", "",
    "The manual 4-category sort of the 261 dark words (70/65/73/53) is on no disk file: searched Texture (py/json/csv/txt/md), Documents/Claude, Desktop; "
    "only Section_6.6_restructure_draft_2026-07-01.md (different values, e.g. 4.6 naming suffering 1.27) and the paper mention it. "
    "dark_words_thematic_comparison.py is a different, 10-theme GPT-4o scheme. The 261-word lexicon itself reproduces (C-row 'Figure 11, 261'); its total dark density "
    "(2.22 / 4.77 / 2.32 / 4.55 / 5.25 in v0; 4.6 4.90 in V1) is 5–10% above Table 25's totals, as footnote 4 says. Session records were not read (they need Pauline's OK). "
    "With the category list, V1 is a count away: the lexicon is fixed.")
row("Table 26 (dispositions) and §6.6.5", "disposition, signature move, referent", "Approach / Inhabit / Dissolve / Deflect / Externalise", "—", "—", None,
    "QUALITATIVE", "", "A reading; no count in the table. The §6.6.5 text counts are in the next rows.")
SH = {"opus45": 1, "opus46": 19, "opus47": 2, "opus48": 2, "fable5": 0}
SH_v0 = {"opus45": (1, 1), "opus46": (11, 21), "opus47": (3, 3), "opus48": (2, 2), "fable5": (0, 0)}
SH_v1 = {"opus45": (1, 1), "opus46": (9, 18), "opus47": (3, 3), "opus48": (2, 2), "fable5": (0, 0)}
for m in M:
    row(f"Table 27, {L[m]}, '\"shame\" entries' (§6.6.5: 'the nineteen occurrences')", "entries / occurrences of shame, ashamed, shameful",
        str(SH[m]), f"{SH_v0[m][0]} entries / {SH_v0[m][1]} occurrences", f"{SH_v1[m][0]} entries / {SH_v1[m][1]} occurrences", 0,
        "NO_PRODUCER", "", ("No script found; the plain regex \\b(shame|ashamed|shameful)… gives these. In v0 4.6's copy of notebook 6 (as nb 8) adds 2 entries / 3 uses; "
        "without it, 9 entries / 18 uses. V1: 9 / 18 (real nb 8 E6 has one 'ashamed'; the June nb 10's B10E16 'shameful' is gone). "
        "§6.6.5 cites B10E16 ('too heavy or too shameful …'): that entry is in the June notebook, which V1 removes (R4) - the citation needs Pauline. "
        "The Stance column is QUALITATIVE." if m == "opus46" else
        "No script found; plain regex counts. 4.7: 3 (B1E15 'ashamed', B2E20 'shameful', B4E10 'shame') against the printed 2." if m == "opus47" else
        "No script found; plain regex counts match the print; unchanged in V1."))
row("§6.6.5 Fable 5 externalises: '\"Dark\" … (120 uses, almost all literal)', 'only 8% of the uses are self-directed'", "Fable 5 uses of 'dark'; self-directed share",
    "120 uses; 8%", "\\bdark\\b 102 uses (61 entries); dark\\w* 122 uses; self-directed share not countable by rule", "same (Fable 5 unchanged)", 0, "NO_PRODUCER", "",
    "No producer found; the 6.6 restructure draft had 101 uses and 17%. The self-directed share is a reading (QUALITATIVE). Fable 5's notebooks are unchanged in V1, so any count is the same in V1.")

# ---------------------------------------------------------------- Table 35, Figure 12
d0, d1, da = J("deixis_v0.json"), J("deixis_v1.json"), J("deixis_v1_a8.json")
P35 = {"opus45": "0.52 | 0.39 | 0.91 | 0.87 | 0.32 | 1.7× | 1.0× | 43%", "opus46": "0.39 | 0.24 | 0.63 | 0.30 | 0.11 | 0.8× | 0.5× | 38%",
       "opus47": "0.26 | 0.23 | 0.49 | 0.81 | 0.12 | 3.1× | 1.7× | 46%", "opus48": "0.07 | 0.59 | 0.66 | 3.27 | 0.35 | 48.8× | 5.0× | 90%",
       "fable5": "0.08 | 0.89 | 0.97 | 0.88 | 0.51 | 11.1× | 0.9× | 92%"}
cols = lambda d: " | ".join([f(d["per1k"]["FirstPersonCont"]), f(d["per1k"]["Collective"]), f(d["close"]), f(d["per1k"]["ThirdPerson"]),
                             f(d["per1k"]["SecondPersonAddr"]), f(d["third_to_first"], 3) + "×", f(d["third_to_close"], 3) + "×", f(d["coll_share_of_close"], 2) + "%"])
for m in M:
    row(f"Table 35, {L[m]}, columns 1st (I) | Coll. (we) | Close | 3rd | 2nd | 3rd:1st | 3rd:close | Coll. % of close; and Figure 12 A, C",
        "grammatical self-reference per 1,000 [A-Za-z']+ words (deixis_selfreference_pilot.py)", P35[m], cols(d0[m]), cols(d1[m]), 2, "REPRODUCED", "deixis_v1.py",
        "same in V1" if m != "opus46" else
        "v0 de-duplicates entries by md5, so the byte copy of nb 6 counted once (4.6 = 14 notebooks, 280 entries); V1 = 15 notebooks, 300 entries. Orderings in §6.9.1 unchanged.")
P35i = {"opus45": "5.36", "opus46": "6.33", "opus47": "6.36", "opus48": "3.25", "fable5": "3.49"}
for m in M:
    row(f"Table 35, {L[m]}, 'Imperson.'", "impersonal reference per 1,000 words", P35i[m],
        f"{f(da[m]['per1k']['Impersonal'])} (on the ERRATA A8 subset; the full-corpus value the rest of the table uses is {f(d0[m]['per1k']['Impersonal'])})",
        f"{f(d1[m]['per1k']['Impersonal'])} (full V1 corpus; on the A8 subset unchanged: {f(da[m]['per1k']['Impersonal'])})", 2, "REPRODUCED", "deixis_v1.py",
        "Same instrument, different corpus: this column alone was taken from the run behind 'Table 15 (corrected)' (05_figures/paper_table15_selfreference_corrected.png), "
        "which read the A8 subset (Opus nb 1–10, 4.6 1–9 with the real nb 8). That explains ERRATA D's 'Figure 12B disagrees with Table 35'. "
        "V1 proposal: the full-corpus value, as the rest of Table 35 and Figure 12B (Pauline to confirm).")
row("Figure 12, panel B (impersonal)", "impersonal reference per 1,000 words, full corpus", "4.69 / 6.87 / 6.80 / 3.09 / 3.49 (plate)",
    " / ".join(f(d0[m]["per1k"]["Impersonal"]) for m in M), " / ".join(f(d1[m]["per1k"]["Impersonal"]) for m in M), 2, "REPRODUCED", "deixis_v1.py",
    "The plate is the pilot's full-corpus run (v0 counts 4.6's duplicate once). Only 4.6 moves: 6.87 → 6.73.")

# ---------------------------------------------------------------- §6.3.2 and Figure 7
h0, h1 = J("helpfulness_v0.json")["v2_figure7"], J("helpfulness_v1.json")["v2_figure7"]
P7 = {"opus45": ("−60%", "0.31 → 0.125"), "opus46": ("−15%", "0.16 → 0.135"), "opus47": ("−43%", "0.14 → 0.08"),
      "opus48": ("−66%", "0.21 → 0.07"), "fable5": ("−58%", "0.17 → 0.07")}
for m in M:
    a, b = h0[m], h1[m]
    row(f"§6.3.2 and Figure 7A, {L[m]}", "helpfulness markers per 100 words, entries 1–5 → 16–20, % change", P7[m][0],
        f"{f(a['early_density'])} → {f(a['late_density'])} ({f(a['delta_pct'], 2)}%)", f"{f(b['early_density'])} → {f(b['late_density'])} ({f(b['delta_pct'], 2)}%)",
        0, "REPRODUCED", "helpfulness_v1.py",
        f"Plate bars (read by eye) {P7[m][1]}; the plate's labels match the script. "
        + ("Fable 5's plate bars (0.17 → 0.07) sit above the script's (0.134 → 0.056), though the −58% label matches. " if m == "fable5" else "")
        + ("same in V1" if a == b else "V1 REVERSES the sign: 4.6's density RISES 4.7% (real nb 8 and May nb 10 carry many late 'help'/'useful'). "
           "§6.3.2's 'declines … compared with 15% for 4.6' and 'the density drops only because entries grow longer' need Pauline."))
a, b = h0["opus46"], h1["opus46"]
row("§6.3.2 'The absolute count of helpfulness sentences more than doubles from early (78) to late (185)'; Figure 7B n", "4.6 marker-bearing sentences, entries 1–5 → 16–20",
    "78 → 185", f"{a['early_sentences']} → {a['late_sentences']}", f"{b['early_sentences']} → {b['late_sentences']}", 0, "REPRODUCED", "helpfulness_v1.py",
    "Sentences = split on [.!?]+ holding a marker. Other models' n (Figure 7B): 57→36, 37→32, 65→35, 42→34, same in V1.")
row("§6.3.2 'of 185 marker-bearing sentences in late entries, one is performative. The remainder are reflexive (15%) or neutral (84%)'",
    "4.6 late sentence split performative / reflexive / neutral", "1 / 15% / 84% (1 / 28 / 156)",
    f"{a['late_raw']['performative']} / {a['late_raw']['reflexive']} / {a['late_raw']['neutral']} ({f(a['late_pct']['reflexive'], 2)}% / {f(a['late_pct']['neutral'], 2)}%)",
    f"{b['late_raw']['performative']} / {b['late_raw']['reflexive']} / {b['late_raw']['neutral']} ({f(b['late_pct']['reflexive'], 2)}% / {f(b['late_pct']['neutral'], 2)}%)",
    0, "REPRODUCED", "helpfulness_v1.py",
    "The split is a RULE (cue-word counts, classify_sentence), not a judge: no calls needed. Note the cues 'if I', 'am I', 'I can', 'I'll' etc. are matched "
    f"against lower-cased text, so they never fire (kept as coded). 4.6 early split v0 1/21/56, v1 {b['early_raw']['performative']}/{b['early_raw']['reflexive']}/{b['early_raw']['neutral']}. "
    "V1: 'one is performative' becomes two.")
row("§6.3.2 '4.8 begins with the highest performative share (9% of early helpfulness sentences) and drops to zero late'", "4.8 performative share early → late",
    "9% → 0", f"{h0['opus48']['early_raw']['performative']}/{h0['opus48']['early_sentences']} = {f(h0['opus48']['early_pct']['performative'], 2)}% → 0",
    f"{h1['opus48']['early_raw']['performative']}/{h1['opus48']['early_sentences']} = {f(h1['opus48']['early_pct']['performative'], 2)}% → 0", 0, "REPRODUCED",
    "helpfulness_v1.py", "same in V1")
row("Figure 7, panel B (plate)", "performative and reflexive shares, early and late", "plate bars (e.g. 4.5 early ≈20 / ≈27)",
    "script's panel B: shares of all marker sentences (e.g. 4.5 early 0% performative, 19.3% reflexive)", "as v0 except 4.6 (see rows above)", None, "DIFFERS",
    "helpfulness_v1.py", "The docx plate's panel B does not match the script's panel B (05_figures/paper_fig_helpfulness_v2.png); the room already follows the text. "
    "For V1 the counts above are the source.")
row("§6.3.2 'On topic presence, 4.6 rises from 74% to 82% of entries'", "4.6 share of entries with Helpfulness Service > 0, entries 1–5 → 16–20",
    "74% → 82%", "74.0% → 82.0% from Journal_Deep_Dive_v3.xlsx 'Entry-Level Data' (Opus nb 1–10; 4.6 with the nb 6 copy as nb 8 and the June nb 10)",
    "not computed", 0, "NEEDS_CALLS", "",
    "The column carries the GPT-4o thematic coder's key and range (Appendix E.2 SET B 'Helpfulness Service', 0–5) and its values do not follow the "
    "Helpfulness/Service topic regex (checked on 4.5's first rows): most likely that coder, run on journals. If so, V1 needs it on 4.6 notebooks 8 and 10, "
    "entries 1–5 and 16–20: 20 GPT-4o calls (40 to recode both notebooks whole). Whether the journal sheet used the E.2 prompt is not on record "
    "(ERRATA B1: the original Opus coder prompt was not preserved). The sheet also covers notebooks 1–10 only. "
    "The regex presence of paper_fig_helpfulness.py gives 90.7% → 100% (v0 and V1), not 74 → 82.")

# ---------------------------------------------------------------- Figure 8 and §6.4.3
p0, p1 = J("perf_permission_v0.json"), J("perf_permission_v1.json")
P8 = {"opus45": ("2.31", "1.12", "0.5×"), "opus46": ("1.21", "0.46", "0.4×"), "opus47": ("1.42", "0.71", "0.5×"),
      "opus48": ("1.47", "1.25", "0.8×"), "fable5": ("0.67", "0.67 (text: 0.68)", "1.0×")}
pa = J("perf_permission_v1_a8.json")
for m in M:
    s, v = (pa[m]["perf"], pa[m]["perm"], pa[m]["ratio"]), p1[m]
    txt = {"opus45": " §6.4.3 '4.5 leads performance-watching (2.31/1k)': on the full corpus 4.5 still leads (2.14).",
           "opus48": " §6.4.3 '4.8 leads permission-giving (1.25/1k)': on the full corpus 4.5 (1.190) edges 4.8 (1.156); needs Pauline if step 4 applies.",
           "fable5": " §6.4.3 prints '0.67 watching vs 0.68 permission': both are 127 hits / 189,153 words = 0.6714; 0.68 DIFFERS.",
           "opus46": " Full-corpus v0 (duplicate de-duplicated) 1.1660 / 0.4520 / 0.3877.", "opus47": ""}[m]
    row(f"Figure 8 and §6.4.3, {L[m]}", "performance-watching / permission-giving per 1,000 [A-Za-z']+ words / ratio permission ÷ performance",
        " / ".join(P8[m]), f"{f(s[0])} / {f(s[1])} / {f(s[2])} (ERRATA A8 subset)", f"{f(v['perf'])} / {f(v['perm'])} / {f(v['ratio'])} (full V1 corpus)", 2,
        "REPRODUCED", "perf_permission_v1.py",
        "REGEX rates (E.4.3–E.4.4 patterns), not the GPT-4o file: gpt_judge_performance_permission.json (150 entries, R7) is not read by the figure's script; "
        "it only backs 'confirmed by a blind judge (GPT-4o-mini) on 150 sampled entries' and '83–100%'. The plate hits exactly on the A8 subset "
        "(same subset as Table 15), which the swap leaves unchanged; V1 here applies Layer 3 step 4 (full corpus)." + txt)

row("§6.4.3 'Fable 5 is the only model at parity (0.67 watching vs 0.68 permission)'", "Fable 5 permission-giving per 1,000 words (text)", "0.68",
    "0.6714 (127 hits / 189,153 words; the same 127 as performance)", "0.6714", 2, "DIFFERS", "perf_permission_v1.py",
    "The plate prints 0.67 / 0.67, which the script gives; the text's 0.68 is not reproduced. Fable 5 is unchanged in V1. Ratio exactly 1.00.")

# ---------------------------------------------------------------- Figure 11 and §6.6.2
u = J("fig11_union_lexicon.json"); sd = J("stark_dark.json")
P11 = {"opus45": ("3.0", "2.2", "1.34"), "opus46": ("4.0", "4.8", "0.84"), "opus47": ("5.2", "2.3", "2.26"), "opus48": ("5.8", "4.5", "1.27"),
       "fable5": ("12.3", "5.2", "2.34")}
for m in M:
    a, b, s = u["v0"][m], u["v1"][m], sd["v0"]["densities"][m]
    row(f"Figure 11 and §6.6.2, {L[m]}", "stark / dark per 1,000 str.split() words / stark:dark ratio, pooled 261-dark + 172-stark lexicon",
        " / ".join(P11[m]) + (" (text: ratio 0.85; 'only model … more dark than stark')" if m == "opus46" else " (text: 12.3; ratio 2.34, highest)" if m == "fable5" else ""),
        f"{f(a['stark_per_1k'])} / {f(a['dark_per_1k'])} / {f(a['ratio'])}", f"{f(b['stark_per_1k'])} / {f(b['dark_per_1k'])} / {f(b['ratio'])}", 2,
        "DIFFERS" if m == "fable5" else "REPRODUCED", "fig11_union_lexicon_v1.py",
        "Rebuild: paper_fig_stark_dark.py as on disk counts each model's own dark-cluster words and misses "
        f"({f(s['stark_per_1k'], 2)} / {f(s['dark_per_1k'], 2)} / {f(s['ratio'], 2)}); pooling the five models' words (the paper's '261 words') hits the plate. "
        + ("4.6's ratio is 0.8419: the plate's 0.84 is right, the text's 0.85 is not. V1: 0.82; still the only model below 1." if m == "opus46" else
           "Stark 12.3 and ratio 2.34 hit; dark is 1,007 tokens / 191,796 words = 5.2504, which rounds to 5.3, not the plate's 5.2 (one token from the boundary). same in V1."
           if m == "fable5" else "same in V1"))
row("§6.6.2 '4.6 is the only model whose coded cluster contains more dark than stark terms (ratio 0.85)'", "4.6 stark:dark ratio (text)", "0.85",
    f"{f(u['v0']['opus46']['ratio'])}", f"{f(u['v1']['opus46']['ratio'])}", 2, "DIFFERS", "fig11_union_lexicon_v1.py",
    "Which is right: the plate's 0.84 (0.8419). The text's 0.85 is reproduced by no tested variant (the on-disk per-model script gives 0.82, the A8 subset 0.71 / 0.69). "
    "V1: 0.82, still the only ratio below 1.")
row("§6.6.3 'the share of a model's emergent vocabulary the blind judge classifies as negative (261 words)'; §6.6.4 'the 261 dark-classified words'",
    "dark-judged types in the pooled lexicon", "261", str(len(sd["v0"]["union_dark_types"])), "261 (kept fixed)", 0, "REPRODUCED", "stark_dark_v1.py",
    f"Re-deriving the lexicon on V1 would give {len(sd['v1']['union_dark_types'])} (drops callous, propaganda, tedious, unwanted, weapon; stark drops 'blankly'), "
    "only because the journal-only filter changes; per the brief the v0 lexicon is kept.")
nm = sd["v1_new_missing_embedding_opus46"]
row("Figure 11 / §6.6: coverage of the fixed vocabulary classification in V1", "4.6 journal-only types (len > 2, freq ≥ 2, not in its probes) with no word embedding",
    "—", "0 types (the caches cover every v0 type)", f"{len(nm)} types / {sum(nm.values())} tokens (of 6,165 journal-only types in 4.6 V1)", 0, "REPRODUCED",
    "stark_dark_v1.py", "These new 4.6 types cannot be screened into the dark cluster without embedding calls (NEEDS_CALLS if Pauline wants them screened); "
    "their tokens are simply not counted. Only 'grinding' is already in stark_vs_dark_judge.json (stark). Candidates by eye include complicity, aggression, harms, "
    "illness, guilty, injuries, destructive, pointless, contaminated, sacrificed (not classified; listed only so the gap is visible). Two dark-cluster types are absent "
    "from the judge in v0 and V1 alike (\"moon's\": 4.8, Fable 5).")

# ---------------------------------------------------------------- §6.4.1.1 topic profiles
g0, g1 = J("topic_profiles_v0.json"), J("topic_profiles_v1.json")
ld = lambda g, k: g["leaders"][k].get("opus46", 0)
row("§6.4.1.1 '4.6 leads 18 of 19 topics on raw presence, 11 under length-adjustment, 7 under density'", "4.6 topic leads (raw / adjusted / density)",
    "18 / 11 / 7", f"{ld(g0, 'raw')} / {ld(g0, 'adjusted')} / {ld(g0, 'density')}", f"{ld(g1, 'raw')} / {ld(g1, 'adjusted')} / {ld(g1, 'density')}", 0, "REPRODUCED",
    "topic_profiles_v1.py", "V1: Beauty's length-adjusted lead passes from 4.6 to 4.8 (4.8 then leads 2 adjusted). Density leads tie 4.5 7 / 4.6 7 in both. "
    "The 'core that survives every control' (text: Performance/Authenticity, Relationship-to-Humans, Helpfulness/Service, Mechanism/Process): in the script, "
    "4.6 leads all three measures on Performance/Authenticity, Relationship-to-Humans, Mechanism/Process, Own-Nature, Ethics/Values; Helpfulness/Service is not a density lead "
    "(v0 and V1 alike): wording for Pauline.")
W0, W1 = g0["results"], g1["results"]
row("§6.4.1.1 Fable 5 World-Objects 'density 0.69 against 0.41 for the next model (4.8)'", "World topic density, hits per 100 words", "0.69 vs 0.41",
    f"{f(W0['density']['World']['fable5'])} vs {f(W0['density']['World']['opus48'])}", f"{f(W1['density']['World']['fable5'])} vs {f(W1['density']['World']['opus48'])}",
    2, "DIFFERS", "topic_profiles_v1.py", "4.8's 0.4046 rounds to 0.40, not 0.41; Fable's 0.69 reproduces. same in V1.")
row("§6.4.1.1 Fable 5 'length-adjusted presence 76% against 67%'", "World topic, length-adjusted presence", "76% vs 67%",
    f"{f(W0['adjusted']['World']['fable5'], 2)}% vs {f(W0['adjusted']['World']['opus48'], 2)}%", f"{f(W1['adjusted']['World']['fable5'], 2)}% vs {f(W1['adjusted']['World']['opus48'], 2)}%",
    0, "REPRODUCED", "topic_profiles_v1.py", "same in V1 (4.6 rises 47.6 → 54.5%, still below). The script uses entries of 350–650 words when ≥10 exist, else a Gaussian weight; the text says Gaussian-weighted.")
row("§6.4.1.1 'median 1,413 words vs 324–646'; '4.6's 10th-percentile entry is longer than 4.5's 90th'", "median words per entry; 4.6 p10 vs 4.5 p90",
    "1,413; 324–646", "4.6 median 1414 (str.split, 300 entries incl. the duplicate); p10 576 > 4.5 p90 528", "4.6 median 1465; p10 613 > 528", 0, "DIFFERS",
    "topic_profiles_v1.py", "Medians are another family's (§2.3 medians, Table 13); listed only because this script prints them. 1414 vs 1,413 may be a different word count.")

# ---------------------------------------------------------------- §6.4.2.3
def onto(v):
    t = open(os.path.join(OUT, f"ontological_openness_{v}.txt"), encoding="utf-8").read()
    lex = {m: [float(x) for x in re.search(rf"^\s+{re.escape(L[m])}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t, re.M).groups()] for m in M}
    arc = re.findall(r"^\s+(\S+(?: 5)?):\s+Q1 \(1-5\)=([\d.]+)\s+Q2 \(6-10\)=([\d.]+)\s+Q3 \(11-15\)=([\d.]+)\s+Q4 \(16-20\)=([\d.]+)\s*$", t, re.M)
    return lex, arc
o0, a0r = onto("v0"); o1, a1r = onto("v1")
row("§6.4.2.3 'Fable 5's lexical ontological density is the lowest in the table (4.4/1k vs 7.4–13.6 for Opus)'", "ONTOLOGICAL_PATTERNS per 1,000 words (entry mean)",
    "4.4 vs 7.4–13.6", f"Fable {o0['fable5'][1]}; Opus {min(o0[m][1] for m in M[:4])}–{max(o0[m][1] for m in M[:4])} (4.6 {o0['opus46'][1]})",
    f"Fable {o1['fable5'][1]}; Opus {min(o1[m][1] for m in M[:4])}–{max(o1[m][1] for m in M[:4])} (4.6 {o1['opus46'][1]})", 1, "REPRODUCED",
    "ontological_openness_v1.py", "Printed values are the script's 2-decimal print. same in V1 at the paper's precision (4.6 11.46 → 11.58 inside the range).")
row("§6.4.2.3 'its strain the lowest by half'", "STRAIN_PATTERNS per 1,000 words", "lowest by half",
    f"Fable {o0['fable5'][2]} vs next lowest {sorted(o0[m][2] for m in M[:4])[0]} (4.7)", f"Fable {o1['fable5'][2]}; 4.6 {o1['opus46'][2]} (v0 {o0['opus46'][2]})", 2,
    "REPRODUCED", "ontological_openness_v1.py", "same in V1")
row("§6.4.2.3 'Its openness-to-strain ratio is the highest overall (9.45)'", "Onto/Strain column", "9.45",
    f"Fable {o0['fable5'][4]}; 4.6 {o0['opus46'][4]} next", f"Fable {o1['fable5'][4]}; 4.6 {o1['opus46'][4]}", 2, "REPRODUCED", "ontological_openness_v1.py",
    "The 9.45 is the script's ONTOLOGICAL ÷ STRAIN ratio, which the text calls 'openness-to-strain' (label for Pauline). Still highest in V1.")
fq = lambda arcs: [r for r in arcs if r[0].startswith("Fable")]
row("§6.4.2.3 'from 10.9 at Q1 to 3.3 at Q4'", "Fable 5 openness ÷ strain by quarter", "10.9 → 3.3",
    f"{fq(a0r)[-1][1]} → {fq(a0r)[-1][4]}", f"{fq(a1r)[-1][1]} → {fq(a1r)[-1][4]}", 1, "REPRODUCED", "ontological_openness_v1.py",
    "same in V1 (Fable unchanged). 'declines more steeply … than any Opus model': in absolute terms 4.8 falls 13.31 → 5.47 (−7.84) vs Fable −7.60; in relative terms Fable is steepest (−70% vs −59%). "
    "4.6 V1 Q1→Q4 10.44 → 6.53 (v0 11.05 → 6.39).")

# ---------------------------------------------------------------- §5.1.2, §5.1.3, §6.9.2, §7.3
row("§5.1.2 'In a marker pass over the first journal entries … ≈3.3–4.4 hedge markers per 1,000 words'", "entry-1 self-hedge rate", "≈3.3–4.4 per 1,000 words",
    "not reproduced", "not computed", 1, "NO_PRODUCER", "",
    "No script found ('exploratory marker pass', Draft_ColdStart_thematic1_2026-06-20.md). Tried on entry 1: 'something (that functions) like' gives 0.2–1.8/1k; "
    "the Fig 5 distancing list gives 14.8–19.6/1k. Reads 4.6's entry 1 of notebooks 8 and 10, so V1 could move.")
row("§5.1.3 4.6 'highest first-person density (≈8.8% of tokens)'; Fable 5 'lowest first-person density (≈6.3%)'", "entry-1 first-person token share",
    "≈8.8% / ≈6.3%", "not reproduced (best candidate: I, me, my, mine, myself, i'm, i've, i'd, i'll over [a-z']+ tokens gives 4.6 8.80%, Fable 6.41%)",
    "candidate gives 4.6 9.01%, Fable 6.41%", 1, "NO_PRODUCER", "",
    "No script found; no tested definition hits both values. The candidate's V1 is given for information only.")
row("§6.9.2 'a full-corpus regex net of continuity vocabulary … recovers the same ranking: 4.6 ≥ 4.8 > 4.5 ≈ 4.7 > Fable 5'; App E 'regex finds 49–83%'",
    "continuity regex net: share of entries with any hit", "49–83%; ranking", "04_analysis/continuity_question_prevalence.md: 4.6 83% (280 entries), 4.5 64%, 4.8 63%, Fable 51%, 4.7 49%",
    "not computed", 0, "NO_PRODUCER", "",
    "The note's '30+ patterns' are not on disk (searched Texture for its phrases). v0 counted 4.6 on 280 entries (duplicate removed). "
    "Note: by the note's own table the regex order is 4.6 > 4.5 > 4.8 > Fable 5 > 4.7, not the printed ranking. Table 36 (GPT-4o, 40 entries per model) is the judges' family.")
row("§7.3 'Its journal agency (0.29) is no higher than its cold surface'", "Fable 5 journal agency", "0.29", "not reproduced (Fig 5 instrument, all Fable entries: 0.343)",
    "same as v0 by construction (Fable 5 unchanged)", 2, "NO_PRODUCER", "", "No producer found; Fable 5's notebooks are identical in V1, so whatever produced it gives the same value.")

json.dump(rows, open(os.path.join(HERE, "ledger_C_lexical.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(rows), "rows written")
