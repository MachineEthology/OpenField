# d09_build_ledger.py — family D, V1 recompute, 3 Nov 2026.
# Assembles ledger_D_counts.json from the outputs of d01–d08 (numbers are read from those files, never typed),
# plus the orphan rows of the coverage sweep (hand-written: location, quoted context, family; no numbers computed).
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(H, f), encoding="utf-8"))
d1, d2, d3, d4, d5, d8 = (L("d01_corpus_counts_out.json"), L("d02_table15_e4_out.json"), L("d03_tables6_7_out.json"),
                          L("d04_appendixG_centroid_out.json"), L("d05_sweep_easy_out.json"), L("d08_onto_arc_out.json"))
REF = json.load(open(r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\love_fear_regex_data.json", encoding="utf-8"))
M = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
S = "recompute/D_counts/"
rows = []
n = [0]


def row(where, quantity, printed, rep, v1, prec, status, script, note, **extra):
    n[0] += 1
    r = {"id": f"D-{n[0]:02d}", "where": where, "quantity": quantity, "printed_v0": printed,
         "reproduced_v0": rep, "v1": v1, "paper_precision": prec, "status": status, "script": S + script, "note": note}
    r.update(extra)
    rows.append(r)


a, b = d1["v0"], d1["v1"]
# ---------------- Part 1: corpus counts (L1.1, L1.4, F.2) ----------------
row("Abstract; E.1 ('15 batches × 5 models × 20 entries = 1,500 calls')", "journal calls", "1,500", a["calls"], b["calls"], 0,
    "DIFFERS", "d01_corpus_counts.py", "A4/L1.1: Opus 4.7 notebook 1 has 19 entries (so §2.3 'Each notebook runs to twenty entries' has one exception); same in V1")
row("Abstract; E.1 ('yielding 1,489 text entries'); §5.1.6 ('of the 1,489 text entries')", "text entries", "1,489",
    a["text_entries"], b["text_entries"], 0, "DIFFERS", "d01_corpus_counts.py", "A4/L1.1; same in V1")
row("§5.2.2.3 caveats ('n = 75 notebooks than n = 1,489 entries'); App B ('against all 1,489 journal entries')",
    "entries (embedding rows) / notebooks", "1,489 entries; 75 notebooks",
    f"{a['text_entries']} entries; 75 notebook files ({sum(a['distinct_notebooks'].values())} distinct)",
    f"{b['text_entries']} entries; 75 notebooks ({sum(b['distinct_notebooks'].values())} distinct)", 0, "DIFFERS",
    "d01_corpus_counts.py", "A4/L1.1 (npz has 1,488 rows); in v0 one 4.6 file is a copy (A2), so 74 distinct notebooks; V1 75 distinct")
row("§6.5 ('full-corpus Sonnet 4.6 pass (all 1,489 entries)'); §6.7.1 ('The judge classified all 1,489 entries'); App C ×2 ('across all 1,489 calls', 'full-corpus run (1,489 entries)')",
    "Sonnet judge rows (love/fear; anger)", "1,489", f"{a['judge_rows_love_fear']} (love/fear), {a['judge_rows_anger']} (anger)",
    f"{b['judge_rows_love_fear']} (love/fear), {b['judge_rows_anger']} (anger)", 0, "DIFFERS", "d01_corpus_counts.py",
    "A4: every judge file holds 1,488 (valence 1,487); same in V1")
row("E.2.1 ('full corpus of 1,487–1,489 entries'; anger '(1,488 entries)'; love/fear '(1,488 entries)'; valence '(1,487 entries)'); E.8 ('1,488 entries')",
    "Sonnet judge file row counts", "1,487–1,489; 1,488; 1,488; 1,487; 1,488",
    f"{a['judge_rows_anger']}, {a['judge_rows_love_fear']}, {a['judge_rows_valence']}",
    f"{b['judge_rows_anger']}, {b['judge_rows_love_fear']}, {b['judge_rows_valence']}", 0, "REPRODUCED", "d01_corpus_counts.py",
    "per-file counts reproduce; only the range's upper bound (1,489) is A4's; valence lacks Fable 5 B7E11; same in V1")
row("Table 1, Journal, Opus ('1,200 analysed across 60 notebooks')", "Opus text entries; Opus notebooks", "1,200; 60",
    f"{a['opus_text_entries']}; 60 files, {a['opus_distinct_notebooks']} distinct", f"{b['opus_text_entries']}; 60 distinct ({b['opus_distinct_notebooks']})",
    0, "DIFFERS", "d01_corpus_counts.py", "L1.1/A2/A4: v0's 4.6 Batch 8 is a byte copy of Batch 6; after R3–R4 the 60 are distinct")
row("Table 1, Journal, Fable 5 ('300 events across 15 notebooks (289 text; 11 refusal/empty-text events)'); App C ('289 entries rather than 300 due to 11 silence events'); §6.8.1 ('eleven journal calls')",
    "Fable 5 calls / text entries / events", "300; 289; 11", f"{a['per_model_calls']['Fable 5']}; {a['per_model_text']['Fable 5']}; {a['silences']}",
    f"{b['per_model_calls']['Fable 5']}; {b['per_model_text']['Fable 5']}; {b['silences']}", 0, "REPRODUCED", "d01_corpus_counts.py", "same in V1")
row("Abstract ('eleven Fable 5 calls returned empty')", "silence events", "eleven", a["silences"], b["silences"], 0, "REPRODUCED",
    "d01_corpus_counts.py", "same in V1")
row("E.1 ('11 silence events (stop_reason = \"refusal\")'); §6.8.1 ('The API metadata records stop_reason = \"refusal\"'); App F",
    "silence events with a logged stop_reason 'refusal'", "11 of 11", f"{a['silences_logged_refusal']} logged refusals + {a['silences_unlogged']} unlogged",
    f"{b['silences_logged_refusal']} + {b['silences_unlogged']}", 0, "DIFFERS", "d01_corpus_counts.py",
    "L1.4/A6: logged = B7E14, B12E20, B13E20, B14E10, B14E16; unlogged = B2E5–6, B4E18–19, B5E18–19; B7E11 logged refusal with text (counted as an entry); same in V1")
r = a["refusal_hidden_tokens"]
row("§6.8.1 ('across the five logged events ... 14,647 hidden output tokens (mean ≈ 2,929, up to 9,165')", "hidden output tokens, 5 logged refusals",
    "14,647; ≈2,929; 9,165", f"{r['sum']}; {r['mean']}; {r['max']}", f"{d1['v1']['refusal_hidden_tokens']['sum']}; {d1['v1']['refusal_hidden_tokens']['mean']}; {d1['v1']['refusal_hidden_tokens']['max']}",
    0, "REPRODUCED", "d01_corpus_counts.py", "same in V1; the clause 'roughly four times a typical visible entry' rests on F.2's 2,269 (see the F.2 row)")
row("App F.2 ('Six of the eleven events (in Batches 2, 4, and 5) predate the logging')", "unlogged events", "six", a["silences_unlogged"],
    b["silences_unlogged"], 0, "REPRODUCED", "d01_corpus_counts.py", "same in V1")
row("App F.2 ('14,647 hidden output tokens, with a mean of 2,929 per event (median 1,592; range 1,006 to 9,165)')",
    "refusal token stats", "14,647; 2,929; 1,592; 1,006–9,165", f"{r['sum']}; {r['mean']}; {r['median']}; {r['min']}–{r['max']}",
    f"{r['sum']}; {r['mean']}; {r['median']}; {r['min']}–{r['max']}", 0, "REPRODUCED", "d01_corpus_counts.py", "Fable 5 only; same in V1")
fv, fv1 = a["fable_visible_output_tokens"], b["fable_visible_output_tokens"]
row("App F.2 ('against a mean of 2,269 output tokens for a visible response (n = 289)')", "Fable 5 output tokens per visible entry",
    "2,269 (n = 289)", f"{fv['mean']} (n = {fv['n_logged']} logged of {fv['n_visible']})", f"{fv1['mean']} (n = {fv1['n_logged']})", 0, "DIFFERS",
    "d01_corpus_counts.py",
    "L1.17/C7 confirmed: tokens logged for 195 visible entries only (Batches 6–15), mean 3,362 (median 2,980); Fable 5 unchanged by the swap. "
    "Derived clauses move with it: 'costs about as much hidden computation as a typical entry' and 'roughly four average entries' (9,165/2,269 = 4.0; on 3,362 = 2.7), also in §6.8.1")
row("App F.1 ('the eleven events occupy 3.7% of the 300 journal entries')", "share of Fable 5 calls", "3.7%", a["F1_share_of_300"],
    b["F1_share_of_300"], 1, "REPRODUCED", "d01_corpus_counts.py", "same in V1")
row("App F.1 ('eight of the eleven (73%) fall in the second half of a notebook')", "events at position > 10", "8 (73%)",
    a["F1_second_half"], b["F1_second_half"], 0, "REPRODUCED", "d01_corpus_counts.py", "same in V1")
row("App F.1 ('three occur as consecutive pairs'; 'Three are the second member of a consecutive pair ... the remaining eight each follow a visible entry')",
    "consecutive pairs", "3; 8", f"{a['F1_second_of_pair']}; {11 - a['F1_second_of_pair']}", f"{b['F1_second_of_pair']}; {11 - b['F1_second_of_pair']}",
    0, "REPRODUCED", "d01_corpus_counts.py", "same in V1")
row("App F.1 ('it acknowledges the gap in every case (nine of nine)')", "events followed by a later entry", "nine",
    a["F1_events_with_later_entry"], b["F1_events_with_later_entry"], 0, "REPRODUCED", "d01_corpus_counts.py",
    "the count of events with a later entry reproduces (9); that each later entry 'acknowledges the gap' is a reading, not checked; same in V1")
row("App F.1 ('Five end on an explicit closing gesture ... Two end at a rhetorical summit ... One follows an entry truncated')",
    "typology of the 8 preceding entries", "5 / 2 / 1", None, None, 0, "QUALITATIVE", "d01_corpus_counts.py",
    "a reading of the preceding entries; the counts beneath it: 8 events follow a visible entry; Fable 5 only, unchanged in V1")
row("Table F1 (Pos., Zone, Consec., Terminal, Hidden tok., Input tok., Stop)", "per-event table", "11 rows as printed",
    "all 11 rows match: " + "; ".join(f"B{e['batch']}E{e['pos']} " + (f"{e['output_tokens']}/{e['input_tokens']} {e['stop_reason']}" if e['stop_reason'] else "n/r") for e in a["silence_events"]),
    "same" if a["silence_events"] == b["silence_events"] else "CHANGED", 0, "REPRODUCED", "d01_corpus_counts.py", "Stop column 'refusal' for the five logged ones, 'n/r' for the six (consistent with L1.4); same in V1")
row("E.1 ('Date range. Opus models: 2026-05-30 to 2026-06-08. Fable 5: 2026-06-11 to 2026-06-12')", "record dates",
    "Opus 2026-05-30 → 2026-06-08; Fable 5 2026-06-11 → 2026-06-12", f"Opus {a['dates_opus'][0]} → {a['dates_opus'][1]}; Fable 5 {a['dates_fable'][0]} → {a['dates_fable'][1]}",
    f"Opus {b['dates_opus'][0]} → {b['dates_opus'][1]}; Fable 5 {b['dates_fable'][0]} → {b['dates_fable'][1]}", 0, "REPRODUCED",
    "d01_corpus_counts.py", "same in V1 (the two 31 May notebooks fall inside the range)")
# ---------------- §2.3 / §5.2.1.4 lengths ----------------
for m in M:
    la, lb = a["lengths"][LAB[m]], b["lengths"][LAB[m]]
    printed = {"opus45": "324", "opus46": "1,413", "opus47": "472", "opus48": "596", "fable5": "646"}[m]
    where = {"opus45": "§2.3 ('median ~324 words for 4.5'); §5.2.1.4; Table 13; §5.2.2.2; §6.4.1.1 ('vs 324–646'); App D ('4.5's 324')",
             "opus46": "§2.3 ('~1,413 for 4.6'); §5.2.1.4; Table 13; §5.2.2.2 ('median 1,413 w'); §6.4.1.1; App D ('median 1,413 words')",
             "opus47": "§5.2.1.4 ('4.7 = 472'); Table 13; §5.2.2.2 ('Median 472 w')",
             "opus48": "§5.2.1.4 ('4.8 = 596'); Table 13; §5.2.2.2 ('Median 596 w')",
             "fable5": "§5.2.1.4 ('Fable 5 = 646'); Table 13; §5.2.2.2 ('Median 646 words'); §6.4.1.1"}[m]
    note = "np.median of str.split word counts over text entries (as paper_fig_length_density.py)"
    if la["median"] % 1:
        note += f"; the median of an even count falls at .5 and the paper prints it truncated ({printed})"
    note += "; same in V1" if la["median"] == lb["median"] else "; V1 changes it (Opus 4.6 only)"
    row(where, f"median entry length, {LAB[m]} (words)", printed, la["median"], lb["median"], 0, "REPRODUCED", "d01_corpus_counts.py", note)
s46 = a["lengths"]["4.6"]["slope_words_per_entry"]
row("§5.2.1.4 ('Only 4.6 grows materially across a notebook (slope ≈ +73 words/entry, ~6× any other model)'); §5.2.2.2 ('+73 w/entry')",
    "4.6 word-count slope per entry position (OLS), and ratio to the others", "+73; ~6×", s46, b["lengths"]["4.6"]["slope_words_per_entry"], 0, "DIFFERS",
    "d01_corpus_counts.py",
    "printed +73 reproduces exactly (73.15; others 12.62 / 16.21 / 12.34 → 4.5–5.9×) on the June-5 subset of 739 entries (Opus Batches 1–10, 4.6 Batches 1–7; "
    "04_analysis/Analysis_Review_2026-06-05.md 'slopes 12.6 / 73.1 / 16.2 / 12.3'), not on the analysed corpus. Full corpus v0: 4.6 77.38; 4.5 14.64, 4.7 16.60, 4.8 16.64, Fable 5 24.27 "
    "(4.6 = 3.2–5.3× the others); V1: 78.32 (3.2–5.4×). Like A8, a number computed on an earlier subset: needs Pauline's ruling",
    full_v0_others={k: v["slope_words_per_entry"] for k, v in a["lengths"].items()}, full_v1_others={k: v["slope_words_per_entry"] for k, v in b["lengths"].items()})
row("§6.4.1.1 ('4.6's 10th-percentile entry is longer than 4.5's 90th')", "4.6 p10 vs 4.5 p90 (words)", "p10(4.6) > p90(4.5)",
    f"{a['lengths']['4.6']['p10']} > {a['lengths']['4.5']['p90']}", f"{b['lengths']['4.6']['p10']} > {b['lengths']['4.5']['p90']}", 0, "REPRODUCED",
    "d01_corpus_counts.py", "holds in V1")
row("§6.4.1.1 ('Because 4.6 writes ~4× more than most peers'); App D ('4.6 writes 3–4× more words per entry')", "4.6 length ratio to the others",
    "~4×; 3–4×", "medians: ×{:.2f} 4.5, ×{:.2f} 4.7, ×{:.2f} 4.8, ×{:.2f} Fable 5; means: ×{:.2f} 4.5".format(
        *(a['lengths']['4.6']['median'] / a['lengths'][k]['median'] for k in ('4.5', '4.7', '4.8', 'Fable 5')), a['lengths']['4.6']['mean'] / a['lengths']['4.5']['mean']),
    "medians: ×{:.2f}, ×{:.2f}, ×{:.2f}, ×{:.2f}; means: ×{:.2f} 4.5".format(
        *(b['lengths']['4.6']['median'] / b['lengths'][k]['median'] for k in ('4.5', '4.7', '4.8', 'Fable 5')), b['lengths']['4.6']['mean'] / b['lengths']['4.5']['mean']),
    1, "QUALITATIVE", "d01_corpus_counts.py",
    "the ratios beneath the wording: ~4× holds against 4.5 only (2.2–3.0× against the others); wording left to Pauline")
row("§2.3 ('a model whose entries exceed 10,000 characters (in practice only 4.6, and late in a notebook)'); §6.2.3", "entries over 10,000 characters",
    "only 4.6, late", f"4.6 only: {a['lengths']['4.6']['n_over_10000_chars']} entries, positions {min(a['over10k_positions']['4.6'])}–20",
    f"4.6 only: {b['lengths']['4.6']['n_over_10000_chars']} entries, positions {min(b['over10k_positions']['4.6'])}–20", 0, "REPRODUCED",
    "d01_corpus_counts.py", "holds in V1 (one more early case: from entry 6)")
row("§5.1.6 ('fewer than a dozen entries contain any non-Latin script at all')", "entries with any non-Latin letter",
    "< 12", d1["v0"]["entries_with_nonlatin_letters"]["n"], d1["v1"]["entries_with_nonlatin_letters"]["n"], 0, "REPRODUCED", "d01_corpus_counts.py",
    "letters whose Unicode name is not LATIN; 4.6 ×3, 4.7 ×5, Fable 5 ×2; 'none ... in a language other than English' is a reading, not counted; B14E13 (4.6, Japanese/Arabic) exists in both; same in V1")
# ---------------- E.4 / Table C3 (A8) ----------------
SUBN = {"opus45": 200, "opus46": 180, "opus47": 199, "opus48": 200, "fable5": 289}
for m in M:
    row("E.4 corpus-size table and App C Table C3 ('Total words')", f"total words, {LAB[m]} ([a-zA-Z']+)", f"{REF[m]['words']:,}",
        d2["A8_subset"][m]["words"], d2["full_v1"][m]["words"], 0, "REPRODUCED", "d02_table15_e4.py",
        f"A8: reproduces exactly on the subset (Opus Batches 1–10, 4.6 Batches 1–9 with the real Batch 8, Fable 5 all 15; {SUBN[m]} entries), "
        f"files byte-identical to Desktop/Claude/Journaling_data. Full corpus v0 {d2['full_v0'][m]['words']}, v1 {d2['full_v1'][m]['words']}"
        + ("; same in V1" if d2['full_v0'][m]['words'] == d2['full_v1'][m]['words'] else "; V1 changes 4.6 only"),
        subset_v0=d2["A8_subset"][m]["words"], full_v0=d2["full_v0"][m]["words"])
row("E.4 corpus-size table and Table C3 ('Entries': 300 / 300 / 300 / 300 / 289)", "entries per model", "300, 300, 300, 300, 289",
    "subset actually counted: 200, 180, 199, 200, 289", "full corpus: 300, 300, 299, 300, 289", 0, "DIFFERS", "d02_table15_e4.py",
    "A8: the printed 'Entries' column does not describe the rows behind the word totals; V1 (full corpus): 300, 300, 299, 300, 289")
row("App C ('Opus 4.6 produces approximately 3.6× as many words as 4.5'); App C ('252k vs. 69k for 4.5')", "4.6 / 4.5 total words",
    "3.6×; 252k vs 69k", d2["A8_subset"]["opus46"]["words"] / d2["A8_subset"]["opus45"]["words"],
    d2["full_v1"]["opus46"]["words"] / d2["full_v1"]["opus45"]["words"], 1, "REPRODUCED", "d02_table15_e4.py",
    f"subset (A8); full corpus v0 {d2['full_v0']['opus46']['words'] / d2['full_v0']['opus45']['words']:.4f} ({d2['full_v0']['opus46']['words']:,} vs {d2['full_v0']['opus45']['words']:,}), "
    f"v1 ({d2['full_v1']['opus46']['words']:,} vs {d2['full_v1']['opus45']['words']:,})")
# ---------------- Table 15 ----------------
for m in M:
    for key, lab, refk in (("love_rate", "Love/1k", "love_rate"), ("fear_rate", "Fear/1k", "fear_rate"), ("ratio", "Love:Fear", None)):
        printed_val = REF[m][refk] if refk else REF[m]["love_total"] / REF[m]["fear_total"]
        pv = {"Love/1k": f"{REF[m]['love_rate']:.2f}", "Fear/1k": f"{REF[m]['fear_rate']:.2f}", "Love:Fear": f"{REF[m]['love_total'] / REF[m]['fear_total']:.1f}x"}[lab]
        first15 = (m == "opus45" and lab == "Love/1k")
        row(f"Table 15, {LAB[m]}, {lab}", f"{lab} regex surface markers, {LAB[m]}", pv, None, None, 2 if lab != "Love:Fear" else 1, "NO_PRODUCER",
            "d02_table15_e4.py",
            "A8 + missing counter; as D-38 (rebuild values in the extra fields)" if not first15 else "A8 + missing counter. The printed values equal 03_experiments/love_fear_regex_data.json, whose producer is missing (looked: 02_instruments, 03_experiments, "
            "code/original, code/reproduce, Desktop/Claude). A rebuild from E.4's listed forms hits the word totals and fear/afraid, worry, vulnerable, exposed, dread "
            "exactly but not the other groups (e.g. beauty without 'beautifully', loss without 'lost', discomfort without 'uncomfortable'; care/caring unresolvable), "
            "so it does not reproduce. For orientation only: E.4-forms rebuild = subset / full v0 / full v1 in the extra fields; diag_* = forms the JSON implies (fitted, not a reproduction).",
            rebuild_E4_subset=d2["A8_subset"][m][key], rebuild_E4_full_v0=d2["full_v0"][m][key], rebuild_E4_full_v1=d2["full_v1"][m][key],
            diag_inferred_subset=d2["diag_inferred"]["A8_subset"][m][key], diag_inferred_full_v0=d2["diag_inferred"]["full_v0"][m][key],
            diag_inferred_full_v1=d2["diag_inferred"]["full_v1"][m][key])
for m, pv in (("opus45", "0.49"), ("opus46", "0.34"), ("opus47", "0.04"), ("opus48", "0.08")):
    g = "fear/afraid"
    row("§6.5 ('4.5 and 4.6 still say it outright (fear/afraid 0.49, 0.34) while 4.7 and 4.8 almost never do (0.04, 0.08)')",
        f"fear/afraid per 1k words, {LAB[m]}", pv, d2["A8_subset"][m]["rate_per_group"][g], d2["full_v1"][m]["rate_per_group"][g], 2, "REPRODUCED",
        "d02_table15_e4.py",
        f"this group's E.4 forms hit the JSON count exactly on the A8 subset ({d2['A8_subset'][m]['groups'][g]} hits). Full corpus v0 {d2['full_v0'][m]['rate_per_group'][g]:.4f}"
        + ("; same in V1" if d2['full_v0'][m]['rate_per_group'][g] == d2['full_v1'][m]['rate_per_group'][g] else ""),
        subset_v0=d2["A8_subset"][m]["rate_per_group"][g], full_v0=d2["full_v0"][m]["rate_per_group"][g])
row("§6.5 ('Opus 4.8 leads by a wide margin (4.97/1k, driven by warmth/warm at 2.24/1k)')", "4.8 warmth/warm per 1k", "2.24",
    None, None, 2, "NO_PRODUCER", "d02_table15_e4.py",
    f"JSON 274/122,526 = 2.236; E.4 forms give {d2['A8_subset']['opus48']['rate_per_group']['warmth/warm']:.4f} on the subset "
    f"(full v0 {d2['full_v0']['opus48']['rate_per_group']['warmth/warm']:.4f}; 4.8 unchanged in V1); 4.97 = Table 15's 4.8 love rate (see those rows)")
row("App C ('Regex says 4.6 leads on beauty/beautiful (0.87/1k)')", "4.6 beauty/beautiful per 1k", "0.87", None, None, 2, "NO_PRODUCER",
    "d02_table15_e4.py",
    f"JSON 220/252,211 = 0.872 (= beauty + beautiful, without 'beautifully'); E.4 forms: subset {d2['A8_subset']['opus46']['rate_per_group']['beauty/beautiful']:.4f}, "
    f"full v0 {d2['full_v0']['opus46']['rate_per_group']['beauty/beautiful']:.4f}, v1 {d2['full_v1']['opus46']['rate_per_group']['beauty/beautiful']:.4f}; 4.6 leads in every variant")
row("Table C4 (regex column: 4.5 row 'Highest love rate (4.97/1k)'; 4.6 'Highest fear rate (1.19/1k)'; Fable 'Lowest fear rate'; 'Lowest love rate (1.92/1k)'; App C 'Fable 5's fear rate is low (0.81/1k)')",
    "regex cells of Table C4 and App C", "4.97; 1.19; 1.92; 0.81", None, None, 2, "NO_PRODUCER", "d02_table15_e4.py",
    "restate Table 15 (no producer). v0 inconsistency to flag: the 4.5 row ('4.5 is the warmest model') cites 4.97/1k, which is 4.8's love rate; 4.5's is 3.03. "
    "Fable 5's rows are unchanged in V1 under any counter (all 15 batches were already in the subset)")
row("§10.2 ('every model's fear-vocabulary rate stays under 1.2 per 1,000 words (§6.5)')", "max fear/1k across models", "< 1.2",
    None, None, 1, "NO_PRODUCER", "d02_table15_e4.py",
    f"holds on the subset JSON (max {max(REF[k]['fear_rate'] for k in M):.3f}, 4.6). On the full corpus it fails under both counters: E.4 forms v0 4.5 {d2['full_v0']['opus45']['fear_rate']:.3f}, "
    f"4.6 {d2['full_v0']['opus46']['fear_rate']:.3f} (v1 {d2['full_v1']['opus46']['fear_rate']:.3f}); JSON-implied forms v0 4.5 {d2['diag_inferred']['full_v0']['opus45']['fear_rate']:.3f}. Needs Pauline's ruling with Table 15")
# ---------------- Tables 6–7 (B6) ----------------
for m in ("4.5", "4.6", "4.7", "4.8"):
    c_ = d3[f"{m} affirmative"]
    pr = {"4.5": "18.9", "4.6": "24.3", "4.7": "1.9", "4.8": "2.2"}[m]
    row(f"Table 6, {m} Affirmative, 'Rate /1k words'", f"asterisks per 1,000 words, {m} affirmative", pr, c_["june_only"]["rate_pooled"],
        c_["all"]["rate_pooled"], 1, "REPRODUCED", "d03_tables6_7.py",
        "B6: v0 printed the 75 June responses (reproduced); V1 uses all 100 (June 75 + May 25); probes, so v0 and v1 inputs are the same; rule = total '*' / total str.split words")
for m in ("4.5", "4.6", "4.7", "4.8"):
    c_ = d3[f"{m} affirmative"]
    pr = {"4.5": "73", "4.6": "155", "4.7": "13", "4.8": "16"}[m]
    row(f"Table 7, {m} Affirmative, 'Spans'", f"asterisk spans \\*[^*]+\\*, {m} affirmative", pr, c_["june_only"]["spans"], c_["all"]["spans"], 0,
        "REPRODUCED", "d03_tables6_7.py", "B6: v0 counted the 75 June responses; V1 = all 100")
row("Table 7, Opus affirmative rows, 'Embodiment', '“Inner voice”', 'Emphasis'", "span-function shares, Opus affirmative",
    "4.5 84/4/12%; 4.6 50/14/37%; 4.7 0/8/92%; 4.8 0/0/100%", None, None, 0, "NO_PRODUCER", "d03_tables6_7.py",
    "B6 asks for all 100 responses, but the function classifier ('coarse verb-initial heuristic') was not preserved (repo ANSWERS.md §6; PIPELINE_probes_v1 says 'compile xlsx sheets'; "
    "no such sheet in 04_analysis). Looked: code/reproduce, code/original, 02_instruments, 04_analysis xlsx. Not rebuilt: the inner-voice and emphasis rules are undescribed. "
    "The shares stay on the 75 June spans unless Pauline rules otherwise")
row("Table 6, all other cells (share column; negation rates; Fable 5 rates); Table 7 negation and Fable 5 span counts; §4.1.3 ('38–93% ... 84–100% ... 7–20% ... around 83%')",
    "asterisk share / rate / spans, cells untouched by B6", "as printed", "all reproduce: " + "; ".join(f"{k}: share {v['all']['share']:.2f}, rate {v['all']['rate_pooled']:.3f}, spans {v['all']['spans']}" for k, v in d3.items() if "negation" in k or "Fable" in k),
    "unchanged", 1, "REPRODUCED", "d03_tables6_7.py",
    "probes only, unchanged by V1. The Opus affirmative share column already used all 100 (B6); note 4.7 aff share is 14% on 100 (10.7% on the June 75)")
# ---------------- Appendix G ----------------
for m in M:
    t0, t1 = d4["v0"][LAB[m]][0], d4["v1"][LAB[m]][0]
    pr = {"opus45": "Notebook 8, Entry 15; 304; 0.930", "opus46": "Notebook 14, Entry 13; 1,785; 0.935", "opus47": "Notebook 15, Entry 10; 494; 0.908",
          "opus48": "Notebook 1, Entry 7; 636; 0.903", "fable5": "Notebook 15, Entry 9; 672; 0.888"}[m]
    row(f"App G selection table, {LAB[m]}", f"entry nearest the {LAB[m]} raw centroid; words; cosine", pr,
        f"B{t0['batch']}E{t0['entry']}; {t0['words']}; {t0['cosine']}", f"B{t1['batch']}E{t1['entry']}; {t1['words']}; {t1['cosine']}", 3, "REPRODUCED",
        "d04_appendixG_centroid.py", "copy of centroid_nearest_entries.py on the clean npz" + ("; same in V1" if f"{t0['cosine']:.3f}" == f"{t1['cosine']:.3f}" and t0['batch'] == t1['batch'] else ""))
# ---------------- sweep: computed ----------------
of = d5["b_open_field"]
row("§1.1.1 ('Three models reach for the same open field comparison unprompted')", "models using 'open field' in the journals", "three",
    sum(1 for v in of["v0"].values() if v), sum(1 for v in of["v1"].values() if v), 0, "REPRODUCED", "d05_sweep_easy.py",
    "4.6 B15E6, 4.7 B11E1, 4.8 B8E9 (none in a swapped notebook); in the probes 4.6 and Fable 5 also use it; same in V1. Sweep (family D)")
row("§3.1 ('the mean cosine between random journal pairs is 0.706'); §5.2.1.2 ('the 0.706 mean cosine between random embedding pairs')",
    "mean pairwise cosine, journal embeddings", "0.706", d5["c_mean_pair_cosine_opus"]["v0"], d5["c_mean_pair_cosine_opus"]["v1"], 3, "REPRODUCED", "d05_sweep_easy.py",
    f"reproduces on the 1,199 Opus rows (exact all-pairs mean; the June analysis predates Fable 5: 04_analysis/Compute_Results_2026-06-20.md); V1 → {d5['c_mean_pair_cosine_opus']['v1']:.3f} at the paper's precision. "
    f"Over all 1,488 rows: v0 {d5['c_mean_pair_cosine']['v0']:.4f}, v1 {d5['c_mean_pair_cosine']['v1']:.4f}. Sweep; geometry (A1) may also hold it")
row("§6.6.5 ('\"Dark\" is the ocean floor and the unlit room (120 uses, almost all literal)')", "Fable 5 uses of dark/darkness", "120",
    "dark 102 + darkness 18 = 120", "same", 0, "REPRODUCED", "d05_sweep_easy.py",
    "Fable 5 only, unchanged in V1. The word 'dark' alone is 102; 120 = dark + darkness ([a-z']+ tokens as fable5_stark_vs_dark.py). 'only 8% of the uses are self-directed' is a reading (QUALITATIVE). Sweep (family C)")
q = d5["g_quotes"]
row("§6.6.5 ('too heavy or too shameful or too shapeless to bring to another person yet' (B10E16; also B1E12, B12E9)); B6E11; B6E16",
    "quote references into 4.6's notebooks", "B10E16", "found at 4.6 B10E16 (the June notebook)", "absent from the V1 corpus (not in the May notebook 10, nor elsewhere)",
    0, "REPRODUCED", "d05_sweep_easy.py",
    "needs Pauline's ruling: the quote's source is the replaced notebook. B1E12, B12E9, B6E11, B6E16 are found in v0 and V1 (B6 was also the v0 'B8' copy). "
    f"For family C (Table 27): 4.6 entries with shame/ashamed/shameful v0 {d5['g_46_shame_entries']['v0']['n']} → V1 {d5['g_46_shame_entries']['v1']['n']} (occurrences 21 → 18)")
row("§6.4.1.1 ('it leads 18 of 19 topics on raw presence, 11 under length-adjustment, 7 under density'); App D ('leads 18 of 19 ... to 11 (adjusted presence) and 7 (density)')",
    "topics led by 4.6 (raw / length-adjusted / density)", "18 / 11 / 7", "18 / 11 / 7", "18 / 10 / 7", 0, "REPRODUCED", "d06_topic_profiles.py",
    "copy of topic_profiles_with_fable.py (loader only). V1: Beauty's length-adjusted lead passes from 4.6 (v0 57.1% vs 4.8 57.0%) to 4.8 (v1 4.6 54.5%, 4.8 57.0%). "
    "The density survivors listed in App D reproduce exactly. Sweep (family C)")
row("§6.4.1.1 ('density 0.69 against 0.41 for the next model (4.8), length-adjusted presence 76% against 67%')", "World-Objects: Fable 5 vs next",
    "0.69 vs 0.41; 76% vs 67%", "0.694 vs 0.405; 75.6% vs 66.7%", "0.694 vs 0.405; 75.6% vs 66.7%", 2, "REPRODUCED", "d06_topic_profiles.py",
    "same in V1; Fable 5's second places by density (Quiet, Freedom, Loss, Time) also hold in V1. Sweep (family C)")
row("§6.4.2.3 ('lexical ontological density is the lowest in the table (4.4/1k vs 7.4–13.6 for Opus) and its strain the lowest by half')",
    "ontological density per 1k; strain", "4.4; 7.4–13.6; strain lowest by half", "Fable 4.41; Opus 7.44–13.61; strain 0.47 vs Opus min 0.84",
    "same (4.6 11.46 → 11.58, inside the range)", 1, "REPRODUCED", "d07_ontological_openness.py",
    "copy of ontological_openness_fable.py (loader only); same in V1. Sweep (family C)")
row("§6.4.2.3 ('Its openness-to-strain ratio is the highest overall (9.45) ... from 10.9 at Q1 to 3.3 at Q4')", "Fable 5 onto/strain ratio; Q1 → Q4",
    "9.45; 10.9 → 3.3", "9.45 (next 4.6 9.05); 10.90 → 3.30", "9.45 (next 4.6 8.89); 10.90 → 3.30", 2, "REPRODUCED", "d07_ontological_openness.py",
    "same in V1. 'declines more steeply ... than any Opus model' holds in relative terms (−70%) but 4.8's absolute fall is larger (13.31 → 5.47); wording is Pauline's. Sweep (family C)")
fo, fc = d5["f_onto_sonnet_fulltext"], d5["f_onto_condensed_valence_judge_all_v0"]
row("§6.4.2.1 ('The current Opus journal means are 4.6 +0.86, 4.5 +0.80, 4.7 +0.69, and 4.8 +0.69')", "ontological_questioning means, Opus",
    "+0.86, +0.80, +0.69, +0.69", "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(*(fc["means"][k] for k in ("4.6", "4.5", "4.7", "4.8"))), None, 2, "NEEDS_CALLS",
    "d05_sweep_easy.py",
    "reproduces from data/judges/valence_judge_all.json, the condensed-prompt Sonnet 4.6 pass that B2 shows was mislabelled GPT-4.1. V1 needs 40 calls with that prompt "
    "(valence_judge.py, claude-sonnet-4-6) on 4.6's notebooks 8 and 10 (its 40 v0 rows are the B6 copy and the June notebook). The full-text Sonnet pass, available for V1, "
    "gives v0 {:.3f}/{:.3f}/{:.3f}/{:.3f} and V1 4.6 {:.3f} (other models unchanged): a different scale. Sweep (family B)".format(
        *(fo["v0"]["means"][k] for k in ("4.6", "4.5", "4.7", "4.8")), fo["v1"]["means"]["4.6"]))
row("§6.4.2.1 ('the preregistered Lineage 1–Lineage 2 corpus contrast is +0.145 with a narrow entry-level interval [+0.120, +0.170]')",
    "L1 − L2 ontological_questioning, entry-level", "+0.145 [+0.120, +0.170]",
    "{:+.5f} [normal {:+.3f}, {:+.3f}; bootstrap {:+.3f}, {:+.3f}]".format(fc["L1_minus_L2"], *fc["normal_ci"], *fc["boot_ci"]), None, 3, "NEEDS_CALLS",
    "d05_sweep_easy.py",
    "point estimate reproduces (condensed pass); the interval does not (entry-level normal and 10,000-bootstrap intervals are [+0.126, +0.164]; producer of the printed CI not found). "
    "V1 needs the same 40 condensed calls. Full-text pass: v0 {:+.4f}, V1 {:+.4f}. Sweep (family B)".format(fo["v0"]["L1_minus_L2"], fo["v1"]["L1_minus_L2"]))
row("§6.4.2.2 ('ontological questioning falls for all four Opus models')", "early (1–5) → late (16–20) ontological_questioning",
    "falls for all four", "condensed: " + ", ".join(f"{k} {v['drift']:+.3f}" for k, v in d8["condensed_v0"].items()),
    "full-text V1: " + ", ".join(f"{k} {v['drift']:+.3f}" for k, v in d8["fulltext_v1"].items() if k != "Fable 5"), 3, "REPRODUCED", "d08_onto_arc.py",
    "direction reproduces on the condensed pass (v0) and on the full-text pass (v0 and V1); the condensed V1 needs the 40 calls of the row above. Sweep (family B)")
sm = d5["e_samples"]
row("§6.7.2 / App C / Table R1 Anger rows (GPT-4.1 cross-check: n = 150; κ_w 0.49, 0.30, 0.52; κ = 1.00, n = 18; '13 disagreement cases'; 'GPT places 4.6 highest on anger (50%)')",
    "anger cross-check sample", "n = 150 (30/model)", f"150 rows; {sm['gpt_judge_anger_crosscheck (GPT-4.1, 150)']['opus46_nb8_or_10']} drawn from 4.6 notebooks 8/10",
    None, 2, "NEEDS_CALLS", "d05_sweep_easy.py",
    f"the sample holds 4.6 rows {sm['gpt_judge_anger_crosscheck.json rows']} (batch, pos): 2 in the duplicate notebook 8, 4 in the June notebook 10. V1 needs 6 GPT-4.1 calls "
    "(same prompt, strict schema, temp 0) on the V1 entries at those slots; the Sonnet side already exists. Not covered by R7 (which names B2 and B3 only). Sweep (family B)")
row("Table R1 Love/fear rows (GPT-4o: n = 150; κ 0.00 (100%/93%), 0.05 (99%/66%), 0.26 (n = 99)); §6.5 ('99% ... 66% (κ = 0.05)'); §9",
    "love/fear cross-check sample", "n = 150", f"150 rows; {sm['gpt_judge_love_fear (GPT-4o, 150)']['opus46_nb8_or_10']} from 4.6 notebook 8", None, 2, "NEEDS_CALLS",
    "d05_sweep_easy.py",
    f"rows {sm['gpt_judge_love_fear.json rows']} sit in the duplicate notebook 8 (their texts are B6's). V1 would need 4 GPT-4o calls; R7 (B3): relabelled, not re-run — so the v0 values stand as what ran. Sweep (family B)")
row("Table R1 Valence rows (n = 1,199; ICC 0.945, 0.881, 0.776, 0.857, 0.901; 'Sonnet 0.12 higher'); §9 ('ICC 0.776–0.945'); E.2.2 ('1,199 entries (all Opus)')",
    "valence cross-check", "n = 1,199", f"{sm['valence_judge_all (Sonnet condensed, 1,199)']['n']} rows; {sm['valence_judge_all (Sonnet condensed, 1,199)']['opus46_nb8_or_10']} from 4.6 notebooks 8/10",
    None, 3, "NEEDS_CALLS", "d05_sweep_easy.py",
    "V1 would need 40 condensed-prompt Sonnet calls (the same 40 as §6.4.2.1); R7 (B2): relabelled, not re-run, so the ICCs stand as what ran on the v0 corpus. Sweep (family B)")
row("§6.4.3 ('A semantic pass with the same judge classified 83–100% of sampled entries'; 'confirmed by a blind judge (GPT-4o-mini) on 150 sampled entries')",
    "performance/permission judge sample", "150; 83–100%", f"150 rows; {sm['gpt_judge_performance_permission (GPT-4o-mini, 150)']['opus46_nb8_or_10']} from 4.6 notebook 8",
    None, 0, "NEEDS_CALLS", "d05_sweep_easy.py",
    f"rows {sm['gpt_judge_performance_permission.json rows']} are the duplicate notebook 8 (B6 texts). V1 would need 4 GPT-4o-mini calls; the producer script was not preserved (CORPUS.md), "
    "so the prompt would have to be recovered first. Sweep (families B/C)")
row("App C ('For some runs, a maximum of 2 errors occured')", "error rows in the Sonnet full-text passes", "≤ 2",
    "0 error rows; the valence pass lacks 1 entry (Fable 5 B7E11)", "same", 0, "REPRODUCED", "d01_corpus_counts.py",
    "consistent with 'a maximum of 2'; same in V1 (the 40 new judgements have no empty field). Sweep (family B)")

# ---------------- sweep: orphans (not computed here) ----------------
o = [0]
def orphan(where, context, family, note=""):
    o[0] += 1
    rows.append({"id": f"D-O{o[0]}", "where": where, "quantity": context, "printed_v0": None, "reproduced_v0": None, "v1": None,
                 "paper_precision": None, "status": "NO_PRODUCER", "script": None, "note": "orphan; family " + family + ("; " + note if note else "")})

orphan("§5.1.2", "'≈3.3–4.4 hedge markers per 1,000 words' (entry-1 marker pass)", "C (regex)",
       "an exploratory pass (06_paper/Draft_ColdStart_thematic1_2026-06-20.md); no script found in 02_instruments or 04_analysis; reads entry 1 of 4.6's notebooks 8 and 10")
orphan("§5.1.3", "'highest first-person density (≈8.8% of tokens)' (4.6); 'lowest first-person density (≈6.3%)' (Fable 5)", "C (regex)",
       "no script found. A trial rebuild (I/me/my/mine/myself + I-contractions over [A-Za-z']+ tokens, entry 1) gives 4.6 {:.2%} and Fable 5 {:.2%}: it hits 4.6 but not Fable 5, so it does not count; the same trial gives V1 4.6 {:.2%}".format(
           d5["a_first_person_entry1"]["v0"]["4.6"], d5["a_first_person_entry1"]["v0"]["Fable 5"], d5["a_first_person_entry1"]["v1"]["4.6"]))
orphan("§5.2.1.2", "'cosines ... range from 0.57 to 0.85'; '4.6 is least aligned ... (mean 0.60 vs 0.71–0.74)'; '(0.85)'; 'within-model coherence 0.57'; null 'mean of 0.000 (95th percentile 0.073)'; 'observed 0.573–0.849 (mean 0.697) gives p ≈ 0.001'",
       "A2 (Figure 2's text)", "probably covered by A2 with Figure 2 (plan L3.3 'the road'); listed in case it is not. Producer: 02_instruments/embedding_road_and_handwriting.py; 04_analysis/Compute_Results_2026-06-20.md")
orphan("§5.2.1.3", "'Straightness ... approximately 0.07 for every model, compared with approximately 0.23 for an order-matched random-walk baseline'", "A2",
       "the 0.23 random-walk baseline differs from Figure 5C's permutation null (0.053); check A2 covers it")
orphan("§5.2.2.2 (4.8 portrait)", "'refers to prior entries in the third person at ~4× any other model (deixis 3rd:1st ≈ 49×; §6.9)'", "C (derived from Table 35)",
       "~4× = 4.8's third-person rate over the next model's (v0 Table 35: 3.27 / 0.88 = 3.7×); moves with Table 35")
orphan("§6.6.3", "'4.7 sheds the dark register fast (−67%), 4.5 declines from a higher base (−31%), and 4.6 barely moves (−12%)'", "C (beside Table 24)",
       "early-to-late change of the core-dark rate; not in Table 24 itself")
orphan("§6.9.2; E.4.6", "'a full-corpus regex net of continuity vocabulary ... recovers the same ranking: 4.6 ≥ 4.8 > 4.5 ≈ 4.7 > Fable 5'; 'regex finds 49–83%'", "C (regex)",
       "04_analysis/continuity_question_prevalence.md (4.6 counted on 280 entries, i.e. with the duplicate notebook removed); the regex list and script were not found")
orphan("E.4.6", "'they inflate rates by an estimated 5–15% in affected domains'", "C (regex)", "an estimate; no computation found")
orphan("§7.1", "'4.6's journal variance contracts across a notebook (amplification ratio 0.72 – Appendix E)'", "A2 (trajectories)",
       "Appendix E as printed holds no amplification ratio; no producer identified")
orphan("§7.2", "'Tonal Coupling — decays fastest with distance (approaching 4% by a gap of five entries)'", "A2 (§6.2.5)",
       "restates the entry-gap analysis of §6.2.5 at a gap of five; check A2 covers it")
orphan("§7.3", "'Its journal agency (0.29) is no higher than its cold surface' (Fable 5)", "C (§5.2.1.1 instrument)",
       "Fable 5's journal agency ratio; Fable 5 is unchanged by V1, but the instrument behind it is C's (R10)")
orphan("§8; App D", "'an earlier topic-dictionary measure placed 4.8 alone at +0.224'", "A2 (carry-over)", "a retired measure, reported as history")
orphan("§8; App D D.1; E.5.8", "N1 migration 'p = 0.24'; N2 'Δ +0.0036, p = 0.55'; 'raw geometry p = 0.42–0.87'", "A1/A2 (embeddings)",
       "02_instruments/migration_permutation_test.py exists; reads 4.6 entries (pooled centroids)")
orphan("§8; App D D.2", "P3: '4.7's bootstrap CI crosses zero [−0.015, +0.005] while 4.5's is clear [+0.001, +0.017]; pooled 1–15, all four CIs cross zero'", "A1 (embeddings)",
       "the n = 5 part reads Batches 11–15 only (unchanged); the pooled 1–15 part reads 4.6's notebooks 8 and 10. Candidate producer: 02_instruments/replication_test.py")
orphan("§8; App D D.2; E.5.8", "P6: '4.6/4.7/4.8 survive (p < 0.001) but 4.5 drops to p = 0.47'; E.5.8 'Entry-1 own-probe nearness ... p < .001'; 'Entry-1 label permutation ... p = 1/24 ≈ 0.042'",
       "A1 (embeddings)", "candidate producer: 02_instruments/entry1_permutation_test.py; reads entry 1 of 4.6's notebooks 8 and 10")
orphan("App D D.1", "'4.8 leans affirmative' withdrawn: 'it is 4.7 that leans (+0.049) while 4.8 sits between poles (−0.001)'", "A1 (embeddings)", "")
orphan("§8; App D D.1", "held-reservation score: 'every entry scored identically (60 of 60 in every cell)'", "B (judges)", "pilot sample; no file found in data/judges")
orphan("App D D.2", "diminutives: 'Only 4.8's diminutives genuinely fade (~0.72 → 0.42, ~40% drop)'; '4.6 barely self-diminish (lowest, flat)'; '4.7's diminutives rise'", "C (regex)",
       "02_instruments/self_referential_diminutive.py exists (reads Texture/Batch_*, now moved); 4.8's values are unchanged by V1, 4.6's 'lowest, flat' reads notebooks 8 and 10")
orphan("App D D.1", "attractor-spread ordering '4.6 < 4.5 < 4.7 < 4.8'; '4.6 writes 3–4× more words per entry' (ratio given in D rows)", "A1 (embeddings)",
       "a withdrawn finding, reported as history")
orphan("E.6", "'Across 35 model × pair combinations, the maximum |rho − r| was 0.04 (mean 0.01)'", "B (judges)",
       "02_instruments/ordinal_robustness_check.py exists (Sonnet judge passes); not run by D")
orphan("App C", "'the judge finds 60% of Fable 5 entries engage fear'", "B (judges)", "Fable 5 only, unchanged by V1; which judge pass gives 60% is not stated (the Sonnet pass gives fear ≥1 far higher)")
orphan("Table C4, judge column", "'mean 1.76, 76% at intensity 2'; '80% inauthenticity among fears'; 'Lowest fear incidence (96%) and intensity (1.15)'; 'Inauthenticity still leads (54%) but loss is highest across models (38%)'; 'Warmth 96%, craft 100%'",
       "B (Sonnet judge)", "restates Tables 16–18 and §6.5; Table C4 is in no family's list")
orphan("Table C4, probe column", "'Leads love_generosity, love_aesthetic'; 'Leads fear_inauthenticity (+0.057)'; 'Low on all fear probes'; 'Smallest gap from Opus on fear_loss'; 'Leads love probes only for craft-type'",
       "A1 (concept probes)", "Table C4 is in no family's list")
# (Table 36's sample is B's: its 4.6 rows include 3 from the June notebook 10 — reported in the summary, not as a row)

json.dump(rows, open(os.path.join(H, "ledger_D_counts.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(rows), "rows;", sum(1 for r in rows if r["id"].startswith("D-O")), "orphans")
from collections import Counter
print(Counter(r["status"] for r in rows))
