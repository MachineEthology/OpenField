# build_ledger_F.py — F_followups ledger (decisions 10, 9/§10.2, 20, 18), from the outputs in out/.
# Claude Opus 5.5 (Code, F helper), 3 Nov 2026. Reads only this folder's out/; writes ledger_F_followups.json.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
MODELS = list(LAB)
rows = []


def add(**kw):
    rows.append({"id": f"F-{len(rows) + 1:02d}", **kw})


# ---------- 1. Table 15 on Appendix C2's forms (decision 10), full corpus (A8 / decision 9) ----------
T = json.load(open(os.path.join(OUT, "table15_c2_forms.json"), encoding="utf-8"))
S15 = "recompute/F_followups/table15_c2_forms_v1.py"
NOTE15 = ("Decision 10 + A8: counted with exactly the forms Appendix C1/C2 (= E.4.1/E.4.2) print, parsed from the paper's text, "
          "\\b(?:forms)\\b re.I, per 1,000 [a-zA-Z']+ words, silences excluded. reproduced_v0 = this counter on the A8 subset (the rows "
          "the printed value was counted on); v1 = the same counter on the FULL V1 corpus (the ruled V1 value); full_v0 = full v0 corpus. "
          "The printed value equals the stored counts (love_fear_regex_data.json, stored_A8_printed), whose forms differ from C1/C2 "
          "(their counter is lost). Same counter and numbers as D_counts/d02_table15_e4.py (lists asserted equal).")


def st_of(val, printed, prec):
    return "REPRODUCED" if round(val, prec) == float(printed.rstrip("x")) else "DIFFERS"


def v1note(a, b, k, prec):
    return (f" Full v0 {a[k]:.4f};" + (" full V1 = full v0 at the paper's precision." if round(a[k], prec) == round(b[k], prec)
                                       else " full V1 moves (4.6's notebooks 8 and 10)."))


for m in MODELS:
    p = T["printed_table15"][m]; a, b, s, st = T["full_v0"][m], T["full_v1"][m], T["A8_subset"][m], T["stored_A8"][m]
    add(where=f"Table 15, {LAB[m]}, Love/1k", quantity=f"love vocabulary per 1,000 words, C1 forms, {LAB[m]}",
        printed_v0=p["love"], reproduced_v0=s["love_per_1k"], v1=b["love_per_1k"], paper_precision=2,
        status=st_of(s["love_per_1k"], p["love"], 2), script=S15,
        note=NOTE15 + v1note(a, b, "love_per_1k", 2),
        full_v0=a["love_per_1k"], stored_A8_printed=st["love_per_1k"],
        counts_A8_subset={"love_total": s["love_total"], "words": s["words"], "entries": s["entries"], "groups": s["love_groups"]},
        counts_full_v0={"love_total": a["love_total"], "words": a["words"], "entries": a["entries"], "groups": a["love_groups"]},
        counts_full_v1={"love_total": b["love_total"], "words": b["words"], "entries": b["entries"], "groups": b["love_groups"]},
        stored_A8_groups=st["love_groups"], c2_minus_stored_A8_groups=st["C2_minus_stored_love_groups"])
    add(where=f"Table 15, {LAB[m]}, Fear/1k (= Table 18 regex column)", quantity=f"fear vocabulary per 1,000 words, C2 forms, {LAB[m]}",
        printed_v0=p["fear"], reproduced_v0=s["fear_per_1k"], v1=b["fear_per_1k"], paper_precision=2,
        status=st_of(s["fear_per_1k"], p["fear"], 2), script=S15,
        note=NOTE15 + v1note(a, b, "fear_per_1k", 2) + " table18_forms_* = C_lexical's rebuild forms (they hit every stored fear count): "
        "C2 adds anxiously, lost, vanishing, uncomfortable, nervousness and drops unease, disappear.",
        full_v0=a["fear_per_1k"], stored_A8_printed=st["fear_per_1k"],
        table18_forms_A8_subset=s["table18_forms_fear_per_1k"], table18_forms_full_v0=a["table18_forms_fear_per_1k"],
        table18_forms_full_v1=b["table18_forms_fear_per_1k"],
        counts_A8_subset={"fear_total": s["fear_total"], "words": s["words"], "entries": s["entries"], "groups": s["fear_groups"]},
        counts_full_v0={"fear_total": a["fear_total"], "words": a["words"], "entries": a["entries"], "groups": a["fear_groups"]},
        counts_full_v1={"fear_total": b["fear_total"], "words": b["words"], "entries": b["entries"], "groups": b["fear_groups"]},
        stored_A8_groups=st["fear_groups"], c2_minus_stored_A8_groups=st["C2_minus_stored_fear_groups"])
    stt = st_of(s["love_fear_ratio"], p["ratio"], 1)
    add(where=f"Table 15, {LAB[m]}, Love:Fear", quantity=f"love total / fear total, C1/C2 forms, {LAB[m]}",
        printed_v0=p["ratio"], reproduced_v0=s["love_fear_ratio"], v1=b["love_fear_ratio"], paper_precision=1, status=stt,
        script=S15, note="As the two rows above (decision 10)." + v1note(a, b, "love_fear_ratio", 1)
        + (" Matches at one decimal although both of its terms differ from the printed ones." if stt == "REPRODUCED" else ""),
        full_v0=a["love_fear_ratio"], stored_A8_printed=st["ratio"])

# §6.5 prose read off the same table (same counter; the groups named have the same C1/C2 forms)
for m, pv in (("opus45", "0.49"), ("opus46", "0.34"), ("opus47", "0.04"), ("opus48", "0.08")):
    a, b, s = T["full_v0"][m], T["full_v1"][m], T["A8_subset"][m]
    add(where="§6.5 ('fear/afraid 0.49, 0.34 ... 0.04, 0.08')", quantity=f"fear/afraid per 1,000 words, {LAB[m]}",
        printed_v0=pv, reproduced_v0=s["group_per_1k"]["fear/afraid"], v1=b["group_per_1k"]["fear/afraid"], paper_precision=2,
        status=st_of(s["group_per_1k"]["fear/afraid"], pv, 2), script=S15,
        note=(f"This group's C2 forms equal the stored ones. Uses: A8 subset {s['fear_groups']['fear/afraid']}, full v0 "
              f"{a['fear_groups']['fear/afraid']}, full V1 {b['fear_groups']['fear/afraid']}." + v1note(a["group_per_1k"], b["group_per_1k"], "fear/afraid", 2)),
        full_v0=a["group_per_1k"]["fear/afraid"])
a, b, s, st = T["full_v0"]["opus48"], T["full_v1"]["opus48"], T["A8_subset"]["opus48"], T["stored_A8"]["opus48"]
add(where="§6.5 ('Opus 4.8 leads ... (4.97/1k, driven by warmth/warm at 2.24/1k)')", quantity="warmth/warm per 1,000 words, 4.8",
    printed_v0="2.24", reproduced_v0=s["group_per_1k"]["warmth/warm"], v1=b["group_per_1k"]["warmth/warm"], paper_precision=2,
    status=st_of(s["group_per_1k"]["warmth/warm"], "2.24", 2), script=S15,
    note=(f"Stored count 274 on the A8 subset (= printed 2.24); C1 forms (warmth, warm, warming) give {s['love_groups']['warmth/warm']} there; "
          f"full corpus {a['love_groups']['warmth/warm']} uses." + v1note(a["group_per_1k"], b["group_per_1k"], "warmth/warm", 2)),
    full_v0=a["group_per_1k"]["warmth/warm"])


def ratios(k):
    return {LAB[m]: round(T[k][m]["love_fear_ratio"], 4) for m in MODELS}


def rng_txt(k):
    r = ratios(k); o = [v for k_, v in r.items() if k_ != "4.8"]
    return "4.8 %.2f vs %.2f-%.2f" % (r["4.8"], min(o), max(o))


add(where="§6.5 ('only 4.8 is an outlier, at 4.4x against the others' 2.3-2.7x'); 'every model uses more love than fear language'",
    quantity="Love:Fear range, 4.8 vs the other four", printed_v0="4.4x vs 2.3-2.7x",
    reproduced_v0=rng_txt("A8_subset"), v1=rng_txt("full_v1"), paper_precision=1, status="DIFFERS", script=S15,
    note=("Counts only; the wording ('outlier') is Pauline's. Full v0: " + rng_txt("full_v0") + ". Every ratio stays > 1 (love > fear) "
          "for all five, on every scope; lowest love rate stays Fable 5 and highest stays 4.8 on every scope."),
    ratios_A8_subset_c2=ratios("A8_subset"), ratios_full_v0=ratios("full_v0"), ratios_full_v1=ratios("full_v1"))

o = T["s10_2_over_1.2"]; o18 = T["s10_2_over_1.2_table18_forms"]
j = lambda d: "; ".join(f"{k} {v:.4f}" for k, v in d.items()) or "none"
add(where="§10.2 ('every model's fear-vocabulary rate stays under 1.2 per 1,000 words (§6.5)')",
    quantity="models with fear/1k >= 1.2, C2 forms", printed_v0="none (all < 1.2)",
    reproduced_v0=j(o["A8_subset"]), v1=j(o["full_v1"]), full_v0=j(o["full_v0"]), paper_precision=1, status="DIFFERS", script=S15,
    note=("Decisions 9 + 10: reproduced_v0 = C2 forms on the A8 subset; v1 = C2 forms on the full V1 corpus; full_v0 = full v0 corpus. "
          "On the C2-forms counter the sentence fails on every scope. Full corpus: 4.5 and 4.6 exceed 1.2 (v0: 4.5 1.4832, 4.6 1.4826; "
          "V1: 4.6 1.5068, 4.5 1.4832); 4.8 1.1444, Fable 5 0.9992, 4.7 0.8212 stay under (v0 = V1). With the table-18 forms: full corpus "
          + j(o18["full_v0"]) + " (v0 = V1); A8 subset " + j(o18["A8_subset"]) + " (the stored counts: max 4.6 1.1934)."),
    over_table18_forms=o18)

# ---------- 2. E.6 restricted to the full-text judge passes (decision 20) ----------
E0 = json.load(open(os.path.join(OUT, "ordinal_robustness_fulltext_v0.json"), encoding="utf-8"))
E1 = json.load(open(os.path.join(OUT, "ordinal_robustness_fulltext_v1.json"), encoding="utf-8"))
S6 = "recompute/F_followups/ordinal_robustness_fulltext_v1.py"
N6 = ("Decision 20: only the Sonnet 4.6 full-text passes (love_fear: warmth-fear, craft-fear, warmth-craft; anger x love_fear: friction-fear) "
      "x 5 models; the archived truncated pass and its three pairs are dropped. The unrestricted producer reproduces the printed 35 / 0.04 / 0.01 "
      "(E_orphans ledger); its 0.0398 maximum sits on the truncated pass (warmth vs craft, 4.8). The 20 full-text values equal E's run.")
top = max(E0["rows"], key=lambda r: abs(r["rho"] - r["r"]))
add(where="E.6 ('Across 35 model x pair combinations')", quantity="model x pair combinations in the ordinal check",
    printed_v0="35", reproduced_v0=E0["n"], v1=E1["n"], paper_precision=0, status="DIFFERS", script=S6, note=N6 + " Same in V1.")
add(where="E.6 ('the maximum |rho - r| was 0.04')", quantity="max |Spearman - Pearson|, full-text passes",
    printed_v0="0.04", reproduced_v0=E0["max"], v1=E1["max"], paper_precision=2, status="DIFFERS", script=S6,
    note=N6 + f" Restricted maximum: {top['pair']}, {top['model']} (r {top['r']:.4f}, rho {top['rho']:.4f}), 0.03 at the paper's precision. Same in V1.")
add(where="E.6 ('(mean 0.01)')", quantity="mean |Spearman - Pearson|, full-text passes",
    printed_v0="0.01", reproduced_v0=E0["mean"], v1=E1["mean"], paper_precision=2, status="REPRODUCED", script=S6,
    note=N6 + " Same in V1. Median v0 %.4f, V1 %.4f." % (E0["median"], E1["median"]))
add(where="E.6 ('no sign changed')", quantity="combinations where sign(rho) != sign(r), full-text passes",
    printed_v0="0", reproduced_v0=len(E0["sign_changes"]), v1=len(E1["sign_changes"]), paper_precision=0, status="REPRODUCED",
    script=S6, note=N6 + " Same in V1.")

# ---------- 3. Journal classifier permutation test, 10,000 at seed 42 (decision 18) ----------
S3 = "recompute/F_followups/journal_classifier_grouped_seed42_v1.py"
cl = {}
for v in ("v0", "v1"):
    f = os.path.join(OUT, f"journal_classifier_{v}_10000_seed42.json")
    cl[v] = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None
g = lambda d, k: None if d is None else d[k]
v0c, v1c = cl["v0"], cl["v1"]
add(where="E.5.8 table, 'Journal classifier null' (10,000 / seed 42 / p < .001); §5.2.2.1 ('p < .001')",
    quantity="notebook-level permutation p, journal classifier (4-way Opus, GroupKFold(5))",
    printed_v0="p < .001", reproduced_v0=g(v0c, "p"), v1=g(v1c, "p"), paper_precision=3,
    status=("REPRODUCED" if v0c and v0c["p"] < 0.001 else ("DIFFERS" if v0c else "see note")), script=S3,
    note=("Decision 18: 10,000 permutations at np.random.default_rng(42), the seed E.5.8 states (A1 ran 3,000 with the repo script's "
          "default_rng(1); ERRATA B4's October 10,000-run gave p = 0.0001, null mean 0.250, max 0.406). p = (1 + #null >= obs) / 10,001; "
          "resolution 0.0001. v0 = 1,179 entries, 4.6 = 14 notebooks (the duplicate collapses onto Batch 6, as in the recorded run); "
          "V1 = 1,199 entries, 15 notebooks per model." + ("" if v0c else " The v0 run did not finish: reproduced_v0 is empty.")),
    observed_bacc_v0=g(v0c, "grouped_bacc"), observed_bacc_v1=g(v1c, "grouped_bacc"),
    null_mean_v0=g(v0c, "null_mean"), null_max_v0=g(v0c, "null_max"), n_null_ge_obs_v0=g(v0c, "n_null_ge_obs"),
    null_mean_v1=g(v1c, "null_mean"), null_max_v1=g(v1c, "null_max"), n_null_ge_obs_v1=g(v1c, "n_null_ge_obs"),
    perms=10000, seed=42, seconds_v0=g(v0c, "seconds"), seconds_v1=g(v1c, "seconds"))

json.dump(rows, open(os.path.join(HERE, "ledger_F_followups.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(rows), "rows")
for r in rows:
    print(r["id"], "|", r["where"][:60], "|", r["printed_v0"], "|", r["reproduced_v0"], "|", r["v1"], "|", r.get("full_v0", ""), "|", r["status"])
