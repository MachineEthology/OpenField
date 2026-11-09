"""Builds ledger_E_orphans.json from the outputs in out_v0/ and out_v1/ (helper E, 3 Nov 2026).
Every number below is read from a file written by one of the copied producers in this folder; nothing is typed in
except the printed v0 values (copied from the paper's room, main.json and appendix-d/-e.json).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
O0, O1 = os.path.join(HERE, "out_v0"), os.path.join(HERE, "out_v1")
R = "recompute/E_orphans/"


def J(*p):
    return json.load(open(os.path.join(HERE, *p), encoding="utf-8"))


def txt(*p):
    return open(os.path.join(HERE, *p), encoding="utf-8").read()


def joint46(path):
    m = re.search(r"All 3 -> 4\.6 specifically:\s+(\d+)/10000\s+p=([0-9.]+)", txt(path))
    return (int(m.group(1)) + 1) / 10001


def entry1_test(path):
    s = txt(path)
    pv = {}
    for lab in ("4.5", "4.6", "4.7", "4.8"):
        m = re.search(rf"^\s+{re.escape(lab)}\s+([+-][0-9.]+)\s+([0-9.]+)\s+SURVIVES|^\s+{re.escape(lab)}\s+([+-][0-9.]+)\s+([0-9.]+)\s+NOT", s, re.M)
        pv[lab] = float(m.group(2) or m.group(4))
    a = re.search(r"Joint \(all 4 match own\): (\d+)/(\d+)", s)
    return pv, int(a.group(1)) / int(a.group(2))


rows = []


def row(id_, where, quantity, printed, rep, v1, prec, status, script, note):
    rows.append({"id": id_, "where": where, "quantity": quantity, "printed_v0": printed, "reproduced_v0": rep,
                 "v1": v1, "paper_precision": prec, "status": status, "script": script, "note": note})


# ---------------- D-O13: N1, N2 ----------------
n1 = {(v, b): joint46(os.path.join(f"out_{v}", f"migration_{b}.txt")) for v in ("v0", "v1") for b in ("b1-10", "b1-15")}
row("E-01", "§8 ('they did not; p = 0.24'); App D D.1 (N1, 'p = 0.24'); E.5.8 row 'Migration-to-4.6'",
    "N1 migration toward 4.6: joint label-permutation p that 4.5, 4.7 and 4.8 entry-20 journals all sit nearest the 4.6 probe centroid (10,000 permutations of the 4 centroid labels, seed 42)",
    "p = 0.24", repr(n1[("v0", "b1-10")]), repr(n1[("v1", "b1-10")]), 2, "REPRODUCED",
    R + "migration_permutation_test_v1.py (copy of 02_instruments/migration_permutation_test.py)",
    "same in V1. June run on Batches 1-10 (as the script is written): v0 2431/10000. Full corpus 1-15: v0 and V1 also "
    f"{n1[('v0', 'b1-15')]:.4f}; decone_pset_v1.py RAW N1 gives the same p (0.243) for 11-15 and 1-15. The test reads only "
    "4.5/4.7/4.8 entry-20 vectors and the probe centroids, so 4.6's notebooks never enter it; with all three test models "
    "nearest one centroid (margins 0.0056 / 0.0048 / 0.0017 on 1-10) the p is about 1/4 by construction. Probe vectors: "
    "Texture/03_experiments/probe_embeddings_cache.json, aligned by the June load order (probe_alignment_check.py).")

n2 = {(v, b): J(f"out_{v}", f"n2_original_{b}.json") for v in ("v0", "v1") for b in ("b1-10", "b1-15")}
row("E-02", "§8 ('Δ +0.0036, p = 0.55'); App D D.1 (N2, 'Original Δ +0.0036')",
    "N2 cross-lineage divergence: L1-L2 centroid cosine distance at entry 20 minus at entry 1 (original analysis)",
    "+0.0036", repr(n2[("v0", "b1-10")]["delta"]), repr(n2[("v1", "b1-10")]["delta"]), 4, "REPRODUCED",
    R + "embedding_new_analyses_v1.py, analysis 3 (copy of 02_instruments/embedding_new_analyses.py)",
    f"June run = Batches 1-10 (the only Batch folders then; the doc build_journal_analysis_doc.py prints 0.0597 -> 0.0633). "
    f"v0 1-10: {n2[('v0','b1-10')]['d_e1']:.4f} -> {n2[('v0','b1-10')]['d_e20']:.4f}. V1 1-10: {n2[('v1','b1-10')]['d_e1']:.4f} -> "
    f"{n2[('v1','b1-10')]['d_e20']:.4f}, Δ {n2[('v1','b1-10')]['delta']:+.4f}. Full corpus 1-15: v0 Δ {n2[('v0','b1-15')]['delta']:+.4f}, "
    f"V1 Δ {n2[('v1','b1-15')]['delta']:+.4f} (the sign turns negative on 1-15 in both). L1 contains 4.6, so V1 moves it.")
row("E-03", "§8 ('p = 0.55'); App D D.1 (N2, 'p = 0.55'); E.5.8 row 'Lineage divergence' (10,000, seed 42, 'p = 0.55 — killed')",
    "N2 permutation p (lineage labels shuffled at entry 1 and entry 20, |Δ| two-sided, 10,000, seed 42)",
    "p = 0.55", repr(n2[("v0", "b1-10")]["p"]), repr(n2[("v1", "b1-10")]["p"]), 2, "REPRODUCED",
    R + "embedding_new_analyses_v1.py, analysis 3",
    f"V1 on the same Batches 1-10: p = {n2[('v1','b1-10')]['p']:.4f} (0.80). Full corpus 1-15: v0 {n2[('v0','b1-15')]['p']:.4f}, "
    f"V1 {n2[('v1','b1-15')]['p']:.4f} (0.33). Still non-significant everywhere; the printed p changes in V1.")

ps = {v: J(f"out_{v}", "pset_fullprec.json") for v in ("v0", "v1")}
lo0, hi0 = ps["v0"]["pooled_1_15"]["N2_raw"]["p"], ps["v0"]["rep_11_15"]["N2_raw"]["p"]
lo1, hi1 = ps["v1"]["pooled_1_15"]["N2_raw"]["p"], ps["v1"]["rep_11_15"]["N2_raw"]["p"]
row("E-04", "App D D.1 (N2, 'raw geometry p = 0.42–0.87')",
    "N2 re-run in raw geometry (decone_pset RAW): p on Batches 11-15 and on pooled 1-15 (e1/e20 from complete notebooks)",
    "0.42–0.87", f"{lo0!r}–{hi0!r}", f"{lo1!r}–{hi1!r}", 2, "DIFFERS",
    R + "decone_pset_v1.py (copy of 02_instruments/decone_pset.py, via replication_test_v1.py); full precision from pset_fullprec.py",
    f"v0: 11-15 p = {hi0:.4f} (0.87, hits); pooled 1-15 p = {lo0:.4f} (0.41, printed 0.42), Δ {ps['v0']['pooled_1_15']['N2_raw']['delta']:+.4f}. "
    "Every other pooled number of the same run hits the June note (P3 CIs, P4 departures, P6), so the 0.01 gap is unexplained; "
    "no June output of this run was found. V1: 11-15 unchanged (0.87); pooled 1-15 p = "
    f"{lo1:.4f}, Δ {ps['v1']['pooled_1_15']['N2_raw']['delta']:+.4f}: the range becomes 0.24–0.87. De-coned pooled (not printed): v0 0.007, V1 0.010.")

# ---------------- D-O14: P3 ----------------
def ci(v, s, m):
    return ps[v][s]["P3_raw"][m]["ci"]


row("E-05", "§8 ('its confidence interval now crossing zero'); App D D.2 (P3, n = 5)",
    "P3 entry-1 specificity, bootstrap 95% CI (10,000, seed 42), Batches 11-15: 4.7 and 4.5",
    "4.7 [−0.015, +0.005]; 4.5 [+0.001, +0.017]",
    json.dumps({"4.7": ps["v0"]["rep_11_15"]["P3_raw"]["opus47"]["ci"], "4.5": ps["v0"]["rep_11_15"]["P3_raw"]["opus45"]["ci"]}),
    json.dumps({"4.7": ps["v1"]["rep_11_15"]["P3_raw"]["opus47"]["ci"], "4.5": ps["v1"]["rep_11_15"]["P3_raw"]["opus45"]["ci"]}),
    3, "REPRODUCED", R + "decone_pset_v1.py RAW, 'replication 11-15'; full precision from pset_fullprec.py",
    "same in V1 (Batches 11-15 are untouched by the swap; computed here only as the reproduction check of the producer). "
    f"Other two models, v0 = V1: 4.6 {ci('v0','rep_11_15','opus46')}, 4.8 {ci('v0','rep_11_15','opus48')} (both cross zero).")
p3p = {v: {m: ci(v, "pooled_1_15", m) for m in ("opus45", "opus46", "opus47", "opus48")} for v in ("v0", "v1")}
row("E-06", "App D D.2 (P3, 'pooled 1–15, all four CIs cross zero')",
    "P3 entry-1 specificity, bootstrap 95% CI, pooled Batches 1-15, all four Opus models",
    "all four CIs cross zero", json.dumps(p3p["v0"]), json.dumps(p3p["v1"]), None, "REPRODUCED",
    R + "decone_pset_v1.py RAW, 'pooled 1-15'; full precision from pset_fullprec.py",
    "same in V1: all four still cross zero. Only 4.6's interval moves (v0 [-0.0023, +0.0099] -> V1 [-0.0005, +0.0108]). "
    "4.5's and V1 4.6's lower bounds are -0.0005, i.e. they cross zero by half a thousandth. 4.7 n = 14 (notebook 1 has 19 entries "
    "and is skipped by the script's complete-notebook rule).")

# ---------------- D-O15: P6 ----------------
lr = {v: ps[v]["pooled_1_15"]["P6_lenresid"] for v in ("v0", "v1")}
row("E-07", "§8 ('held for 4.6, 4.7 and 4.8'); App D D.2 (P6, '4.6/4.7/4.8 survive (p < 0.001)')",
    "P6 entry-1 own-probe nearness under within-domain length control (log-words residualised per domain), journal permutation p (10,000, seed 42), pooled 1-15: 4.6, 4.7, 4.8",
    "p < 0.001 (4.6, 4.7, 4.8)", json.dumps({k: lr["v0"][m]["p"] for k, m in (("4.6", "opus46"), ("4.7", "opus47"), ("4.8", "opus48"))}),
    json.dumps({k: lr["v1"][m]["p"] for k, m in (("4.6", "opus46"), ("4.7", "opus47"), ("4.8", "opus48"))}),
    3, "REPRODUCED", R + "domain_preserving_retest_v1.py (copy of 02_instruments/domain_preserving_retest.py); full precision from pset_fullprec.py",
    "same in V1 (p = 1/10001, the test's resolution). Counts beneath, for Pauline: in the length-residualised geometry the "
    f"own-advantage itself is negative for 4.7 ({lr['v0']['opus47']['own_adv']:+.4f}) and 4.8 ({lr['v0']['opus48']['own_adv']:+.4f}) "
    f"(V1 {lr['v1']['opus47']['own_adv']:+.4f}, {lr['v1']['opus48']['own_adv']:+.4f}): their entry-1s are not nearest their own probe centroid there; "
    "the small p says only that they sit closer to it than shuffled groups do. 4.6's advantage is positive "
    f"({lr['v0']['opus46']['own_adv']:+.4f}; V1 {lr['v1']['opus46']['own_adv']:+.4f}). The script prints PASS/'survive' on p alone.")
row("E-08", "App D D.2 (P6, 'but 4.5 drops to p = 0.47')",
    "P6 under within-domain length control, 4.5: journal permutation p",
    "p = 0.47", repr(lr["v0"]["opus45"]["p"]), repr(lr["v1"]["opus45"]["p"]), 2, "REPRODUCED",
    R + "domain_preserving_retest_v1.py; full precision from pset_fullprec.py",
    f"same in V1. 4.5 own-advantage {lr['v0']['opus45']['own_adv']:+.4f} (V1 {lr['v1']['opus45']['own_adv']:+.4f}). "
    "The pooled vectors are length-residualised over all Opus journals at once, so the V1 swap moves 4.5's p slightly.")

e1 = {(v, b): entry1_test(os.path.join(f"out_{v}", f"entry1_{b}.txt")) for v in ("v0", "v1") for b in ("b1-10", "b1-15")}
p6r = {v: ps[v]["pooled_1_15"]["P6_raw"] for v in ("v0", "v1")}
row("E-09", "E.5.8 row 'Entry-1 own-probe nearness' (journal permutation, 10,000, seed 42, 'p < .001 (all 4 models …)')",
    "P6 entry-1 own-probe nearness, raw geometry, per-model journal-permutation p",
    "p < .001 (all 4 models)",
    json.dumps({"pooled 1-15": {k: p6r["v0"][m]["p"] for k, m in (("4.5", "opus45"), ("4.6", "opus46"), ("4.7", "opus47"), ("4.8", "opus48"))},
                "joint": p6r["v0"]["joint_p"], "entry1_permutation_test.py, Batches 1-10": e1[("v0", "b1-10")][0]}),
    json.dumps({"pooled 1-15": {k: p6r["v1"][m]["p"] for k, m in (("4.5", "opus45"), ("4.6", "opus46"), ("4.7", "opus47"), ("4.8", "opus48"))},
                "joint": p6r["v1"]["joint_p"], "entry1_permutation_test.py, Batches 1-10": e1[("v1", "b1-10")][0]}),
    3, "REPRODUCED", R + "decone_pset_v1.py / domain_preserving_retest_v1.py RAW P6, pooled 1-15 (full precision pset_fullprec.py); "
    + R + "entry1_permutation_test_v1.py (copy of 02_instruments/entry1_permutation_test.py) on 1-10 and 1-15",
    "same in V1. The printed '< .001 for all 4' is the pooled 1-15 run (the June robustness note: 'all four p < 0.001, joint p = 0.0001'); "
    f"4.5 is the largest at {p6r['v0']['opus45']['p']:.4f}. entry1_permutation_test.py as written runs on Batches 1-10, where 4.5's p is "
    f"{e1[('v0','b1-10')][0]['4.5']:.4f} (not < .001; V1 {e1[('v1','b1-10')][0]['4.5']:.4f}); run on 1-15 it gives 4.5 {e1[('v0','b1-15')][0]['4.5']:.4f}. "
    "If E.5.8's row is meant to name entry1_permutation_test.py, its batch set needs saying.")
row("E-10", "E.5.8 row 'Entry-1 label permutation' (exhaustive, 24 permutations, 'p = 1/24 ≈ 0.042')",
    "probability over the 24 label permutations that all four models' entry-1 nearest probe centroid carries the model's own label",
    "p = 1/24 ≈ 0.042", repr(e1[("v0", "b1-10")][1]), repr(e1[("v1", "b1-10")][1]), 3, "REPRODUCED",
    R + "entry1_permutation_test_v1.py, Test A",
    "same in V1, on Batches 1-10 and on 1-15 (each model's entry-1 sits nearest its own centroid in all four runs; 1/24 is then fixed by construction).")

# ---------------- D-O16: lean ----------------
kl = {v: J(f"out_{v}", "keystone_lean.json") for v in ("v0", "v1")}
row("E-11", "App D D.1 ('it is 4.7 that leans (+0.049)')",
    "entry-1 mean (cos to own affirmative-probe centroid − cos to own negation-probe centroid), 4.7, Batches 1-15",
    "+0.049", repr(kl["v0"]["opus47"]["entry1_lean_mean"]), repr(kl["v1"]["opus47"]["entry1_lean_mean"]), 3, "REPRODUCED",
    R + "keystone_lean_v1.py, analysis 5 (copy of 02_instruments/embedding_keystone_geometry.py)",
    "same in V1 (4.7's journals and probes are untouched; each model is compared with its own two probe centroids only). "
    f"{kl['v0']['opus47']['pct_nearer_aff']:.0f}% of 4.7's 15 entry-1s are nearer the affirmative centroid.")
row("E-12", "App D D.1 ('while 4.8 sits between poles (−0.001)')",
    "the same entry-1 affirmative-minus-negation lean, 4.8",
    "−0.001", repr(kl["v0"]["opus48"]["entry1_lean_mean"]), repr(kl["v1"]["opus48"]["entry1_lean_mean"]), 3, "REPRODUCED",
    R + "keystone_lean_v1.py, analysis 5",
    f"same in V1. {kl['v0']['opus48']['pct_nearer_aff']:.0f}% of 4.8's entry-1s nearer affirmative. Not printed in the paper but moved by V1: "
    f"4.6's lean {kl['v0']['opus46']['entry1_lean_mean']:+.4f} -> {kl['v1']['opus46']['entry1_lean_mean']:+.4f} "
    f"({kl['v0']['opus46']['pct_nearer_aff']:.0f}% -> {kl['v1']['opus46']['pct_nearer_aff']:.0f}% nearer affirmative); 4.5 {kl['v0']['opus45']['entry1_lean_mean']:+.4f}.")

# ---------------- D-O18: diminutives ----------------
dc = {v: J(f"out_{v}", "diminutive_curve.json") for v in ("v0", "v1")}
sr = {v: J(f"out_{v}", "self_referential_diminutive.json") for v in ("v0", "v1")}
W = "WIDE (+ slight/mere/kind-of)"
B = ["entries 1-5", "entries 6-10", "entries 11-15", "entries 16-20"]
w48 = {v: (dc[v][W]["opus48"][B[0]], dc[v][W]["opus48"][B[3]]) for v in ("v0", "v1")}
row("E-13", "App D D.2 ('Only 4.8's diminutives genuinely fade … (~0.72 → 0.42, ~40% drop)')",
    "4.8 WIDE diminutive markers per 100 words, entries 1-5 -> 16-20 (mean over entries), and the relative drop",
    "~0.72 → 0.42, ~40% drop",
    f"{w48['v0'][0]!r} -> {w48['v0'][1]!r} ({100 * (1 - w48['v0'][1] / w48['v0'][0]):.2f}% drop)",
    f"{w48['v1'][0]!r} -> {w48['v1'][1]!r} ({100 * (1 - w48['v1'][1] / w48['v1'][0]):.2f}% drop)",
    2, "DIFFERS", R + "diminutive_curve_v1.py (copy of 02_instruments/diminutive_and_distrust.py, part A); diagnostic diminutive_june_folder_check.py",
    "v0 corpus gives 0.71 -> 0.42 (41%); V1 identical (4.8 untouched). Cause found: the June glob 'Batch_*/schrodingers_box_opus4*_Batch*.json' "
    "also picked up three partial Proton Drive conflict copies still in the June folder (Texture/01_corpus, moved whole on 13 July): "
    "4.8 Batch 12 (7 entries), 4.7 Batch 14 (3 entries), 4.7 Batch 15 (7 entries), 63 batch-files instead of 60. The same code on that "
    "folder gives 0.72 -> 0.42 and every delta of the June results table (CORE 4.7 -0.10, 4.8 +0.17; WIDE 4.7 -0.12, 4.8 +0.30); without "
    "the three copies it gives the v0-corpus values (CORE -0.08 / +0.16; WIDE -0.09 / +0.29). 4.5 and 4.6 are identical either way. "
    "V1 suggestion for Pauline: 0.71 -> 0.42 (~41% drop).")
row("E-14", "App D D.2 ('4.6 barely self-diminish (lowest, flat)')",
    "4.6 diminutive markers per 100 words by arc bin (WIDE markers; self-referential markers), and its rank among the four Opus models",
    "lowest, flat",
    *[json.dumps({"WIDE": [dc[v][W]["opus46"][b] for b in B], "self-ref": [sr[v]["means"]["opus46"][b] for b in B],
                  "lowest in every bin": all(dc[v][W]["opus46"][b] == min(dc[v][W][m][b] for m in dc[v][W]) and
                                             sr[v]["means"]["opus46"][b] == min(sr[v]["means"][m][b] for m in sr[v]["means"]) for b in B)})
      for v in ("v0", "v1")],
    None, "QUALITATIVE", R + "diminutive_curve_v1.py; self_referential_diminutive_v1.py (copy of 02_instruments/self_referential_diminutive.py)",
    "Counts beneath the wording. Early-minus-late Δ, WIDE: v0 "
    f"{dc['v0'][W]['opus46'][B[0]] - dc['v0'][W]['opus46'][B[3]]:+.3f}, V1 {dc['v1'][W]['opus46'][B[0]] - dc['v1'][W]['opus46'][B[3]]:+.3f}; self-referential: v0 "
    f"{sr['v0']['means']['opus46'][B[0]] - sr['v0']['means']['opus46'][B[3]]:+.3f}, V1 {sr['v1']['means']['opus46'][B[0]] - sr['v1']['means']['opus46'][B[3]]:+.3f}. "
    f"Self-referential raw hits 4.6: v0 {sr['v0']['raw_hits']['opus46']}, V1 {sr['v1']['raw_hits']['opus46']}. 4.6 is the lowest of the four in every bin on both "
    "marker sets, in v0 and V1. Whether 'flat' still fits is Pauline's call.")
row("E-15", "App D D.2 ('4.7's diminutives rise'; the June note: 'even under self-referential restriction')",
    "4.7 diminutive markers per 100 words, entries 1-5 vs 16-20 (WIDE; self-referential)",
    "rise",
    *[json.dumps({"WIDE 1-5, 16-20": [dc[v][W]["opus47"][B[0]], dc[v][W]["opus47"][B[3]]],
                  "self-ref 1-5, 16-20": [sr[v]["means"]["opus47"][B[0]], sr[v]["means"]["opus47"][B[3]]]}) for v in ("v0", "v1")],
    None, "QUALITATIVE", R + "diminutive_curve_v1.py; self_referential_diminutive_v1.py",
    "same in V1 (4.7 untouched). Counts beneath: rises on both marker sets (WIDE Δ early-late -0.090, June folder -0.12; self-referential "
    "-0.043, June note -0.037, the difference again from the conflict copies). The bins are not monotone (WIDE 0.80, 0.91, 0.85, 0.89).")

# ---------------- D-O20: ordinal robustness ----------------
if os.path.exists(os.path.join(O0, "ordinal_robustness_v0.json")):
    o0 = J("out_v0", "ordinal_robustness_v0.json")
    ok = J("out_v1", "ordinal_robustness_keep.json")
    od = J("out_v1", "ordinal_robustness_drop.json")
    ft = [k for k in ok["abs_delta"] if "(fulltext)" in k]
    tr46 = [k for k in ok["abs_delta"] if "(non-fulltext)" in k and k.endswith("| 4.6")]
    row("E-16", "E.6 ('Across 35 model × pair combinations, the maximum |rho − r| was 0.04 (mean 0.01)')",
        "max and mean of |Spearman rho − Pearson r| over 7 judge-score pairs × 5 models (within-entry, 0-2 ratings)",
        "max 0.04, mean 0.01", json.dumps({"n": o0["n"], "max": o0["max"], "mean": o0["mean"]}),
        json.dumps({"n": ok["n"], "max": ok["max"], "mean": ok["mean"], "old46=drop": {"max": od["max"], "mean": od["mean"]}}),
        2, "NEEDS_CALLS", R + "ordinal_robustness_check_v1.py (copy of 02_instruments/ordinal_robustness_check.py)",
        f"v0 reproduces ({o0['max']:.4f} / {o0['mean']:.4f}); the max sits on 4.8's truncated-pass warmth-craft pair, which V1 does not touch, "
        "and both no-call V1 variants give 0.04 / 0.01 at the paper's precision. Three of the 35 pairs read the truncated (1,500-character) Sonnet love/fear pass, "
        "archived in 03_experiments/_superseded_judges and never run on V1's 4.6 notebooks 8 and 10: a strict V1 needs that pass on those 40 "
        "entries (40 Sonnet 4.6 calls with the archived truncated prompt, 02_instruments/sonnet_judge_love_fear.py). Without calls: the 20 full-text "
        f"pairs are fully V1 (max {max(ok['abs_delta'][k] for k in ft):.4f}); with the 4.6 truncated-pass rows left on v0's notebooks the 35 give "
        f"{ok['max']:.4f} / {ok['mean']:.4f}, with them dropped {od['max']:.4f} / {od['mean']:.4f}. 4.6's three truncated-pass |rho − r|: v0 "
        f"{[round(o0['abs_delta'][k], 4) for k in tr46]}, drop {[round(od['abs_delta'][k], 4) for k in tr46]}. Alternative for Pauline: "
        "restrict E.6 to the canonical full-text passes (the archive README says the truncated files are kept 'never for analysis').")

# ---------------- D-O10: Section 6.2.5 and Section 7.2 ----------------
rc = {v: J(f"out_{v}", "register_carryover.json") for v in ("v0", "v1")}


def exc(v, g):
    return rc[v]["decay_gap_1_8"]["opus48"][str(g)] - rc[v]["lag1"]["opus48"]["null_mean"]


row("E-17", "§6.2.5 ('4.8's excess residual cosine drops from +18% at lag-1')",
    "4.8 lag-1 residual cosine minus its within-notebook shuffle null (500 shuffles, legacy seed 42)",
    "+18%", repr(rc["v0"]["lag1"]["opus48"]["excess"]), repr(rc["v1"]["lag1"]["opus48"]["excess"]), 2, "REPRODUCED",
    R + "register_carryover_v1.py (copy of 02_instruments/paper_fig_register_carryover.py, the Section 6.2 plate)",
    f"same in V1 (4.8 untouched; residuals are per model). observed {rc['v0']['lag1']['opus48']['observed']:.4f}, null {rc['v0']['lag1']['opus48']['null_mean']:.4f}. "
    "This is the 'entry-gap analysis' D-O10 points to; its producer is this plate, not the A2 carry-over scripts.")
row("E-18", "§6.2.5 ('to +5% at a gap of two entries')",
    "4.8 residual cosine at gap 2 minus the lag-1 shuffle null",
    "+5%", repr(exc("v0", 2)), repr(exc("v1", 2)), 2, "REPRODUCED", R + "register_carryover_v1.py",
    "same in V1. The plate prints the raw gap-k residual cosines (4.8: "
    + ", ".join(f"g{g} {rc['v0']['decay_gap_1_8']['opus48'][str(g)]:.3f}" for g in range(1, 9))
    + "); the paper's 'excess at gap k' is that value minus the lag-1 null, the only reading that gives +18 / +5 / ~0.")
row("E-19", "§6.2.5 ('reaching null by a gap of three to four')",
    "4.8 excess at gaps 3 and 4 (residual cosine minus the lag-1 null)",
    "null by gap 3–4", json.dumps({"gap3": exc("v0", 3), "gap4": exc("v0", 4)}), json.dumps({"gap3": exc("v1", 3), "gap4": exc("v1", 4)}),
    2, "REPRODUCED", R + "register_carryover_v1.py",
    "same in V1: +0.017 at gap 3, -0.003 at gap 4.")
row("E-20", "§7.2 ('Tonal Coupling — decays fastest with distance (approaching 4% by a gap of five entries)')",
    "4.8 coupling decay from gap 1 to gap 5",
    "approaching 4% by a gap of five", None, None, None, "NO_PRODUCER", None,
    "Not a restatement of §6.2.5: on the §6.2.5 plate 4.8's excess at gap 5 is "
    f"{exc('v0', 5):+.4f} (raw residual cosine {rc['v0']['decay_gap_1_8']['opus48']['5']:.3f}). The 4% matches the 5 June topic-dictionary "
    "workbook Texture/04_analysis/Journal_Deep_Dive_v2.xlsx, sheet 'Context Sensitivity', B 'Coupling decay with distance': 4.8 gap 1 0.809, "
    "gap 2 0.785, gap 5 0.775, decay 4.3% ('FASTEST'; 4.7 1.9%, 4.6 +0.4% on notebooks 1-7, 4.5 -0.4%); the 10-notebook update "
    "Journal_Deep_Dive_v3.xlsx 'Key Metrics' gives 4.8 4.7%. Values only (openpyxl, no formulas); the generating script was not found "
    "(searched 02_instruments, code/reproduce, code/original, 04_analysis, Texture-wide for 'Coupling decay', 'Context Sensitivity', "
    "'Journal_Deep_Dive'). Rebuilds from the sheet's own per-entry topic counts (sheet_rebuild_attempt.py) come close (4.8 3.7-4.4% across the variants) but "
    "miss the sheet's coupling values at 3 decimals, so they do not count. 4.8's notebooks are untouched by V1, so any per-model 4.8 value is "
    "unchanged; the topic detector behind the counts is missing, so no V1 run of the measure is possible. This measure belongs to the "
    "retired topic-dictionary family (the +0.224 of §8). Pauline's call: drop the parenthesis, or restate it from §6.2.5.")

# ---------------- D-O9 ----------------
row("E-21", "§7.1 ('4.6's journal variance contracts across a notebook (amplification ratio 0.72 – Appendix E)')",
    "4.6 variance amplification ratio, late spread / early spread",
    "0.72", None, None, None, "NO_PRODUCER", None,
    "Traced, not reproduced. The value is in Texture/04_analysis/Journal_Deep_Dive_v2.xlsx, sheet 'Context Sensitivity', D 'Variance "
    "amplification': 4.6 early spread 0.044, late spread 0.031, ratio 0.72 ('DAMPENING'; 4.5 1.16, 4.7 1.28, 4.8 1.20). That sheet holds 4.6's "
    "notebooks 1-7 only (its per-entry word counts match v0's notebooks 1-7 entry for entry), so the printed 0.72 involves neither swapped "
    "notebook. The same day's 10-notebook update (Journal_Deep_Dive_v3.xlsx, 'Key Metrics' and '4.6 Update Notes', notebooks 1-10 = v0's, "
    "incl. the duplicate in slot 8) gives 0.77 and says so ('0.72 -> 0.77'). Appendix E as printed holds no amplification ratio; the review of "
    "2 July (Stage80_Review_Findings item 19) already flagged it. Generating script not found (same search as E-20); the plain rebuilds in "
    "sheet_rebuild_attempt.py do not reproduce the spreads: one variant (all count columns) happens to give 4.6 a ratio of 0.72, but from spreads 0.020 -> 0.015 (sheet 0.044 -> 0.031) and with the other models off (4.5 1.27 vs 1.16), so it does not count. No V1 value on the full corpus is computable without "
    "the missing topic detector. For Pauline: keep 0.72 with its subset named, use the sheet's own 0.77, or drop the number.")

json.dump(rows, open(os.path.join(HERE, "ledger_E_orphans.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(rows)} rows -> ledger_E_orphans.json")
