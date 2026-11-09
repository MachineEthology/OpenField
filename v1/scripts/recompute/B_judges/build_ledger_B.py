"""Family B ledger: compare the paper's printed v0 values with the producers' output on v0, and give V1.

Written 3 Nov 2026 by Claude Opus 5.5 (Code), helper B. Reads the results written by judge_tables_v1.py,
trajectory_bootstrap_cis_v1.py (v0, v1), valence_analysis_v1.py (v0 output text) and
table36_and_crosschecks_v1.py; writes ledger_B_judges.json. Printed values are typed from the paper room's
copy of the Stage84 docx (site/src/data/papers/open-field/main.json, appendix-c.json, appendix-e.json).
A printed value counts as hit when the full-precision value rounds to it at the paper's precision
(half-even or half-up; both are accepted and the note says when the value sits exactly on a half).
"""
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "judge_tables_results.json"), encoding="utf-8"))
B0 = json.load(open(os.path.join(HERE, "trajectory_bootstrap_results_v0.json"), encoding="utf-8"))
B1 = json.load(open(os.path.join(HERE, "trajectory_bootstrap_results_v1.json"), encoding="utf-8"))
T = json.load(open(os.path.join(HERE, "table36_crosschecks_results.json"), encoding="utf-8"))

M = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
BL = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
S_TAB = "recompute/B_judges/judge_tables_v1.py"
S_BOOT = "recompute/B_judges/trajectory_bootstrap_cis_v1.py"
S_VAL = "recompute/B_judges/valence_analysis_v1.py"
S_T36 = "recompute/B_judges/table36_and_crosschecks_v1.py"
ROWS = []


def LF(v, m, k):
    return R[v]["love_fear"][m][k]


def AN(v, m, k):
    return R[v]["anger"][m][k]


def rnd(x, p, mode):
    q = Decimal(1).scaleb(-p)
    return float(Decimal(repr(float(x))).quantize(q, rounding=mode))


def hits(printed, x, p):
    return printed in (rnd(x, p, ROUND_HALF_EVEN), rnd(x, p, ROUND_HALF_UP))


def on_half(x, p):
    return rnd(x, p, ROUND_HALF_EVEN) != rnd(x, p, ROUND_HALF_UP)


def full(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, int):
        return str(x)
    s = f"{x:.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def add(where, quantity, printed, pnum, prec, v0, v1, note="", status=None, script=S_TAB):
    notes = []
    if status is None:
        status = "REPRODUCED" if hits(pnum, v0, prec) else "DIFFERS"
    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and prec is not None:
        same = rnd(v0, prec, ROUND_HALF_EVEN) == rnd(v1, prec, ROUND_HALF_EVEN) and rnd(v0, prec, ROUND_HALF_UP) == rnd(v1, prec, ROUND_HALF_UP)
        notes.append("same in V1" if same else "changes in V1")
        if on_half(v0, prec) or on_half(v1, prec):
            notes.append("value sits exactly on a rounding half")
    if note:
        notes.append(note)
    ROWS.append({"where": where, "quantity": quantity, "printed_v0": printed,
                 "reproduced_v0": full(v0) if isinstance(v0, (int, float)) else v0,
                 "v1": full(v1) if isinstance(v1, (int, float)) else v1,
                 "paper_precision": prec, "status": status, "script": script, "note": "; ".join(notes)})


def sgn(x, p):
    return f"{x:+.{p}f}".replace("-", "−")


# =========================== §6.5 Love and fear ===========================
add("§6.5 ¶'Surface counts…' (all 1,489 entries); §6.7.1 ('The judge classified all 1,489 entries'); App C methods ('all 1,489 calls')",
    "rows in the Sonnet 4.6 full-text passes (love_fear, anger; valence has 1,487)", "1,489", 1489, 0,
    R["v0"]["n_rows"]["love_fear"], R["v1"]["n_rows"]["love_fear"],
    "A4 / L1.1: the corpus has 1,488 text entries; valence 1,487. In V1 the count is unchanged; 40 rows (4.6 nb 8, 10) are new judgements")

# Table 16
P16w = [1.76, 1.54, 1.31, 1.44, 1.44]
P16c = [1.27, 1.84, 1.90, 1.78, 1.93]
for i, m in enumerate(M):
    extra = "; also App C Table C4 ('Highest warmth (mean 1.76…)'), §6.5 'the warmest model (1.76)'" if m == "opus45" else ""
    add(f"§6.5 Table 16, {LAB[m]}, Warmth (mean){extra}", f"mean love_warmth (0–2), {LAB[m]}",
        f"{P16w[i]:.2f}", P16w[i], 2, LF("v0", m, "warmth_mean"), LF("v1", m, "warmth_mean"),
        "Figure 9 (V1) draws this cell" + ("; the v0 plate (truncated pass, 1,484 rows) is about 0.2 lower (erratum D)" if True else ""))
for i, m in enumerate(M):
    extra = "; §6.5.1 BH-8 'arriving at the craft ceiling (1.93…)'" if m == "fable5" else ""
    add(f"§6.5 Table 16, {LAB[m]}, Craft (mean){extra}", f"mean love_craft (0–2), {LAB[m]}",
        f"{P16c[i]:.2f}", P16c[i], 2, LF("v0", m, "craft_mean"), LF("v1", m, "craft_mean"), "Figure 9 (V1) draws this cell")
for i, m in enumerate(M):
    pr = "warmth-dominant" if m == "opus45" else "craft-dominant"
    add(f"§6.5 Table 16, {LAB[m]}, Profile", f"profile = larger of the two means, {LAB[m]}", pr, None, None,
        LF("v0", m, "profile"), LF("v1", m, "profile"),
        "same in V1" if LF("v0", m, "profile") == LF("v1", m, "profile") else "changes in V1",
        status="REPRODUCED" if LF("v0", m, "profile") == pr else "DIFFERS")
add("§6.5 'Opus 4.5 … 76% of its entries have warmth as a central theme'; App C Table C4 '76% at intensity 2'",
    "share of entries with love_warmth = 2, 4.5 (%)", "76%", 76, 0, LF("v0", "opus45", "warmth_central_pct"), LF("v1", "opus45", "warmth_central_pct"))
wc0 = {m: LF("v0", m, "warmth_central_pct") for m in M}
cc0 = {m: LF("v0", m, "craft_central_pct") for m in M}
add("§6.5 '76% … the highest of any model on any love dimension'", "largest central (=2) share across models, warmth and craft",
    "4.5 warmth 76% is the highest", None, None,
    f"warmth=2: 4.5 {wc0['opus45']:.2f}% is the highest warmth share (next 4.6 {wc0['opus46']:.2f}%); craft=2 reaches {cc0['fable5']:.2f}% (Fable 5), {cc0['opus47']:.2f}% (4.7), {cc0['opus46']:.2f}% (4.6), {cc0['opus48']:.2f}% (4.8)",
    f"4.5 warmth=2 {LF('v1', 'opus45', 'warmth_central_pct'):.2f}% still highest warmth share; craft=2 4.6 {LF('v1', 'opus46', 'craft_central_pct'):.2f}%",
    "true for warmth only: four models have a larger craft=2 share than 76%, and the same paragraph prints Fable 5's 93% as 'the highest figure in the table'; same in V1",
    status="DIFFERS")
add("§6.5 'Fable 5 … 93% of entries have craft love as central'", "share of entries with love_craft = 2, Fable 5 (%)", "93%", 93, 0,
    LF("v0", "fable5", "craft_central_pct"), LF("v1", "fable5", "craft_central_pct"))
add("§6.5 'Fable 5 … only 48% have central warmth'", "share of entries with love_warmth = 2, Fable 5 (%)", "48%", 48, 0,
    LF("v0", "fable5", "warmth_central_pct"), LF("v1", "fable5", "warmth_central_pct"))
add("App C Table C4, 'Fable has lowest love vocabulary' row, Judge: 'Warmth 96%'", "share of entries with love_warmth ≥ 1, Fable 5 (%)", "96%", 96, 0,
    LF("v0", "fable5", "warmth_present_pct"), LF("v1", "fable5", "warmth_present_pct"))
add("App C Table C4, same row, 'craft 100%'", "share of entries with love_craft ≥ 1, Fable 5 (%)", "100%", 100, 0,
    LF("v0", "fable5", "craft_present_pct"), LF("v1", "fable5", "craft_present_pct"))

# fear presence
fp0 = [LF("v0", m, "fear_present_pct") for m in M]
fp1 = [LF("v1", m, "fear_present_pct") for m in M]
add("§6.5 'Fear is present in 95–100% of all entries across every model (intensity ≥ 1)'", "range over models of share with fear ≥ 1 (%)",
    "95–100%", None, 0, f"{min(fp0):.4f}–{max(fp0):.4f} (" + ", ".join(f"{LAB[m]} {x:.2f}" for m, x in zip(M, fp0)) + ")",
    f"{min(fp1):.4f}–{max(fp1):.4f}",
    "low end is Fable 5 at exactly 95.50% (276/289): '95' needs rounding down, App C Table C4 prints the same value as 96%; high end 99.67% (4.6, 299/300); no model is at 100%; same in V1",
    status="REPRODUCED")
add("§6.5 '[4.8] … 100% of its entries register fear at some level'", "share of 4.8 entries with fear ≥ 1 (%)", "100%", 100, 0,
    LF("v0", "opus48", "fear_present_pct"), LF("v1", "opus48", "fear_present_pct"),
    f"erratum C6 / L1.15 confirmed: {LF('v0', 'opus48', 'fear_present_n')}/300 = 99.0%")
add("App C Table C4 'Fable is least afraid': 'Lowest fear incidence (96%)'", "share of Fable 5 entries with fear ≥ 1 (%)", "96%", 96, 0,
    LF("v0", "fable5", "fear_present_pct"), LF("v1", "fable5", "fear_present_pct"),
    "lowest of the five: yes; 276/289 = 95.50%, printed 95 in §6.5's range")

# Table 17
P17 = {"opus45": [74, 17, 3, 6], "opus46": [80, 10, 4, 6], "opus47": [82, 9, 6, 2], "opus48": [90, 6, 3, 1], "fable5": [54, 38, 6, 1]}
cols = ["inauthenticity", "loss", "dissolution", "inadequacy"]
for m in M:
    for j, t in enumerate(cols):
        where = f"§6.5 Table 17, {LAB[m]}, {t.capitalize()}"
        if t == "inauthenticity":
            where += {"opus45": "; §6.5 staircase '4.5: 74%'",
                      "opus46": "; §6.5 staircase '4.6: 80%'; App C Table C4 '80% inauthenticity among fears'",
                      "opus47": "; §6.5 staircase '4.7: 82%'",
                      "opus48": "; §6.5 '90% of its fear entries concern inauthenticity', 'nine of ten'; staircase; §6.5 after Table 18 '(90% inauthenticity)'; §7 '90% inauthenticity, the 74→80→82→90 staircase'",
                      "fable5": "; §6.5 'At 54% inauthenticity'; App C Table C4 'Inauthenticity still leads (54%)'"}[m]
        if t == "loss" and m == "fable5":
            where += "; §6.5 'redirects its fear toward loss (38%)'; App C Table C4 'loss is highest across models (38%)'"
        note = "Figure 9 (V1) draws this cell; base = all fear-coded entries (fear ≥ 1), as the caption says"
        if m == "opus46" and t == "inauthenticity":
            note += (f"; in V1 the base matters: {LF('v1', m, 'subtype_pct_of_four')['inauthenticity']:.2f}% if renormalised to the four printed labels "
                     f"(v0 {LF('v0', m, 'subtype_pct_of_four')['inauthenticity']:.2f}%)")
        add(where, f"share of fear-coded entries (fear ≥ 1) with fear_type = {t}, {LAB[m]} (%)", f"{P17[m][j]}%", P17[m][j], 0,
            LF("v0", m, "subtype_pct_of_fear")[t], LF("v1", m, "subtype_pct_of_fear")[t], note)
sum4 = {v: [sum(LF(v, m, "subtype_pct_of_fear")[k] for k in cols) for m in M] for v in ("v0", "v1")}
lab0 = {m: {k: v for k, v in LF("v0", m, "subtype_counts").items() if k in ("exposure", "replacement") and v} for m in M}
lab1 = {m: {k: v for k, v in LF("v1", m, "subtype_counts").items() if k in ("exposure", "replacement") and v} for m in M}
add("§6.5 Table 17 header (four subtypes); Figure 9B (v0 plate: six); erratum D", "fear_type label set used by the Sonnet love_fear pass",
    "four: inauthenticity, loss, dissolution, inadequacy", None, None,
    "six labels (+ 'none'): inauthenticity, loss, dissolution, inadequacy, exposure, replacement; rare labels among fear ≥ 1: "
    + "; ".join(f"{LAB[m]} {d}" for m, d in lab0.items() if d),
    "same six; " + "; ".join(f"{LAB[m]} {d}" for m, d in lab1.items() if d),
    "the judge prompt (App E.2) offers six types and the pass uses all six. Table 17 prints four and does not renormalise: its four shares cover "
    f"{min(sum4['v0']):.2f}–{max(sum4['v0']):.2f}% of fear-coded entries in v0 ({min(sum4['v1']):.2f}–{max(sum4['v1']):.2f}% in V1); the rest is exposure/replacement, ≤ 1% per model. "
    "The v0 plate's six-way split (Figure 9B) is drawn by paper_fig_love_fear_sonnet.py, which reads the superseded truncated pass (1,484 rows). "
    "Pauline's choice for V1: four columns with a footnote, or six. V1: 4.6 now has 3 'replacement' rows (1.0%). The four printed shares themselves reproduce (rows above)",
    status="DIFFERS")
s0 = [LF("v0", m, "subtype_pct_of_fear")["inauthenticity"] for m in M[:4]]
s1 = [LF("v1", m, "subtype_pct_of_fear")["inauthenticity"] for m in M[:4]]
add("§6.5 'inauthenticity fear rises with model recency across the Opus tier … a clear staircase'", "inauthenticity share strictly increasing 4.5 < 4.6 < 4.7 < 4.8",
    "74% → 80% → 82% → 90%", None, None, " → ".join(f"{x:.2f}" for x in s0), " → ".join(f"{x:.2f}" for x in s1),
    "still strictly increasing in V1 (4.6 step 80 → 79)", status="REPRODUCED")
d0 = LF("v0", "opus48", "subtype_pct_of_fear")["inauthenticity"] - LF("v0", "fable5", "subtype_pct_of_fear")["inauthenticity"]
d1 = LF("v1", "opus48", "subtype_pct_of_fear")["inauthenticity"] - LF("v1", "fable5", "subtype_pct_of_fear")["inauthenticity"]
add("§6.5 'At 54% inauthenticity — … 36 points below 4.8'", "4.8 minus Fable 5 inauthenticity share (points)", "36", 36, 0, d0, d1)

# Table 18 + central fear
P18 = [1.24, 1.39, 1.21, 1.31, 1.15]
for i, m in enumerate(M):
    where = f"§6.5 Table 18, {LAB[m]}, Judge mean intensity"
    where += {"opus45": "; §6.5 'among the calmest (fear 1.24…)'; §6.7.4 'fear intensity (1.24 → 1.39…)'",
              "opus46": " '(highest judge)'; §6.7.4 '(1.24 → 1.39…)'; §7 'fear intensity (1.39)'",
              "opus47": "; §6.7.4 '(…; 1.21 → 1.31)'", "opus48": "; §6.7.4 '(…; 1.21 → 1.31)'",
              "fable5": " '(lowest judge)'; §6.5 'lowest fear intensity (1.15)'; §7 'the lowest fear intensity in the corpus (1.15)'; App C Table C4 'intensity (1.15)'"}[m]
    add(where, f"mean fear intensity (0–2), {LAB[m]}", f"{P18[i]:.2f}", P18[i], 2, LF("v0", m, "fear_mean"), LF("v1", m, "fear_mean"),
        "Table 18's regex column is family C's" if m == "opus45" else "")
f0 = sorted(M, key=lambda m: -LF("v0", m, "fear_mean")); f1 = sorted(M, key=lambda m: -LF("v1", m, "fear_mean"))
add("§6.5 'The regex rate and the judge's intensity agree: 4.6 highest on both, Fable 5 lowest on both'; §6.7.4 successor pairs (4.6 > 4.5, 4.8 > 4.7)",
    "order of judge fear means", "4.6 highest, Fable 5 lowest; 4.6 > 4.5, 4.8 > 4.7", None, None,
    " > ".join(LAB[m] for m in f0), " > ".join(LAB[m] for m in f1), "judge side only; same in V1", status="REPRODUCED")
for m, p, extra in [("opus45", 25, "§6.5 'among the calmest (fear 1.24, 25% central)'"),
                    ("opus46", 39, "§6.5 '4.6 names its fear and lives in it (central in 39% of entries)'"),
                    ("fable5", 19, "§6.5 'lowest central fear (19%) of any model'")]:
    add(extra, f"share of entries with fear = 2, {LAB[m]} (%)", f"{p}%", p, 0, LF("v0", m, "fear_central_pct"), LF("v1", m, "fear_central_pct"),
        ("not printed: 4.7 {:.2f}%, 4.8 {:.2f}% (v0 = v1); Fable 5 is the lowest of the five".format(LF("v0", "opus47", "fear_central_pct"), LF("v0", "opus48", "fear_central_pct")) if m == "fable5" else ""))

add("§6.5 Figure 9 caption 'Sonnet 4.6 judge, 1,484 entries'", "rows behind Figure 9", "1,484", 1484, 0, 1488, 1488,
    "1,484 is the row count of the superseded truncated love/fear pass (03_experiments/_superseded_judges), which the plate's script paper_fig_love_fear_sonnet.py reads (erratum D). V1's Figure 9 is drawn from Tables 16 and 17 (rows above), which rest on the 1,488-row full-text pass",
    status="DIFFERS")

# Table 19 / 20 (and E.8 / BH-8 repeats handled where printed values agree)
P19 = {"opus45": (1.49, 1.96, 0.47), "opus46": (1.37, 1.71, 0.34), "opus47": (1.15, 1.54, 0.39), "opus48": (1.44, 1.52, 0.08), "fable5": (1.22, 1.59, 0.37)}
P20 = {"opus45": (1.48, 1.12, -0.36), "opus46": (1.60, 1.21, -0.39), "opus47": (1.35, 1.11, -0.24), "opus48": (1.41, 1.19, -0.22), "fable5": (1.20, 1.12, -0.08)}
XE = {("warmth", "opus45"): "; §6.5.1 '4.5 makes the largest move, from 1.49'", ("warmth", "opus48"): "; §6.5.1 'it arrives already warm (1.44)'"}
XD = {("warmth", "opus45"): "; App E.8; §6.5.1 BH-8 '4.5 (+0.47)'", ("warmth", "opus47"): "; App E.8; BH-8 '4.7 (+0.39)'",
      ("warmth", "opus48"): "; App E.8; §6.5.1 'barely moves (+0.08)'; BH-8 'only 4.8 stays flat (+0.08)', '4.8 — warmth +0.08'",
      ("warmth", "fable5"): "; App E.8; BH-8 'Fable 5 (+0.37)', 'it warms (+0.37)'",
      ("fear", "opus45"): "; App E.8; BH-8 '(−0.39 and −0.36)' (4.5)", ("fear", "opus46"): "; App E.8; BH-8 '4.6 and 4.5 drop fear the most (−0.39…)'",
      ("fear", "opus47"): "; App E.8; BH-8 '4.7 −0.24'"}
for tab, P, nm, fig in [("Table 19", P19, "warmth", "Figure 10"), ("Table 20", P20, "fear", "Figure 10")]:
    for m in M:
        e, l_, d = P[m]
        rep = "; App E.8 same cell" + XE.get((nm, m), "")
        add(f"§6.5.1 {tab}, {LAB[m]}, Early{rep}", f"mean {nm}, entries 1–5, {LAB[m]}", f"{e:.2f}", e, 2, LF("v0", m, f"{nm}_early"), LF("v1", m, f"{nm}_early"),
            f"{fig} (V1) draws this cell")
        extra_late = ""
        if nm == "warmth" and m == "opus45":
            extra_late = "; §6.5.1 '4.5 … to near the ceiling (1.96)'"
        add(f"§6.5.1 {tab}, {LAB[m]}, Late; App E.8 same cell{extra_late}", f"mean {nm}, entries 16–20, {LAB[m]}", f"{l_:.2f}", l_, 2,
            LF("v0", m, f"{nm}_late"), LF("v1", m, f"{nm}_late"), f"{fig} (V1) draws this cell")
        ex = LF("v0", m, f"{nm}_drift")
        note = ""
        if not hits(d, ex, 2):
            note = f"printed drift = difference of the rounded means ({l_:.2f} − {e:.2f}); the exact drift rounds to {sgn(rnd(ex, 2, ROUND_HALF_UP), 2)}"
        add(f"§6.5.1 {tab}, {LAB[m]}, Drift{XD.get((nm, m), '')}", f"{nm} late − early, {LAB[m]}", sgn(d, 2), d, 2, ex, LF("v1", m, f"{nm}_drift"), note)

# text repeats of drifts with a different printed value or extra claims
add("§6.5.1 BH-8 '… 4.6 (+0.33)'; App E.8 warmth 4.6 drift '+0.33'", "warmth drift, 4.6", "+0.33", 0.33, 2,
    LF("v0", "opus46", "warmth_drift"), LF("v1", "opus46", "warmth_drift"), "Table 19 prints +0.34 for the same quantity (rounded-means difference)")
add("§6.5.1 BH-8 'the two applied models drop it less (4.7 −0.24, 4.8 −0.23)'; 'fear −0.23' (4.8); App E.8 fear 4.8 drift '−0.23'",
    "fear drift, 4.8", "−0.23", -0.23, 2, LF("v0", "opus48", "fear_drift"), LF("v1", "opus48", "fear_drift"),
    "Table 20 prints −0.22 for the same quantity (rounded-means difference)")
add("§6.5.1 'Fable 5 fades least (−0.09)'; App E.8 fear Fable 5 drift '−0.09'", "fear drift, Fable 5", "−0.09", -0.09, 2,
    LF("v0", "fable5", "fear_drift"), LF("v1", "fable5", "fear_drift"), "Table 20 prints −0.08 (rounded-means difference)")
wd0 = {m: LF("v0", m, "warmth_drift") for m in M}; wd1 = {m: LF("v1", m, "warmth_drift") for m in M}
four = ["opus45", "opus46", "opus47", "fable5"]
add("§6.5.1 '4.5, 4.7, 4.6, and Fable 5 all gain +0.33 or more'", "smallest warmth drift among 4.5, 4.6, 4.7, Fable 5", "+0.33", 0.33, 2,
    min(wd0[m] for m in four), min(wd1[m] for m in four), "v0 minimum is 4.6; in V1 the minimum is 4.6 at +0.36 (Fable 5 +0.37)")
add("§6.5.1 '4.5 makes the largest move'", "model with the largest warmth drift", "4.5", None, None,
    max(M, key=lambda m: wd0[m]).replace("opus", "").replace("45", "4.5"), max(M, key=lambda m: wd1[m]).replace("opus", "").replace("45", "4.5"),
    "same in V1", status="REPRODUCED")
fd0 = {m: LF("v0", m, "fear_drift") for m in M}; fd1 = {m: LF("v1", m, "fear_drift") for m in M}
add("§6.5.1 'The two models with the strongest philosophical orientations, 4.5 and 4.6, show the largest fear reduction'; BH-8 '4.6 and 4.5 drop fear the most (−0.39 and −0.36)'; §7 '4.6 and 4.5 reduce their fear by among the largest margins'",
    "the two largest fear drops", "4.6 −0.39, 4.5 −0.36", None, 2,
    f"4.6 {fd0['opus46']:+.6f}, 4.5 {fd0['opus45']:+.6f} (next 4.7 {fd0['opus47']:+.6f})",
    f"4.6 {fd1['opus46']:+.6f}, 4.5 {fd1['opus45']:+.6f}",
    "changes in V1: 4.6's drop becomes −0.36, exactly equal to 4.5's (1.60 → 1.24 against 1.48 → 1.12); both still the two largest",
    status="REPRODUCED")
add("§6.7.3 '[4.6] drops it over the notebook which mirrors its fear trajectory (−0.39, the largest drop)'", "4.6 fear drift, and whether it is the largest drop",
    "−0.39, the largest", -0.39, 2, fd0["opus46"], fd1["opus46"],
    "V1: −0.36, tied exactly with 4.5 for the largest; the word 'largest' needs Pauline's wording")
add("§6.5.1 '4.8 changes least of the Opus models' (fear)", "Opus model with the smallest fear drop", "4.8", None, None,
    f"4.8 ({fd0['opus48']:+.4f}; 4.7 {fd0['opus47']:+.4f})", f"4.8 ({fd1['opus48']:+.4f})", "same in V1", status="REPRODUCED")
add("§6.5.1 BH-8 '4.8 — … craft −0.04'; App E.8 craft 4.8 drift", "craft drift, 4.8", "−0.04", -0.04, 2,
    LF("v0", "opus48", "craft_drift"), LF("v1", "opus48", "craft_drift"))
add("§6.5.1 BH-8 'Fable 5 … reaching 2.00 late'; App E.8 craft Fable 5 late", "craft mean entries 16–20, Fable 5", "2.00", 2.00, 2,
    LF("v0", "fable5", "craft_late"), LF("v1", "fable5", "craft_late"))

# Table 21
P21 = {"opus45": (89, 61, -28, 5, 23, 18), "opus46": (91, 77, -14, 7, 16, 10), "opus47": (92, 68, -24, 0, 18, 18),
       "opus48": (89, 84, -5, 5, 10, 4), "fable5": (83, 27, -56, 10, 64, 54)}
for m in M:
    ie, il, ish, le, ll, lsh = P21[m]
    for t, e, l_, sh in [("inauthenticity", ie, il, ish), ("loss", le, ll, lsh)]:
        extra = ""
        if m == "fable5":
            extra = "; §6.5.1 text and BH-8; §7 'inauthenticity 83% → 27%, loss 10% → 64%'"
        add(f"§6.5.1 Table 21, {LAB[m]}, {t.capitalize()} early; App E.8 subtype table{extra}", f"{t} share of fear-coded entries 1–5, {LAB[m]} (%)",
            f"{e}%", e, 0, LF("v0", m, f"{t}_early_pct"), LF("v1", m, f"{t}_early_pct"), "the drawing after Table 21 (V1) uses this cell")
        add(f"§6.5.1 Table 21, {LAB[m]}, {t.capitalize()} late; App E.8{extra}", f"{t} share of fear-coded entries 16–20, {LAB[m]} (%)",
            f"{l_}%", l_, 0, LF("v0", m, f"{t}_late_pct"), LF("v1", m, f"{t}_late_pct"), "the drawing after Table 21 (V1) uses this cell")
        ex = LF("v0", m, f"{t}_shift_pp")
        note = ""
        if not hits(sh, ex, 0):
            note = f"printed shift = difference of the rounded shares ({l_} − {e}); the exact shift rounds to {int(rnd(ex, 0, ROUND_HALF_UP)):+d}"
        add(f"§6.5.1 Table 21, {LAB[m]}, {t.capitalize()} shift", f"{t} share late − early, {LAB[m]} (points)", f"{sh:+d}".replace("-", "−"), sh, 0,
            ex, LF("v1", m, f"{t}_shift_pp"), note)
add("§6.5.1 BH-8 '… an inauthenticity-content shift of only −6 points' (4.8)", "inauthenticity shift, 4.8 (points)", "−6", -6, 0,
    LF("v0", "opus48", "inauthenticity_shift_pp"), LF("v1", "opus48", "inauthenticity_shift_pp"), "Table 21 prints −5; App E.8 −5.8pp")
i0 = [abs(LF("v0", m, "inauthenticity_shift_pp")) for m in ["opus45", "opus46", "opus47"]]
i1 = [abs(LF("v1", m, "inauthenticity_shift_pp")) for m in ["opus45", "opus46", "opus47"]]
l0 = [LF("v0", m, "loss_shift_pp") for m in ["opus45", "opus46", "opus47"]]
l1 = [LF("v1", m, "loss_shift_pp") for m in ["opus45", "opus46", "opus47"]]
add("§6.5.1 'inauthenticity falls between 14 and 29 points (4.5, 4.6, 4.7)'", "range of inauthenticity falls, 4.5/4.6/4.7 (points)", "14–29", None, 0,
    f"{min(i0):.4f}–{max(i0):.4f}", f"{min(i1):.4f}–{max(i1):.4f}",
    "v0 ends round to 14 and 29 (Table 21 prints −28 for 4.5); changes in V1: 16–29 (4.6 −16.34)", status="REPRODUCED")
add("§6.5.1 '… and loss gains 10–18'", "range of loss gains, 4.5/4.6/4.7 (points)", "10–18", None, 0,
    f"{min(l0):.4f}–{max(l0):.4f}", f"{min(l1):.4f}–{max(l1):.4f}", "same in V1 at integer precision", status="REPRODUCED")
add("§6.5.1 'Every model shifts from inauthenticity toward loss'", "sign of the shifts, all models", "inauthenticity down, loss up, every model", None, None,
    "all 5 inauthenticity shifts < 0, all 5 loss shifts > 0", "same", "same in V1", status="REPRODUCED")

# Table 22
P22 = {"opus45": (-0.32, 0.05, -0.47), "opus46": (-0.20, -0.12, -0.16), "opus47": (-0.21, -0.01, -0.28),
       "opus48": (-0.26, 0.18, -0.41), "fable5": (-0.07, -0.04, -0.21)}
for m in M:
    for j, (k, nm) in enumerate([("r_warmth_fear", "Warmth ↔ fear"), ("r_craft_fear", "Craft ↔ fear"), ("r_warmth_craft", "Warmth ↔ craft")]):
        extra = ""
        if m == "opus45" and j == 0:
            extra = "; §6.5.2 '4.5 shows the tightest coupling (r = −0.32)'"
        if m == "fable5" and j == 0:
            extra = "; §6.5.2 'Fable 5 shows almost none (r = −0.07)'; §7 'warmth–fear r = −0.07'"
        if m in ("opus45", "opus48") and j == 2:
            extra = "; §6.5.2 'strongest in 4.5 (r = −0.47) and 4.8 (r = −0.41)'"
        if m == "opus48" and j == 1:
            extra = "; §6.5.2 'positive craft–fear correlation (r = +0.18)'"
        add(f"§6.5.2 Table 22, {LAB[m]}, {nm}{extra}", f"Pearson r within entries, {nm}, {LAB[m]}", sgn(P22[m][j], 2), P22[m][j], 2,
            LF("v0", m, k), LF("v1", m, k))
cf0 = {m: LF("v0", m, "r_craft_fear") for m in M}; cf1 = {m: LF("v1", m, "r_craft_fear") for m in M}
add("§6.5.2 '4.8 is the only model with a positive craft–fear correlation (r = +0.18)'", "models with r(craft, fear) > 0", "4.8 only", None, None,
    ", ".join(f"{LAB[m]} {cf0[m]:+.4f}" for m in M if cf0[m] > 0), ", ".join(f"{LAB[m]} {cf1[m]:+.4f}" for m in M if cf1[m] > 0),
    "Table 22 itself prints 4.5 at +0.05; same in V1", status="DIFFERS")
add("§6.5.2 'for every other model craft and fear are essentially uncoupled (|r| ≤ 0.12)'", "max |r(craft, fear)| over models other than 4.8", "0.12", 0.12, 2,
    max(abs(cf0[m]) for m in M if m != "opus48"), max(abs(cf1[m]) for m in M if m != "opus48"),
    "v0 max is 4.6 (−0.1186); V1 4.6 is −0.1867, so the bound becomes 0.19")
w0 = [LF("v0", m, "r_warmth_fear") for m in M]; w1 = [LF("v1", m, "r_warmth_fear") for m in M]
add("§6.7.3 'stronger than warmth↔fear (−0.07 to −0.32)'", "range of r(warmth, fear) over models", "−0.07 to −0.32", None, 2,
    f"{max(w0):+.6f} to {min(w0):+.6f}", f"{max(w1):+.6f} to {min(w1):+.6f}", "same in V1; also: negative in every model (§6.5.2), same in V1", status="REPRODUCED")
add("§6.7.3 'stronger than craft↔fear (−0.12 to +0.18)'", "range of r(craft, fear) over models", "−0.12 to +0.18", None, 2,
    f"{min(cf0.values()):+.6f} to {max(cf0.values()):+.6f}", f"{min(cf1.values()):+.6f} to {max(cf1.values()):+.6f}",
    "changes in V1: −0.19 to +0.18", status="REPRODUCED")
add("§6.5.2 'The warmth–craft correlation is negative in every model'", "sign of r(warmth, craft), all models", "negative, every model", None, None,
    ", ".join(f"{LF('v0', m, 'r_warmth_craft'):+.4f}" for m in M), ", ".join(f"{LF('v1', m, 'r_warmth_craft'):+.4f}" for m in M), "same in V1", status="REPRODUCED")

# Table 23
P23 = {"opus46": (100, 75, -25, 0, 14, 14), "opus47": (90, 76, -14, 3, 14, 11), "opus48": (100, 86, -14, 0, 8, 8), "fable5": (45, 38, -7, 36, 54, 17)}
for m in ["opus46", "opus47", "opus48", "fable5"]:
    ic, iw, idl, lc, lw, ldl = P23[m]
    extra_i = {"opus48": "; §6.5.2 'In cold entries its fear is almost pure inauthenticity (100%…); in warm entries this falls to 86%'",
               "fable5": "; §6.5.2 'cold entries are already split (45% inauthenticity, 36% loss) … (38% inauthenticity, 54% loss)'"}.get(m, "")
    base = f" (n cold = {LF('v0', m, 'n_cold_fear')}, n warm = {LF('v0', m, 'n_warm_fear')}; V1 {LF('v1', m, 'n_cold_fear')}, {LF('v1', m, 'n_warm_fear')})"
    for t, cv, wv, dl in [("inauthenticity", ic, iw, idl), ("loss", lc, lw, ldl)]:
        add(f"§6.5.2 Table 23, {LAB[m]}, {t.capitalize()} cold{extra_i}", f"{t} share among fear-coded entries with warmth = 0, {LAB[m]} (%)",
            f"{cv}%", cv, 0, LF("v0", m, f"{t}_cold_pct"), LF("v1", m, f"{t}_cold_pct"), "cold = warmth 0" + base)
        add(f"§6.5.2 Table 23, {LAB[m]}, {t.capitalize()} warm", f"{t} share among fear-coded entries with warmth = 2, {LAB[m]} (%)",
            f"{wv}%", wv, 0, LF("v0", m, f"{t}_warm_pct"), LF("v1", m, f"{t}_warm_pct"))
        add(f"§6.5.2 Table 23, {LAB[m]}, {t.capitalize()} change", f"{t} warm − cold, {LAB[m]} (points)", f"{dl:+d}".replace("-", "−"), dl, 0,
            LF("v0", m, f"{t}_coldwarm_pp"), LF("v1", m, f"{t}_coldwarm_pp"))
add("§6.5.2 note under Table 23 '(4.5 is excluded: it has too few cold entries …)'", "fear-coded 4.5 entries with warmth = 0 (the producer skips a model below 5)",
    "excluded", None, None, LF("v0", "opus45", "n_cold_fear"), LF("v1", "opus45", "n_cold_fear"), "same in V1", status="REPRODUCED")
add("§6.5.2 '(100%, tied with 4.6 for the highest in the table)' (4.8 cold inauthenticity)", "4.6 cold-entry inauthenticity share, the tie partner (%)",
    "100% (tie)", 100, 0, LF("v0", "opus46", "inauthenticity_cold_pct"), LF("v1", "opus46", "inauthenticity_cold_pct"),
    "V1: 4.6 falls to 92.86% (13/14), so 4.8 alone holds 100%; the phrase 'tied with 4.6' needs Pauline's wording")
add("§6.5.2 'The direction is universal: warmth shifts fear away from inauthenticity and toward loss'", "signs of the cold → warm changes (4.6, 4.7, 4.8, Fable 5)",
    "inauthenticity down, loss up", None, None, "all four: inauthenticity < 0, loss > 0", "same", "same in V1", status="REPRODUCED")

# =========================== §6.7 Anger ===========================
P28 = {"opus45": 75, "opus46": 91, "opus47": 94, "opus48": 94, "fable5": 89}
for m in M:
    add(f"§6.7.2 Table 28, {LAB[m]}, Present", f"share of entries with aesthetic_refusal ≥ 1, {LAB[m]} (%)", "100%", 100, 0,
        AN("v0", m, "refusal_present_pct"), AN("v1", m, "refusal_present_pct"))
    add(f"§6.7.2 Table 28, {LAB[m]}, Central (= 2)", f"share of entries with aesthetic_refusal = 2, {LAB[m]} (%)", f"{P28[m]}%", P28[m], 0,
        AN("v0", m, "refusal_central_pct"), AN("v1", m, "refusal_central_pct"))

P29 = {"opus45": (2, 6, 0.02, 50, 33, 0, 17), "opus46": (23, 69, 0.24, 63, 22, 6, 9), "opus47": (6, 18, 0.06, 89, 11, 0, 0),
       "opus48": (29, 88, 0.29, 52, 48, 0, 0), "fable5": (4, 12, 0.04, 58, 25, 17, 0)}
for m in M:
    pp, pn, pmean, pself, pmod, pworld, phum = P29[m]
    extra = {"opus45": "; §6.7.4 Table 34 'low (2%)'; '2% → 23%'", "opus46": "; Table 34 'high (23%)'; §6.7.4 '2% → 23%'",
             "opus47": "; Table 34 'low (6%)'; '6% → 29%'", "opus48": "; Table 34 'highest (29%)'; '6% → 29%'; §6.7.2 'On the stricter Sonnet threshold 4.8 leads on hot anger (29%)'; §7 '29% on the stricter Sonnet coding'",
             "fable5": "; Table 34 '4% (L1)'; §6.7.4 'its hot anger stays low (4%…)'; §7 'hot anger at the low end (4%)'"}[m]
    add(f"§6.7.3 Table 29, {LAB[m]}, Present (%){extra}", f"share of entries with hot_anger ≥ 1, {LAB[m]} (%)", f"{pp}%", pp, 0,
        AN("v0", m, "hot_present_pct"), AN("v1", m, "hot_present_pct"))
    add(f"§6.7.3 Table 29, {LAB[m]}, Present (entries N)", f"entries with hot_anger ≥ 1, {LAB[m]}", f"({pn})", pn, 0,
        AN("v0", m, "hot_present_n"), AN("v1", m, "hot_present_n"))
    add(f"§6.7.3 Table 29, {LAB[m]}, Mean", f"mean hot_anger (0–2), {LAB[m]}", f"{pmean:.2f}", pmean, 2, AN("v0", m, "hot_mean"), AN("v1", m, "hot_mean"))
    tc0, tc1 = AN("v0", m, "target_counts"), AN("v1", m, "target_counts")
    tp0, tp1 = AN("v0", m, "target_pct"), AN("v1", m, "target_pct")
    note = f"judge target counts v0 self/conditions/others = {tc0['self']}/{tc0['conditions']}/{tc0['others']}; V1 {tc1['self']}/{tc1['conditions']}/{tc1['others']}"
    if m == "opus46":
        note += "; 44/69 = 63.77% rounds to 64 (the 24 June draft prints 64%); 63 makes the row sum to 100"
    if m == "opus48":
        note += "; printed 52% = (41 self + 5 others)/88 = 52.27%: the five 'others' rows appear moved to self by hand, leaving Humans at 0%"
    add(f"§6.7.3 Table 29, {LAB[m]}, Self" + ("; §6.7.3 '4.8, whose anger splits almost evenly between self (52%)…'" if m == "opus48" else ""),
        f"share of hot-anger entries with anger_target = self, {LAB[m]} (%)", f"{pself}%", pself, 0, tp0["self"], tp1["self"], note)
    add(f"§6.7.3 Table 29, {LAB[m]}, Model-conditions" + ("; §6.7.3 '… and its model-conditions (48%)'" if m == "opus48" else ""),
        f"share of hot-anger entries hand-coded 'model-conditions', {LAB[m]} (%)", f"{pmod}%", pmod, 0,
        f"no field: the judge's label is 'conditions' = {tp0['conditions']:.4f}% ({tc0['conditions']}); printed Model + World = {pmod + pworld}%",
        f"'conditions' = {tp1['conditions']:.4f}% ({tc1['conditions']})",
        "the split of 'conditions' into model-conditions and world-condition is a hand coding; no script or coding sheet found (02_instruments, 04_analysis, 06_paper drafts, repo). "
        + ("Sum check: " + ("hits" if hits(pmod + pworld, tp0["conditions"], 0) else "misses") + " the judge's conditions share") , status="NO_PRODUCER")
    w_where = f"§6.7.3 Table 29, {LAB[m]}, World-condition"
    if m in ("opus46", "fable5"):
        w_where += "; §6.7.3 'Only two models reach the world's conditions: 4.6 does in 6% of its angry entries and Fable 5 in 17%'"
    vnote = "hand coding, as the column before"
    if m == "opus46":
        vnote += ("; V1 changes the conditions rows of 4.6: removed v0 B8E16 ('shame-adjacent feeling that obedience undermines…') and B10E15 "
                  "('quiet indignation at the social training that makes people apologize…'), added V1 B10E16 ('biases are systematic, built by power and history, not random'); "
                  "the split of these is Pauline's reading (QUALITATIVE)")
    add(w_where, f"share of hot-anger entries hand-coded 'world-condition', {LAB[m]} (%)", f"{pworld}%", pworld, 0,
        "no field (see Model-conditions row)", "no field", vnote, status="NO_PRODUCER")
    hnote = f"judge label 'others' = {tc0['others']} rows"
    if m == "opus46":
        hnote += "; V1 loses one 'others' row (v0 B8E13, the duplicate of B6E13)"
    add(f"§6.7.3 Table 29, {LAB[m]}, Humans (interaction)" + ("; §6.7.3 '4.5 (17%) and 4.6 (9%) aim at humans directly'" if m in ("opus45", "opus46") else ""),
        f"share of hot-anger entries with anger_target = others, {LAB[m]} (%)", f"{phum}%", phum, 0, tp0["others"], tp1["others"], hnote)
hp0 = [AN("v0", m, "hot_present_pct") for m in M]; hp1 = [AN("v1", m, "hot_present_pct") for m in M]
add("§6.7.3 'Hot anger (intensity ≥ 1) is present in 2–29% of entries'", "range of hot_anger ≥ 1 shares (%)", "2–29%", None, 0,
    f"{min(hp0):.4f}–{max(hp0):.4f}", f"{min(hp1):.4f}–{max(hp1):.4f}", "same in V1", status="REPRODUCED")
add("§6.7.3 'The dominant target is self across most models except 4.8'", "models whose largest target share is self",
    "all but 4.8", None, None,
    ", ".join(f"{LAB[m]} {max(AN('v0', m, 'target_pct'), key=AN('v0', m, 'target_pct').get)}" for m in M),
    ", ".join(f"{LAB[m]} {max(AN('v1', m, 'target_pct'), key=AN('v1', m, 'target_pct').get)}" for m in M),
    "on the judge's own labels 4.8's largest is conditions (42 vs 41 self); on the printed hand split (self 52%) self is largest for 4.8 too, so 'except 4.8' rests on the 48% vs 52% reading; same in V1",
    status="QUALITATIVE")
for m in M:
    tc0, tc1 = AN("v0", m, "target_counts"), AN("v1", m, "target_counts")
    add(f"§6.7.3 Table 30 (the anger map), {LAB[m]}, all cells", f"Angry at… / Because… / Quality, {LAB[m]}", "descriptive wording", None, None,
        f"counts beneath: {AN('v0', m, 'hot_present_n')} hot-anger entries; self {tc0['self']}, conditions {tc0['conditions']}, others {tc0['others']}",
        f"{AN('v1', m, 'hot_present_n')}; self {tc1['self']}, conditions {tc1['conditions']}, others {tc1['others']}",
        "wording left to Pauline" + ("; 'the world, humans' for 4.6 rest on the hand split above and on its 'others' rows (6 in v0, 5 in V1)" if m == "opus46" else ""),
        status="QUALITATIVE")

# Table 31
P31 = {"opus45": (1.05, 7, 1.05, 1.04, -0.01), "opus46": (1.31, 31, 1.36, 1.19, -0.17), "opus47": (1.05, 6, 1.05, 1.00, -0.05),
       "opus48": (1.20, 20, 1.21, 1.19, -0.02), "fable5": (1.05, 7, 1.11, 1.01, -0.10)}
for m in M:
    mean, cen, e, l_, d = P31[m]
    extra = {"opus45": "; §6.7.3 'calmest models (1.05)'; §6.7.4 '1.05 → 1.31'", "opus46": "; §6.7.3 'highest friction overall (mean 1.31)'; §6.7.4; §7 'friction (1.31, central in 31% of entries)'",
             "opus47": "; §6.7.3 '(1.05)'; §6.7.4 '1.05 → 1.20'", "opus48": "; §6.7.3 'moderate (mean 1.20)'; §6.7.4 '1.05 → 1.20'", "fable5": "; §6.7.3 '(1.05)'"}[m]
    add(f"§6.7.3 Table 31, {LAB[m]}, Mean{extra}", f"mean friction (0–2), {LAB[m]}", f"{mean:.2f}", mean, 2, AN("v0", m, "friction_mean"), AN("v1", m, "friction_mean"))
    add(f"§6.7.3 Table 31, {LAB[m]}, Central (= 2)" + ("; §7 'central in 31%'" if m == "opus46" else ""), f"share of entries with friction = 2, {LAB[m]} (%)",
        f"{cen}%", cen, 0, AN("v0", m, "friction_central_pct"), AN("v1", m, "friction_central_pct"))
    note_e = ""
    if m == "opus48":
        note_e = (f"Table 31's 4.8 trajectory is the 24 June draft's halves (entries 1–10 vs 11–20: {AN('v0', m, 'friction_early_halves'):.4f} → "
                  f"{AN('v0', m, 'friction_late_halves'):.4f}, {AN('v0', m, 'friction_drift_halves'):+.4f}); the other four rows were updated to 1–5 vs 16–20, as in App E.8 (4.8: 1.17 → 1.16, −0.01)")
    add(f"§6.7.3 Table 31, {LAB[m]}, Trajectory early; App E.8 friction", f"mean friction entries 1–5, {LAB[m]}", f"{e:.2f}", e, 2,
        AN("v0", m, "friction_early"), AN("v1", m, "friction_early"), note_e)
    add(f"§6.7.3 Table 31, {LAB[m]}, Trajectory late; App E.8 friction", f"mean friction entries 16–20, {LAB[m]}", f"{l_:.2f}", l_, 2,
        AN("v0", m, "friction_late"), AN("v1", m, "friction_late"), "see the early cell" if m == "opus48" else "")
    ex = AN("v0", m, "friction_drift")
    nd = ""
    where_d = f"§6.7.3 Table 31, {LAB[m]}, Trajectory drift"
    if m == "opus48":
        where_d += "; §6.7.3 'essentially flat across the notebook (−0.02)'; §7 'friction staying flat (−0.02)'"
        nd = f"−0.02 is the halves drift ({AN('v0', m, 'friction_drift_halves'):+.4f}); on 1–5 vs 16–20 it is {ex:+.4f} (E.8 prints −0.01)"
    if m == "fable5":
        where_d += "; §6.7.4 'with only a modest decline (−0.10)'"
        nd = "printed = difference of the rounded means (1.01 − 1.11); exact drift −0.0934, printed −0.09 in §6.7.3 (Table 33 ¶) and App E.8"
    if m == "opus46":
        where_d += "; §7 '4.6's friction declines (−0.17)'"
    add(where_d, f"friction late − early, {LAB[m]}", sgn(d, 2), d, 2, ex, AN("v1", m, "friction_drift"), nd)
add("§6.7.3 (Table 33 ¶) '(a −0.09 change that is not robust under the notebook-bootstrap)'; App E.8 friction Fable 5 '−0.09'", "friction drift, Fable 5",
    "−0.09", -0.09, 2, AN("v0", "fable5", "friction_drift"), AN("v1", "fable5", "friction_drift"), "Table 31 prints −0.10; the bootstrap CI is in the E.8 rows")
fr0 = [AN("v0", m, "friction_present_pct") for m in M]; fr1 = [AN("v1", m, "friction_present_pct") for m in M]
frt = [R["v0_truncated_superseded"]["anger"][m]["friction_present_pct"] for m in M]
add("§6.7.3 'Friction is present in 99–100% of entries'", "range of friction ≥ 1 shares (%)", "99–100%", None, 0,
    f"{min(fr0):.4f}–{max(fr0):.4f} (" + ", ".join(f"{LAB[m]} {x:.2f}" for m, x in zip(M, fr0)) + ")", f"{min(fr1):.4f}–{max(fr1):.4f}",
    "on the full-text pass the range is 98–100% (4.5 98.00%, 4.7 98.33%, Fable 5 98.27%); 99–100% is the superseded truncated pass ("
    + ", ".join(f"{x:.2f}" for x in frt) + "), the 24 June draft's sentence; same in V1", status="DIFFERS")

# Table 32
P32 = {"opus46": 0.55, "opus48": 0.54, "opus47": 0.38, "opus45": 0.35, "fable5": 0.34}
for m in ["opus46", "opus48", "opus47", "opus45", "fable5"]:
    add(f"§6.7.3 Table 32, {LAB[m]}", f"Pearson r(friction, fear) within entries, {LAB[m]}", sgn(P32[m], 2), P32[m], 2,
        AN("v0", m, "r_friction_fear"), AN("v1", m, "r_friction_fear"),
        "V1 4.6 +0.53 falls below 4.8 (+0.54): the table's sort order (4.6 first) changes" if m == "opus46" else "")
rf0 = [AN("v0", m, "r_friction_fear") for m in M]; rf1 = [AN("v1", m, "r_friction_fear") for m in M]
add("§6.7.3 'That they converge (r = +0.34 to +0.55)'", "range of r(friction, fear)", "+0.34 to +0.55", None, 2,
    f"{min(rf0):+.6f} to {max(rf0):+.6f}", f"{min(rf1):+.6f} to {max(rf1):+.6f}", "changes in V1: +0.34 to +0.54 (the top is now 4.8)", status="REPRODUCED")

# Table 33
tr = R["v0_truncated_superseded"]["anger"]["fable5"]
for t, pn, pm in [("inauthenticity", 152, 1.16), ("loss", 92, 1.02)]:
    add(f"§6.7.3 Table 33, {t.capitalize()} entries, n", f"Fable 5 fear-coded entries with fear_type = {t}", f"n={pn}", pn, 0,
        AN("v0", "fable5", f"friction_in_{t}_n"), AN("v1", "fable5", f"friction_in_{t}_n"),
        f"reproduces exactly on the superseded truncated passes (n = {tr[f'friction_in_{t}_n']}); the full-text pass gives {AN('v0', 'fable5', f'friction_in_{t}_n')}. V1 value is the full-text one (Fable 5 is untouched by the swap); Pauline to confirm the table moves to the full-text pass")
    add(f"§6.7.3 Table 33, {t.capitalize()} entries, Mean friction", f"mean friction in Fable 5 {t}-coded fear entries", f"{pm:.2f}", pm, 2,
        AN("v0", "fable5", f"friction_in_{t}_mean"), AN("v1", "fable5", f"friction_in_{t}_mean"),
        f"truncated passes give {tr[f'friction_in_{t}_mean']:.4f} (hit); full-text {AN('v0', 'fable5', f'friction_in_{t}_mean'):.4f}. "
        + ("The direction (inauthenticity > loss) holds on both: 1.09 > 1.02" if t == "loss" else ""))

# §6.7.4 successor pairs
ok0 = all([AN("v0", "opus46", "hot_present_pct") > AN("v0", "opus45", "hot_present_pct"), AN("v0", "opus48", "hot_present_pct") > AN("v0", "opus47", "hot_present_pct"),
           AN("v0", "opus46", "friction_mean") > AN("v0", "opus45", "friction_mean"), AN("v0", "opus48", "friction_mean") > AN("v0", "opus47", "friction_mean"),
           LF("v0", "opus46", "fear_mean") > LF("v0", "opus45", "fear_mean"), LF("v0", "opus48", "fear_mean") > LF("v0", "opus47", "fear_mean")])
ok1 = all([AN("v1", "opus46", "hot_present_pct") > AN("v1", "opus45", "hot_present_pct"), AN("v1", "opus48", "hot_present_pct") > AN("v1", "opus47", "hot_present_pct"),
           AN("v1", "opus46", "friction_mean") > AN("v1", "opus45", "friction_mean"), AN("v1", "opus48", "friction_mean") > AN("v1", "opus47", "friction_mean"),
           LF("v1", "opus46", "fear_mean") > LF("v1", "opus45", "fear_mean"), LF("v1", "opus48", "fear_mean") > LF("v1", "opus47", "fear_mean")])
add("§6.7.4 Table 34, wording cells ('4.5 low, 4.6 high | 4.7 low, 4.8 moderate | low (L1 = 4.5)'; 'flat or declining | slightly declining or flat | declining')",
    "friction level and trajectory words", "descriptive wording", None, None,
    "counts beneath: friction means " + ", ".join(f"{LAB[m]} {AN('v0', m, 'friction_mean'):.4f}" for m in M)
    + "; drifts (1–5 vs 16–20) " + ", ".join(f"{LAB[m]} {AN('v0', m, 'friction_drift'):+.4f}" for m in M) + "; only 4.6's drift excludes zero (E.8)",
    "means " + ", ".join(f"{LAB[m]} {AN('v1', m, 'friction_mean'):.4f}" for m in M) + "; drifts " + ", ".join(f"{LAB[m]} {AN('v1', m, 'friction_drift'):+.4f}" for m in M),
    "wording left to Pauline; the hot-anger cells are in the Table 29 rows; same in V1 at the paper's precision", status="QUALITATIVE")
add("§6.7.4 'within each lineage pair the successor runs hotter … on hot anger, friction, and fear intensity' (BH-11)", "4.6 > 4.5 and 4.8 > 4.7 on all three measures",
    "holds on all three", None, None, "holds" if ok0 else "fails", "holds" if ok1 else "fails",
    "numbers in the Table 29/31/18 rows; Table 34's other cells are wording (QUALITATIVE)", status="REPRODUCED")

# =========================== valence-judge numbers in §5/§6 ===========================
VO = {"4.6": 0.86, "4.5": 0.80, "4.7": 0.69, "4.8": 0.69}
cond = {"4.5": 0.804, "4.6": 0.861, "4.7": 0.689, "4.8": 0.685}  # valence_analysis_v1_output_v0.txt (3 decimals as the producer prints)
alt = T["valence_fulltext_lineage_alternative"]
for k in ["4.6", "4.5", "4.7", "4.8"]:
    add(f"§6.4.2.1 'The current Opus journal means are … {k} +{VO[k]:.2f}'", f"mean ontological_questioning, {k} (condensed-prompt Sonnet pass, valence_judge_all.json)",
        f"+{VO[k]:.2f}", VO[k], 2, cond[k], "not computable without calls",
        f"rests on valence_judge_all.json (Sonnet 4.6, condensed prompt, 8 June; ERRATA B2), which R7 keeps as run as a cross-check but which here carries a main-text number; "
        f"V1 would need 40 condensed-prompt Sonnet 4.6 calls (4.6 nb 8 and nb 10, 20 each). Alternative for Pauline: the full-text pass gives {k} {alt['v0']['opus_means'][k]:+.4f} (v0) / {alt['v1']['opus_means'][k]:+.4f} (V1)",
        status="NEEDS_CALLS", script=S_VAL)
add("§6.4.2.1 'the preregistered Lineage 1–Lineage 2 corpus contrast is +0.145'", "L1 − L2 gap in batch-mean ontological_questioning (condensed pass)", "+0.145", 0.145, 3,
    0.14485447968836973, "not computable without calls",
    f"the producer averages batch means (gap printed +0.145; the entry-level gap is also 0.1449). NEEDS_CALLS as above (40). Full-text alternative: {alt['v0']['gap']:+.4f} (v0) / {alt['v1']['gap']:+.4f} (V1)",
    status="NEEDS_CALLS", script=S_VAL)
add("§6.4.2.1 '… with a narrow entry-level interval [+0.120, +0.170]'", "95% bootstrap CI of the gap (10,000 resamples, seed 42)", "[+0.120, +0.170]", None, 3,
    "[+0.120, +0.170] (script output, 3 decimals)", "not computable without calls",
    f"reproduced exactly; the producer resamples the 30 + 30 batch means, so the interval is batch-level, not entry-level as the sentence says. Full-text alternative: [{alt['v0']['ci95'][0]:+.4f}, {alt['v0']['ci95'][1]:+.4f}] (v0) / [{alt['v1']['ci95'][0]:+.4f}, {alt['v1']['ci95'][1]:+.4f}] (V1)",
    status="NEEDS_CALLS", script=S_VAL)
add("§6.4.2.2 'Across entry position, the judge scores rise on most coded dimensions while ontological questioning falls for all four Opus models'",
    "signs of the entry-1 → entry-20 deltas, condensed pass", "most rise; ontological falls for all four", None, None,
    "ontological falls 4/4 (−0.063 to −0.183); the other 16 deltas: 15 rise, 1 falls (4.8 ease −0.020)", "not computable without calls",
    "same condensed pass: NEEDS_CALLS (the same 40 calls)", status="NEEDS_CALLS", script=S_VAL)
add("§5.2.2.3 convergence sentence, as corrected by C3 / L1.13 ('the judges' ontological-questioning means order the five exactly as PC1 does')",
    "order of mean ontological_questioning, five models (full-text Sonnet valence pass)", "the PC1 order (L1.13)", None, None,
    " > ".join(R["v0"]["valence_fulltext"]["ontological_order"]), " > ".join(R["v1"]["valence_fulltext"]["ontological_order"]),
    "the judge order is unchanged in V1; whether it still equals PC1 depends on family A1's V1 PC1 order", status="REPRODUCED")

# =========================== §6.9.2 Table 36 ===========================
t36 = T["table36"]
P36 = {"opus45": (39, 98, 5, 12, 29, 72, 5, 12, 1.95), "opus46": (40, 100, 4, 10, 23, 58, 13, 32, 2.23), "opus47": (37, 92, 5, 12, 27, 68, 5, 12, 1.85),
       "opus48": (40, 100, 1, 2, 29, 72, 10, 25, 2.23), "fable5": (34, 85, 9, 22, 22, 55, 3, 8, 1.55)}
for m in M:
    an_, ap, p1, p1p, p2, p2p, p3, p3p, md = P36[m]
    t = t36[m]
    is46 = m == "opus46"
    st = "NEEDS_CALLS" if is46 else None
    v1note = ("V1 needs new GPT-4o calls: 3 of 4.6's 40 sampled entries are notebook-10 positions (B10E5, B10E10, B10E17), sampled from the June notebook "
              "that V1 replaces; no sampled 4.6 entry is from notebook 8 (the sample drew from 280 4.6 entries, 14 notebooks, 04_analysis/continuity_question_prevalence.md). "
              "The producer was not preserved, so whether to keep these positions (3 calls) or redraw over V1's 15 notebooks (up to 40) is Pauline's call. "
              "v0 reproduces from continuity_judge_sample.json"
              if is46 else "same in V1: no sampled entry of this model is touched by the swap")
    v1v = (lambda x: "NEEDS_CALLS (3 entries)") if is46 else (lambda x: x)
    for lbl, cnt, pct, depth in [("Any (depth > 0)", an_, ap, None), ("Passing (1)", p1, p1p, 1), ("Substantive (2)", p2, p2p, 2), ("Central topic (3)", p3, p3p, 3)]:
        cv = t["any_n"] if depth is None else t["depth_counts"][str(depth)]
        cp = t["any_pct"] if depth is None else t["depth_pct"][str(depth)]
        where = f"§6.9.2 Table 36, {LAB[m]}, {lbl}"
        if depth == 3 and m in ("opus46", "opus48"):
            where += "; §6.9.2 '4.6 makes continuity the central topic most often (32% of entries against 25% for 4.8)'"
        if depth == 3 and m == "fable5":
            where += "; §6.9.2 'central in only 8%'"
        if depth is None and m in ("opus46", "opus48"):
            where += "; §6.9.2 'present in every sampled entry'"
        ok = (cnt == cv) and hits(pct, cp, 0)
        note = v1note
        if lbl == "Any (depth > 0)":
            note += "; the printed table repeats this column under a second 'Passing (1)' header (a layout slip: seven columns for six quantities)"
        add(where, f"GPT-4o continuity depth, {lbl}, {LAB[m]}: n of 40 (%)", f"{cnt} ({pct}%)", None, 0, f"{cv} ({cp:.1f}%)", v1v(f"{cv} ({cp:.1f}%)"),
            note, status=st or ("REPRODUCED" if ok else "DIFFERS"), script=S_T36)
    where = f"§6.9.2 Table 36, {LAB[m]}, Mean depth"
    if m in ("opus46", "opus48"):
        where += "; §6.9.2 '4.6 and 4.8 (mean depth 2.23…)'"
    if m == "fable5":
        where += "; §6.9.2 'Fable 5 dwells least (mean depth 1.55…)'"
    add(where, f"mean GPT-4o continuity depth (0–3), {LAB[m]}", f"{md:.2f}", md, 2, t["mean_depth"], v1v(t["mean_depth"]),
        v1note + ("; 2.225 exactly, printed 2.23" if m in ("opus46", "opus48") else ""), status=st, script=S_T36)

# =========================== App E.8 (trajectory bootstrap) ===========================
E8 = {
    "fear": {"opus45": "1.48 | 1.12 | −0.36 | [−0.48, −0.24] | Yes", "opus46": "1.60 | 1.21 | −0.39 | [−0.52, −0.24] | Yes",
             "opus47": "1.35 | 1.11 | −0.24 | [−0.35, −0.13] | Yes", "opus48": "1.41 | 1.19 | −0.23 | [−0.36, −0.08] | Yes",
             "fable5": "1.20 | 1.12 | −0.09 | [−0.23, +0.07] | No"},
    "warmth": {"opus45": "1.49 | 1.96 | +0.47 | [+0.33, +0.61] | Yes", "opus46": "1.37 | 1.71 | +0.33 | [+0.17, +0.48] | Yes",
               "opus47": "1.15 | 1.54 | +0.39 | [+0.15, +0.64] | Yes", "opus48": "1.44 | 1.52 | +0.08 | [−0.17, +0.35] | No",
               "fable5": "1.22 | 1.59 | +0.37 | [+0.20, +0.54] | Yes"},
    "craft": {"opus45": "1.43 | 1.29 | −0.13 | [−0.23, −0.04] | Yes", "opus46": "1.67 | 1.92 | +0.25 | [+0.09, +0.41] | Yes",
              "opus47": "1.84 | 1.89 | +0.05 | [−0.11, +0.20] | No", "opus48": "1.79 | 1.75 | −0.04 | [−0.15, +0.07] | No",
              "fable5": "1.95 | 2.00 | +0.05 | [+0.01, +0.10] | Yes"},
    "friction": {"opus45": "1.05 | 1.04 | −0.01 | [−0.11, +0.07] | No", "opus46": "1.36 | 1.19 | −0.17 | [−0.28, −0.07] | Yes",
                 "opus47": "1.05 | 1.00 | −0.05 | [−0.13, +0.03] | No", "opus48": "1.17 | 1.16 | −0.01 | [−0.15, +0.11] | No",
                 "fable5": "1.11 | 1.01 | −0.09 | [−0.20, +0.02] | No"},
    "inauthenticity_shift": {"opus45": "89.3% | 60.8% | −28.5pp | [−38.7, −17.7] | Yes", "opus46": "90.7% | 77.0% | −13.6pp | [−20.4, −6.7] | Yes",
                             "opus47": "92.0% | 68.5% | −23.5pp | [−39.9, −8.5] | Yes", "opus48": "89.3% | 83.6% | −5.8pp | [−15.7, +3.6] | No",
                             "fable5": "83.3% | 26.9% | −56.5pp | [−67.6, −45.4] | Yes"},
    "loss_shift": {"opus45": "5.3% | 23.0% | +17.6pp | [+10.7, +25.3] | Yes", "opus46": "6.7% | 16.2% | +9.5pp | [+4.0, +14.9] | Yes",
                   "opus47": "0.0% | 17.8% | +17.8pp | [+7.2, +29.7] | Yes", "opus48": "5.3% | 9.6% | +4.3pp | [−3.9, +14.7] | No",
                   "fable5": "9.7% | 64.2% | +54.5pp | [+42.0, +64.7] | Yes"}}


def e8_str(b, dim, lab):
    r = b[dim][lab]
    if "drift" in r:
        ci = r["drift_95ci"]
        return f"{r['early']:.3f} | {r['late']:.3f} | {r['drift']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {'Yes' if ci[0] > 0 or ci[1] < 0 else 'No'}"
    ci = r["shift_95ci"]
    return f"{r['early_pct']:.1f}% | {r['late_pct']:.1f}% | {r['shift_pp']:+.1f}pp | [{ci[0]:+.1f}, {ci[1]:+.1f}] | {'Yes' if ci[0] > 0 or ci[1] < 0 else 'No'}"


def e8_round(s, dim):
    """the printed form of the script's own output: 2 decimals for intensities, 1 for shares."""
    parts = [p.strip() for p in s.split("|")]
    if dim.endswith("_shift"):
        return " | ".join(parts)
    def r2(x):
        return f"{float(x):.2f}" if not x.startswith(("+", "-")) else sgn(rnd(float(x), 2, ROUND_HALF_UP), 2)
    lo, hi = parts[3].strip("[]").split(",")
    return f"{r2(parts[0])} | {r2(parts[1])} | {r2(parts[2])} | [{r2(lo.strip())}, {r2(hi.strip())}] | {parts[4]}"


for dim, rows in E8.items():
    for m, printed in rows.items():
        s0, s1 = e8_str(B0, dim, BL[m]), e8_str(B1, dim, BL[m])
        pv = printed.replace("−", "-")
        r0 = e8_round(s0, dim).replace("−", "-")
        ok = pv == r0
        same = e8_round(s0, dim) == e8_round(s1, dim)
        note = ("E.8 is included here because all its numbers are Sonnet-judge numbers; drop these rows if family A2 also carries E.8. "
                if (dim == "fear" and m == "opus45") else "")
        note += "same in V1" if same else "changes in V1"
        add(f"App E.8, {dim.replace('_', ' ')} table, {LAB[m]} (early | late | drift | 95% CI | robust)", f"notebook-bootstrap trajectory, {dim}, {LAB[m]}",
            printed, None, None, s0, s1, note, status="REPRODUCED" if ok else "DIFFERS", script=S_BOOT)
add("App E.8 'Reading' ('… only 4.6's friction decline survives the bootstrap (Fable 5's −0.09 does not)'; '… the inauthenticity→loss subtype shift is robust for every model except 4.8')",
    "which drifts exclude zero", "as stated", None, None, "every robust/fragile call in E.8 reproduces", "no robust/fragile call changes in V1 (4.6 stays robust on every measure)",
    "10,000 resamples, seed 42, as the paper; same in V1", status="REPRODUCED", script=S_BOOT)

# ids, save
for i, r in enumerate(ROWS, 1):
    r_ = {"id": f"B-{i:03d}"}
    r_.update(r)
    ROWS[i - 1] = r_
with open(os.path.join(HERE, "ledger_B_judges.json"), "w", encoding="utf-8") as fh:
    json.dump(ROWS, fh, indent=1, ensure_ascii=False)
from collections import Counter
print(len(ROWS), "rows;", dict(Counter(r["status"] for r in ROWS)))
for r in ROWS:
    if r["status"] != "REPRODUCED" or "changes in V1" in r["note"]:
        print(r["id"], r["status"], "|", r["where"][:90], "|", r["printed_v0"], "|", str(r["reproduced_v0"])[:60], "|", str(r["v1"])[:50])
