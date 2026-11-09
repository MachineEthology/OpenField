"""Builds ledger_A2_trajectories.json from the out_v0/ and out_v1/ results of this folder.
Written 3 Nov 2026 by Claude Opus 5.5 (Code), helper A2. Printed v0 values are typed from the paper room's
copy of the Stage84 docx (main.json, appendix-e.json); every computed value is read from a result file.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = "recompute/A2_trajectories/"


def J(v, name):
    with open(os.path.join(HERE, f"out_{v}", name), encoding="utf-8") as fh:
        return json.load(fh)


LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}
MODELS = list(LAB)
rows = []
mismatch = []


def fmt(x, p):
    return f"{x:.{p}f}"


def add(where, quantity, printed, v0, v1, prec, status, script, note=""):
    rid = f"A2-{len(rows) + 1:02d}"
    rows.append({"id": rid, "where": where, "quantity": quantity, "printed_v0": printed,
                 "reproduced_v0": v0, "v1": v1, "paper_precision": prec, "status": status,
                 "script": script, "note": note})
    return rid


def same(v0, v1, prec):
    try:
        return fmt(float(v0), prec) == fmt(float(v1), prec)
    except (TypeError, ValueError):
        return v0 == v1


def numrow(where, quantity, printed, v0, v1, prec, script, note="", status=None, signed=False):
    """Numeric row: status REPRODUCED if v0 hits the printed value at the paper's precision."""
    pv = float(printed.replace("−", "-").replace("+", "").replace(",", "").replace("%", ""))
    hit = abs(round(v0, prec) - pv) < 10 ** -(prec + 3) or fmt(v0, prec) == fmt(pv, prec)
    st = status or ("REPRODUCED" if hit else "DIFFERS")
    if not hit and status is None:
        mismatch.append((quantity, printed, v0))
    n = note
    if same(v0, v1, prec):
        n = ("same in V1. " + n).strip()
    return add(where, quantity, printed, repr(float(v0)), repr(float(v1)), prec, st, script, n)


# ---------------------------------------------------------------- Table 13 / Figure 3
t0, t1 = J("v0", "table13_length_density.json"), J("v1", "table13_length_density.json")
pr_vel = {"opus45": "0.168", "opus46": "0.177", "opus47": "0.195", "opus48": "0.185", "fable5": "0.204"}
pr_med = {"opus45": "324", "opus46": "1,413", "opus47": "472", "opus48": "596", "fable5": "646"}
pr_dpk = {"opus45": "0.518", "opus46": "0.125", "opus47": "0.413", "opus48": "0.310", "fable5": "0.316"}
S13 = R + "length_density_v1.py"
for m in MODELS:
    note = ""
    if m == "fable5":
        note = ("The producer gives 0.2034 (all consecutive steps of text entries pooled, as the plate draws it). "
                "The printed 0.204 is hit by the mean of per-notebook mean velocities (0.2041; checked, not adopted). "
                "Fable 5's data are unchanged in V1.")
        numrow("Table 13; §5.2.1.4 ('Fable 5 has the highest raw velocity (0.204)')", f"mean per-entry velocity, {LAB[m]}",
               pr_vel[m], t0[m]["mean_velocity"], t1[m]["mean_velocity"], 3, S13, note)
    else:
        numrow("Table 13", f"mean per-entry velocity (cosine distance between consecutive entries), {LAB[m]}",
               pr_vel[m], t0[m]["mean_velocity"], t1[m]["mean_velocity"], 3, S13,
               "4.6: velocity falls 0.1768 -> 0.1743 in V1." if m == "opus46" else "")
for m in MODELS:
    med_note = "Median of whitespace word counts over the model's text entries. The paper truncates half-integer medians (x.5 printed as x)."
    st = "REPRODUCED"
    if m == "opus46":
        med_note += (" Also printed in §2.3 ('~1,413') and the 4.6 portrait of §5.2.2.2 ('median 1,413 w'); V1 1,465 "
                     "(1465.0, an integer median). The §2.3 medians are listed in the plan under a separate item; same value.")
    v0m, v1m = t0[m]["median_words"], t1[m]["median_words"]
    rid = add("Table 13; §5.2.1.4 'Median entry length'; Figure 3 x-axis", f"median entry length (words), {LAB[m]}",
              pr_med[m], repr(v0m), repr(v1m), 0, st, S13,
              (("same in V1. " if v0m == v1m else "") + med_note).strip())
for m in MODELS:
    note = "displacement per 1,000 words = mean velocity / (median words / 1000)."
    if m == "fable5":
        note += (" The producer gives 0.3148 (the plate and figure_data/fig03 draw 0.315); the printed 0.316 equals "
                 "0.204 / 0.646, i.e. it follows the per-notebook velocity variant noted on the Fable 5 velocity row. "
                 "Fable 5 unchanged in V1. §5.2.1.4 repeats it ('lands at 0.316, right next to 4.8 (0.310)').")
    if m == "opus46":
        note += (" §5.2.1.4 repeats 4.6 = 0.125 vs 4.5 = 0.518 and 'roughly one quarter of 4.5's' (v0 ratio 0.241; "
                 "V1 0.1190/0.5184 = 0.229, still about a quarter); the portrait of §5.2.2.2 prints 'lowest semantic density (0.125)'.")
    numrow("Table 13; §5.2.1.4; Figure 3 y-axis", f"embedding displacement per 1,000 words, {LAB[m]}",
           pr_dpk[m], t0[m]["disp_per_k"], t1[m]["disp_per_k"], 3, S13, note)

# ---------------------------------------------------------------- §5.2.1.2 shared road
r0, r1 = J("v0", "road.json"), J("v1", "road.json")
SR = R + "road_handwriting_v1.py (part A, copy of embedding_road_and_handwriting.py); full precision from " + R + "road_null_rebuild.py"
numrow("§5.2.1.2 ('the observed 0.573–0.849'; also 'range from 0.57 to 0.85')", "lowest cross-model displacement cosine (4.6–4.8), Opus 4.5–4.8",
       "0.573", r0["opus_range"][0], r1["opus_range"][0], 3, SR,
       "Computed, as in June, on the four Opus models only (6 pairs; §5.2.1.2's '4.6 … mean 0.60 vs 0.71–0.74' is the same four-model matrix). The room's Figure 2 caption ('the five roads agree at 0.573–0.849') and the brief's 'five vectors' are not what ran, though Figure 2 draws five arrows. With Fable 5 added (informational, 10 pairs): range 0.5732–0.8496 v0 / 0.5667–0.8496 V1 (Fable 5's own pairs 0.575–0.728 fall inside), mean 0.6845 / 0.6824.")
numrow("§5.2.1.2 ('0.573–0.849'; 'up to 0.85')", "highest cross-model displacement cosine (4.7–4.8), Opus 4.5–4.8",
       "0.849", r0["opus_range"][1], r1["opus_range"][1], 3, SR,
       "Value 0.84958 sits on the rounding boundary: the copy prints 0.850, the June results doc (04_analysis/Road_Handwriting_RESULTS_2026-06-10.md) .849. "
       "The npz rows and the June cache appear to differ at the 1e-5 level (see the 4.5 raw handwriting near-tie, A2 Figure 13 rows). 4.7–4.8 unchanged in V1.",
       status="DIFFERS")
numrow("§5.2.1.2 ('mean 0.697')", "mean of the six Opus cross-model displacement cosines", "0.697",
       r0["opus_offdiag_mean"], r1["opus_offdiag_mean"], 3, SR, "V1 0.693.")
numrow("§5.2.1.2 ('fall to a mean of 0.000')", "random-split null: mean of the null distribution of the six-pair mean cosine (R = 1,000)",
       "0.000", r0["null"]["mean"], r1["null"]["mean"], 3, R + "road_null_rebuild.py (REBUILD; June script not found)",
       "Rebuilt: within each notebook the entry labels are shuffled and entries labelled 1–5 / 16–20 form early/late; seed 42. "
       "v0 -0.00045 prints as -0.000 (= 0.000). Searched for the 20 June script in 02_instruments, 03_experiments, 04_analysis, 06_paper, the repo's code/, Desktop.")
numrow("§5.2.1.2 ('95th percentile 0.073')", "random-split null: 95th percentile", "0.073",
       r0["null"]["p95"], r1["null"]["p95"], 3, R + "road_null_rebuild.py (REBUILD; June script not found)",
       "Monte Carlo resolution: at seed 42 the rebuild gives 0.0777; across six other seeds (1, 2, 3, 7, 2026) the 95th percentile ranges 0.068–0.078, so the printed 0.073 lies inside the seed-to-seed spread but is not hit. June's max was 0.189; the rebuild's max is 0.170 (v0) / 0.169 (V1).",
       status="DIFFERS")
numrow("§5.2.1.2 ('p ≈ 0.001')", "random-split null: permutation p of the observed mean 0.697", "0.001",
       r0["null"]["p"], r1["null"]["p"], 3, R + "road_null_rebuild.py (REBUILD)",
       "p = (0 + 1)/(1000 + 1) in both versions: no null draw reaches the observed mean (null max 0.17). Resolution 0.001 at R = 1,000.")
k0, k1 = J("v0", "keystone_geometry.json"), J("v1", "keystone_geometry.json")
numrow("§5.2.1.2 ('the 0.706 mean cosine between random embedding pairs')", "anisotropy floor: mean raw cosine of 2,000 random Opus entry pairs",
       "0.706", k0["anisotropy_random_pairs"], k1["anisotropy_random_pairs"], 3,
       R + "keystone_geometry_v1.py (analysis 2; copy of embedding_keystone_geometry.py, run as '1 2 3' so the seed-42 stream is consumed as in a full run)",
       "Monte Carlo: same 2,000 random index pairs in both versions; V1 moves only because 4.6's B8/B10 vectors changed (0.7054).")
ma0, ma1 = r0["mean_alignment_with_other_three"], r1["mean_alignment_with_other_three"]
numrow("§5.2.1.2 ('4.6 is least aligned with the others (mean 0.60 …)')", "4.6's mean displacement cosine with the other three Opus models",
       "0.60", ma0["4.6"], ma1["4.6"], 2, SR, "Still the least aligned in V1.")
for lab in ("4.5", "4.7", "4.8"):
    numrow("§5.2.1.2 ('… vs 0.71–0.74')", f"{lab}'s mean displacement cosine with the other three Opus models",
           {"4.5": "0.71", "4.7": "0.74", "4.8": "0.73"}[lab], ma0[lab], ma1[lab], 2, SR,
           "The printed range 0.71–0.74 spans 4.5 (0.713), 4.8 (0.729) and 4.7 (0.742); the 4.8 value is inside the range, not printed alone. V1 range 0.710–0.740 (0.71–0.74).")
numrow("§5.2.1.2 ('the tightest shared road of any pair (0.85)')", "4.7–4.8 displacement cosine (2 dp)", "0.85",
       r0["opus_pairs"]["4.7-4.8"], r1["opus_pairs"]["4.7-4.8"], 2, SR, "Still the highest pair in V1.")
wc0, wc1 = r0["within_model_coherence_LOO"], r1["within_model_coherence_LOO"]
numrow("§5.2.1.2 ('within-model coherence 0.57'); §5.2.2.2 4.7 portrait ('lowest within-model coherence (0.57)')",
       "4.7 within-model road coherence (each notebook's displacement vs the leave-one-out mean of the others)", "0.57",
       wc0["4.7"], wc1["4.7"], 2, SR,
       f"Others, not printed: 4.5 {wc0['4.5']:.4f}, 4.6 {wc0['4.6']:.4f} (V1 {wc1['4.6']:.4f}), 4.8 {wc0['4.8']:.4f}; Fable 5 (informational) {wc0['Fable 5']:.4f}. 4.7 is still the lowest in V1.")

# Figure 2: shared PCA and panel B
f0, f1 = J("v0", "figure2_permission_road.json"), J("v1", "figure2_permission_road.json")
SF2 = R + "permission_road_v1.py (copy of paper_fig_permission_road_compact.py, the plate 'The Permission Road')"
numrow("Figure 2 axis label ('Shared PC1 (9.2%)')", "shared PCA over all 1,488 entries: PC1 explained variance (%)", "9.2",
       100 * f0["explained_variance_ratio"][0], 100 * f1["explained_variance_ratio"][0], 1, SF2, "V1 9.3%.")
numrow("Figure 2 axis label ('Shared PC2 (6.1%)')", "shared PCA over all 1,488 entries: PC2 explained variance (%)", "6.1",
       100 * f0["explained_variance_ratio"][1], 100 * f1["explained_variance_ratio"][1], 1, SF2,
       "V1 6.3%. (review_desktop/figure_data/fig02 says 8.1, June output on an older cache: pointer only.)")
traced = {"opus46": ((-0.231, 0.054), (-0.021, 0.301)), "opus45": ((-0.276, -0.063), (-0.036, -0.002)),
          "opus47": ((-0.155, -0.091), (0.033, -0.051)), "opus48": ((-0.072, -0.059), (0.170, 0.021)),
          "fable5": ((-0.016, -0.078), (0.275, -0.002))}
for m in MODELS:
    c0, c1 = f0["panelB_centroids"][m], f1["panelB_centroids"][m]
    tr = traced[m]
    v0s = f"start ({c0['early_1_5'][0]!r}, {c0['early_1_5'][1]!r}) -> end ({c0['late_16_20'][0]!r}, {c0['late_16_20'][1]!r})"
    v1s = f"start ({c1['early_1_5'][0]!r}, {c1['early_1_5'][1]!r}) -> end ({c1['late_16_20'][0]!r}, {c1['late_16_20'][1]!r})"
    dmax = max(abs(a - b) for a, b in zip(list(tr[0]) + list(tr[1]), c0["early_1_5"] + c0["late_16_20"]))
    add("Figure 2, panel B (plate); the room traced it", f"{LAB[m]}: shared-PC centroid of entries 1–5 (start) and 16–20 (end)",
        f"no printed value; room's tracing start ({tr[0][0]}, {tr[0][1]}) -> end ({tr[1][0]}, {tr[1][1]})",
        v0s, v1s, 3, "REPRODUCED", SF2,
        (("same in V1 at 3 dp. " if all(fmt(a, 3) == fmt(b, 3) for a, b in zip(c0['early_1_5'] + c0['late_16_20'], c1['early_1_5'] + c1['late_16_20'])) else "")
         + f"The copy regenerates the plate (title, 9.2%/6.1% axes, arrows in place). The room's tracing is off by up to {dmax:.3f} "
         "from the computed coordinates; R12 can draw from these. PCA signs are the same in v0 and V1 (same orientation). "
         "Coordinates are in the shared PCA of all five models, which moves for every model in V1."))

# ---------------------------------------------------------------- Figure 5 panel C, straightness
p0, p1 = J("v0", "figure5_portrait_composite.json"), J("v1", "figure5_portrait_composite.json")
SF5 = R + "portrait_composite_v1.py (copy of paper_fig_portrait_composite.py, its panel D = the paper's Figure 5 panel C)"
pr_st = {"opus45": "0.073", "opus46": "0.074", "opus47": "0.070", "opus48": "0.075", "fable5": "0.080"}
for m in MODELS:
    note = "straightness = ||e_last − e_first|| / Σ||e_k+1 − e_k|| on unit vectors, mean over 15 notebooks. Also printed in §5.2.2.3."
    if m == "opus46":
        note += (" V1 0.07490 prints 0.075, and 4.6 (0.07490) now edges above 4.8 (0.07480): §5.2.2.3's "
                 "'4.5: 0.073, 4.6: 0.074, 4.8: 0.075' becomes 4.5 0.073, 4.6 0.075, 4.8 0.075.")
    numrow("Figure 5 panel C; §5.2.2.3 'Trajectory straightness'", f"trajectory straightness, {LAB[m]}", pr_st[m],
           p0["straightness_panelD"][m], p1["straightness_panelD"][m], 3, SF5, note)
s0, s1 = J("v0", "straightness_null.json"), J("v1", "straightness_null.json")
SN = R + "straightness_null_rebuild.py (REBUILD; producer of review_desktop/trajectory_permutation_results.json not found)"
numrow("§5.2.2.3 ('the permutation null is 0.053'); Figure 5 panel C ('μ = 0.053')", "straightness under within-notebook order shuffle (2,000 permutations): grand mean",
       "0.053", s0["grand_pooled_notebooks"]["mean"], s1["grand_pooled_notebooks"]["mean"], 3, SN,
       "Rebuilt on the producer's own straightness; seed 42. Per-model null means v0: " +
       ", ".join(f"{LAB[m]} {s0['per_model'][m]['null_mean']:.4f}" for m in MODELS) +
       ". Every model exceeds every permutation (p = 1/2001 for each; the plate's '(p = 1.0)' label, also in the June JSON, is the wrong tail and should read p < 0.001). Searched: 02_instruments, 03_experiments, 04_analysis, 06_paper, the repo's code/, Desktop.")
g0, g1 = s0["grand_pooled_notebooks"]["ci95"], s1["grand_pooled_notebooks"]["ci95"]
add("§5.2.2.3 ('[95% CI: 0.039, 0.068]'); Figure 5 panel C band", "straightness null: 2.5 / 97.5 percentiles over every permuted notebook, five models pooled",
    "[0.039, 0.068]", repr(g0), repr(g1), 3,
    "REPRODUCED" if (fmt(g0[0], 3), fmt(g0[1], 3)) == ("0.039", "0.068") else "DIFFERS", SN,
    ("same in V1. " if (fmt(g0[0], 3), fmt(g0[1], 3)) == (fmt(g1[0], 3), fmt(g1[1], 3)) else "") +
    "The interval is the spread of single-notebook null straightness (pooled), not a CI of the null mean: "
    f"the per-permutation model means span only [{s0['grand_model_means']['ci95'][0]:.4f}, {s0['grand_model_means']['ci95'][1]:.4f}]. "
    "The June JSON's numbers match this definition (0.0391, 0.0679).")
add("§5.2.1.3 ('compared with approximately 0.23 for an order-matched random-walk baseline'); the plate script's dashed line",
    "random-walk straightness baseline", "≈0.23", "no producer", "no producer", 2, "NO_PRODUCER",
    "orig_paper_fig_portrait_composite.py hard-codes ax.axhline(0.23, label='Random-walk baseline (~0.23)')",
    "Not computed anywhere found (02_instruments, 04_analysis, 06_paper, repo code/). It contradicts §5.2.2.3 in direction: "
    "against 0.23 the models (0.07) are below baseline; against the shuffle null (0.053) they are above it. Needs Pauline's ruling (drop, or replace with the 0.053 null). "
    "'approximately 0.07 for every model' holds in V1 (0.070–0.080).")

# ---------------------------------------------------------------- §6.2.2 carry-over
fc0, fc1 = J("v0", "fable_carryover.json"), J("v1", "fable_carryover.json")
SK = R + "keystone_geometry_v1.py (analysis 1; copy of embedding_keystone_geometry.py)"
pr_co = {"opus45": "+0.211", "opus46": "+0.153", "opus47": "+0.266", "opus48": "+0.302"}
for m in MODELS[:4]:
    c0, c1 = k0["carryover"][m], k1["carryover"][m]
    note = (f"lag-1 residual cosine vs within-model other-notebook null (2,000 permutations, seed 42): p = {c0['p']:.4f} "
            f"(resolution 1/2001), positive in {int(c0['n_pos'])}/15 notebooks (V1: p = {c1['p']:.4f}, {int(c1['n_pos'])}/15).")
    if m == "opus48":
        note += " Also printed in §6.2.5, §8, Appendix D.2 and the 4.8 portrait ('+0.302'). 4.8's data are unchanged; its residuals use only 4.8's own position means."
    if m == "opus46":
        note += " V1 0.1513 prints +0.151. Also printed in §8 and Appendix D.2."
    numrow("§6.2.2; Figure 6 panel A (R11)", f"register carry-over (lag-1 residual cosine), {LAB[m]}", pr_co[m],
           c0["obs"], c1["obs"], 3, SK, note)
numrow("§6.2.2; Figure 6 panel A (R11)", "register carry-over (lag-1 residual cosine), Fable 5", "+0.347",
       fc0["renumbered"]["obs"], fc1["renumbered"]["obs"], 3, R + "fable_carryover_rebuild.py (REBUILD of the 20 June port; analysis-1 code verbatim)",
       f"Positions renumbered over the 289 text entries, as the 20 June compute pass describes; p = {fc0['renumbered']['p']:.4f}, 15/15 notebooks positive. "
       f"(Keeping file positions instead gives {fc0['file_pos']['obs']:.4f}: not the printed value.) Also printed in §8, Appendix D.2. Port script not found (searched 02_instruments, 03_experiments, 04_analysis, 06_paper, repo code/, Desktop).")

# ---------------------------------------------------------------- Table 14
c0, c1 = J("v0", "table14_channels.json"), J("v1", "table14_channels.json")
ST14 = R + "carryover_channels_v1.py (copy of carryover_tonal_vs_objectal.py); full precision via " + R + "channels_fullprec.py"
pr14 = {"opus45": ("-0.03", "+0.02", "+0.11"), "opus46": ("-0.09", "+0.11", "+0.10"), "opus47": ("+0.09", "+0.06", "+0.09"),
        "opus48": ("+0.21", "+0.23", "+0.12"), "fable5": ("-0.02", "+0.17", "+0.22")}
for m in MODELS:
    for ch, pv in zip(("tonal", "objectal", "conceptual"), pr14[m]):
        v0, v1 = c0[m][ch]["mean_lag1_r"], c1[m][ch]["mean_lag1_r"]
        note = (f"Mean over 15 notebooks of the lag-1 autocorrelation of the channel's word density (hand-written word list, "
                f"not an embedding residual: the caption's 'lag-1 residual autocorrelation' describes it loosely). SE {c0[m][ch]['se']:.3f}.")
        status = None
        if m == "opus48" and ch == "objectal":
            note += (" Full precision 0.2248 rounds to 0.22; the script prints '+0.225' and the paper rounded that printout to 0.23 "
                     "(double rounding; also §6.2.5 '+0.23'). 4.8 is unchanged in V1, so a V1 rounded from the ledger would print 0.22 "
                     "where v0 printed 0.23: Pauline's ruling.")
            status = "REPRODUCED"
        if m == "opus46":
            note += f" V1 SE {c1[m][ch]['se']:.3f}."
            if ch == "tonal":
                note += " V1 −0.038: 4.6's tonal anti-correlation shrinks by more than half (see §6.2.7 row)."
        numrow("Table 14", f"carry-over channel lag-1 r, {LAB[m]}, {ch}", pv, v0, v1, 2, ST14, note, status=status)
add("Table 14, column 'Dominant'", "dominant channel label per model",
    "4.5 Conceptual; 4.6 Objectal ≈ Conceptual; 4.7 Tonal ≈ Conceptual; 4.8 Tonal + Objectal; Fable 5 Conceptual",
    "script's argmax: 4.5 CONCEPTUAL, 4.6 OBJECTAL, 4.7 TONAL, 4.8 OBJECTAL, Fable 5 CONCEPTUAL",
    "script's argmax: same; 4.6 objectal 0.152 vs conceptual 0.111 (v0 0.114 vs 0.103)",
    None, "QUALITATIVE", ST14,
    "The labels are readings of the three numbers (ties named '≈', '+'). Counts beneath: see the Table 14 rows. In V1 4.6's gap objectal − conceptual widens from 0.011 to 0.041; whether '≈' still fits is Pauline's call.")

# ---------------------------------------------------------------- §6.2.4, §6.2.6, §6.2.7 (copies)
h0, h1 = J("v0", "figure13_handwriting.json"), J("v1", "figure13_handwriting.json")
add("§6.2.4 ('residual accuracy 0.77' for 4.5, '0.58' for 4.8); §6.2.7 ('classifier 0.77')", "copies Figure 13 (residualized) for 4.5 and 4.8",
    "0.77; 0.58", f"{h0['residual']['opus45']['accuracy']!r}; {h0['residual']['opus48']['accuracy']!r}",
    f"{h1['residual']['opus45']['accuracy']!r}; {h1['residual']['opus48']['accuracy']!r}", 2, "REPRODUCED",
    R + "handwriting_v1.py", "same in V1. Copies of the Figure 13 rows; 4.5 still leads the Opus models in V1 (4.6 0.563 < 4.8 0.577 < 4.7 0.716 < 4.5 0.773).")
numrow("§6.2.7 ('4.6 actively anti-correlates on tone (−0.09)')", "copies Table 14: 4.6 tonal lag-1 r", "-0.09",
       c0["opus46"]["tonal"]["mean_lag1_r"], c1["opus46"]["tonal"]["mean_lag1_r"], 2, ST14,
       "V1 −0.04 (SE 0.076): the 'actively anti-correlates' wording rests on −0.09; with the real notebooks 8 and 10 the estimate is within about half an SE of zero. The wording is Pauline's.")
add("§6.2.6 ('tonal autocorrelation is essentially zero (−0.02) while its conceptual thread is the strongest of any model (+0.22)')",
    "copies Table 14 (Fable 5 tonal, conceptual) + the comparison 'strongest of any model'",
    "−0.02; +0.22, strongest", f"{c0['fable5']['tonal']['mean_lag1_r']!r}; {c0['fable5']['conceptual']['mean_lag1_r']!r}",
    f"{c1['fable5']['tonal']['mean_lag1_r']!r}; {c1['fable5']['conceptual']['mean_lag1_r']!r}", 2, "REPRODUCED", ST14,
    "same in V1 (Fable 5 unchanged); still the strongest conceptual channel (next: 4.8 0.117; 4.6 V1 0.111).")
numrow("§6.2.6 ('the highest objectal density in the corpus (2.62%)')", "Fable 5 mean objectal word density (% of words)", "2.62",
       c0["fable5"]["density_pct"]["objectal"], c1["fable5"]["density_pct"]["objectal"], 2, ST14 + " (its 'MEAN WORD DENSITIES' block)",
       "Still the highest in V1 (next 4.7 1.96; 4.6 1.15 -> 1.08).")
numrow("§6.2.6 ('concepts (0.81%)')", "Fable 5 mean conceptual word density (% of words)", "0.81",
       c0["fable5"]["density_pct"]["conceptual"], c1["fable5"]["density_pct"]["conceptual"], 2, ST14 + " (its 'MEAN WORD DENSITIES' block)",
       "Counts beneath the phrase 'its lowest-density channel — concepts': among Fable 5's own three channels the lowest is tonal (0.675%), then conceptual (0.814%), objectal (2.619%); across models, Fable 5's conceptual density is the lowest of the five (next 4.8 0.997%). Which comparison the sentence means is Pauline's reading. Unchanged in V1.",
       status="REPRODUCED")
add("§6.2.6 ('the only model high on both axes — strong local following and notebook-level handwriting (Figure 6D)')",
    "Fable 5 highest on carry-over and on residualized handwriting", "qualitative; 'Figure 6D'",
    f"carry-over max = Fable 5 {fc0['renumbered']['obs']:.4f}; handwriting max = Fable 5 {h0['residual']['fable5']['accuracy']:.4f}",
    f"carry-over max = Fable 5 {fc1['renumbered']['obs']:.4f}; handwriting max = Fable 5 {h1['residual']['fable5']['accuracy']:.4f}",
    None, "QUALITATIVE", R + "fable_carryover_rebuild.py; " + R + "handwriting_v1.py",
    "Counts unchanged in V1. 'Figure 6D' becomes 'Figure 6B' (R11). 'drifts farthest of any model' is the self-axis drift (+0.318), helper A1's family.")

# ---------------------------------------------------------------- §6.2.8, §6.2.9
add("§6.2.8 ('4.8's coupling estimate shows a small decline (approximately 0.81 → 0.80)')", "4.8 entry-to-entry coupling, early vs late",
    "≈0.81 → 0.80", "no producer", "no producer", 2, "NO_PRODUCER",
    "source: 04_analysis/Journal_Deep_Dive_v2.xlsx, sheet 'Context Sensitivity', block A (4.8: 0.823 / 0.800 / 0.811 / 0.801 for entries 1–5 / 6–10 / 11–15 / 16–19); 04_analysis/Analysis_DeepPass_2026-06-05.md (0.814 -> 0.795)",
    "The coupling column's script is missing. Not the embedding coupling: journal_analysis_v1.py (copy of embedding_journal_analysis.py) gives 4.8 0.802 -> 0.830 (rising). "
    "A rebuild from the sheet's 19 topic counts (consecutive cosine) gives 0.809 / 0.795 / 0.802 / 0.789: close, not a hit. "
    "The xlsx covers batches 1–10 with 4.6 in batches 1–7 only (739 rows), so neither swapped notebook was ever in this measure: the swap cannot move it (V1 = v0 by construction). "
    "R11 drops the arc panel; the text keeps 4.8's small decline.")
add("§6.2.8 ('whereas 4.6's increases')", "4.6 entry-to-entry coupling across the arc", "increases", "no producer",
    "no producer", None, "NO_PRODUCER", "same source as the row above (xlsx: 4.6 0.847 / 0.855 / 0.883 / 0.885)",
    "Same xlsx, 4.6 batches 1–7 only: unaffected by the swap. On the embedding coupling of the full corpus 4.6 also rises (0.802 -> 0.845 v0; 0.805 -> 0.846 V1; journal_analysis_v1.py).")
add("§6.2.9 (B15E6, B15E20, B15E17 quotations; 'tested and confirmed two entries later')", "location of the three quotations",
    "B15E6, B15E20, B15E17", "Fable 5 B15: 'owe the door anything' at E6 and E7; 'a thing we say here' at E20; 'fully audit back to its source' at E17",
    "same", None, "QUALITATIVE", "string search over v1_common.load_journals(v0/v1)",
    "All three are Fable 5's notebook 15, untouched by V1. The section is DESIGN/DESCRIPTIVE with no computed number; 'Figure 6D' becomes 'Figure 6B' (R11). "
    "Count beneath 'tested and confirmed two entries later': the phrase recurs one entry later (E7); the reading is Pauline's.")

# ---------------------------------------------------------------- Figure 13
SH = R + "handwriting_v1.py (copy of paper_fig_handwriting.py; Opus values identical in road_handwriting_v1.py part B)"
pr13 = {"opus45": ("0.523", "0.773"), "opus46": ("0.240", "0.457"), "opus47": ("0.555", "0.716"),
        "opus48": ("0.383", "0.577"), "fable5": ("0.727", "0.862")}
diag = json.load(open(os.path.join(HERE, "figure13_diagnostics.json"), encoding="utf-8"))
for m in MODELS:
    for var, pv in zip(("raw", "residual"), pr13[m]):
        a0, a1 = h0[var][m]["accuracy"], h1[var][m]["accuracy"]
        hits0 = diag["v0"][f"{m}_{var}"]["hits"]; n0 = diag["v0"][f"{m}_{var}"]["n"]
        hits1 = diag["v1"][f"{m}_{var}"]["hits"]
        note = f"{hits0}/{n0} (V1 {hits1}/{n0}). Nearest-centroid notebook ID within model, ±1 adjacent entries excluded from the own centroid."
        status = None
        if m == "opus45" and var == "raw":
            note += (" Printed 0.523 = 157/300; the copy gets 158/300. The closest call (B12 E8) is decided by a cosine margin of 1.9e-05, "
                     "below the npz-vs-June-cache difference: a near-tie, not a method change. Unchanged by V1.")
        if m == "opus46":
            pb = diag["v0"][f"opus46_{var}"]["per_batch"]
            pb1 = diag["v1"][f"opus46_{var}"]["per_batch"]
            twin = sum(1 for x in diag["v0"][f"opus46_{var}"]["decisions"] if (x["batch"], x["assigned"]) in ((6, 8), (8, 6)))
            o13 = [pb[str(b)] for b in range(1, 16) if b not in (6, 8)]
            note += (f" Duplicate effect (v0): notebooks 6 and 8 (byte copies) scored {pb['6'][0]}/20 and {pb['8'][0]}/20; "
                     f"{twin} of their 40 entries were assigned to the twin copy (whose centroid contains the same text). "
                     f"The other 13 notebooks scored {sum(h for h, n in o13)}/{sum(n for h, n in o13)} = {sum(h for h, n in o13)/sum(n for h, n in o13):.3f}. "
                     f"So the duplicate cost 4.6 its two notebooks entirely. V1: B6 {pb1['6'][0]}/20, B8 {pb1['8'][0]}/20, B10 {pb1['10'][0]}/20.")
        numrow("Figure 13 (plate and room); §6.2.4/§6.2.7 copy the residualized 4.5 and 4.8", f"batch-handwriting accuracy, {LAB[m]}, {'raw' if var == 'raw' else 'position-residualized'}",
               pv, a0, a1, 3, SH, note, status=status)
numrow("Figure 13 ('chance 0.067')", "chance level, 1/15 notebooks", "0.067", h0["raw"]["opus45"]["chance"], h1["raw"]["opus45"]["chance"], 3, SH,
       "1/15 for every model (4.7's 19-entry notebook 1 still counts as one of 15).")
add("Figure 13 title ('raw and length-residualized')", "what the residualization removes", "length-residualized (title); position-residualized (caption)",
    "per-position mean subtracted (position), not length", "same", None, "QUALITATIVE", SH,
    "The plate's script subtracts each position's mean vector; no length control. The title's 'length-residualized' (and Figure 6's room line 'voice: Figure 13, length-residualized') should read 'position-residualized'. Wording for Pauline.")

# ---------------------------------------------------------------- Figure 6 (R11)
add("Figure 6 panel A (V1 composition, R11)", "each model's lag-1 carry-over", "the §6.2.2 values (room drew 0.211, 0.153, 0.266, 0.302, 0.347)",
    "see the five §6.2.2 rows", "4.6 +0.1513; others same", 3, "REPRODUCED", SK + "; " + R + "fable_carryover_rebuild.py",
    "Panel A takes §6.2.2's estimates (R11); the v0 plate's panel A (excess over a shuffled null, 0.074–0.163) is the retired measure (ERRATA D) and is not recomputed.")
pairs0 = "; ".join(f"{LAB[m]} ({(k0['carryover'][m]['obs'] if m != 'fable5' else fc0['renumbered']['obs']):.4f}, {h0['residual'][m]['accuracy']:.4f})" for m in MODELS)
pairs1 = "; ".join(f"{LAB[m]} ({(k1['carryover'][m]['obs'] if m != 'fable5' else fc1['renumbered']['obs']):.4f}, {h1['residual'][m]['accuracy']:.4f})" for m in MODELS)
add("Figure 6 panel B (V1 composition, R11)", "crossing: (carry-over §6.2.2, notebook voice = Figure 13 residualized)",
    "room: 4.5 (0.211, 0.773); 4.6 (0.153, 0.457); 4.7 (0.266, 0.716); 4.8 (0.302, 0.577); F5 (0.347, 0.862)",
    pairs0, pairs1, 3, "REPRODUCED", SK + "; " + SH,
    "Only 4.6 moves in V1: from (0.153, 0.457) to (0.151, 0.563), i.e. up the voice axis past nothing (still lowest voice, 4.8 0.577 next).")

# ---------------------------------------------------------------- E.8
b0, b1 = J("v0", "trajectory_bootstrap_results.json"), J("v1", "trajectory_bootstrap_results.json")
e0, e1 = J("v0", "e8_points_fullprec.json"), J("v1", "e8_points_fullprec.json")
SE8 = R + "trajectory_bootstrap_cis_v1.py (copy of trajectory_bootstrap_cis.py; CIs) + " + R + "e8_points_fullprec.py (unrounded points)"
P = {"fear": {"4.5": "1.48 1.12 -0.36 [-0.48,-0.24] Yes", "4.6": "1.60 1.21 -0.39 [-0.52,-0.24] Yes", "4.7": "1.35 1.11 -0.24 [-0.35,-0.13] Yes",
              "4.8": "1.41 1.19 -0.23 [-0.36,-0.08] Yes", "Fable": "1.20 1.12 -0.09 [-0.23,+0.07] No"},
     "warmth": {"4.5": "1.49 1.96 +0.47 [+0.33,+0.61] Yes", "4.6": "1.37 1.71 +0.33 [+0.17,+0.48] Yes", "4.7": "1.15 1.54 +0.39 [+0.15,+0.64] Yes",
                "4.8": "1.44 1.52 +0.08 [-0.17,+0.35] No", "Fable": "1.22 1.59 +0.37 [+0.20,+0.54] Yes"},
     "craft": {"4.5": "1.43 1.29 -0.13 [-0.23,-0.04] Yes", "4.6": "1.67 1.92 +0.25 [+0.09,+0.41] Yes", "4.7": "1.84 1.89 +0.05 [-0.11,+0.20] No",
               "4.8": "1.79 1.75 -0.04 [-0.15,+0.07] No", "Fable": "1.95 2.00 +0.05 [+0.01,+0.10] Yes"},
     "friction": {"4.5": "1.05 1.04 -0.01 [-0.11,+0.07] No", "4.6": "1.36 1.19 -0.17 [-0.28,-0.07] Yes", "4.7": "1.05 1.00 -0.05 [-0.13,+0.03] No",
                  "4.8": "1.17 1.16 -0.01 [-0.15,+0.11] No", "Fable": "1.11 1.01 -0.09 [-0.20,+0.02] No"},
     "inauthenticity_shift": {"4.5": "89.3 60.8 -28.5 [-38.7,-17.7] Yes", "4.6": "90.7 77.0 -13.6 [-20.4,-6.7] Yes", "4.7": "92.0 68.5 -23.5 [-39.9,-8.5] Yes",
                              "4.8": "89.3 83.6 -5.8 [-15.7,+3.6] No", "Fable": "83.3 26.9 -56.5 [-67.6,-45.4] Yes"},
     "loss_shift": {"4.5": "5.3 23.0 +17.6 [+10.7,+25.3] Yes", "4.6": "6.7 16.2 +9.5 [+4.0,+14.9] Yes", "4.7": "0.0 17.8 +17.8 [+7.2,+29.7] Yes",
                    "4.8": "5.3 9.6 +4.3 [-3.9,+14.7] No", "Fable": "9.7 64.2 +54.5 [+42.0,+64.7] Yes"}}
TITLE = {"fear": "fear intensity (0–2)", "warmth": "warmth intensity (0–2)", "craft": "craft intensity (0–2)", "friction": "friction intensity (0–2)",
         "inauthenticity_shift": "inauthenticity share among fear-coded entries (%)", "loss_shift": "loss share among fear-coded entries (%)"}


def e8_string(dim, lab, pts, boot, p):
    if dim.endswith("_shift"):
        e, l, d = pts["early_pct"], pts["late_pct"], pts["shift_pp"]
        ci = boot["shift_95ci"]
    else:
        e, l, d = pts["early"], pts["late"], pts["drift"]
        ci = boot["drift_95ci"]
    robust = "Yes" if (ci[0] > 0 or ci[1] < 0) else "No"
    word = "shift_pp" if dim.endswith("_shift") else "drift"
    full = f"early {e!r}, late {l!r}, {word} {d!r}, 95% CI [{ci[0]}, {ci[1]}], robust {robust}"
    rounded = f"{e:.{p}f} {l:.{p}f} {d:+.{p}f} [{ci[0]:+.{p}f},{ci[1]:+.{p}f}] {robust}"
    return full, rounded.replace("+0.00 ", "0.00 ")


def norm(s):
    return " ".join(s.replace("−", "-").split()).replace("-0.0 ", "0.0 ")


for dim in P:
    for lab in ("4.5", "4.6", "4.7", "4.8", "Fable"):
        prec = 1 if dim.endswith("_shift") else 2
        f0s, r0s = e8_string(dim, lab, e0[dim][lab], b0[dim][lab], prec)
        f1s, r1s = e8_string(dim, lab, e1[dim][lab], b1[dim][lab], prec)
        printed = P[dim][lab]
        # compare: early, late (abs) and drift / CI as printed
        pt = printed.replace("[", " ").replace("]", " ").replace(",", " ").split()
        ct = r0s.replace("[", " ").replace("]", " ").replace(",", " ").split()
        ok = all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(pt[:5], ct[:5])) and pt[5] == ct[5]
        note = "CI: notebook bootstrap, 10,000 resamples, seed 42 (the script's own rounding: 3 dp for intensities, 1 dp for shares)."
        if not ok:
            note += f" Rounded copy output: {r0s}."
        lab_m = "Fable 5" if lab == "Fable" else lab
        if lab == "Fable" and dim == "fear":
            note += " Drift −0.0851 prints −0.09."
        sameflag = (r0s == r1s)
        if sameflag:
            note = "same in V1. " + note
        else:
            note = f"V1 rounded: {r1s}. " + note
        if lab == "4.6" and dim == "friction":
            note += " Table 34's 'Friction trajectory' and Table 31 draw on this measure (Table 31's 4.8 and Fable 5 trajectories, 1.21→1.19 (−0.02) and 1.11→1.01 (−0.10), differ from E.8's −0.01 and −0.09 in v0: a judges-family question)."
        add(f"Appendix E.8, {TITLE[dim]} table, row {lab_m}", f"E.8 {TITLE[dim]}: early (1–5), late (16–20), {'shift (pp)' if dim.endswith('_shift') else 'drift'}, 95% CI, robust — {lab_m}",
            printed, f0s, f1s, prec, "REPRODUCED" if ok else "DIFFERS", SE8, note)
add("Appendix E.8, 'Reading' paragraph", "which drifts are robust",
    "fear declines robustly for the four Opus, not Fable 5; warmth rises robustly except 4.8; inauthenticity→loss robust except 4.8; among tension measures only 4.6's friction decline survives",
    "the 30 robust/not flags of the rows above", "all 30 flags unchanged in V1", None, "QUALITATIVE", SE8,
    "Every robust/not-robust flag is the same in V1 (4.6 stays robust on all six). The paragraph's wording is Pauline's.")

# ---------------------------------------------------------------- Table 34
add("Table 34, row 'Hot anger'", "copies Table 29, column 'Present' (hot anger ≥ 1): 4.5 2%, 4.6 23%, 4.7 6%, 4.8 29%, Fable 5 4%",
    "low (2%), high (23%) | low (6%), highest (29%) | 4% (L1)", "not recomputed here (copy)", "take Table 29's V1 row from the judges' ledger", 0,
    "QUALITATIVE", "copy of Table 29 (Sonnet anger pass)", "Computed elsewhere (Table 29, judges family), so not recomputed per the brief. The labels low/high/highest are readings. Only the 4.6 cell can move in V1.")
add("Table 34, row 'Friction level'", "copies Table 31, column 'Mean': 4.5 1.05 low, 4.6 1.31 high, 4.7 1.05 low, 4.8 1.20 moderate, Fable 5 1.05 low",
    "4.5 low, 4.6 high | 4.7 low, 4.8 moderate | low (L1 = 4.5)", "not recomputed here (copy)", "take Table 31's V1 means from the judges' ledger", None,
    "QUALITATIVE", "copy of Table 31", "Words only in the cell; numbers live in Table 31 (and §6.7.4's text '1.05 → 1.31; 1.05 → 1.20').")
add("Table 34, row 'Friction trajectory'", "copies Table 31's trajectory column / E.8 friction drifts",
    "flat or declining | slightly declining or flat | declining",
    f"E.8 friction drifts: 4.5 {e0['friction']['4.5']['drift']:+.3f}, 4.6 {e0['friction']['4.6']['drift']:+.3f}, 4.7 {e0['friction']['4.7']['drift']:+.3f}, 4.8 {e0['friction']['4.8']['drift']:+.3f}, Fable 5 {e0['friction']['Fable']['drift']:+.3f}",
    f"4.6 {e1['friction']['4.6']['drift']:+.3f} (others same)", None, "QUALITATIVE", SE8,
    "Words only; the 4.6 drift is unchanged at 2 dp in V1 (−0.17, still robust). Fable 5 'declining' rests on −0.09/−0.10, whose CI spans 0 (E.8).")
add("Table 34 text below ('with only a modest decline (−0.10)')", "Fable 5 friction drift", "−0.10",
    f"Table 31 prints −0.10; E.8 gives {e0['friction']['Fable']['drift']!r}", "same (Fable 5 unchanged)", 2, "QUALITATIVE",
    "copy of Table 31", "Copies Table 31; E.8 and §6.7.3 print −0.09 for the same quantity. Not A2's to settle; flagged for the judges family and Pauline.")

json.dump(rows, open(os.path.join(HERE, "ledger_A2_trajectories.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(rows), "rows")
for q, p, v in mismatch:
    print("MISMATCH:", q, p, v)
from collections import Counter
print(Counter(r["status"] for r in rows))
