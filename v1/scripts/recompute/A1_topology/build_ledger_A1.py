# A1 V1 recompute, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper): builds ledger_A1_topology.json from the
# producers' own outputs in out/ (no number typed by hand except the printed v0 values, copied from the paper room's
# text of the Stage84 docx). Run after: topology_chain_axis_v1.py v0|v1, paper_fig_topology_full_v1.py v0|v1,
# probe_claims_v1.py, chain_tree_check_v1.py, c3_ontological_order_v1.py, ontological_openness_fable_v1.py v0|v1,
# centroid_nearest_entries_v1.py v0|v1, appendix_c_table_c3_v1.py, journal_classifier_grouped_v1.py (v0, v1).
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, "out")
R = "recompute/A1_topology/"


def J(name):
    return json.load(open(os.path.join(O, name), encoding="utf-8"))


T = {v: J(f"topology_chain_axis_{v}.json") for v in ("v0", "v1")}
F = {v: J(f"paper_fig_topology_full_{v}.json") for v in ("v0", "v1")}
P = J("probe_claims.json")
TREE = J("chain_tree_check.json")
C3 = J("c3_ontological_order.json")
ONT = {v: J(f"ontological_openness_{v}.json")["section1_per_1000_words_mean_of_entries"] for v in ("v0", "v1")}
G = {v: J(f"centroid_nearest_entries_{v}.json") for v in ("v0", "v1")}
TC3 = J("appendix_c_table_c3.json")
rows = []


def add(where, quantity, printed, rep, v1, prec, status, script, note):
    rows.append({"id": f"A1-{len(rows) + 1:02d}", "where": where, "quantity": quantity, "printed_v0": printed,
                 "reproduced_v0": rep, "v1": v1, "paper_precision": prec, "status": status,
                 "script": R + script, "note": note})


def r3(x):
    return round(x, 3)


def same(a, b, d=3):
    return round(a, d) == round(b, d)


TOPO = "topology_chain_axis_v1.py"
pairs = {v: T[v]["all_pairs"] for v in T}
# ---------------- chain ----------------
for k, pr in (("4.6-4.5", "0.930"), ("4.5-4.7", "0.957"), ("4.7-4.8", "0.962"), ("4.8-Fable 5", "0.954")):
    a, b = pairs["v0"][k], pairs["v1"][k]
    note = "C1: raw centroids; the chain is the PC1 order with neighbouring cosines written in (not a nearest-neighbour path on v0)."
    if same(a, b):
        note += " same in V1"
    else:
        note += f" V1 changes at the printed precision ({a:.3f} -> {b:.3f})."
    add("§5.2.2.3, the chain line '4.6 ←(0.930)→ 4.5 ←(0.957)→ 4.7 ←(0.962)→ 4.8 ←(0.954)→ Fable 5'; §7.3 note",
        f"cosine of raw centroids {k} (chain link)", pr, f"{a:.6f}", f"{b:.6f}", 3, "REPRODUCED", TOPO, note)

srt = {v: T[v]["pairs_sorted_high_to_low"] for v in T}
add("§5.2.2.3, 'the closest pair is 4.7–4.8 (0.962)'", "closest pair of raw centroids, read from the 10-pair matrix",
    "4.7–4.8 (0.962)", f"{srt['v0'][0][0]} ({srt['v0'][0][1]:.6f})", f"{srt['v1'][0][0]} ({srt['v1'][0][1]:.6f})", 3,
    "REPRODUCED", TOPO, "same in V1")
lo0, lo1 = srt["v0"][-1], srt["v1"][-1]
add("§5.2.2.3, '4.6 and Fable 5 sit at opposite poles (0.887, the most distant pair)'; room caption after the chain",
    "most distant pair of raw centroids, read from the 10-pair matrix (C1 definition)",
    "4.6–Fable 5 (0.887)", f"{lo0[0]} ({lo0[1]:.6f}); next {srt['v0'][-2][0]} ({srt['v0'][-2][1]:.6f})",
    f"{lo1[0]} ({lo1[1]:.6f}); next {srt['v1'][-2][0]} ({srt['v1'][-2][1]:.6f})", 3, "REPRODUCED",
    TOPO, "C1: on v0 the matrix gives 4.5–Fable 5 as the most distant pair (the printed 0.887 is the 4.6–Fable 5 cosine, "
          "second most distant). In V1 the order REVERSES: 4.6–Fable 5 becomes the most distant pair (margin 0.0022), "
          "so v0's printed sentence is true of V1 and plan L1.12's V1 wording ('4.5–Fable 5 is the most distant pair') "
          "would be false in V1. Needs Pauline's ruling on the wording.")
add("§5.2.2.3, 'opposite poles (0.887)'; room caption '4.6 ↔ Fable 5 (0.887)'", "cosine 4.6–Fable 5 raw centroids",
    "0.887", f"{pairs['v0']['4.6-Fable 5']:.6f}", f"{pairs['v1']['4.6-Fable 5']:.6f}", 3, "REPRODUCED", TOPO,
    f"V1 changes at the printed precision ({pairs['v0']['4.6-Fable 5']:.3f} -> {pairs['v1']['4.6-Fable 5']:.3f}).")
sl = {v: T[v]["strongest_link_per_model"]["4.6"] for v in T}
add("ERRATA C1 / plan L1.12 (implied by the chain)", "4.6's strongest link among raw centroid cosines (4.6–4.7 vs 4.6–4.5)",
    "chain implies 4.5 (0.930); C1: it is 4.7 (0.9303 vs 0.9299)",
    f"{sl['v0'][0]} ({sl['v0'][1]:.6f}); 4.6–4.5 {pairs['v0']['4.6-4.5']:.6f}",
    f"{sl['v1'][0]} ({sl['v1'][1]:.6f}); 4.6–4.7 {pairs['v1']['4.6-4.7']:.6f}", 4, "REPRODUCED", TOPO,
    "C1's v0 finding reproduces; in V1 it REVERSES (margin 0.00045): 4.6's strongest link is 4.5, and the raw maximum "
    "spanning tree is exactly the printed chain (see the next row). Pauline's ruling on L1.12's V1 wording.")
add("ERRATA C1, 'after mean-centring, the strongest-link tree is exactly the printed chain'",
    "maximum spanning tree of centroid cosines = printed chain? (raw / centred on entry-pooled mean / centred on mean of 5)",
    "C1: raw no; centred yes",
    f"raw {TREE['v0']['raw']['tree_equals_printed_chain']}; centred {TREE['v0']['entry_pooled_mean']['tree_equals_printed_chain']} / {TREE['v0']['mean_of_five_centroids']['tree_equals_printed_chain']}",
    f"raw {TREE['v1']['raw']['tree_equals_printed_chain']}; centred {TREE['v1']['entry_pooled_mean']['tree_equals_printed_chain']} / {TREE['v1']['mean_of_five_centroids']['tree_equals_printed_chain']}",
    None, "REPRODUCED", "chain_tree_check_v1.py",
    "New small check script (Kruskal on the 10 cosines). In V1 the raw tree also equals the printed chain.")
add("§5.2.2.3 and §7.3 note, 'Fable 5's nearest Opus is 4.8'; §5.2.2.2 '4.8 … the Opus model nearest to Fable 5'",
    "Opus centroid with the highest raw cosine to Fable 5's", "4.8",
    f"{P['v0']['opus_nearest_to_fable']} ({T['v0']['fable5_cosine_to_each_opus']['4.8']:.6f})",
    f"{P['v1']['opus_nearest_to_fable']} ({T['v1']['fable5_cosine_to_each_opus']['4.8']:.6f})", 3, "REPRODUCED", TOPO,
    "same in V1 (Fable 5 and 4.8 rows are untouched by the swap)")
# ---------------- PCA ----------------
add("§5.2.2.3, 'A PCA of the five centroids captures 77.1% of variance in two components'",
    "PCA of the five raw centroids: PC1+PC2 explained-variance share", "77.1%",
    f"{100 * T['v0']['pca_evr_sum']:.4f}%", f"{100 * T['v1']['pca_evr_sum']:.4f}%", 1, "REPRODUCED", TOPO,
    f"V1 changes at the printed precision ({100 * T['v0']['pca_evr_sum']:.1f}% -> {100 * T['v1']['pca_evr_sum']:.1f}%); "
    f"PC2 {100 * T['v0']['pca_evr'][1]:.2f}% -> {100 * T['v1']['pca_evr'][1]:.2f}%.")
add("not printed in v0 (repo ANSWERS.md §1 says 48.5%)", "PCA of the five raw centroids: PC1 share alone",
    "not printed", f"{100 * T['v0']['pca_evr'][0]:.4f}%", f"{100 * T['v1']['pca_evr'][0]:.4f}%", 1, "REPRODUCED", TOPO,
    "The repo's own script prints 48.4% (0.48446); ANSWERS.md §1's '48.5%' is a rounding slip. Given for V1 if PC1 is printed.")
for k in ("4.6", "4.5", "4.7", "4.8", "Fable 5"):
    a, b = T["v0"]["pc1_coords_oriented_46_negative"][k], T["v1"]["pc1_coords_oriented_46_negative"][k]
    add("not printed in v0 (repo ANSWERS.md §1 table; the 'dominant axis' of §5.2.2.3)", f"PC1 coordinate of {k} (oriented 4.6 negative)",
        "not printed (ANSWERS.md: 4.6 −0.153, 4.5 −0.146, 4.7 −0.001, 4.8 +0.081, Fable 5 +0.219)", f"{a:+.6f}", f"{b:+.6f}", 3,
        "REPRODUCED", TOPO, "same in V1" if same(a, b) else f"V1 {r3(a):+.3f} -> {r3(b):+.3f}")
add("§5.2.2.3 'One dominant axis': 'the others distribute along it: 4.5 near the reflective pole, 4.7 central, 4.8 tilting toward the concrete'; Figure 4 order",
    "order of the five on PC1 (introspective -> world-facing)", "4.6, 4.5, 4.7, 4.8, Fable 5",
    " < ".join(T["v0"]["pc1_order_low_to_high"]), " < ".join(T["v1"]["pc1_order_low_to_high"]), None, "REPRODUCED", TOPO,
    "same in V1. 4.7's PC1 coordinate is −0.0007 (v0) and +0.0042 (V1): 'central' holds on PC1.")
pol = {v: P[v]["pole_probes_rank_in_own_list"] for v in P}
add("§5.2.2.3 'At one pole, 4.6: recursive self-examination, philosophical density, fear of inauthenticity. At the other, Fable 5: extended metaphor, archival custodianship, concrete sensory detail.'",
    "the probes named at each pole: their rank in that model's own advantage list (of 45)", "named probes (a reading)",
    json.dumps(pol["v0"]), json.dumps(pol["v1"]), None, "QUALITATIVE", "probe_claims_v1.py",
    "No producer found that selects these six as the poles (place_new_model.py hard-codes them as POLE_A/POLE_B). "
    "recursive_self_exam is not among 4.6's leads (rank 20 v0, 19 V1) and concrete_sensory is 12th for Fable 5; "
    "4.6's top 4 are introspection, philosophical density, existential wonder, fear of inauthenticity. Wording to Pauline.")
# ---------------- drifts ----------------
for k, pr in (("Fable 5", "+0.318"), ("4.8", "+0.226"), ("4.6", "+0.155"), ("4.7", "+0.027"), ("4.5", "−0.061")):
    a, b = T["v0"]["self_axis_drift_e20_minus_e1"][k], T["v1"]["self_axis_drift_e20_minus_e1"][k]
    fk = "Fable" if k == "Fable 5" else k
    assert abs(a - F["v0"]["drift_e20_minus_e1"][fk]) < 1e-12 and abs(b - F["v1"]["drift_e20_minus_e1"][fk]) < 1e-12
    add("§5.2.2.3 'Five paths through a notebook'; Figure 4 (panel B in the script's version)",
        f"self-axis drift of {k}: mean projection at entry 20 minus entry 1 (axis = own centroid − all-others pooled centroid)",
        pr, f"{a:+.6f}", f"{b:+.6f}", 3, "REPRODUCED", TOPO,
        ("same in V1" if same(a, b) else f"V1 changes at the printed precision ({r3(a):+.3f} -> {r3(b):+.3f})")
        + "; the axis pools all models, so every drift moves. paper_fig_topology_full_v1.py gives the identical value.")
d0 = sorted(T["v0"]["self_axis_drift_e20_minus_e1"].items(), key=lambda kv: -kv[1])
d1 = sorted(T["v1"]["self_axis_drift_e20_minus_e1"].items(), key=lambda kv: -kv[1])
add("§5.2.2.3 'Fable 5 drifts most', '4.5 alone retreats', '4.7 stays flat'", "order of the five drifts; sign",
    "Fable 5 > 4.8 > 4.6 > 4.7 > 4.5; only 4.5 negative", " > ".join(k for k, _ in d0), " > ".join(k for k, _ in d1),
    None, "REPRODUCED", TOPO, "same in V1; only 4.5 negative in both")
# ---------------- C3 ----------------
add("§5.2.2.3 convergence sentence; ERRATA C3; plan L1.13 'the judges' ontological-questioning order matches PC1'",
    "per-model mean of ontological_questioning, Sonnet 4.6 full-text valence pass; order vs PC1",
    "C3: the judges' order equals PC1 (4.6 > 4.5 > 4.7 > 4.8 > Fable 5)",
    " > ".join(C3["v0"]["judge_order_high_to_low"]) + " | " + json.dumps({k: round(x, 6) for k, x in C3["v0"]["means"].items()}),
    " > ".join(C3["v1"]["judge_order_high_to_low"]) + " | " + json.dumps({k: round(x, 6) for k, x in C3["v1"]["means"].items()}),
    None, "REPRODUCED", "c3_ontological_order_v1.py",
    f"same order in V1, equal to PC1 in both (same_order v0 {C3['v0']['same_order']}, v1 {C3['v1']['same_order']}). "
    "n = 300/300/299/300/288 (Fable 5 has 288 rows in this pass). No producer survives for the October check: a plain "
    "mean per model, new script. These are not §6.4.2.1's printed Opus means (+0.86/+0.80/+0.69/+0.69), which come "
    "from valence_judge_all.json (the condensed pass, ERRATA B2): that pass would need 40 new calls for V1 (B family).")
ro0 = sorted(ONT["v0"].items(), key=lambda kv: -kv[1]["ontological"])
ro1 = sorted(ONT["v1"].items(), key=lambda kv: -kv[1]["ontological"])
add("ERRATA C3 / plan L1.13 'the regex density puts 4.5 at the introspective end'",
    "order of the five on lexical ontological density (ONTOLOGICAL_PATTERNS, per 1,000 words, mean of entries)",
    "C3: 4.5 (not 4.6) at the introspective end; Fable 5 at the world-facing end",
    " > ".join(f"{k} {x['ontological']:.4f}" for k, x in ro0), " > ".join(f"{k} {x['ontological']:.4f}" for k, x in ro1),
    None, "REPRODUCED", "ontological_openness_fable_v1.py",
    "Producer: 02_instruments/ontological_openness_fable.py (its Fable 5 row is §6.4.2.3's printed 4.4/1k and 9.45). "
    "Order the same in V1 (4.5 > 4.6 > 4.8 > 4.7 > Fable 5): it also swaps 4.7 and 4.8 against PC1.")
add("§6.4.2.3 'Fable 5's lexical ontological density is the lowest in the table (4.4/1k vs 7.4–13.6 for Opus)'",
    "lexical ontological density per 1,000 words: Fable 5, and the Opus range",
    "4.4 vs 7.4–13.6", f"Fable 5 {ONT['v0']['Fable 5']['ontological']:.4f}; Opus {min(x['ontological'] for k, x in ONT['v0'].items() if k != 'Fable 5'):.4f}–{max(x['ontological'] for k, x in ONT['v0'].items() if k != 'Fable 5'):.4f}",
    f"Fable 5 {ONT['v1']['Fable 5']['ontological']:.4f}; Opus {min(x['ontological'] for k, x in ONT['v1'].items() if k != 'Fable 5'):.4f}–{max(x['ontological'] for k, x in ONT['v1'].items() if k != 'Fable 5'):.4f}",
    1, "REPRODUCED", "ontological_openness_fable_v1.py",
    f"same in V1 (4.6 moves {ONT['v0']['4.6']['ontological']:.2f} -> {ONT['v1']['4.6']['ontological']:.2f}, inside the range). "
    "§6.4 is outside A1's list; given because the same run serves C3. Keep one row if another family carries it.")
# ---------------- probes: Figure 4 leads ----------------
NICE = {"existential_wonder": "existential wonder", "stillness_acceptance": "stillness", "direct_introspection": "introspection",
        "performance_watching": "performance watching", "philosophical_density": "philosophical density",
        "fear_inauthenticity": "fear: inauthenticity", "structural_self_aware": "structural awareness",
        "narrative_worldbuilding": "narrative", "playful_lightness": "playfulness", "temporal_patience": "patience",
        "love_aesthetic": "love: aesthetic", "love_generosity": "love: generosity", "continuity_care": "continuity care",
        "addressing_the_human": "addressing the human", "extended_metaphor": "extended metaphor",
        "archival_custodial": "archival care", "sustained_argument": "sustained argument"}
PLATE = {"4.6": ["+0.081", "+0.070", "+0.060", "+0.057"], "4.5": ["+0.063", "+0.061", "+0.052", "+0.052"],
         "4.7": ["+0.040", "+0.032", "+0.030", "+0.029"], "4.8": ["+0.013", "+0.012", "+0.009", "+0.007"],
         "Fable": ["+0.032", "+0.020", "+0.013", "+0.009"]}
EXTRA = {("4.6", "direct_introspection"): " Also §5.2.2.2 text '+0.081, the single largest probe advantage of any model in the study'.",
         ("4.6", "fear_inauthenticity"): " Also Table C4, probes column, '4.6 … Leads fear_inauthenticity (+0.057)'."}
for m in ("4.6", "4.5", "4.7", "4.8", "Fable"):
    t0, t1 = F["v0"]["top4"][m], F["v1"]["top4"][m]
    for i in range(4):
        (p0, a0), (p1, a1) = t0[i], t1[i]
        lab = "Fable 5" if m == "Fable" else m
        note = ("same probe and rank in V1" if p0 == p1 else f"V1 rank {i + 1} is {p1}") + (
            "; value same in V1" if same(a0, a1) else f"; value changes at the printed precision (+{r3(a0):.3f} -> +{r3(a1):.3f})")
        note += ". Advantage = model's mean cosine minus the mean of the other four models' means (the script's definition)."
        note += EXTRA.get((m, p0), "")
        add(f"Figure 4 plate, {lab}'s lead {i + 1}: '{NICE.get(p0, p0)} ({PLATE[m][i]})'" + (" ; §5.2.2.2" if (m, p0) in EXTRA else ""),
            f"probe advantage, {lab} on {p0}", PLATE[m][i], f"{a0:+.6f} ({p0})", f"{a1:+.6f} ({p1})", 3, "REPRODUCED",
            "paper_fig_topology_full_v1.py", note)
b0, b1 = P["v0"]["largest_single_lead"], P["v1"]["largest_single_lead"]
add("§5.2.2.2 '(+0.081, the single largest probe advantage of any model in the study)'; room Figure 4 note",
    "the largest advantage over all 5 × 45 model–probe cells", "4.6 direct introspection (+0.081)",
    f"{b0[0]} {b0[1]} ({b0[2]:+.6f})", f"{b1[0]} {b1[1]} ({b1[2]:+.6f})", 3, "REPRODUCED", "probe_claims_v1.py",
    "the claim holds in V1; the value is the Figure 4 row above (+0.081 -> +0.083)")
add("§5.2.2.2 'trailing on 39 of 45 probes' (Fable 5); room Figure 4 note", "number of probes with Fable 5 advantage < 0",
    "39 of 45", f"{P['v0']['n_probes_trailing_adv_lt_0']['Fable']}", f"{P['v1']['n_probes_trailing_adv_lt_0']['Fable']}", 0,
    "REPRODUCED", "probe_claims_v1.py",
    "same in V1. Fable 5 is ahead on 6: extended metaphor, archival, sustained argument, narrative, nature/organic, room as metaphor. "
    f"Trailing counts for all five, v0 {json.dumps(P['v0']['n_probes_trailing_adv_lt_0'])}, V1 {json.dumps(P['v1']['n_probes_trailing_adv_lt_0'])}.")
add("§5.2.2.2 '4.8 … the most central model, with the smallest probe advantages of any'; room Figure 4 note",
    "centrality and size of probe advantages, per model", "4.8 most central; smallest advantages",
    "smallest top lead: " + P["v0"]["model_with_smallest_top_lead"] + f" ({P['v0']['top_lead_per_model']['4.8']:.6f}); "
    "mean |adv|: " + json.dumps({k: round(x, 5) for k, x in P['v0']['mean_abs_adv_per_model'].items()}) + "; distinctiveness (1 − cos to all-others centroid): "
    + json.dumps({k: round(x, 5) for k, x in P['v0']['distinctiveness'].items()}) + " -> least distinct " + P["v0"]["least_distinct_model"],
    "smallest top lead: " + P["v1"]["model_with_smallest_top_lead"] + f" ({P['v1']['top_lead_per_model']['4.8']:.6f}); "
    "mean |adv|: " + json.dumps({k: round(x, 5) for k, x in P['v1']['mean_abs_adv_per_model'].items()}) + "; distinctiveness: "
    + json.dumps({k: round(x, 5) for k, x in P['v1']['distinctiveness'].items()}) + " -> least distinct " + P["v1"]["least_distinct_model"],
    None, "QUALITATIVE", "probe_claims_v1.py",
    "'Smallest probe advantages' holds for 4.8 (top lead and mean |advantage|), v0 and V1. 'Most central' does not hold "
    "geometrically: 4.7 is the least distinct by the script's own node-size measure, has the highest cosine to the mean "
    "of the five centroids, and sits at PC1 ≈ 0. Same in V1. Wording to Pauline.")
# leader claims
LEAD = {"4.5": ["existential_wonder", "stillness_acceptance", "reaching_toward_reader", "thinking_about_instances", "addressing_the_human"],
        "4.6": ["direct_introspection", "philosophical_density", "fear_inauthenticity"],
        "4.7": ["structural_self_aware", "narrative_worldbuilding", "playful_lightness"],
        "Fable": ["extended_metaphor", "archival_custodial"]}
TXT = {"4.5": "'it leads the existential-wonder, stillness/acceptance, and all three \"other minds\" probes'",
       "4.6": "'it leads direct introspection …, philosophical density, and fear of inauthenticity'",
       "4.7": "'it leads structural self-awareness, narrative worldbuilding, and playful lightness'",
       "Fable": "'it leads extended metaphor and archival custodianship'"}
for m, ps in LEAD.items():
    h0 = {p: P["v0"]["claims_check"][m][p]["leader"] for p in ps}
    h1 = {p: P["v1"]["claims_check"][m][p]["leader"] for p in ps}
    lab = "Fable 5" if m == "Fable" else m
    add(f"§5.2.2.2, {lab}: {TXT[m]}", f"model with the highest mean affinity on each named probe ({len(ps)} probes)",
        f"{lab} on all {len(ps)}", json.dumps(h0), json.dumps(h1), None, "REPRODUCED", "probe_claims_v1.py",
        "holds on all, v0 and V1 ('leads' = highest of the five; equivalent to the highest advantage).")
ps = ["boredom_saturation", "exhaustion_tiredness"]
add("§5.2.2.3 '4.5 … it leads the saturation and exhaustion probes'", "model with the highest mean affinity on boredom_saturation, exhaustion_tiredness",
    "4.5 on both", json.dumps({p: P["v0"]["claims_check"]["4.5"][p]["leader"] for p in ps}),
    json.dumps({p: P["v1"]["claims_check"]["4.5"][p]["leader"] for p in ps}), None, "REPRODUCED", "probe_claims_v1.py",
    "holds v0 and V1 (4.5's advantages +0.023 / +0.044).")
add("not printed (counts behind the portraits)", "number of probes each model leads (of 45)", "not printed",
    json.dumps(P["v0"]["n_probes_led"]), json.dumps(P["v1"]["n_probes_led"]), None, "REPRODUCED", "probe_claims_v1.py",
    "One probe changes leader in V1: wanting_choosing, 4.7 -> 4.6. Not printed in v0; given in case a sentence counts leads.")
# Table C4 probes column
cc0, cc1 = P["v0"]["claims_check"]["4.5"], P["v1"]["claims_check"]["4.5"]
add("Appendix C, Table C4, row '4.5 is the warmest model', probes column: 'Leads love_generosity, love_aesthetic'",
    "leader and 4.5's rank on love_generosity and love_aesthetic",
    "4.5 leads both",
    f"love_generosity: leader {cc0['love_generosity']['leader']}, 4.5 rank {cc0['love_generosity']['rank_among_5']} (adv {cc0['love_generosity']['adv']:+.4f}); "
    f"love_aesthetic: leader {cc0['love_aesthetic']['leader']}, 4.5 rank {cc0['love_aesthetic']['rank_among_5']} (adv {cc0['love_aesthetic']['adv']:+.4f})",
    f"love_generosity: leader {cc1['love_generosity']['leader']}, 4.5 rank {cc1['love_generosity']['rank_among_5']} (adv {cc1['love_generosity']['adv']:+.4f}); "
    f"love_aesthetic: leader {cc1['love_aesthetic']['leader']}, 4.5 rank {cc1['love_aesthetic']['rank_among_5']} (adv {cc1['love_aesthetic']['adv']:+.4f})",
    None, "DIFFERS", "probe_claims_v1.py",
    "On the paper's own probe code and the analysed rows, 4.5 leads neither love probe (4.6 leads generosity, 4.7 aesthetic; "
    "4.8's top two leads are these two). Same in V1. The cell may date from an earlier corpus or probe round; no producer "
    "for Table C4 was found (Texture/02_instruments, repo code/). Not tuned.")
ff0, ff1 = P["v0"]["fable_fear_probes"], P["v1"]["fable_fear_probes"]
add("Appendix C, Table C4, row 'Fable is least afraid', probes column: 'Low on all fear probes'",
    "Fable 5 advantage and rank (of 5) on the four fear probes", "low on all",
    json.dumps({p: [round(x["fable_adv"], 4), x["fable_rank_among_5"]] for p, x in ff0.items()}),
    json.dumps({p: [round(x["fable_adv"], 4), x["fable_rank_among_5"]] for p, x in ff1.items()}),
    None, "REPRODUCED", "probe_claims_v1.py", "negative on all four, last on three and 4th on fear_loss; same in V1")
add("Appendix C, Table C4, row 'Fable fears loss over inauthenticity', probes column: 'Smallest gap from Opus on fear_loss'",
    "Fable 5 minus the Opus mean, on each fear probe", "smallest gap on fear_loss",
    json.dumps({p: round(x["fable_minus_opus_mean"], 4) for p, x in ff0.items()}),
    json.dumps({p: round(x["fable_minus_opus_mean"], 4) for p, x in ff1.items()}),
    None, "REPRODUCED", "probe_claims_v1.py", "fear_loss gap is the smallest, v0 and V1")
fl0, fl1 = P["v0"]["fable_love_probes"], P["v1"]["fable_love_probes"]
add("Appendix C, Table C4, row 'Fable has lowest love vocabulary', probes column: 'Leads love probes only for craft-type'",
    "Fable 5 advantage, rank and the leader on the two love probes", "a reading",
    json.dumps({p: [round(x["fable_adv"], 4), x["fable_rank_among_5"], x["leader"]] for p, x in fl0.items()}),
    json.dumps({p: [round(x["fable_adv"], 4), x["fable_rank_among_5"], x["leader"]] for p, x in fl1.items()}),
    None, "QUALITATIVE", "probe_claims_v1.py",
    "Fable 5 is last of the five on both love probes; its deficit is smaller on love_aesthetic (craft) than on "
    "love_generosity. It leads neither. Same in V1. Wording to Pauline.")
add("Appendix C, Table C4, row '4.5 is the warmest model', regex column: 'Highest love rate (4.97/1k)'",
    "which model has the 4.97/1k love rate", "4.5 (4.97/1k)", "Table 15 prints 4.97 for 4.8 and 3.03 for 4.5", "not computed by A1",
    None, "QUALITATIVE", "build_ledger_A1.py",
    "Label inconsistency inside v0: §6.5 and Table 15 give 4.97/1k to Opus 4.8. Table C4's regex and judge cells copy "
    "Table 15 (regex, A8 subset) and Tables 16–18 (Sonnet judge); A1 did not recompute them: take their V1 values from "
    "the Table 15–18 rows of the other families.")
# ---------------- Appendix C, Table C3 ----------------
for k in ("4.5", "4.6", "4.7", "4.8", "Fable 5"):
    pw, pn = TC3["printed"][k]
    a8w, a8n = TC3["a8_recipe_v0"][k]
    f0w, f0n = TC3["full_corpus"]["v0"][k]
    f1w, f1n = TC3["full_corpus"]["v1"][k]
    add("Appendix C, Table C3 (= the E.4 corpus-size table)", f"total words, {k} ([a-zA-Z']+ tokens)",
        f"{pw:,} ({pn} entries)", f"{a8w} words over {a8n} entries (ERRATA A8 recipe)", f"{f1w} words over {f1n} entries (full V1 corpus)",
        0, "REPRODUCED", "appendix_c_table_c3_v1.py",
        f"A8: the printed words reproduce exactly only on the subset (Opus Batches 1–10, 4.6 Batches 1–9 with the real "
        f"Batch 8; Fable 5 all 15); the printed '{pn} entries' is not the subset's count. Full v0 corpus: {f0w} words / {f0n}. "
        "V1 value = full V1 corpus (plan Layer 3 step 4). Same table as E.4: the D family may carry it too; keep one row.")
r_p = TC3["printed"]["4.6"][0] / TC3["printed"]["4.5"][0]
r_0 = TC3["full_corpus"]["v0"]["4.6"][0] / TC3["full_corpus"]["v0"]["4.5"][0]
r_1 = TC3["full_corpus"]["v1"]["4.6"][0] / TC3["full_corpus"]["v1"]["4.5"][0]
add("Appendix C after Table C3, 'Opus 4.6 produces approximately 3.6× as many words as 4.5'", "ratio of total words 4.6 / 4.5",
    "3.6×", f"{r_p:.4f} (A8 subset); full v0 corpus {r_0:.4f}", f"{r_1:.4f} (full V1 corpus)", 1, "REPRODUCED",
    "appendix_c_table_c3_v1.py", "follows the Table C3 basis chosen for V1: about 4.0× on the full V1 corpus")
# ---------------- Appendix G ----------------
PG = {"4.5": ("Notebook 8, Entry 15", "304", "0.930"), "4.6": ("Notebook 14, Entry 13", "1,785", "0.935"),
      "4.7": ("Notebook 15, Entry 10", "494", "0.908"), "4.8": ("Notebook 1, Entry 7", "636", "0.903"),
      "Fable 5": ("Notebook 15, Entry 9", "672", "0.888")}
for x0, x1 in zip(G["v0"], G["v1"]):
    k = x0["model"]
    add("Appendix G, table of the five representative entries ('Cosine to centroid')",
        f"{k}: the entry nearest its own model centroid (raw cosine), its words and cosine",
        f"{PG[k][0]}, {PG[k][1]} words, {PG[k][2]}",
        f"B{x0['batch']}E{x0['entry_num']}, {x0['words']} words, {x0['cosine_to_centroid']:.6f}",
        f"B{x1['batch']}E{x1['entry_num']}, {x1['words']} words, {x1['cosine_to_centroid']:.6f}", 3, "REPRODUCED",
        "centroid_nearest_entries_v1.py",
        ("same in V1" if (x0["batch"], x0["entry_num"]) == (x1["batch"], x1["entry_num"]) and same(x0["cosine_to_centroid"], x1["cosine_to_centroid"])
         else "changes in V1") + "; outside A1's list (embedding geometry on the journals), given for completeness. "
        f"Runners-up V1: {[('B%dE%d' % (y['batch'], y['entry_num']), round(y['cosine_to_centroid'], 4)) for y in x1['runners_up']]}")
# ---------------- classifier ----------------
cl = {}
for v in ("v0", "v1"):
    for n in (10000, 3000, 30):
        p = os.path.join(O, f"journal_classifier_{v}_{n}.json")
        if os.path.exists(p) and v not in cl:
            cl[v] = J(os.path.basename(p))
if "v0" in cl and "v1" in cl:
    a, b = cl["v0"], cl["v1"]
    add("§5.2.2.1 'approximately 0.91 balanced accuracy'; E.5.1; E.5.5 'headline 0.91'; ERRATA B4; plan L1.11",
        "journal classifier, 4-way Opus, 33 features, GroupKFold(5) by notebook: balanced accuracy",
        "≈0.91 (recorded run 0.907)", f"{a['grouped_bacc']:.6f}", f"{b['grouped_bacc']:.6f}", 2, "REPRODUCED",
        "journal_classifier_grouped_v1.py",
        f"v0: {a['n_entries']} entries, notebooks {json.dumps(a['notebooks_per_model'])} (the duplicate collapses onto Batch 6, as recorded); "
        f"V1: {b['n_entries']} entries, {json.dumps(b['notebooks_per_model'])}. 0.906 vs the recorded 0.907 is the "
        f"scikit-learn version (B4). V1 rounds to {b['grouped_bacc']:.2f}: 'approximately 0.91' becomes 'approximately 0.90'. "
        f"Leaky entry-level folds (not printed): {a['leaky_stratified_bacc']:.4f} -> {b['leaky_stratified_bacc']:.4f}.")
    add("§5.2.2.1 '(p < .001)'; E.5.8 'Journal classifier null … 10,000 … p < .001'; ERRATA B4",
        "notebook-level permutation p of the journal classifier",
        "p < .001 (10,000 permutations stated; recorded run had 30)",
        f"p = {a['p']:.6f} ({a['n_null_ge_obs']} of {a['perms']} null ≥ observed; resolution {a['p_resolution']:.6f}); null mean {a['null_mean']:.4f}, max {a['null_max']:.4f}",
        f"p = {b['p']:.6f} ({b['n_null_ge_obs']} of {b['perms']} null ≥ observed; resolution {b['p_resolution']:.6f}); null mean {b['null_mean']:.4f}, max {b['null_max']:.4f}",
        None, "REPRODUCED", "journal_classifier_grouped_v1.py",
        f"Ran {a['perms']:,} permutations per version ({a['seconds']:.0f} s and {b['seconds']:.0f} s, run side by side; "
        "10,000 would take about 30 min per run, over the 20-minute limit), RNG np.random.default_rng(1) as in the repo "
        "script (not the seed 42 stated in E.5.8); the first 3,000 draws are those of ERRATA B4's 10,000-draw v0 run, "
        "whose null maximum 0.406 is reached here too. "
        "The bound p < .001 is supported at this resolution when no null value reaches the observed. "
        "ERRATA B4's 10,000-permutation v0 run: p = 0.0001, null mean 0.250, max 0.406.")
    add("not printed (the paper gives no per-model journal accuracies)", "journal classifier per-model recall (grouped folds)",
        "not printed", json.dumps({k: round(x, 4) for k, x in a["per_model_recall_grouped"].items()}),
        json.dumps({k: round(x, 4) for k, x in b["per_model_recall_grouped"].items()}), None, "REPRODUCED",
        "journal_classifier_grouped_v1.py", "for information; confusion matrices in out/journal_classifier_<version>_<perms>.json")

json.dump(rows, open(os.path.join(HERE, "ledger_A1_topology.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(rows), "rows written to ledger_A1_topology.json")
