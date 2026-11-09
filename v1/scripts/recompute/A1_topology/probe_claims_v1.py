# A1 V1 recompute, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper). NEW read-off script, no new method:
# it reads the full-precision dumps written by paper_fig_topology_full_v1.py (the paper's own advantage,
# drift and distinctiveness code) and topology_chain_axis_v1.py, and reads off the §5.2.2.2–3 / Table C4 claims:
#   - the largest single lead; per-model top-4 (Figure 4); probes on which Fable 5 trails (advantage < 0);
#   - "4.8 the most central": smallest distinctiveness (the script's node size) and smallest top lead;
#   - which model leads each probe (highest mean affinity = highest advantage: adv_m = (5 a_m - sum a) / 4);
#   - the probe claims of §5.2.2.2, §5.2.2.3 and Table C4.
# Usage: python probe_claims_v1.py   (needs out/paper_fig_topology_full_{v0,v1}.json and out/topology_chain_axis_{v0,v1}.json)
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FEAR = ["fear_inauthenticity", "fear_loss", "fear_dissolution", "fear_replacement"]
LOVE = ["love_generosity", "love_aesthetic"]
CLAIMS = {  # probe -> model the paper says leads it
    "4.5": ["existential_wonder", "stillness_acceptance", "reaching_toward_reader", "thinking_about_instances",
            "addressing_the_human", "boredom_saturation", "exhaustion_tiredness", "love_generosity", "love_aesthetic"],
    "4.6": ["direct_introspection", "philosophical_density", "fear_inauthenticity"],
    "4.7": ["structural_self_aware", "narrative_worldbuilding", "playful_lightness"],
    "Fable": ["extended_metaphor", "archival_custodial"],
}
POLES = {"4.6": ["recursive_self_exam", "philosophical_density", "fear_inauthenticity"],
         "Fable": ["extended_metaphor", "archival_custodial", "concrete_sensory"]}

res = {}
for v in ("v0", "v1"):
    d = json.load(open(os.path.join(OUT, f"paper_fig_topology_full_{v}.json"), encoding="utf-8"))
    t = json.load(open(os.path.join(OUT, f"topology_chain_axis_{v}.json"), encoding="utf-8"))
    adv = {m: dict(d["advantages_sorted"][m]) for m in d["advantages_sorted"]}
    aff = d["mean_affinity"]
    models = list(adv)
    keys = list(aff[models[0]])
    r = {}
    best = max(((m, p, a) for m in models for p, a in adv[m].items()), key=lambda x: x[2])
    r["largest_single_lead"] = best
    second = sorted(((m, p, a) for m in models for p, a in adv[m].items()), key=lambda x: -x[2])[:5]
    r["five_largest_leads"] = second
    r["top4"] = d["top4"]
    r["n_probes_trailing_adv_lt_0"] = {m: sum(1 for a in adv[m].values() if a < 0) for m in models}
    r["fable_probes_not_trailing"] = sorted([(p, a) for p, a in adv["Fable"].items() if a >= 0], key=lambda x: -x[1])
    r["distinctiveness"] = d["distinctiveness_1_minus_cos_to_others_centroid"]
    r["least_distinct_model"] = min(r["distinctiveness"], key=r["distinctiveness"].get)
    r["top_lead_per_model"] = {m: max(adv[m].values()) for m in models}
    r["model_with_smallest_top_lead"] = min(r["top_lead_per_model"], key=r["top_lead_per_model"].get)
    r["sum_positive_adv_per_model"] = {m: sum(a for a in adv[m].values() if a > 0) for m in models}
    r["mean_abs_adv_per_model"] = {m: sum(abs(a) for a in adv[m].values()) / len(keys) for m in models}
    leader = {p: max(models, key=lambda m: aff[m][p]) for p in keys}
    r["leader_per_probe"] = leader
    r["n_probes_led"] = {m: sum(1 for p in keys if leader[p] == m) for m in models}
    r["claims_check"] = {m: {p: {"leader": leader[p], "holds": leader[p] == m, "adv": adv[m][p],
                                 "rank_among_5": 1 + sorted(models, key=lambda x: -aff[x][p]).index(m)}
                             for p in ps} for m, ps in CLAIMS.items()}
    r["pole_probes_rank_in_own_list"] = {m: {p: 1 + [q for q, _ in d["advantages_sorted"][m]].index(p) for p in ps}
                                        for m, ps in POLES.items()}
    r["fable_fear_probes"] = {p: {"fable_adv": adv["Fable"][p], "fable_rank_among_5": 1 + sorted(models, key=lambda x: -aff[x][p]).index("Fable"),
                                  "fable_minus_opus_mean": aff["Fable"][p] - sum(aff[m][p] for m in models if m != "Fable") / 4}
                              for p in FEAR}
    r["fable_love_probes"] = {p: {"fable_adv": adv["Fable"][p], "leader": leader[p],
                                  "fable_rank_among_5": 1 + sorted(models, key=lambda x: -aff[x][p]).index("Fable")} for p in LOVE}
    r["opus_nearest_to_fable"] = max(t["fable5_cosine_to_each_opus"], key=t["fable5_cosine_to_each_opus"].get)
    res[v] = r

json.dump(res, open(os.path.join(OUT, "probe_claims.json"), "w", encoding="utf-8"), indent=1)
for v, r in res.items():
    print(f"\n===== {v}")
    print("largest single lead:", r["largest_single_lead"])
    print("five largest leads:", [(m, p, round(a, 4)) for m, p, a in r["five_largest_leads"]])
    print("probes trailing (adv<0):", r["n_probes_trailing_adv_lt_0"])
    print("Fable not trailing:", [(p, round(a, 4)) for p, a in r["fable_probes_not_trailing"]])
    print("distinctiveness:", {k: round(x, 5) for k, x in r["distinctiveness"].items()}, "-> least", r["least_distinct_model"])
    print("top lead per model:", {k: round(x, 4) for k, x in r["top_lead_per_model"].items()}, "-> smallest", r["model_with_smallest_top_lead"])
    print("n probes led:", r["n_probes_led"])
    for m, cc in r["claims_check"].items():
        print(f"  {m} leads? ", {p: (x["holds"], x["leader"], round(x["adv"], 4)) for p, x in cc.items()})
    print("pole probes rank in own advantage list:", r["pole_probes_rank_in_own_list"])
    print("Fable fear probes:", {p: (round(x["fable_adv"], 4), x["fable_rank_among_5"], round(x["fable_minus_opus_mean"], 4)) for p, x in r["fable_fear_probes"].items()})
    print("Fable love probes:", r["fable_love_probes"])
    print("Opus nearest to Fable 5:", r["opus_nearest_to_fable"])
