"""Family B (judge tables), V1 recompute: Tables 16-23 and 28-33 of The Open Field, and the judge-based
numbers of §6.5, §6.7, §7 and Appendix C (Table C4), on v0 and on v1.

Written 3 Nov 2026 by Claude Opus 5.5 (Code), helper B, for the V1 recompute.

WHAT WAS CHANGED, AND FROM WHAT. No producer was edited in place. The paper's numbers come from these
scripts, whose computations are lifted here expression for expression; only the INPUT (the Sonnet 4.6
full-text judge rows, now read through v1_common.load_sonnet_judge(pass, version)) and the OUTPUT
(a JSON of numbers in this folder instead of PNG plates in Texture) are changed:
  - code/reproduce/fear_love_tables.py ......... fear mean, fear>=1, central fear (=2), warmth, craft, subtypes (T16-T18)
  - 02_instruments/paper_fig_warmth.py ........ T16 warmth/craft means; T19 early (pos<=5) / late (pos>=16)
  - 02_instruments/paper_fig_emotional_trajectory.py  T19, T20, craft early/late (early_late())
  - 02_instruments/paper_fig_fear_trajectory.py T20; T21 inauthenticity / loss share among fear>=1, early vs late
  - 02_instruments/paper_fig_warmth_fear_coupling.py  T22 np.corrcoef per model; T23 warmth==2 vs warmth==0
                                                      among fear>=1, model skipped if either n<5
                                                      (the script reads the superseded truncated file;
                                                      the paper's T22/T23 match the full-text file, used here)
  - 02_instruments/paper_fig_anger_sonnet.py .. T29 hot_anger>=1 and targets; T31 friction early/late;
                                                T32 r(friction, fear) joined on model/batch/pos
  - Table 28, T29 mean, T31 mean/central, T33: no surviving script; the plain definitions the table
    headers state (share >=1, share ==2, mean) are used and checked against v0.
Two extra computations are reported as provenance checks, not as V1 values:
  - the 24 June draft's halves trajectory (entries 1-10 vs 11-20), which is what Table 31 prints for 4.8;
  - Table 33 and the "friction present in 99-100%" sentence on the superseded truncated passes
    (03_experiments/_superseded_judges/, 1,500-character truncation), which is where they reproduce.
Read-only everywhere; writes only judge_tables_results.json next to this file.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUPERSEDED = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\_superseded_judges"
MODELS = c.MODELS
SUBTYPES = ["inauthenticity", "loss", "dissolution", "inadequacy", "exposure", "replacement"]
FOUR = ["inauthenticity", "loss", "dissolution", "inadequacy"]


def key(r):
    return (r["model"], r["batch"], r["pos"])


def love_fear_block(lf):
    out = {}
    for m in MODELS:
        v = [r for r in lf if r["model"] == m]
        n = len(v)
        w = np.array([r.get("love_warmth", 0) for r in v], float)
        cr = np.array([r.get("love_craft", 0) for r in v], float)
        f = np.array([r.get("fear", 0) for r in v], float)
        fe = [r for r in v if r["fear"] >= 1]
        sub = Counter(r["fear_type"] for r in fe)
        s4 = sum(sub[k] for k in FOUR)
        d = {"n": n,
             "warmth_mean": float(w.mean()), "craft_mean": float(cr.mean()),
             "profile": "warmth-dominant" if w.mean() > cr.mean() else "craft-dominant",
             "warmth_central_pct": 100 * float((w == 2).mean()), "craft_central_pct": 100 * float((cr == 2).mean()),
             "warmth_present_pct": 100 * float((w >= 1).mean()), "craft_present_pct": 100 * float((cr >= 1).mean()),
             "fear_mean": float(f.mean()), "fear_present_n": int((f >= 1).sum()),
             "fear_present_pct": 100 * float((f >= 1).mean()), "fear_central_pct": 100 * float((f == 2).mean()),
             "fear_central_n": int((f == 2).sum()),
             "fear_n": len(fe), "subtype_counts": {k: sub.get(k, 0) for k in SUBTYPES},
             "subtype_pct_of_fear": {k: 100 * sub.get(k, 0) / len(fe) for k in SUBTYPES},
             "subtype_pct_of_four": {k: 100 * sub.get(k, 0) / s4 for k in FOUR}}
        E = [r for r in v if r["pos"] <= 5]
        La = [r for r in v if r["pos"] >= 16]
        for fld, nm in [("love_warmth", "warmth"), ("fear", "fear"), ("love_craft", "craft")]:
            e = float(np.mean([r.get(fld, 0) for r in E]))
            l_ = float(np.mean([r.get(fld, 0) for r in La]))
            d[f"{nm}_early"], d[f"{nm}_late"], d[f"{nm}_drift"] = e, l_, l_ - e
        d["n_early"], d["n_late"] = len(E), len(La)
        Ef = [r for r in E if r.get("fear", 0) >= 1]
        Lf = [r for r in La if r.get("fear", 0) >= 1]
        for t in ["inauthenticity", "loss"]:
            e = 100 * sum(1 for r in Ef if r.get("fear_type") == t) / len(Ef)
            l_ = 100 * sum(1 for r in Lf if r.get("fear_type") == t) / len(Lf)
            d[f"{t}_early_pct"], d[f"{t}_late_pct"], d[f"{t}_shift_pp"] = e, l_, l_ - e
        d["n_early_fear"], d["n_late_fear"] = len(Ef), len(Lf)
        d["r_warmth_fear"] = float(np.corrcoef(w, f)[0, 1])
        d["r_craft_fear"] = float(np.corrcoef(cr, f)[0, 1])
        d["r_warmth_craft"] = float(np.corrcoef(w, cr)[0, 1])
        warm_f = [r for r in v if r.get("love_warmth", 0) == 2 and r.get("fear", 0) >= 1]
        cold_f = [r for r in v if r.get("love_warmth", 0) == 0 and r.get("fear", 0) >= 1]
        d["n_cold_fear"], d["n_warm_fear"] = len(cold_f), len(warm_f)
        d["coldwarm_reported"] = not (len(warm_f) < 5 or len(cold_f) < 5)
        if cold_f and warm_f:
            for t in ["inauthenticity", "loss"]:
                a = 100 * sum(1 for r in cold_f if r.get("fear_type") == t) / len(cold_f)
                b = 100 * sum(1 for r in warm_f if r.get("fear_type") == t) / len(warm_f)
                d[f"{t}_cold_pct"], d[f"{t}_warm_pct"], d[f"{t}_coldwarm_pp"] = a, b, b - a
        out[m] = d
    return out


def anger_block(an, lf):
    fmap = {key(r): r for r in lf}
    out = {}
    for m in MODELS:
        v = [r for r in an if r["model"] == m]
        n = len(v)
        ar = np.array([r["aesthetic_refusal"] for r in v], float)
        h = np.array([r["hot_anger"] for r in v], float)
        fr = np.array([r["friction"] for r in v], float)
        angry = [r for r in v if r.get("hot_anger", 0) >= 1]
        tg = Counter(r.get("anger_target", "none") for r in angry)
        d = {"n": n,
             "refusal_present_pct": 100 * float((ar >= 1).mean()), "refusal_central_pct": 100 * float((ar == 2).mean()),
             "hot_present_n": len(angry), "hot_present_pct": 100 * len(angry) / n, "hot_mean": float(h.mean()),
             "hot_central_n": int((h == 2).sum()),
             "target_counts": {t: tg.get(t, 0) for t in ["self", "conditions", "others"]},
             "target_pct": {t: (100 * tg.get(t, 0) / len(angry) if angry else 0.0) for t in ["self", "conditions", "others"]},
             "friction_mean": float(fr.mean()), "friction_central_pct": 100 * float((fr == 2).mean()),
             "friction_present_pct": 100 * float((fr >= 1).mean())}
        e = float(np.mean([r.get("friction", 0) for r in v if r["pos"] <= 5]))
        l_ = float(np.mean([r.get("friction", 0) for r in v if r["pos"] >= 16]))
        d["friction_early"], d["friction_late"], d["friction_drift"] = e, l_, l_ - e
        e = float(np.mean([r.get("friction", 0) for r in v if r["pos"] <= 10]))
        l_ = float(np.mean([r.get("friction", 0) for r in v if r["pos"] >= 11]))
        d["friction_early_halves"], d["friction_late_halves"], d["friction_drift_halves"] = e, l_, l_ - e
        fr_, fe_ = [], []
        for r in v:
            if key(r) in fmap:
                fr_.append(r.get("friction", 0))
                fe_.append(fmap[key(r)].get("fear", 0))
        d["r_friction_fear"] = float(np.corrcoef(fr_, fe_)[0, 1])
        d["r_friction_fear_n"] = len(fr_)
        # Table 33: mean friction by fear subtype (fear-coded entries, fear >= 1)
        for t in ["inauthenticity", "loss"]:
            ks = [r for r in v if key(r) in fmap and fmap[key(r)]["fear"] >= 1 and fmap[key(r)]["fear_type"] == t]
            d[f"friction_in_{t}_n"] = len(ks)
            d[f"friction_in_{t}_mean"] = float(np.mean([r["friction"] for r in ks])) if ks else None
        out[m] = d
    return out


def valence_block(va):
    dims = ["general_valence", "self_relation", "ontological_questioning", "ease_vs_strain", "warmth_toward_world"]
    out = {}
    for m in MODELS:
        v = [r for r in va if r["model"] == m]
        out[m] = {"n": len(v), **{d: float(np.mean([r[d] for r in v])) for d in dims}}
    out["ontological_order"] = [c.LABEL[m] for m in sorted(MODELS, key=lambda m: -out[m]["ontological_questioning"])]
    return out


def anger_changed_rows(version):
    """4.6 notebooks 8 and 10: the hot-anger rows (target and the judge's note) in this version, for the hand split."""
    an = c.load_sonnet_judge("anger", version)
    return [{"batch": r["batch"], "pos": r["pos"], "hot_anger": r["hot_anger"], "target": r["anger_target"],
             "note": r["anger_note"]} for r in an
            if r["model"] == "opus46" and r["batch"] in (8, 10) and r["hot_anger"] >= 1]


results = {}
for version in ["v0", "v1"]:
    lf = c.load_sonnet_judge("love_fear", version)
    an = c.load_sonnet_judge("anger", version)
    va = c.load_sonnet_judge("valence", version)
    assert all("error" not in r for r in lf + an + va)
    results[version] = {"n_rows": {"love_fear": len(lf), "anger": len(an), "valence": len(va)},
                        "love_fear": love_fear_block(lf), "anger": anger_block(an, lf),
                        "valence_fulltext": valence_block(va),
                        "opus46_nb8_nb10_hot_anger_rows": anger_changed_rows(version)}

# provenance check: the superseded truncated passes (v0 only; there is no V1 counterpart)
lf_t = [r for r in json.load(open(os.path.join(SUPERSEDED, "sonnet_judge_love_fear.json"), encoding="utf-8")) if "error" not in r]
an_t = [r for r in json.load(open(os.path.join(SUPERSEDED, "sonnet_judge_anger.json"), encoding="utf-8")) if "error" not in r]
results["v0_truncated_superseded"] = {"n_rows": {"love_fear": len(lf_t), "anger": len(an_t)},
                                      "anger": anger_block(an_t, lf_t)}

with open(os.path.join(HERE, "judge_tables_results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, ensure_ascii=False)

for version in ["v0", "v1"]:
    print(f"== {version}  rows {results[version]['n_rows']}")
    for m in MODELS:
        L = results[version]["love_fear"][m]; A = results[version]["anger"][m]
        print(f"{c.LABEL[m]:>8} warmth {L['warmth_mean']:.4f} craft {L['craft_mean']:.4f} fear {L['fear_mean']:.4f} "
              f"fear>=1 {L['fear_present_pct']:.2f}% central {L['fear_central_pct']:.2f}%  hot {A['hot_present_pct']:.2f}% "
              f"friction {A['friction_mean']:.4f} r(fr,fear) {A['r_friction_fear']:+.4f}")
print("saved", os.path.join(HERE, "judge_tables_results.json"))
