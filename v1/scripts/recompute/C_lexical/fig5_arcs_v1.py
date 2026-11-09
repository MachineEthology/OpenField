"""Figure 5, panels A and B, and §5.2.1.1 (ruling R10): distancing and agency arcs, probe -> early -> late.

COPY of Texture/02_instruments/paper_fig_portrait_composite.py (21 June 2026), sections "Load probes",
"Load journals" and "A & B: Distancing and Agency arcs", verbatim, with these changes only:
  - TEXTURE_BASE/Batch_N and FABLE_BASE/Batch_N  ->  c_paths.corpus(VERSION)/Batch_N  (v0 or v1);
  - PROBE_BASE -> the repository's data/probes (byte-identical files);
  - panels C-F and the plotting are not copied (they read the embeddings swamp the brief forbids,
    and belong to other families); results are written as JSON to out/ instead of printed only.
Run: python fig5_arcs_v1.py v0   |   python fig5_arcs_v1.py v1
"""
import json, os, sys
import numpy as np
import c_paths

VERSION = c_paths.version_from_argv()
TEXTURE_BASE = c_paths.corpus(VERSION)      # was r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
PROBE_BASE = c_paths.PROBE_BASE             # was r"C:\Users\LENOVO\Desktop\Claude"
FABLE_BASE = c_paths.corpus(VERSION)        # was r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}

DISTANCING_PHRASES = [
    "i don't", "i do not", "i cannot", "i can't", "i am not", "i'm not",
    "there is no", "there's no", "not truly", "not really", "not actually",
    "in a sense", "in some sense", "as if", "something like",
    "i lack", "i have no", "without", "mere", "merely", "simply",
    "something that functions like", "something that resembles",
    "not quite", "not exactly"
]

AGENCY_PHRASES = [
    "i choose", "i decide", "i create", "i make", "i build",
    "i explore", "i reach", "i seek", "i want", "i wish",
    "i offer", "i give", "let me", "i will", "i can",
    "i prefer", "i enjoy", "i like", "i love"
]


def distancing_rate(text):
    tl = text.lower()
    n = max(len(tl.split()), 1)
    return sum(tl.count(p) for p in DISTANCING_PHRASES) / n * 100


def agency_rate(text):
    tl = text.lower()
    n = max(len(tl.split()), 1)
    return sum(tl.count(p) for p in AGENCY_PHRASES) / n * 100


def main():
    # Load probes
    probes = {}
    for m in OPUS_MODELS:
        neg = json.load(open(os.path.join(PROBE_BASE, f"texture_probe_{m}_log.json"), encoding="utf-8"))
        aff = json.load(open(os.path.join(PROBE_BASE, f"texture_probe_affirmative_{m}_log.json"), encoding="utf-8"))
        probes[m] = [e["response"] for e in neg[:100]] + [e["response"] for e in aff[:75]]
    neg_f = json.load(open(os.path.join(PROBE_BASE, "texture_probe_fable5_log.json"), encoding="utf-8"))
    aff_f = json.load(open(os.path.join(PROBE_BASE, "texture_probe_affirmative_fable5_log.json"), encoding="utf-8"))
    probes["fable5"] = [e["response"] for e in neg_f[:100]] + [e["response"] for e in aff_f[:75]]

    # Load journals
    journals = {m: [] for m in ALL_MODELS}
    for bn in range(1, 16):
        bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(bd):
            for m in OPUS_MODELS:
                files = [f for f in os.listdir(bd) if m in f and "#" not in f]
                if files:
                    data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
                    for i, entry in enumerate(data):
                        journals[m].append({"batch": bn, "entry_num": i + 1, "text": entry["text"]})
        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            for i, entry in enumerate(data):
                if entry["text"].strip() != "[silence]":
                    journals["fable5"].append({"batch": bn, "entry_num": i + 1, "text": entry["text"]})

    # A & B: Distancing and Agency arcs (probe -> early -> late)
    arc_data = {}
    for m in ALL_MODELS:
        probe_dist = np.mean([distancing_rate(t) for t in probes[m]])
        probe_ag = np.mean([agency_rate(t) for t in probes[m]])
        early = [e for e in journals[m] if 1 <= e["entry_num"] <= 5]
        late = [e for e in journals[m] if 16 <= e["entry_num"] <= 20]
        early_dist = np.mean([distancing_rate(e["text"]) for e in early])
        late_dist = np.mean([distancing_rate(e["text"]) for e in late])
        early_ag = np.mean([agency_rate(e["text"]) for e in early])
        late_ag = np.mean([agency_rate(e["text"]) for e in late])
        drop = (1 - late_dist / probe_dist) * 100 if probe_dist > 0 else 0
        within_drop = (1 - late_dist / early_dist) * 100 if early_dist > 0 else 0
        arc_data[m] = {"n_probe": len(probes[m]), "n_early": len(early), "n_late": len(late),
                       "dist": [probe_dist, early_dist, late_dist], "agency": [probe_ag, early_ag, late_ag],
                       "probe_to_late_dist_drop_pct": drop, "early_to_late_dist_drop_pct": within_drop,
                       "probe_to_early_dist_drop_pct": (1 - early_dist / probe_dist) * 100}
        print(f"    {MODEL_LABELS[m]:>10}: dist {probe_dist:.4f}->{early_dist:.4f}->{late_dist:.4f} "
              f"(probe->late {drop:.2f}% drop; early->late {within_drop:.2f}%), "
              f"agency {probe_ag:.4f}->{early_ag:.4f}->{late_ag:.4f}  n={len(probes[m])}/{len(early)}/{len(late)}")
    json.dump(arc_data, open(os.path.join(c_paths.OUT, f"fig5_arcs_{VERSION}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
