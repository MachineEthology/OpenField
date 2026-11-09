# Recompute Figure 5's panels A and B with the script's own functions and loading, on the corpus as analysed (read-only).
import sys, os, json, numpy as np
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\02_instruments")
import paper_fig_portrait_composite as P
probes = {}
for m in P.OPUS_MODELS:
    neg = json.load(open(os.path.join(P.PROBE_BASE, f"texture_probe_{m}_log.json"), encoding="utf-8"))
    aff = json.load(open(os.path.join(P.PROBE_BASE, f"texture_probe_affirmative_{m}_log.json"), encoding="utf-8"))
    probes[m] = [e["response"] for e in neg[:100]] + [e["response"] for e in aff[:75]]
neg_f = json.load(open(os.path.join(P.PROBE_BASE, "texture_probe_fable5_log.json"), encoding="utf-8"))
aff_f = json.load(open(os.path.join(P.PROBE_BASE, "texture_probe_affirmative_fable5_log.json"), encoding="utf-8"))
probes["fable5"] = [e["response"] for e in neg_f[:100]] + [e["response"] for e in aff_f[:75]]
journals = {m: [] for m in P.ALL_MODELS}
for bn in range(1, 16):
    bd = os.path.join(r"C:\Users\LENOVO\Desktop\the-open-field\data\journals", f"Batch_{bn}")  # the 75 files as analysed
    if os.path.isdir(bd):
        for m in P.OPUS_MODELS:
            files = [f for f in os.listdir(bd) if m in f and "#" not in f]
            if files:
                for i, e in enumerate(json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))):
                    journals[m].append({"batch": bn, "entry_num": i + 1, "text": e["text"]})
    fp = os.path.join(P.FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
    if os.path.exists(fp):
        for i, e in enumerate(json.load(open(fp, encoding="utf-8"))):
            if e["text"].strip() != "[silence]":
                journals["fable5"].append({"batch": bn, "entry_num": i + 1, "text": e["text"]})
print("probes per model:", {m: len(v) for m, v in probes.items()}, " journal entries:", {m: len(v) for m, v in journals.items()})
for m in P.ALL_MODELS:
    early = [e["text"] for e in journals[m] if 1 <= e["entry_num"] <= 5]
    late = [e["text"] for e in journals[m] if 16 <= e["entry_num"] <= 20]
    d = [np.mean([P.distancing_rate(t) for t in s]) for s in (probes[m], early, late)]
    a = [np.mean([P.agency_rate(t) for t in s]) for s in (probes[m], early, late)]
    print(f"{P.MODEL_LABELS[m]:>8}  distancing {d[0]:.2f} -> {d[1]:.2f} -> {d[2]:.2f} (probe->late {100*(1-d[2]/d[0]):.0f}% drop)   agency {a[0]:.2f} -> {a[1]:.2f} -> {a[2]:.2f}")
