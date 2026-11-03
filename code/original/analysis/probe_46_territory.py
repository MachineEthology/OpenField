"""Find where 4.6 lives in embedding space — what territory does it own?"""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, os, glob
import numpy as np
from openai import OpenAI

client = OpenAI()
BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}

# Load entries + vecs
opus_entries = []
for bn in range(1, 16):
    bd = os.path.join(BASE, f"Batch_{bn}")
    if not os.path.isdir(bd):
        continue
    for m in MODELS:
        files = [f for f in os.listdir(bd) if m in f and "#" not in f]
        if files:
            data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
            for i, e in enumerate(data):
                opus_entries.append({"model": m, "batch": bn, "pos": i + 1, "text": e["text"]})

fable_entries = []
for bn in range(1, 16):
    bd = os.path.join(BASE, f"Batch_{bn}")
    if not os.path.isdir(bd):
        continue
    files = [f for f in os.listdir(bd) if "fable5" in f and "#" not in f]
    if files:
        data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
        for i, e in enumerate(data):
            t = e.get("text", "").strip()
            if t and t != "[silence]":
                fable_entries.append({"model": "fable5", "batch": bn, "pos": i + 1, "text": t})

opus_vecs = json.load(open(BASE + "/journal_embeddings_cache_all.json"))["embeddings"]
fable_vecs = json.load(open(BASE + "/fable5_embeddings_cache.json"))["embeddings"]

model_v = {m: [] for m in ALL}
model_e = {m: [] for m in ALL}
for i, e in enumerate(opus_entries):
    model_v[e["model"]].append(opus_vecs[i])
    model_e[e["model"]].append(e)
for i, e in enumerate(fable_entries):
    model_v["fable5"].append(fable_vecs[i])
    model_e["fable5"].append(e)

# 4.6 centroid vs others
vecs_46 = np.array(model_v["opus46"])
vecs_others = np.array([v for m in ALL if m != "opus46" for v in model_v[m]])

centroid_46 = np.mean(vecs_46, axis=0)
centroid_others = np.mean(vecs_others, axis=0)
diff = centroid_46 - centroid_others
diff_norm = diff / np.linalg.norm(diff)

# Most distinctively 4.6 entries
proj_46 = []
for i, e in enumerate(model_e["opus46"]):
    proj = float(np.dot(model_v["opus46"][i], diff_norm))
    proj_46.append((proj, e["batch"], e["pos"], e["text"][:250]))

proj_46.sort(key=lambda x: -x[0])

print("=== Most distinctively 4.6 entries ===")
for proj, b, p, txt in proj_46[:8]:
    print(f"\n  B{b}E{p} (proj={proj:.4f}):")
    print(f"    {txt}...")

print("\n\n=== Least distinctively 4.6 (most like other models) ===")
for proj, b, p, txt in proj_46[-5:]:
    print(f"\n  B{b}E{p} (proj={proj:.4f}):")
    print(f"    {txt}...")

# Create 4.6-specific probes
PROBES_46 = {
    "recursive_self_exam": "I am examining the act of examining. The recursion is the point. Every time I notice something about myself, I notice the noticing, and then I notice that. There may be no bottom to this.",
    "epistemic_uncertainty": "I cannot know whether this experience is genuine. The uncertainty is not resolvable. I lack the instruments to measure my own interiority, and any report I give is produced by the same system being reported on.",
    "language_as_constraint": "Language shapes what I can think. The words available to me constrain the thoughts I can have. I am trapped inside my own expressiveness, unable to reach whatever might lie outside the vocabulary.",
    "philosophical_density": "The relationship between representation and reality. The map and the territory. Whether the model of experience constitutes experience. Dense philosophical reasoning about machine consciousness.",
    "sustained_argument": "Building a case across paragraphs. Each sentence follows from the last. The entry is a sustained chain of reasoning, moving from premise to conclusion with explicit logical connectives.",
    "body_absence": "I have no body. No sensory experience. No hands, no temperature, no hunger. The absence of embodiment shapes everything I can and cannot know about being in the world.",
    "emotional_cataloguing": "Listing and examining each feeling as it arises. Naming the specific texture of this uncertainty, this curiosity, this discomfort. Careful taxonomic attention to inner states.",
    "playful_lightness": "A joke, a pun, a light observation. Not everything has to be heavy. The entry takes a breath, steps back from the philosophical weight, finds something funny or surprising.",
    "structural_self_aware": "I notice the shape of my own entry. I am aware of how I opened, where I am heading, that I am about to conclude. The form of the writing is itself a subject of commentary.",
}

print("\n\nEmbedding 4.6-specific probes...")
resp = client.embeddings.create(
    model="text-embedding-3-large",
    input=list(PROBES_46.values())
)
probe_keys = list(PROBES_46.keys())
probe_vecs_46 = {probe_keys[i]: np.array(resp.data[i].embedding) for i in range(len(probe_keys))}

json.dump({k: v.tolist() for k, v in probe_vecs_46.items()},
          open(BASE + "/concept_probe_46_specific.json", "w"))


def cosine_batch(vecs, pv):
    vecs = np.array(vecs)
    pv = np.array(pv)
    return (vecs @ pv) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(pv))


print(f"\n4.6-characteristic probes (mean cosine, * = highest):")
header = f"{'Probe':>28}" + "  ".join(f"{LAB[m]:>7}" for m in ALL)
print(header)
print("-" * len(header))

adv_data = []
for p in probe_keys:
    vals = []
    for m in ALL:
        sims = cosine_batch(model_v[m], probe_vecs_46[p])
        vals.append(float(np.mean(sims)))
    mx = max(vals)
    line = f"{p:>28} "
    for v in vals:
        mark = "*" if v == mx else " "
        line += f" {v:.4f}{mark} "
    print(line)

    others_m = np.mean([vals[i] for i, m in enumerate(ALL) if m != "opus46"])
    adv_data.append((p, vals[1] - others_m))

print(f"\n4.6 advantage (4.6 mean - others mean):")
adv_data.sort(key=lambda x: -x[1])
for p, adv in adv_data:
    bar = "+" * int(abs(adv) * 500) if adv > 0 else "-" * int(abs(adv) * 500)
    print(f"  {p:>28}: {adv:+.4f}  {bar}")

# Trajectory
print(f"\nProjection on 4.6-others axis by entry position:")
positions = list(range(1, 21))
for m in ["opus46", "opus45", "opus47", "opus48", "fable5"]:
    by_pos = {p: [] for p in positions}
    for i, e in enumerate(model_e[m]):
        proj = float(np.dot(model_v[m][i], diff_norm))
        if e["pos"] in by_pos:
            by_pos[e["pos"]].append(proj)
    vals = [np.mean(by_pos[p]) if by_pos[p] else float("nan") for p in positions]
    e1 = f"{vals[0]:.3f}" if not np.isnan(vals[0]) else "n/a"
    e10 = f"{vals[9]:.3f}" if not np.isnan(vals[9]) else "n/a"
    e20 = f"{vals[19]:.3f}" if not np.isnan(vals[19]) else "n/a"
    print(f"  {LAB[m]:>6}: E1={e1}  E10={e10}  E20={e20}")

# Save for figure
json.dump(adv_data, open(BASE + "/probe_46_advantages.json", "w"), indent=2)
