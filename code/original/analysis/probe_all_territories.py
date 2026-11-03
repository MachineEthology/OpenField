"""Check whether 4.5, 4.7, 4.8 share territory or have distinct regions."""
import sys; sys.stdout.reconfigure(encoding="utf-8")
import json, os
import numpy as np

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}

# Load vecs
opus_entries = []
for bn in range(1, 16):
    bd = os.path.join(BASE, f"Batch_{bn}")
    if not os.path.isdir(bd): continue
    for m in MODELS:
        files = [f for f in os.listdir(bd) if m in f and "#" not in f]
        if files:
            data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
            for i, e in enumerate(data):
                opus_entries.append({"model": m, "batch": bn, "pos": i + 1})

fable_entries = []
for bn in range(1, 16):
    bd = os.path.join(BASE, f"Batch_{bn}")
    if not os.path.isdir(bd): continue
    files = [f for f in os.listdir(bd) if "fable5" in f and "#" not in f]
    if files:
        data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
        for i, e in enumerate(data):
            t = e.get("text", "").strip()
            if t and t != "[silence]":
                fable_entries.append({"model": "fable5", "batch": bn, "pos": i + 1})

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

# Centroids
centroids = {}
for m in ALL:
    centroids[m] = np.mean(model_v[m], axis=0)

# Pairwise cosine between centroids
print("=== Pairwise cosine similarity between model centroids ===\n")
header = f"{'':>8}" + "".join(f"{LAB[m]:>8}" for m in ALL)
print(header)
for m1 in ALL:
    row = f"{LAB[m1]:>8}"
    for m2 in ALL:
        c = np.dot(centroids[m1], centroids[m2]) / (np.linalg.norm(centroids[m1]) * np.linalg.norm(centroids[m2]))
        row += f"  {c:.4f}"
    print(row)

# Pairwise Euclidean distance between centroids
print("\n\n=== Pairwise Euclidean distance between centroids ===\n")
header = f"{'':>8}" + "".join(f"{LAB[m]:>8}" for m in ALL)
print(header)
for m1 in ALL:
    row = f"{LAB[m1]:>8}"
    for m2 in ALL:
        d = np.linalg.norm(np.array(centroids[m1]) - np.array(centroids[m2]))
        row += f"  {d:.4f}"
    print(row)

# Load ALL probes
probes_orig = json.load(open(BASE + "/concept_probe_embeddings.json"))
probes_fable = json.load(open(BASE + "/concept_probe_fable_specific.json"))
probes_46 = json.load(open(BASE + "/concept_probe_46_specific.json"))
all_probes = {}
all_probes.update(probes_orig)
all_probes.update(probes_fable)
all_probes.update(probes_46)

def cosine_batch(vecs, pv):
    vecs = np.array(vecs)
    pv = np.array(pv)
    return (vecs @ pv) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(pv))

# For each model, compute advantage over all-others-mean on every probe
# Then find each model's "signature probes"
print("\n\n=== Each model's signature probes (highest advantage over others) ===\n")
for m in ALL:
    advs = []
    for p, pv in all_probes.items():
        pv = np.array(pv)
        m_mean = float(np.mean(cosine_batch(model_v[m], pv)))
        others_mean = np.mean([float(np.mean(cosine_batch(model_v[om], pv)))
                               for om in ALL if om != m])
        advs.append((p, m_mean - others_mean, m_mean))
    advs.sort(key=lambda x: -x[1])
    print(f"{LAB[m]} leads on:")
    for p, adv, raw in advs[:6]:
        print(f"  {p:>28}: +{adv:.4f}  (raw={raw:.4f})")
    print(f"{LAB[m]} trails on:")
    for p, adv, raw in advs[-3:]:
        print(f"  {p:>28}: {adv:.4f}  (raw={raw:.4f})")
    print()

# Trajectory comparison: do 4.5, 4.7, 4.8 stay flat or migrate?
# Compute each model's direction vs all-others, then check trajectory
print("\n=== Trajectory: does each model migrate into its own territory? ===\n")
positions = list(range(1, 21))
for m in ALL:
    c_m = np.mean(model_v[m], axis=0)
    c_others = np.mean([v for om in ALL if om != m for v in model_v[om]], axis=0)
    diff = c_m - c_others
    diff_norm = diff / np.linalg.norm(diff)

    by_pos = {p: [] for p in positions}
    for i, e in enumerate(model_e[m]):
        proj = float(np.dot(model_v[m][i], diff_norm))
        if e["pos"] in by_pos:
            by_pos[e["pos"]].append(proj)

    means = [np.mean(by_pos[p]) if by_pos[p] else float("nan") for p in positions]
    e1 = means[0] if not np.isnan(means[0]) else 0
    e20 = means[19] if not np.isnan(means[19]) else 0
    drift = e20 - e1
    direction = "deepens" if drift > 0.02 else ("retreats" if drift < -0.02 else "stable")
    print(f"  {LAB[m]:>6}: E1={e1:.3f} -> E20={e20:.3f}  drift={drift:+.3f}  [{direction}]")

# Uniqueness score: mean distance of each model's centroid from the others-centroid
print("\n\n=== Distinctiveness score (cosine distance from others-centroid) ===\n")
for m in ALL:
    c_others = np.mean([v for om in ALL if om != m for v in model_v[om]], axis=0)
    cos_dist = 1 - np.dot(centroids[m], c_others) / (np.linalg.norm(centroids[m]) * np.linalg.norm(c_others))
    print(f"  {LAB[m]:>6}: {cos_dist:.6f}")
