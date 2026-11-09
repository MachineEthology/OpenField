# COPY of Texture/02_instruments/centroid_nearest_entries.py (the Appendix G table "Cosine to centroid"), for the
# V1 recompute (A1). Changed, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper):
#   - INPUT ONLY: the journals are read from v1_common.corpus_dir(version) (OPUS_BASE and FABLE_BASE), version =
#     argv[1] "v0" | "v1"; the vectors come from v1_common.embeddings_path(version) (the clean npz), looked up by
#     (model, batch, pos) with a text_md5 check, instead of the two JSON caches (the brief forbids the Opus cache).
#   - OUTPUT PATH: out/centroid_nearest_entries_<version>.json in this folder; output only: the two runners-up.
"""
Find the journal entry closest to each model's embedding centroid.
Produces one "most representative" entry per model for Appendix H.
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import hashlib

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as _c
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"
BASE = os.path.dirname(os.path.abspath(__file__))
OPUS_BASE = _c.corpus_dir(VERSION)
FABLE_BASE = _c.corpus_dir(VERSION)
_Z = {k: v for k, v in np.load(_c.embeddings_path(VERSION)).items()}  # read each array once
_IDX = {(str(m), int(b), int(p)): i for i, (m, b, p) in enumerate(zip(_Z["model"], _Z["batch"], _Z["pos"]))}


def _vecs_for(entries):
    out = []
    for e in entries:
        i = _IDX[(e["model"], e["batch"], e["entry_num"])]
        assert _Z["text_md5"][i] == hashlib.md5((e["model"] + e["text"].strip()).encode("utf-8")).hexdigest()
        out.append(_Z["vectors"][i].astype(np.float64))
    return np.array(out)
OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}


def load_opus_journals():
    entries = []
    for batch_num in range(1, 16):
        batch_dir = os.path.join(OPUS_BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        for m in OPUS_MODELS:
            files = [f for f in os.listdir(batch_dir) if m in f and "#" not in f]
            if files:
                with open(os.path.join(batch_dir, files[0]), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for i, entry in enumerate(data):
                    entries.append({
                        "model": m, "batch": batch_num, "entry_num": i + 1,
                        "text": entry["text"],
                        "words": len(entry["text"].split())
                    })
    return entries


def load_fable_journals():
    entries = []
    for batch_num in range(1, 16):
        batch_dir = os.path.join(FABLE_BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        fname = f"schrodingers_box_fable5_Batch{batch_num}.json"
        fpath = os.path.join(batch_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for i, entry in enumerate(data):
            text = entry["text"].strip()
            if text == "[silence]":
                continue
            entries.append({
                "model": "fable5", "batch": batch_num, "entry_num": i + 1,
                "text": text,
                "words": len(text.split())
            })
    return entries


print("Loading Opus journals...")
opus_entries = load_opus_journals()
print(f"  {len(opus_entries)} Opus entries")

print("Loading Fable journals...")
fable_entries = load_fable_journals()
print(f"  {len(fable_entries)} Fable entries")

print("Loading Opus embeddings...")
opus_emb = _vecs_for(opus_entries)   # [A1 V1 copy: from the npz]
print(f"  {opus_emb.shape}")

print("Loading Fable embeddings...")
fable_emb = _vecs_for(fable_entries)  # [A1 V1 copy: from the npz]
print(f"  {fable_emb.shape}")

assert len(opus_entries) == opus_emb.shape[0], f"Opus mismatch: {len(opus_entries)} entries vs {opus_emb.shape[0]} embeddings"
assert len(fable_entries) == fable_emb.shape[0], f"Fable mismatch: {len(fable_entries)} entries vs {fable_emb.shape[0]} embeddings"

all_entries = opus_entries + fable_entries
all_emb = np.vstack([opus_emb, fable_emb])

print(f"\nCombined: {len(all_entries)} entries, {all_emb.shape}")

# Compute centroid-nearest for each model
results = []
for model_key in OPUS_MODELS + ["fable5"]:
    label = LABELS[model_key]
    idx = [i for i, e in enumerate(all_entries) if e["model"] == model_key]
    model_emb = all_emb[idx]
    centroid = model_emb.mean(axis=0)

    # Cosine similarity to centroid
    norms = np.linalg.norm(model_emb, axis=1)
    centroid_norm = np.linalg.norm(centroid)
    cosine_sims = model_emb @ centroid / (norms * centroid_norm)

    best_local = np.argmax(cosine_sims)
    best_global = idx[best_local]
    entry = all_entries[best_global]

    results.append({
        "model": label,
        "batch": entry["batch"],
        "entry_num": entry["entry_num"],
        "words": entry["words"],
        "cosine_to_centroid": float(cosine_sims[best_local]),
        "text": entry["text"],
        # [A1 V1 copy: output only] the next two candidates, to show the margin
        "runners_up": [{"batch": all_entries[idx[j]]["batch"], "entry_num": all_entries[idx[j]]["entry_num"],
                        "words": all_entries[idx[j]]["words"], "cosine_to_centroid": float(cosine_sims[j])}
                       for j in np.argsort(-cosine_sims)[1:3]],
    })

    print(f"\n{'='*80}")
    print(f"  {label}: Batch {entry['batch']}, Entry {entry['entry_num']}  "
          f"({entry['words']} words, cosine={cosine_sims[best_local]:.4f})")
    print(f"{'='*80}")
    print(entry["text"][:500] + ("..." if len(entry["text"]) > 500 else ""))

# Save results
out = os.path.join(BASE, "out", f"centroid_nearest_entries_{VERSION}.json")  # [A1 V1 copy: output path]
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n\nSaved full results to {out}")
