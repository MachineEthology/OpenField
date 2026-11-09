# embed_new_entries.py — Layer 3, step 2 (rulings R4, R9): text-embedding-3-large vectors for
# the 20 entries of Opus 4.6's original notebook 10 (31 May), as v0 embedded journals: one vector
# per whole entry, the raw entry text, no cleaning or truncation, default 3,072 dimensions.
# (The real notebook 8 already has vectors: OpenField data/embeddings/opus46_real_batch8_embeddings.npz.)
# Output: embeddings/opus46_original_batch10_embeddings.npz (vectors, pos), the same shape as the batch 8 file.
import json, os, sys, winreg
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "embeddings", "opus46_original_batch10_embeddings.npz")
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists")

def user_env(name):
    v = os.environ.get(name)
    if v:
        return v
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        return winreg.QueryValueEx(k, name)[0]

data = json.load(open(os.path.join(HERE, "corpus", "Batch_10", "schrodingers_box_opus46_Batch10.json"), encoding="utf-8"))
rows = [(i + 1, e["text"]) for i, e in enumerate(data) if e["text"].strip() and e["text"].strip() != "[silence]"]
assert len(rows) == 20
resp = OpenAI(api_key=user_env("OPENAI_API_KEY"), timeout=300.0).embeddings.create(
    model="text-embedding-3-large", input=[t for _, t in rows])
vec = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)], dtype=np.float32)
assert vec.shape == (20, 3072)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, vectors=vec, pos=np.array([p for p, _ in rows], dtype=np.int64))
print(f"20 vectors, 3,072-d, {resp.usage.total_tokens:,} tokens (~${resp.usage.total_tokens * 0.13 / 1e6:.4f}); saved {OUT}")
