# r9_check.py — ruling R9: before the swap, check that v0's two models are still served.
# 1. claude-sonnet-4-6 (the full-text judge): one tiny call.
# 2. text-embedding-3-large: re-embed one entry that already has a v0-era vector
#    (real 4.6 Batch 8, entry 1, embedded 25 July 2026) and compare the two.
# Keys come from the environment (Pauline's); they are never printed or written.
import json, os, sys, winreg
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from anthropic import Anthropic
from openai import OpenAI

REPO = r"C:\Users\LENOVO\Desktop\the-open-field"

def user_env(name):
    v = os.environ.get(name)
    if v:
        return v
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        return winreg.QueryValueEx(k, name)[0]

a = Anthropic(api_key=user_env("ANTHROPIC_API_KEY"), timeout=120.0)
r = a.messages.create(model="claude-sonnet-4-6", max_tokens=16,
                      messages=[{"role": "user", "content": "Reply with the single word: ready"}])
print(f"Sonnet 4.6: served as {r.model}; stop {r.stop_reason}; "
      f"tokens {r.usage.input_tokens}/{r.usage.output_tokens}; said {r.content[0].text.strip()!r}")

o = OpenAI(api_key=user_env("OPENAI_API_KEY"), timeout=120.0)
nb = json.load(open(os.path.join(REPO, "data", "journals_supplementary", "opus46_real_batch8_2026-05-31",
                                 "schrodingers_box_opus46_Batch8.json"), encoding="utf-8"))
z = np.load(os.path.join(REPO, "data", "embeddings", "opus46_real_batch8_embeddings.npz"))
i = int(np.where(z["pos"] == 1)[0][0])
e = o.embeddings.create(model="text-embedding-3-large", input=[nb[0]["text"]])
v = np.array(e.data[0].embedding, dtype=np.float64); old = z["vectors"][i].astype(np.float64)
cos = v @ old / (np.linalg.norm(v) * np.linalg.norm(old))
print(f"text-embedding-3-large: served as {e.model}; {len(v)} dimensions; {e.usage.total_tokens} tokens; "
      f"cosine to the 25 July vector {cos:.6f}")
