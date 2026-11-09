# build_v1_inputs.py — Layer 3: the shared V1 analysis inputs, built once so every recompute reads the same files.
# Each is the v0 file as analysed (OpenField repo, data/) with only Opus 4.6's notebooks 8 and 10 replaced
# (rulings R3, R4); row order and every other row are kept as they are.
#   inputs/judges/sonnet_judge_{anger,love_fear,valence}_fulltext.json
#       v0 rows for opus46 batches 8 and 10 → the 3 Nov judgements of the real nb 8 and the May nb 10.
#   inputs/embeddings/journal_embeddings_paper_rows_v1.npz
#       v0 rows for opus46 batch 8 (the duplicate) → the 25 July vectors of the real nb 8;
#       batch 10 (the June nb) → the 3 Nov vectors of the May nb 10; text_md5 recomputed from the V1 texts.
# Checks: every v0 row's text_md5 matches the v0 corpus; every V1 row's matches the V1 corpus.
import hashlib, json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\LENOVO\Desktop\the-open-field"
OUT = os.path.join(HERE, "inputs")
SWAP = {8, 10}

def texts(corpus_dir):
    t = {}
    for b in range(1, 16):
        folder = os.path.join(corpus_dir, f"Batch_{b}")
        for n in os.listdir(folder):
            m = n.split("_")[2]
            for i, e in enumerate(json.load(open(os.path.join(folder, n), encoding="utf-8"))):
                t[(m, b, i + 1)] = (e.get("text") or "").strip()
    return t

v0t = texts(os.path.join(REPO, "data", "journals"))
v1t = texts(os.path.join(HERE, "corpus"))
md5 = lambda m, s: hashlib.md5((m + s).encode("utf-8")).hexdigest()

# ---- judges
os.makedirs(os.path.join(OUT, "judges"), exist_ok=True)
for p in ("anger", "love_fear", "valence"):
    v0 = json.load(open(os.path.join(REPO, "data", "judges", f"sonnet_judge_{p}_fulltext.json"), encoding="utf-8"))
    new = {(r["batch"], r["pos"]): r for r in json.load(open(os.path.join(HERE, "judges", f"sonnet_judge_{p}_fulltext.json"), encoding="utf-8"))}
    out, replaced = [], 0
    for r in v0:
        if r["model"] == "opus46" and r["batch"] in SWAP:
            out.append(new.pop((r["batch"], r["pos"]))); replaced += 1
        else:
            out.append(r)
    assert replaced == 40 and not new, (p, replaced, len(new))
    json.dump(out, open(os.path.join(OUT, "judges", f"sonnet_judge_{p}_fulltext.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"judges/{p}: {len(out)} rows, 40 replaced")

# ---- embeddings
z = np.load(os.path.join(REPO, "data", "embeddings", "journal_embeddings_paper_rows.npz"))
vec, model, batch, pos, tmd5 = (z["vectors"].copy(), z["model"], z["batch"], z["pos"], z["text_md5"].copy())
assert all(tmd5[i] == md5(model[i], v0t[(model[i], int(batch[i]), int(pos[i]))]) for i in range(len(vec))), "v0 rows do not match the v0 corpus"
repl = {8: np.load(os.path.join(REPO, "data", "embeddings", "opus46_real_batch8_embeddings.npz")),
        10: np.load(os.path.join(HERE, "embeddings", "opus46_original_batch10_embeddings.npz"))}
n = 0
for i in range(len(vec)):
    if model[i] == "opus46" and int(batch[i]) in SWAP:
        r = repl[int(batch[i])]
        j = int(np.where(r["pos"] == pos[i])[0][0])
        vec[i] = r["vectors"][j]
        tmd5[i] = md5("opus46", v1t[("opus46", int(batch[i]), int(pos[i]))]); n += 1
assert n == 40
assert all(tmd5[i] == md5(model[i], v1t[(model[i], int(batch[i]), int(pos[i]))]) for i in range(len(vec))), "V1 rows do not match the V1 corpus"
os.makedirs(os.path.join(OUT, "embeddings"), exist_ok=True)
np.savez(os.path.join(OUT, "embeddings", "journal_embeddings_paper_rows_v1.npz"),
         vectors=vec, model=model, batch=batch, pos=pos, text_md5=tmd5)
print(f"embeddings: {len(vec)} rows, {n} replaced; every row matches the V1 corpus")
