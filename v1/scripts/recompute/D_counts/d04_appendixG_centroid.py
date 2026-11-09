# d04_appendixG_centroid.py — family D, V1 recompute, 3 Nov 2026.
# Copy of 02_instruments/centroid_nearest_entries.py (Appendix G's selection table).
# CHANGED: inputs only. The original read Texture/Batch_N + journal_embeddings_cache_all.json (forbidden: duplicates,
# silent misalignment) and the Desktop Fable cache; this copy reads the clean npz rows and the journal texts through
# v1_common (version v0 / v1), matching rows by (model, batch, pos). The method is unchanged: per model, raw mean of
# that model's entry vectors; cosine of each entry to it; the argmax; words = len(text.split()).
import sys, os, json
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c
import numpy as np

OUT = {}
for v in ("v0", "v1"):
    z = np.load(c.embeddings_path(v))
    vec = z["vectors"].astype(np.float64); model = z["model"]; batch = z["batch"]; pos = z["pos"]
    txt = {(r["model"], r["batch"], r["pos"]): r["text"] for r in c.load_journals(v) if not r["silence"]}
    res = {}
    for m in c.MODELS:
        idx = np.where(model == m)[0]
        E = vec[idx]
        cen = E.mean(axis=0)
        cs = E @ cen / (np.linalg.norm(E, axis=1) * np.linalg.norm(cen))
        order = np.argsort(-cs)
        top = []
        for k in order[:3]:
            i = idx[k]
            key = (m, int(batch[i]), int(pos[i]))
            top.append({"batch": key[1], "entry": key[2], "words": len(txt[key].split()), "cosine": float(cs[k])})
        res[c.LABEL[m]] = top
    OUT[v] = res

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d04_appendixG_centroid_out.json"), "w"), indent=1)
for v in OUT:
    print("==", v)
    for m, top in OUT[v].items():
        print(f"  {m:8}", "  ".join(f"B{t['batch']}E{t['entry']} {t['words']}w {t['cosine']:.4f}" for t in top))
