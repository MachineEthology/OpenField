"""A2 (trajectories and carry-over): one loader for the copied producers.

Written 3 Nov 2026 by Claude Opus 5.5 (Code), helper A2, for the V1 recompute.
Everything is read through ..\\..\\v1_common.py (version switch "v0" / "v1"); nothing outside
this folder is written.

The paper's June scripts loaded:
  * Opus: Texture\\Batch_N files (every entry, file order) + journal_embeddings_cache_all.json,
    rows in the order  batch 1..15 -> model opus45..opus48 -> entry 1..20;
  * Fable 5: Desktop\\Claude\\Journaling_data\\Batch_N files, text entries only ("[silence]" skipped,
    entry_num = position in the file, silences counted) + fable5_embeddings_cache.json.
This module rebuilds exactly those lists from v1_common.load_journals(version) and the clean npz
of v1_common.embeddings_path(version) (never journal_embeddings_cache_all.json), matched by
(model, batch, pos). Vectors are returned as float64, un-normalised, as the caches gave them.
"""
import os
import sys

import numpy as np

V1_DIR = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1"
sys.path.insert(0, V1_DIR)
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPUS = ["opus45", "opus46", "opus47", "opus48"]
ALL = OPUS + ["fable5"]


def version_from_argv(default="v0"):
    for a in sys.argv[1:]:
        if a in ("v0", "v1"):
            return a
    return default


def out_dir(version):
    d = os.path.join(HERE, f"out_{version}")
    os.makedirs(d, exist_ok=True)
    return d


def _npz(version):
    z = np.load(c.embeddings_path(version), allow_pickle=True)
    keys = list(zip([str(m) for m in z["model"]], z["batch"].tolist(), z["pos"].tolist()))
    vec = z["vectors"].astype(np.float64)
    return {k: vec[i] for i, k in enumerate(keys)}


def load(version):
    """Returns (opus_entries, opus_emb, fable_entries, fable_emb).

    opus_entries: dicts model, batch, entry_num, text — order batch -> model -> entry, as the
    June scripts built them; opus_emb aligned row for row. Same for Fable 5 (text entries only).
    """
    rows = c.load_journals(version)
    vec = _npz(version)
    by_key = {(r["model"], r["batch"], r["pos"]): r for r in rows}
    opus_entries, opus_v, fable_entries, fable_v = [], [], [], []
    for b in range(1, 16):
        for m in OPUS:
            for r in sorted((r for r in rows if r["model"] == m and r["batch"] == b), key=lambda r: r["pos"]):
                opus_entries.append({"model": m, "batch": b, "entry_num": r["pos"],
                                     "text": r["record"]["text"]})
                opus_v.append(vec[(m, b, r["pos"])])
    for b in range(1, 16):
        for r in sorted((r for r in rows if r["model"] == "fable5" and r["batch"] == b), key=lambda r: r["pos"]):
            if r["record"]["text"].strip() != "[silence]":
                fable_entries.append({"model": "fable5", "batch": b, "entry_num": r["pos"],
                                      "text": r["record"]["text"]})
                fable_v.append(vec[("fable5", b, r["pos"])])
    assert len(opus_entries) == 1199 and len(fable_entries) == 289, (len(opus_entries), len(fable_entries))
    assert len(vec) == 1488
    return opus_entries, np.array(opus_v), fable_entries, np.array(fable_v)
