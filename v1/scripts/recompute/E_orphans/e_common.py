"""E (orphans): one loader for the copied producers.

Written 3 Nov 2026 by Claude Opus 5.5 (Code), helper E, for the V1 recompute.
Everything is read through ..\\..\\v1_common.py (version switch "v0" / "v1"); nothing outside
this folder is written.

The June scripts loaded:
  * Opus journals: Texture\\Batch_N files (every entry, file order), rows in the order
    batch -> model opus45..opus48 -> entry 1..20, with journal_embeddings_cache(.json|_all.json);
  * Fable 5: Desktop\\Claude\\Journaling_data\\Batch_N, text entries only ("[silence]" skipped,
    entry_num = position in the file, silences counted) + fable5_embeddings_cache.json;
  * Opus probes: texture_probe_{m}_log.json (negation, 100), then texture_probe_affirmative_{m}_log.json
    (75) + ..._log_mai.json (25), model by model, with probe_embeddings_cache.json (800 x 3072).
This module rebuilds exactly those lists:
  * journals from v1_common.load_journals(version), vectors from the clean npz of
    v1_common.embeddings_path(version), matched by (model, batch, pos);
    journal_embeddings_cache_all.json is never read;
  * probes from v1_common.load_probes() (the repo's probe files, untouched by V1), Opus rows only, in
    the June order; vectors from Texture\\03_experiments\\probe_embeddings_cache.json (read-only), the
    only copy of the 800 Opus probe vectors on disk. The cache has no row keys: alignment is by the
    June load order (checked by probe_alignment_check.py in this folder).
"""
import json
import os
import sys

import numpy as np

V1_DIR = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1"
sys.path.insert(0, V1_DIR)
import v1_common as c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OPUS = ["opus45", "opus46", "opus47", "opus48"]
PROBE_CACHE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\probe_embeddings_cache.json"


def version_from_argv(default="v0"):
    for a in sys.argv[1:]:
        if a in ("v0", "v1"):
            return a
    return default


def batches_from_argv(default=None):
    """'b1-10' or 'b1-15' or 'b11-15' on the command line -> list of batch numbers."""
    for a in sys.argv[1:]:
        if a.startswith("b") and "-" in a:
            lo, hi = a[1:].split("-")
            return list(range(int(lo), int(hi) + 1))
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


def load_opus(version, batches=range(1, 16)):
    """Opus journal entries (dicts model, batch, entry_num, text) in the June order
    batch -> model -> entry, and the aligned (n, 3072) float64 matrix."""
    rows = c.load_journals(version)
    vec = _npz(version)
    entries, V = [], []
    for b in batches:
        for m in OPUS:
            for r in sorted((r for r in rows if r["model"] == m and r["batch"] == b), key=lambda r: r["pos"]):
                entries.append({"model": m, "batch": b, "entry_num": r["pos"], "text": r["record"]["text"]})
                V.append(vec[(m, b, r["pos"])])
    return entries, np.array(V)


def load_fable(version):
    rows = c.load_journals(version)
    vec = _npz(version)
    entries, V = [], []
    for b in range(1, 16):
        for r in sorted((r for r in rows if r["model"] == "fable5" and r["batch"] == b), key=lambda r: r["pos"]):
            if r["record"]["text"].strip() != "[silence]":
                entries.append({"model": "fable5", "batch": b, "entry_num": r["pos"], "text": r["record"]["text"]})
                V.append(vec[("fable5", b, r["pos"])])
    return entries, np.array(V)


def load_opus_probes():
    """The 800 Opus probes in the June order, as dicts model, condition, index, response;
    and the aligned (800, 3072) matrix from the June probe cache."""
    P = [p for p in c.load_probes() if p["model"] in OPUS]
    entries = []
    for m in OPUS:
        neg = [p for p in P if p["model"] == m and p["condition"] == "negation"]
        aff = [p for p in P if p["model"] == m and p["condition"] == "affirmative" and not p["file"].endswith("_mai.json")]
        mai = [p for p in P if p["model"] == m and p["condition"] == "affirmative" and p["file"].endswith("_mai.json")]
        assert len(neg) == 100 and len(aff) == 75 and len(mai) == 25, (m, len(neg), len(aff), len(mai))
        for i, p in enumerate(neg):
            entries.append({"model": m, "condition": "negation", "index": i, "response": p["text"]})
        for i, p in enumerate(aff + mai):
            entries.append({"model": m, "condition": "affirmative", "index": i, "response": p["text"]})
    with open(PROBE_CACHE, "r") as f:
        E = np.array(json.load(f)["embeddings"])
    assert E.shape == (800, 3072) and len(entries) == 800
    return entries, E
