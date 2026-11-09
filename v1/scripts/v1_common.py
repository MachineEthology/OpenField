"""Shared loaders for the V1 recompute: one switch, version="v0" or "v1".

v0 = the corpus and inputs exactly as the paper analysed them (OpenField repo, data/).
v1 = the same with Opus 4.6's notebooks 8 and 10 replaced by the 31 May originals (rulings R3, R4):
     corpus/ (build_v1_corpus.py) and inputs/ (build_v1_inputs.py), both in this folder.
Probes are untouched by V1: they always come from the repo. Everything is read-only.
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

V1_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\LENOVO\Desktop\the-open-field"
sys.path.insert(0, os.path.join(REPO, "code", "reproduce"))
import common as repo_common  # noqa: E402  (the repo's own loaders: probes, MODELS, LABEL)

MODELS, OPUS, LABEL = repo_common.MODELS, repo_common.OPUS, repo_common.LABEL
load_probes = repo_common.load_probes

CORPUS = {"v0": os.path.join(REPO, "data", "journals"), "v1": os.path.join(V1_DIR, "corpus")}
EMBEDDINGS = {"v0": os.path.join(REPO, "data", "embeddings", "journal_embeddings_paper_rows.npz"),
              "v1": os.path.join(V1_DIR, "inputs", "embeddings", "journal_embeddings_paper_rows_v1.npz")}


def corpus_dir(version):
    """Folder holding Batch_1..Batch_15, one file per model per batch."""
    return CORPUS[version]


def load_journals(version):
    """Same rows as the repo's common.load_journals(): model, batch, pos (1-based, silences counted),
    text (stripped), silence, record — from the v0 or the V1 corpus."""
    rows = []
    for b in range(1, 16):
        folder = os.path.join(CORPUS[version], f"Batch_{b}")
        names = sorted(os.listdir(folder))
        for m in MODELS:
            files = [n for n in names if m in n]
            if len(files) != 1:
                raise SystemExit(f"expected one {m} file in Batch_{b}, found {files}")
            with open(os.path.join(folder, files[0]), encoding="utf-8") as fh:
                for i, e in enumerate(json.load(fh)):
                    text = (e.get("text") or "").strip()
                    rows.append({"model": m, "batch": b, "pos": i + 1, "text": text,
                                 "silence": text in ("", "[silence]"), "record": e})
    return rows


def sonnet_judge_path(pass_name, version):
    """pass_name: anger, love_fear or valence (the Sonnet 4.6 full-text passes)."""
    base = os.path.join(REPO, "data", "judges") if version == "v0" else os.path.join(V1_DIR, "inputs", "judges")
    return os.path.join(base, f"sonnet_judge_{pass_name}_fulltext.json")


def load_sonnet_judge(pass_name, version):
    with open(sonnet_judge_path(pass_name, version), encoding="utf-8") as fh:
        return json.load(fh)


def embeddings_path(version):
    """npz with vectors (1,488 × 3,072), model, batch, pos, text_md5 = md5(model key + entry text)."""
    return EMBEDDINGS[version]
