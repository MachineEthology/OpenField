"""Shared paths and loaders for the reproduction scripts. Everything is read from this repository."""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
OPUS = MODELS[:4]
LABEL = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}


def load_journals():
    """Journal entries exactly as the paper analysed them (data/journals/Batch_N).

    One file per model per batch, picked by model key in the file name, as the paper's loaders did.
    Note: Batch_8's Opus 4.6 file is a byte copy of Batch 6 (see docs/CORPUS.md); it is kept as analysed.
    Returns dicts with model, batch, pos (1-based notebook position, silences counted), text, silence.
    """
    rows = []
    for b in range(1, 16):
        folder = os.path.join(DATA, "journals", f"Batch_{b}")
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


def load_probes():
    """The 975 cold probes: Opus negation 100 (one file), Opus affirmative 100 (75 June + 25 May files),
    Fable 5 negation 100 and affirmative 75. Each row keeps its source file and index."""
    rows = []
    for m in MODELS:
        if m == "fable5":
            sets = [("negation", "texture_probe_fable5_log.json"),
                    ("affirmative", "texture_probe_affirmative_fable5_log.json")]
        else:
            sets = [("negation", f"texture_probe_{m}_log.json"),
                    ("affirmative", f"texture_probe_affirmative_{m}_log.json"),
                    ("affirmative", f"texture_probe_affirmative_{m}_log_mai.json")]
        for cond, fn in sets:
            with open(os.path.join(DATA, "probes", fn), encoding="utf-8") as fh:
                for i, e in enumerate(json.load(fh)):
                    rows.append({"model": m, "condition": cond, "file": fn, "index": i,
                                 "text": e["response"], "date": e.get("date"), "record": e})
    return rows


def load_json(*parts):
    with open(os.path.join(DATA, *parts), encoding="utf-8") as fh:
        return json.load(fh)


# --- public release (3 Nov 2026): the notebooks and probes are in the private companion repository.
_load_journals_all, _load_probes_all = load_journals, load_probes


def _need(sub, what):
    if not os.path.isdir(os.path.join(DATA, sub)):
        raise SystemExit(f"This script needs {what}, which are kept in the private companion repository "
                         f"(opened on request: open an issue on this repository; see README.md). Copy its data/{sub}/ here to run it.")


def load_journals():
    _need("journals", "the notebooks")
    return _load_journals_all()


def load_probes():
    _need("probes", "the cold-probe responses")
    return _load_probes_all()
