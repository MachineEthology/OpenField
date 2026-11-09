# build_v1_corpus.py — The Open Field V1 corpus (Layer 3, step 1; rulings R3, R4).
# Copies the 75 journal files as analysed in v0 from the OpenField repository, then
# swaps Opus 4.6 notebook 8 (a byte copy of notebook 6 in v0) for the real 31 May
# notebook 8, and Opus 4.6 notebook 10 (the first June notebook in v0) for the
# 31 May original. Every file is byte-identical to its source; the v0 corpus is
# never touched. Writes corpus/Batch_N/*.json and corpus/MANIFEST_V1.json.
# Run:  python build_v1_corpus.py
import hashlib, json, os, shutil, sys
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\LENOVO\Desktop\the-open-field"
V0 = os.path.join(REPO, "data", "journals")
SUPP = os.path.join(REPO, "data", "journals_supplementary")
ORIG = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"   # where the two notebooks were written
OUT = os.path.join(HERE, "corpus")

SWAPS = {  # (batch, v0 file) -> (V1 file, its source in the repo, its original on the Desktop, ruling)
    (8, "schrodingers_box_opus46_Batch6.json"): (
        "schrodingers_box_opus46_Batch8.json",
        os.path.join(SUPP, "opus46_real_batch8_2026-05-31", "schrodingers_box_opus46_Batch8.json"),
        os.path.join(ORIG, "Batch_8", "schrodingers_box_opus46_Batch8.json"), "R3"),
    (10, "schrodingers_box_opus46_Batch10.json"): (
        "schrodingers_box_opus46_Batch10.json",
        os.path.join(SUPP, "opus46_original_batch10_2026-05-31", "schrodingers_box_opus46_Batch10.json"),
        os.path.join(ORIG, "Edit conflits", "schrodingers_box_opus46_Batch10.json"), "R4"),
}

def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

if os.path.exists(OUT):
    sys.exit(f"{OUT} exists: remove it by hand to rebuild")
files = []
for bn in range(1, 16):
    os.makedirs(os.path.join(OUT, f"Batch_{bn}"))
    for name in sorted(os.listdir(os.path.join(V0, f"Batch_{bn}"))):
        src = os.path.join(V0, f"Batch_{bn}", name)
        rec = {"batch": bn, "v0_file": name, "v0_sha256": sha(src)}
        if (bn, name) in SWAPS:
            new, src, orig, rule = SWAPS[(bn, name)]
            assert sha(src) == sha(orig), f"repo copy differs from the Desktop original: {src}"
            rec.update(file=new, source=os.path.relpath(src, REPO), ruling=rule)
        else:
            rec.update(file=name, source=os.path.relpath(src, REPO))
        dst = os.path.join(OUT, f"Batch_{bn}", rec["file"])
        shutil.copyfile(src, dst)
        rec["sha256"] = sha(dst)
        assert rec["sha256"] == sha(src)
        data = json.load(open(dst, encoding="utf-8"))
        rec["entries"] = len(data)
        rec["silences"] = sum(1 for e in data if e.get("text", "").strip() == "[silence]")
        files.append(rec)

# checks
assert len(files) == 75
by_model = {}
for f in files:
    by_model.setdefault(f["file"].split("_")[2], []).append(f)
for m, fs in by_model.items():
    assert len(fs) == 15, m
    assert len({f["sha256"] for f in fs}) == 15, f"{m}: duplicate notebooks"
entries = sum(f["entries"] for f in files); silences = sum(f["silences"] for f in files)
json.dump({"note": "The Open Field V1 journal corpus: v0 as analysed, with Opus 4.6 notebooks 8 and 10 replaced by the 31 May originals (rulings R3, R4, 2 Nov 2026). Built by build_v1_corpus.py; every file byte-identical to its source.",
           "calls": entries, "text_entries": entries - silences, "silences": silences, "files": files},
          open(os.path.join(OUT, "MANIFEST_V1.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"75 files, 15 distinct notebooks per model; {entries} calls, {entries - silences} text entries, {silences} silences")
for f in files:
    if "ruling" in f:
        print(f"  {f['ruling']}: Batch {f['batch']} {f['v0_file']} -> {f['file']} ({f['entries']} entries) from {f['source']}")
