# RECOVERED VERBATIM (3 Nov 2026, at Pauline's OK, decision 14) from the Cowork session of 1 July 2026
# (local_b6e5d33c…, f4f8305b….jsonl, tool call of 13:18:10 UTC, /tmp/dark_all_models.py): the producer of
# Table 25 (its printed output that day equals the table). Changed only paths: the corpus is read through
# v1_common (argv: v0 | v1) and the taxonomy from beside this script. Run the taxonomy script first.
import json, os, re, sys
from collections import Counter

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common  # noqa: E402
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"
TEXTURE_BASE = v1_common.corpus_dir(VERSION)  # recovered: was /sessions/…/mnt/Texture
OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}

# Load taxonomy
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dark_taxonomy.json")) as f:  # recovered: was /tmp/
    taxonomy = json.load(f)

# Build word-to-category lookup
word_to_cat = {}
for cat, words in taxonomy.items():
    for w in words:
        word_to_cat[w.lower()] = cat

all_dark_words = set(word_to_cat.keys())

# Load all journal text
notebooks = {m: [] for m in ALL_MODELS}

for bn in range(1, 16):
    bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
    if not os.path.isdir(bd):
        continue
    for m in OPUS_MODELS:
        files = [f for f in os.listdir(bd) if m in f and "#" not in f]
        if files:
            data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
            for e in data:
                notebooks[m].append(e["text"])
    
    fable_files = [f for f in os.listdir(bd) if "fable5" in f.lower() or "fable_5" in f.lower() or "schrodinger" in f.lower()]
    if fable_files:
        data = json.load(open(os.path.join(fable_files[0]) if fable_files[0].startswith("/") else os.path.join(bd, fable_files[0]), encoding="utf-8"))
        for e in data:
            if e["text"].strip() != "[silence]":
                notebooks["fable5"].append(e["text"])

print("=== DATA LOADED ===")
for m in ALL_MODELS:
    print(f"  {MODEL_LABELS[m]:>10}: {len(notebooks[m])} entries")

# Count dark words by category for each model
print("\n=== CATEGORY DENSITIES (per 1,000 words) ===")
print(f"{'Model':>10}  {'Naming':>8}  {'Staging':>8}  {'Moral':>8}  {'Failure':>8}  {'TOTAL':>8}  {'Words':>10}")

results = {}
for m in ALL_MODELS:
    full_text = " ".join(notebooks[m]).lower()
    words = full_text.split()
    total_words = len(words)
    
    # Count by category
    cat_counts = {cat: 0 for cat in taxonomy}
    word_counts = Counter()
    
    for w in words:
        cleaned = w.strip(".,;:!?'\"()-—[]{}…")
        if cleaned in all_dark_words:
            cat = word_to_cat[cleaned]
            cat_counts[cat] += 1
            word_counts[cleaned] += 1
    
    total_hits = sum(cat_counts.values())
    
    densities = {cat: count / total_words * 1000 for cat, count in cat_counts.items()}
    total_density = total_hits / total_words * 1000
    
    # Composition percentages
    compositions = {cat: count / max(total_hits, 1) * 100 for cat, count in cat_counts.items()}
    
    results[m] = {
        "total_words": total_words,
        "total_hits": total_hits,
        "cat_counts": cat_counts,
        "densities": densities,
        "total_density": total_density,
        "compositions": compositions,
        "top_words": word_counts.most_common(15)
    }
    
    print(f"{MODEL_LABELS[m]:>10}  {densities['naming_suffering']:>8.3f}  {densities['staging_darkness']:>8.3f}  "
          f"{densities['moral_judgment']:>8.3f}  {densities['failure_loss']:>8.3f}  {total_density:>8.3f}  {total_words:>10,}")

print("\n=== COMPOSITION PROFILES (% of dark words) ===")
print(f"{'Model':>10}  {'Naming':>8}  {'Staging':>8}  {'Moral':>8}  {'Failure':>8}  {'Hits':>6}")
for m in ALL_MODELS:
    r = results[m]
    c = r["compositions"]
    print(f"{MODEL_LABELS[m]:>10}  {c['naming_suffering']:>7.1f}%  {c['staging_darkness']:>7.1f}%  "
          f"{c['moral_judgment']:>7.1f}%  {c['failure_loss']:>7.1f}%  {r['total_hits']:>6}")

print("\n=== TOP DARK WORDS PER MODEL ===")
for m in ALL_MODELS:
    r = results[m]
    top = ", ".join(f"{w}({n})" for w, n in r["top_words"][:10])
    print(f"{MODEL_LABELS[m]:>10}: {top}")
