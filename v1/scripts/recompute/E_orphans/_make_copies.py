"""Builds the V1 copies of the June producers (helper E, 3 Nov 2026): reads each original from
02_instruments read-only, applies only the listed input-path replacements (each asserted to occur),
prepends a header, and writes the copy into this folder."""
import os, sys
SRC = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\02_instruments"
HERE = os.path.dirname(os.path.abspath(__file__))

def make(src, dst, header, reps):
    s = open(os.path.join(SRC, src), encoding="utf-8").read()
    for old, new, n in reps:
        k = s.count(old)
        assert k == n, (src, old, k, n)
        s = s.replace(old, new)
    open(os.path.join(HERE, dst), "w", encoding="utf-8").write(header + s)
    print("wrote", dst)

JOURNAL_LOADER = '''def load_journals():
    entries = []
    for batch_num in range(1, 11):
        batch_dir = os.path.join(BASE, f"Batch_{batch_num}")
        if not os.path.isdir(batch_dir):
            continue
        for m in MODELS:
            files = [f for f in os.listdir(batch_dir) if m in f]
            if files:
                with open(os.path.join(batch_dir, files[0]), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for i, entry in enumerate(data):
                    entries.append({
                        "model": m, "batch": batch_num, "entry_num": i + 1,
                        "text": entry["text"]
                    })
    return entries'''
NEW_JOURNAL_LOADER = '''def load_journals():
    # [E V1] was: Texture\Batch_1..10 files; now e_common (v1_common corpus + clean npz), same order
    global _J_EMB
    entries, _J_EMB = e_common.load_opus(VERSION, BATCHES)
    return entries'''
PROBE_LOADER_START = 'def load_probes():\n    entries = []\n    for m in MODELS:\n        neg_file'

# ---------------- 1. migration_permutation_test.py ----------------
hdr = '''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/migration_permutation_test.py.
Changed: input paths only. Journals + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json); probes via e_common (repo probe files + June probe cache, read-only).
The batch range (June: 1-10, hard-coded) is a command-line switch, default b1-10; version v0|v1.
Usage: python migration_permutation_test_v1.py v0|v1 [b1-10|b1-15]
"""
import e_common
VERSION = e_common.version_from_argv()
BATCHES = e_common.batches_from_argv(list(range(1, 11)))
'''
make("migration_permutation_test.py", "migration_permutation_test_v1.py", hdr, [
    (JOURNAL_LOADER, NEW_JOURNAL_LOADER, 1),
    ('    j_emb = load_cached(os.path.join(BASE, "journal_embeddings_cache.json"))', '    j_emb = _J_EMB  # [E V1]', 1),
    ('    p_emb = load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))', '    p_emb = _P_EMB  # [E V1]', 1),
    ('    p_entries = load_probes()', '    p_entries, _P_EMB = e_common.load_opus_probes()  # [E V1] was load_probes() + probe_embeddings_cache.json', 1),
    ('for batch_num in range(1, 11):', 'for batch_num in BATCHES:  # [E V1] was range(1, 11)', 2),
])

# ---------------- 2. entry1_permutation_test.py ----------------
hdr = hdr.replace("migration_permutation_test", "entry1_permutation_test")
make("entry1_permutation_test.py", "entry1_permutation_test_v1.py", hdr, [
    (JOURNAL_LOADER, NEW_JOURNAL_LOADER, 1),
    ('    j_emb = load_cached(os.path.join(BASE, "journal_embeddings_cache.json"))', '    j_emb = _J_EMB  # [E V1]', 1),
    ('    p_emb = load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))', '    p_emb = _P_EMB  # [E V1]', 1),
    ('    p_entries = load_probes()', '    p_entries, _P_EMB = e_common.load_opus_probes()  # [E V1] was load_probes() + probe_embeddings_cache.json', 1),
    ('for batch_num in range(1, 11):', 'for batch_num in BATCHES:  # [E V1] was range(1, 11)', 2),
])

# ---------------- 3. replication_test.py (imported as a module by decone_pset / domain_preserving_retest) ----
hdr3 = '''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/replication_test.py,
used only as the module that decone_pset_v1.py and domain_preserving_retest_v1.py import (as the June
scripts imported replication_test). Changed: input paths only. load_journals(batch_range) now returns the
entries from e_common (v1_common corpus) and keeps the aligned clean-npz vectors in _J_EMB; load_probes()
returns the repo probes in the June order and keeps the June probe-cache vectors in _P_EMB. The metric
functions are untouched. Its __main__ block (which embeds through the API) is never run here.
Version: v0|v1 from the command line of the importing script.
"""
import e_common
VERSION = e_common.version_from_argv()
_J_EMB = None
_P_EMB = None
'''
REP_LOADER_OLD = '''def load_journals(batch_range):
    entries = []
    for batch_num in batch_range:'''
REP_LOADER_NEW = '''def load_journals(batch_range):
    # [E V1] was: Texture\\Batch_N files; now e_common (v1_common corpus + clean npz), same order
    global _J_EMB
    _entries, _J_EMB = e_common.load_opus(VERSION, batch_range)
    return _entries


def _june_load_journals(batch_range):  # [E V1] the June loader, kept for reference, not called
    entries = []
    for batch_num in batch_range:'''
REP_PROBE_OLD = '''def load_probes():
    entries = []'''
REP_PROBE_NEW = '''def load_probes():
    # [E V1] was: Texture\\texture_probe_* files; now e_common (repo probe files, June order + June cache)
    global _P_EMB
    _entries, _P_EMB = e_common.load_opus_probes()
    return _entries


def _june_load_probes():  # [E V1] the June loader, kept for reference, not called
    entries = []'''
make("replication_test.py", "replication_test_v1.py", hdr3, [
    (REP_LOADER_OLD, REP_LOADER_NEW, 1),
    (REP_PROBE_OLD, REP_PROBE_NEW, 1),
])

# ---------------- 4. decone_pset.py ----------------
hdr4 = '''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/decone_pset.py.
Changed: input paths only. Imports replication_test_v1 (the module copy with e_common loaders) instead of
replication_test; journal vectors = the clean npz (never journal_embeddings_cache_all.json); probe vectors =
the June probe cache. Usage: python decone_pset_v1.py v0|v1
"""
'''
make("decone_pset.py", "decone_pset_v1.py", hdr4, [
    ("import replication_test as rt   # guarded __main__, safe to import",
     "import replication_test_v1 as rt   # [E V1] was replication_test", 1),
    ('all_emb_raw = rt.load_cached(os.path.join(BASE, "journal_embeddings_cache_all.json"))',
     'all_emb_raw = rt._J_EMB  # [E V1] was journal_embeddings_cache_all.json', 1),
    ('p_emb_raw = rt.load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))',
     'p_emb_raw = rt._P_EMB  # [E V1] was BASE/probe_embeddings_cache.json (same file, now in 03_experiments)', 1),
])

# ---------------- 5. domain_preserving_retest.py ----------------
hdr5 = hdr4.replace("decone_pset", "domain_preserving_retest")
make("domain_preserving_retest.py", "domain_preserving_retest_v1.py", hdr5, [
    ("import replication_test as rt\n", "import replication_test_v1 as rt  # [E V1] was replication_test\n", 1),
    ('JE = rt.load_cached(os.path.join(BASE, "journal_embeddings_cache_all.json"))',
     'JE = rt._J_EMB  # [E V1] was journal_embeddings_cache_all.json', 1),
    ('PE = rt.load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))',
     'PE = rt._P_EMB  # [E V1] was BASE/probe_embeddings_cache.json (same file, now in 03_experiments)', 1),
])

# ---------------- 6. embedding_new_analyses.py (analysis 3 only: N2, the original lineage-divergence test) ----
hdr6 = '''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/embedding_new_analyses.py,
run for ANALYSIS 3 only (cross-lineage distance over time: the N2 'DIVERGE' test, Delta +0.0036, p = 0.55).
Changed: input paths only. Journals + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json), restricted to the batch range given on the command line (in June the
script looped over Batch_1..15 but only Batch_1..10 existed: default b1-10); probes via e_common (June probe
cache). PNGs go to out_<version>/. One added line dumps the analysis-3 numbers at full precision (it reads the
variables computed above it and changes nothing). Analyses 1, 2, 4, 5 are not run (5 calls the API).
Usage: python embedding_new_analyses_v1.py 3 v0|v1 [b1-10|b1-15]
"""
import e_common
VERSION = e_common.version_from_argv()
BATCHES = e_common.batches_from_argv(list(range(1, 11)))
'''
NEA_LOADER_OLD = '''def load_journals():
    entries = []
    for batch_num in range(1, 16):'''
NEA_LOADER_NEW = '''def load_journals():
    # [E V1] was: Texture\\Batch_N files; now e_common (v1_common corpus + clean npz), same order
    global _J_EMB
    _entries, _J_EMB = e_common.load_opus(VERSION, BATCHES)
    return _entries


def _june_load_journals():  # [E V1] the June loader, kept for reference, not called
    entries = []
    for batch_num in range(1, 16):'''
make("embedding_new_analyses.py", "embedding_new_analyses_v1.py", hdr6, [
    (NEA_LOADER_OLD, NEA_LOADER_NEW, 1),
    ('j_emb = load_cached(os.path.join(BASE, "journal_embeddings_cache_all.json"))',
     'j_emb = _J_EMB  # [E V1] was journal_embeddings_cache_all.json', 1),
    ('p_entries = load_probes()\np_emb = load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))',
     'p_entries, p_emb = e_common.load_opus_probes()  # [E V1] was load_probes() + BASE/probe_embeddings_cache.json', 1),
    ('    RUN = {1, 2, 3, 4, 5}', '    RUN = {1, 2, 3, 4, 5}\nassert RUN == {3}, "[E V1] only analysis 3 is run here"', 1),
    ('plt.savefig(os.path.join(BASE, "lineage_distance.png")',
     'plt.savefig(os.path.join(e_common.out_dir(VERSION), f"lineage_distance_b{BATCHES[0]}-{BATCHES[-1]}.png")', 1),
    ('    print(f"  Lineage divergence test: {trend}")\n',
     '    print(f"  Lineage divergence test: {trend}")\n'
     '    json.dump({"batches": BATCHES, "d_e1": lineage_dist[0][1], "d_e20": lineage_dist[-1][1], "delta": delta_lineage, "obs_delta_perm_design": obs_delta, "perm_count": perm_count, "p": p_val_lineage}, open(os.path.join(e_common.out_dir(VERSION), f"n2_original_b{BATCHES[0]}-{BATCHES[-1]}.json"), "w"), indent=1)  # [E V1] added: full-precision dump\n', 1),
])

# ---------------- 7. embedding_keystone_geometry.py (analysis 5 only: entry-1 aff-neg lean, D-O16) ----
hdr7 = '''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/embedding_keystone_geometry.py,
run for ANALYSIS 5 only (entry-1 nearest probe-condition: the '4.7 leans +0.049, 4.8 -0.001' numbers).
Changed: input paths only. Journals + vectors via e_common (v1_common corpus + clean npz, batches 1-15, never
journal_embeddings_cache_all.json); probes via e_common (repo probe files in the June order + June probe cache).
PNG to out_<version>/. One added line dumps the analysis-5 numbers at full precision (reads the variables
computed above it; changes nothing). Analyses 1-4 and 6 are run by A2 (keystone_geometry_v1.py) and not here.
Usage: python keystone_lean_v1.py 5 v0|v1
"""
import e_common
VERSION = e_common.version_from_argv()
'''
make("embedding_keystone_geometry.py", "keystone_lean_v1.py", hdr7, [
    ('J = load_journals()\nJE = load_cached(os.path.join(BASE, "journal_embeddings_cache_all.json"))',
     'J, JE = e_common.load_opus(VERSION, range(1, 16))  # [E V1] was load_journals() + journal_embeddings_cache_all.json', 1),
    ('P = load_probes()\nPE = load_cached(os.path.join(BASE, "probe_embeddings_cache.json"))',
     'P, PE = e_common.load_opus_probes()  # [E V1] was load_probes() + BASE/probe_embeddings_cache.json', 1),
    ('RUN = set(int(x) for x in sys.argv[1:] if x.isdigit()) or {1, 2, 3, 4, 5, 6}',
     'RUN = set(int(x) for x in sys.argv[1:] if x.isdigit()) or {1, 2, 3, 4, 5, 6}\nassert RUN == {5}, "[E V1] only analysis 5 is run here"', 1),
    ('plt.savefig(os.path.join(BASE, "entry1_probe_condition.png")',
     'plt.savefig(os.path.join(e_common.out_dir(VERSION), "entry1_probe_condition.png")', 1),
    ('    print("  Saved: entry1_probe_condition.png")\n',
     '    print("  Saved: entry1_probe_condition.png")\n'
     '    json.dump({m: {"entry1_lean_mean": float(e1_store[m].mean()), "pct_nearer_aff": float(100 * np.mean(e1_store[m] > 0)), "n": int(len(e1_store[m])), "arc_1_20": [float(x) for x in arc[m]]} for m in MODELS}, open(os.path.join(e_common.out_dir(VERSION), "keystone_lean.json"), "w"), indent=1)  # [E V1] added: full-precision dump\n', 1),
])

# ---------------- 8. diminutive_and_distrust.py (part A only) and self_referential_diminutive.py (D-O18) ----
hdr8 = '''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/diminutive_and_distrust.py,
part (A) only, the diminutive curve (CORE / WIDE markers; 'wide 0.72 -> 0.42' for 4.8).
Changed: input paths only: BASE -> v1_common.corpus_dir(version) (the Batch_1..15 folders; v0 = repo
data/journals as analysed, v1 = V1/corpus). Part (B) reads the interiority skeptical logs (unchanged by V1,
not a journal measure) and is not run. One added block dumps the per-bin means at full precision.
Usage: python diminutive_curve_v1.py v0|v1
"""
import e_common
VERSION = e_common.version_from_argv()
'''
make("diminutive_and_distrust.py", "diminutive_curve_v1.py", hdr8, [
    (r'BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"',
     r'BASE = e_common.c.corpus_dir(VERSION)  # [E V1] was r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"', 1),
    (r'''        print("  (positive delta = marker FADES with accumulation = stands up)\n")''' + "\n",
     r'''        print("  (positive delta = marker FADES with accumulation = stands up)\n")''' + "\n"
     '''        _DUMP[series] = means  # [E V1] added\n''', 1),
    (r'''if __name__ == "__main__":
    diminutive_curve()
    print("\n" + "=" * 70 + "\n")
    distrust_grammar()''',
     '''_DUMP = {}  # [E V1] added: full-precision dump of the per-bin means

if __name__ == "__main__":
    diminutive_curve()
    json.dump({"files": len(glob.glob(os.path.join(BASE, "Batch_*", "schrodingers_box_opus4*_Batch*.json"))), **_DUMP},
              open(os.path.join(e_common.out_dir(VERSION), "diminutive_curve.json"), "w"), indent=1)  # [E V1] added
    # [E V1] part (B), distrust_grammar(), not run: it reads the interiority logs, not the journals''', 1),
])
hdr9 = hdr8.replace("diminutive_and_distrust.py,\npart (A) only, the diminutive curve (CORE / WIDE markers; 'wide 0.72 -> 0.42' for 4.8).",
                    "self_referential_diminutive.py\n(the tightened, self-bound diminutive curve).").replace(
                    " Part (B) reads the interiority skeptical logs (unchanged by V1,\nnot a journal measure) and is not run.", "").replace(
                    "diminutive_curve_v1.py", "self_referential_diminutive_v1.py")
make("self_referential_diminutive.py", "self_referential_diminutive_v1.py", hdr9, [
    (r'BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"',
     r'BASE = e_common.c.corpus_dir(VERSION)  # [E V1] was r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"', 1),
    ('''        f"{means[m]['entries 1-5'] - means[m]['entries 16-20']:>+8.3f}"
        for m in ['opus45','opus46','opus47','opus48']))''',
     '''        f"{means[m]['entries 1-5'] - means[m]['entries 16-20']:>+8.3f}"
        for m in ['opus45','opus46','opus47','opus48']))
    json.dump({"files": nf, "raw_hits": raw, "means": means}, open(os.path.join(e_common.out_dir(VERSION), "self_referential_diminutive.json"), "w"), indent=1)  # [E V1] added''', 1),
])

# ---------------- 9. ordinal_robustness_check.py (D-O20) ----
hdr10 = r'''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/ordinal_robustness_check.py.
Changed: input paths only. The two full-text Sonnet 4.6 passes come from v1_common.load_sonnet_judge(p, version)
(v0 = repo data/judges, byte-identical to Texture/03_experiments; v1 = V1/inputs/judges). The truncated
(1,500-character) love/fear pass, which the script reads deliberately as its 'non-fulltext' pairs, is the archived
file Texture/03_experiments/_superseded_judges/sonnet_judge_love_fear.json (README there: the script's 'old'
path is to be repointed there). That pass was never re-run on V1's 4.6 notebooks 8 and 10, so in v1 mode its
4.6 rows for batches 8 and 10 still judge v0's notebooks: the switch old46=keep (default) leaves them, old46=drop
removes them (4.6 n = 258 on the three truncated-pass pairs). One added block dumps the 35 |rho - r| values.
Usage: python ordinal_robustness_check_v1.py v0|v1 [old46=keep|old46=drop]
"""
import e_common
VERSION = e_common.version_from_argv()
OLD46 = "drop" if "old46=drop" in __import__("sys").argv else "keep"
_OLD = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\_superseded_judges\sonnet_judge_love_fear.json"
'''
make("ordinal_robustness_check.py", "ordinal_robustness_check_v1.py", hdr10, [
    ('love_fear_ft = load_json("sonnet_judge_love_fear_fulltext.json")',
     'love_fear_ft = [r for r in e_common.c.load_sonnet_judge("love_fear", VERSION) if "error" not in r]  # [E V1] was load_json(BASE/...fulltext.json)', 1),
    ('love_fear_old = load_json("sonnet_judge_love_fear.json")',
     'love_fear_old = [r for r in json.load(open(_OLD, encoding="utf-8")) if "error" not in r]  # [E V1] was load_json(BASE/sonnet_judge_love_fear.json)\n'
     'if VERSION == "v1" and OLD46 == "drop":  # [E V1] switch, see header\n'
     '    love_fear_old = [r for r in love_fear_old if not (r["model"] == "opus46" and r["batch"] in (8, 10))]', 1),
    ('anger_ft = load_json("sonnet_judge_anger_fulltext.json")',
     'anger_ft = [r for r in e_common.c.load_sonnet_judge("anger", VERSION) if "error" not in r]  # [E V1] was load_json(BASE/...fulltext.json)', 1),
    ('print(f"  Median |Spearman − Pearson|:     {np.median(all_deltas):.4f}")',
     'print(f"  Median |Spearman − Pearson|:     {np.median(all_deltas):.4f}")\n'
     '_labels = [(pl, m) for pl, *_ in pairs for m in MODELS]  # [E V1] added: full-precision dump\n'
     'json.dump({"version": VERSION, "old46": OLD46, "n": len(all_deltas), "max": float(max(all_deltas)), "mean": float(np.mean(all_deltas)), "median": float(np.median(all_deltas)),\n'
     '           "abs_delta": {f"{pl} | {LABELS[m]}": float(d) for (pl, m), d in zip(_labels, all_deltas)}},\n'
     '          open(os.path.join(e_common.out_dir(VERSION), f"ordinal_robustness_{OLD46 if VERSION == \'v1\' else \'v0\'}.json"), "w"), indent=1)', 1),
])

# ---------------- 10. paper_fig_register_carryover.py (D-O10: the Section 6.2.5 entry-gap analysis) ----
hdr11 = r'''"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/paper_fig_register_carryover.py
(the plate of Section 6.2: lag-1 residual cosine against a within-notebook shuffle null, and its decay by entry gap 1-8).
Changed: input paths only. Opus and Fable 5 entries + vectors via e_common (v1_common corpus + clean npz, never
journal_embeddings_cache_all.json or fable5_embeddings_cache.json); the PNG goes to out_<version>/. One added block
dumps lag-1 / decay / arc results at full precision (reads the dicts computed above it; changes nothing).
Usage: python register_carryover_v1.py v0|v1
"""
import e_common
VERSION = e_common.version_from_argv()
'''
OLD_LOAD = '''    # Load embeddings
    opus_emb = np.array(json.load(open(
        os.path.join(TEXTURE_BASE, "journal_embeddings_cache_all.json")))["embeddings"])
    fable_emb = np.array(json.load(open(
        os.path.join(TEXTURE_BASE, "fable5_embeddings_cache.json")))["embeddings"])'''
NEW_LOAD = '''    # Load embeddings
    # [E V1] was journal_embeddings_cache_all.json + fable5_embeddings_cache.json; now e_common (clean npz rows)
    _oe, opus_emb = e_common.load_opus(VERSION, range(1, 16))
    _fe, fable_emb = e_common.load_fable(VERSION)'''
OLD_IDX = '''    opus_flat = []
    for bn in range(1, 16):
        bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(bd):
            for om in OPUS_MODELS:
                files = [f for f in os.listdir(bd) if om in f and "#" not in f]
                if files:
                    data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
                    for i, _ in enumerate(data):
                        opus_flat.append({"model": om, "batch": bn, "entry_num": i + 1})

    fable_flat = []
    for bn in range(1, 16):
        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            for i, entry in enumerate(data):
                if entry["text"].strip() != "[silence]":
                    fable_flat.append({"batch": bn, "entry_num": i + 1})'''
NEW_IDX = '''    # [E V1] was: indices rebuilt from Texture\Batch_N and Journaling_data\Batch_N; now from the e_common rows (same order)
    opus_flat = [{"model": e["model"], "batch": e["batch"], "entry_num": e["entry_num"]} for e in _oe]
    fable_flat = [{"batch": e["batch"], "entry_num": e["entry_num"]} for e in _fe]'''
make("paper_fig_register_carryover.py", "register_carryover_v1.py", hdr11, [
    (OLD_LOAD, NEW_LOAD, 1),
    (OLD_IDX, NEW_IDX, 1),
    ('    out = os.path.join(TEXTURE_BASE, "paper_fig_register_carryover.png")',
     '    out = os.path.join(e_common.out_dir(VERSION), "paper_fig_register_carryover.png")  # [E V1] was TEXTURE_BASE\n'
     '    json.dump({"lag1": {m: {k: float(v) for k, v in d.items()} for m, d in lag1_results.items()},\n'
     '               "decay_gap_1_8": {m: {int(g): float(v) for g, v in d.items()} for m, d in decay_results.items()},\n'
     '               "arc": {m: {k: float(v) for k, v in d.items()} for m, d in arc_results.items()}},\n'
     '              open(os.path.join(e_common.out_dir(VERSION), "register_carryover.json"), "w"), indent=1)  # [E V1] added', 1),
])
