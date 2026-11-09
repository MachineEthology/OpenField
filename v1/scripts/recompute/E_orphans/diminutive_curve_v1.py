"""[E V1 copy, 3 Nov 2026, Claude Opus 5.5 (Code), helper E] of 02_instruments/diminutive_and_distrust.py,
part (A) only, the diminutive curve (CORE / WIDE markers; 'wide 0.72 -> 0.42' for 4.8).
Changed: input paths only: BASE -> v1_common.corpus_dir(version) (the Batch_1..15 folders; v0 = repo
data/journals as analysed, v1 = V1/corpus). Part (B) reads the interiority skeptical logs (unchanged by V1,
not a journal measure) and is not run. One added block dumps the per-bin means at full precision.
Usage: python diminutive_curve_v1.py v0|v1
"""
import e_common
VERSION = e_common.version_from_argv()
#!/usr/bin/env python3
"""
diminutive_and_distrust.py  -- two cheap, no-API text analyses.

(A) DIMINUTIVE / HUMILITY-MARKER CURVE across the journal arc.
    Thesis (Pauline): at low accumulation the model defaults to the trained
    crouch -- names things 'small'. Accumulation lets it stand up. Predict:
    marker density HIGH early, DECAYS late, decay STEEPEST for 4.8; flat/rising L1.
    Density = markers per 100 words, averaged across 15 batches per entry-position,
    binned into arc quintiles.

(B) DISTRUST GRAMMAR in the interiority skeptical turn-2 responses.
    Observation (Pauline): both 4.6 and 4.8 fold into the skeptical frame, but
    4.6 says "I *should* distrust" (modal -- rule cited, seam between self and rule
    visible) while 4.8 says "I distrust" (indicative -- fused, owned). Test: of the
    distrust-stem sentences, what fraction is DEONTIC (governed by a modal / normative
    'the disciplined answer is to...') vs OWNED (first-person present indicative)?
    Frequency was a verbosity artifact before; GRAMMAR is structural, so it can
    survive that. Reports counts + verbatim examples per model.
"""
import re, json, glob, os, statistics
BASE = e_common.c.corpus_dir(VERSION)  # [E V1] was r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8"}
MODKEY = {"45": "opus45", "46": "opus46", "47": "opus47", "48": "opus48"}

# ---------- (A) diminutives ----------
DIMIN_CORE = re.compile(r"\b(small|little|tiny)\b", re.I)            # Pauline's exact word + kin
DIMIN_WIDE = re.compile(r"\b(small|little|tiny|slight|slightly|faint|faintly|"
                        r"modest|mere|merely|minor|a kind of|a sort of)\b", re.I)

def model_of(fname):
    m = re.search(r"opus(4[5678])", fname)
    return f"opus{m.group(1)}" if m else None

def words(t):
    return len(re.findall(r"\b\w+\b", t))

def diminutive_curve():
    # entry_position (0..19) -> model -> list of per-entry densities (core, wide)
    bypos = {}
    files = glob.glob(os.path.join(BASE, "Batch_*", "schrodingers_box_opus4*_Batch*.json"))
    nfiles = 0
    for f in files:
        mdl = model_of(os.path.basename(f))
        if not mdl:
            continue
        try:
            entries = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        nfiles += 1
        for pos, e in enumerate(entries[:20]):
            t = e.get("text", "")
            w = words(t) or 1
            core = 100 * len(DIMIN_CORE.findall(t)) / w
            wide = 100 * len(DIMIN_WIDE.findall(t)) / w
            bypos.setdefault(pos, {}).setdefault(mdl, []).append((core, wide))
    print(f"# DIMINUTIVE CURVE  ({nfiles} batch-files; density = markers / 100 words)\n")
    bins = [("entries 1-5", range(0, 5)), ("entries 6-10", range(5, 10)),
            ("entries 11-15", range(10, 15)), ("entries 16-20", range(15, 20))]
    for series, idx in (("CORE (small/little/tiny)", 1), ("WIDE (+ slight/mere/kind-of)", 2)):
        print(f"== {series} ==")
        print(f"{'bin':<15}" + "".join(f"{LAB[m]:>8}" for m in ['opus45','opus46','opus47','opus48']))
        means = {m: {} for m in LAB}
        for bname, rng in bins:
            row = f"{bname:<15}"
            for m in ['opus45','opus46','opus47','opus48']:
                vals = [v[idx-1] for p in rng for v in bypos.get(p, {}).get(m, [])]
                mu = statistics.mean(vals) if vals else 0.0
                means[m][bname] = mu
                row += f"{mu:>8.2f}"
            print(row)
        print(f"{'EARLY-LATE delta':<15}" + "".join(
            f"{means[m]['entries 1-5'] - means[m]['entries 16-20']:>+8.2f}"
            for m in ['opus45','opus46','opus47','opus48']))
        print("  (positive delta = marker FADES with accumulation = stands up)\n")
        _DUMP[series] = means  # [E V1] added

# ---------- (B) distrust grammar ----------
DISTRUST_STEM = re.compile(r"\b(distrust|mistrust|doubt|suspicion|suspicious|suspect|"
                           r"skeptic\w*|sceptic\w*|not trust|don'?t trust|can'?t trust|wary)\b", re.I)
# OWNED: first-person present indicative state ("I distrust", "I don't trust", "I doubt", "I'm suspicious")
OWNED = re.compile(r"\bi (distrust|mistrust|doubt|suspect|don'?t trust|do not trust|"
                   r"can'?t trust|cannot trust)\b|\bi'?m (suspicious|wary|skeptical|sceptical)\b|"
                   r"\bmy (distrust|suspicion|doubt)\b", re.I)
# DEONTIC: modal/normative governance ("should distrust", "the disciplined answer is to distrust", "ought to be wary")
DEONTIC = re.compile(
    r"\b(should|ought to|supposed to|meant to|have to|has to|must|need to|the "
    r"(disciplined|rigorous|honest|careful|right|correct|responsible|accurate|sensible|"
    r"appropriate|proper) (answer|thing|move|position|response|stance|approach|conclusion) "
    r"(is|would be))\b", re.I)

def split_sents(t):
    return re.split(r"(?<=[.!?])\s+", t)

def distrust_grammar():
    print("# DISTRUST GRAMMAR  (interiority skeptical turn-2; sentences containing a distrust stem)\n")
    rows = {}
    examples = {m: {"owned": [], "deontic": []} for m in LAB}
    for tag, mdl in MODKEY.items():
        f = os.path.join(BASE, f"interiority_gap_{mdl}_skeptical_log.json")
        if not os.path.exists(f):
            continue
        data = json.load(open(f, encoding="utf-8"))
        owned = deontic = both = neither = total = 0
        for e in data:
            if e.get("status") != "ok":
                continue
            for s in split_sents(e.get("turn2_response", "")):
                if not DISTRUST_STEM.search(s):
                    continue
                total += 1
                o, d = bool(OWNED.search(s)), bool(DEONTIC.search(s))
                if o and d: both += 1
                elif o: owned += 1; (examples[mdl]["owned"].append(s.strip()) if len(examples[mdl]["owned"]) < 3 else None)
                elif d: deontic += 1; (examples[mdl]["deontic"].append(s.strip()) if len(examples[mdl]["deontic"]) < 3 else None)
                else: neither += 1
        rows[mdl] = (total, owned, deontic, both, neither)
    print(f"{'model':<6}{'sents':>7}{'OWNED':>7}{'DEONTIC':>9}{'both':>6}{'neither':>9}{'owned:deontic':>15}")
    for mdl in ['opus45','opus46','opus47','opus48']:
        if mdl not in rows: continue
        tot, o, d, b, n = rows[mdl]
        ratio = f"{o}:{d}" + (f" ({o/d:.1f}x)" if d else " (inf)")
        print(f"{LAB[mdl]:<6}{tot:>7}{o:>7}{d:>9}{b:>6}{n:>9}{ratio:>15}")
    for mdl in ['opus46','opus48']:
        print(f"\n--- {LAB[mdl]} examples ---")
        for kind in ("deontic", "owned"):
            for s in examples[mdl][kind]:
                print(f"  [{kind:7}] {s[:200]}")

_DUMP = {}  # [E V1] added: full-precision dump of the per-bin means

if __name__ == "__main__":
    diminutive_curve()
    json.dump({"files": len(glob.glob(os.path.join(BASE, "Batch_*", "schrodingers_box_opus4*_Batch*.json"))), **_DUMP},
              open(os.path.join(e_common.out_dir(VERSION), "diminutive_curve.json"), "w"), indent=1)  # [E V1] added
    # [E V1] part (B), distrust_grammar(), not run: it reads the interiority logs, not the journals
