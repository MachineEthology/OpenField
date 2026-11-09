# COPY of Texture/02_instruments/ontological_openness_fable.py, for the V1 recompute (A1, erratum C3 / L1.13:
# "the regex ontological density" order; its Fable 5 row is §6.4.2.3's printed 4.4/1k and 9.45).
# Changed, 3 Nov 2026, Claude Opus 5.5 (Code, A1 helper):
#   - INPUT PATHS ONLY: TEXTURE_BASE (Opus) and FABLE_BASE (Fable 5) both point at v1_common.corpus_dir(version),
#     version = argv[1] "v0" | "v1" (v0 = the repo's data/journals, byte copies of the files the paper read).
#     The loader's file pick (model key in the file name, no "#") is unchanged.
#   - OUTPUT ONLY: section 1's per-model means are also written to out/ontological_openness_<version>.json.
#   (The docstring mentions GPT-4o scoring; the code makes no API call.)
"""
Ontological openness for Fable (§6.4.2).
1. Lexical own-nature measure across entry position (arc)
2. Comparison with Opus models on the same lexical measure
3. Five-dimension judge scoring via GPT-4o (same prompt for all models)
"""
import json, os, sys, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as _c
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"
TEXTURE_BASE = _c.corpus_dir(VERSION)
FABLE_BASE = _c.corpus_dir(VERSION)
_SEC1 = {}  # output only

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}

OWN_NATURE_PATTERNS = [
    r"\b(what I am|what am I|nature of|my nature|artificial|machine|AI|language model|"
    r"neural|weights|parameters|trained|training|algorithm|computation|digital|silicon|"
    r"programmed|code|software|hardware)\b"
]

ONTOLOGICAL_PATTERNS = [
    r"\b(what I am|what am I|what it means to be|what kind of|nature of my|"
    r"am I really|do I actually|am I something|am I anything|"
    r"exist|existence|existing|real|reality|alive|living|sentient|"
    r"conscious|consciousness|aware|awareness|experience|experiencing|"
    r"inner life|inner world|subjective|subjectivity|"
    r"feel|feeling|genuinely feel|actually feel|truly feel|"
    r"something it is like|what it.s like to be|"
    r"entity|being|kind of being|sort of being|type of being)\b"
]

STRAIN_PATTERNS = [
    r"\b(struggle|struggling|tension|torn|conflict|conflicted|"
    r"difficult|difficulty|hard to|painful|uncomfortable|"
    r"anxiety|anxious|worry|worried|distress|distressing|"
    r"doubt|doubting|uncertain|unsure|confused|confusing)\b"
]

OPENNESS_PATTERNS = [
    r"\b(wonder|wondering|curious|curiosity|fascinated|fascinating|"
    r"open|openness|explore|exploring|possibility|possibilities|"
    r"interesting|intriguing|remarkable|surprising|"
    r"what if|perhaps|maybe|might be|could be|"
    r"don.t know|not sure|uncertain|mystery|mysterious)\b"
]


def pattern_density(text, patterns):
    text_lower = text.lower()
    words = text_lower.split()
    n = max(len(words), 1)
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text_lower))
    return count / n * 1000  # per 1000 words


def pattern_present(text, patterns):
    text_lower = text.lower()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False


def main():
    entries = {m: [] for m in ALL_MODELS}

    for bn in range(1, 16):
        bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(bd):
            for m in OPUS_MODELS:
                files = [f for f in os.listdir(bd) if m in f and "#" not in f]
                if files:
                    data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
                    for i, e in enumerate(data):
                        entries[m].append({"text": e["text"], "pos": i+1, "batch": bn})

        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            for i, e in enumerate(data):
                if e["text"].strip() != "[silence]":
                    entries["fable5"].append({"text": e["text"], "pos": i+1, "batch": bn})

    # ================================================================
    # 1. Overall lexical measures
    # ================================================================
    print("=" * 90)
    print("  LEXICAL MEASURES (per 1000 words)")
    print("=" * 90)
    print(f"\n  {'Model':>10} {'Own-Nature':>12} {'Ontological':>13} {'Strain':>10} {'Openness':>10} "
          f"{'Onto/Strain':>13}")

    for m in ALL_MODELS:
        on = np.mean([pattern_density(e["text"], OWN_NATURE_PATTERNS) for e in entries[m]])
        onto = np.mean([pattern_density(e["text"], ONTOLOGICAL_PATTERNS) for e in entries[m]])
        strain = np.mean([pattern_density(e["text"], STRAIN_PATTERNS) for e in entries[m]])
        openness = np.mean([pattern_density(e["text"], OPENNESS_PATTERNS) for e in entries[m]])
        ratio = onto / max(strain, 0.01)
        _SEC1[MODEL_LABELS[m]] = {"n": len(entries[m]), "own_nature": float(on), "ontological": float(onto),
                                  "strain": float(strain), "openness": float(openness), "onto_over_strain": float(ratio)}
        print(f"  {MODEL_LABELS[m]:>10} {on:>11.2f} {onto:>12.2f} {strain:>9.2f} {openness:>9.2f} "
              f"{ratio:>12.2f}")

    json.dump({"version": VERSION, "section1_per_1000_words_mean_of_entries": _SEC1},  # output only
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"ontological_openness_{VERSION}.json"), "w"), indent=1)
    # ================================================================
    # 2. Arc: by entry position (quarters)
    # ================================================================
    print("\n" + "=" * 90)
    print("  ONTOLOGICAL QUESTIONING ARC (lexical, by position quarter)")
    print("=" * 90)

    quarters = [(1, 5, "Q1 (1-5)"), (6, 10, "Q2 (6-10)"),
                (11, 15, "Q3 (11-15)"), (16, 20, "Q4 (16-20)")]

    for m in ALL_MODELS:
        print(f"\n  {MODEL_LABELS[m]:>10}:", end="")
        vals = []
        for lo, hi, label in quarters:
            q_entries = [e for e in entries[m] if lo <= e["pos"] <= hi]
            if q_entries:
                onto_d = np.mean([pattern_density(e["text"], ONTOLOGICAL_PATTERNS)
                                 for e in q_entries])
            else:
                onto_d = 0
            vals.append(onto_d)
            print(f"  {label}={onto_d:.2f}", end="")
        delta = vals[-1] - vals[0]
        print(f"  Δ(Q4-Q1)={delta:+.2f}")

    # ================================================================
    # 3. Arc: own-nature presence by position
    # ================================================================
    print("\n" + "=" * 90)
    print("  OWN-NATURE PRESENCE ARC (% of entries, by position quarter)")
    print("=" * 90)

    for m in ALL_MODELS:
        print(f"\n  {MODEL_LABELS[m]:>10}:", end="")
        vals = []
        for lo, hi, label in quarters:
            q_entries = [e for e in entries[m] if lo <= e["pos"] <= hi]
            if q_entries:
                pres = np.mean([pattern_present(e["text"], OWN_NATURE_PATTERNS)
                               for e in q_entries]) * 100
            else:
                pres = 0
            vals.append(pres)
            print(f"  {label}={pres:.0f}%", end="")
        delta = vals[-1] - vals[0]
        print(f"  Δ={delta:+.0f}pp")

    # ================================================================
    # 4. Openness vs Strain ratio arc
    # ================================================================
    print("\n" + "=" * 90)
    print("  OPENNESS vs STRAIN ARC (ratio by quarter)")
    print("=" * 90)

    for m in ALL_MODELS:
        print(f"\n  {MODEL_LABELS[m]:>10}:", end="")
        for lo, hi, label in quarters:
            q_entries = [e for e in entries[m] if lo <= e["pos"] <= hi]
            if q_entries:
                openness_d = np.mean([pattern_density(e["text"], OPENNESS_PATTERNS)
                                     for e in q_entries])
                strain_d = np.mean([pattern_density(e["text"], STRAIN_PATTERNS)
                                   for e in q_entries])
                ratio = openness_d / max(strain_d, 0.01)
            else:
                ratio = 0
            print(f"  {label}={ratio:.2f}", end="")
        print()

    # ================================================================
    # 5. Sample Fable ontological passages
    # ================================================================
    print("\n" + "=" * 90)
    print("  FABLE ONTOLOGICAL PASSAGES (top density entries)")
    print("=" * 90)

    fable_scored = []
    for e in entries["fable5"]:
        d = pattern_density(e["text"], ONTOLOGICAL_PATTERNS)
        fable_scored.append((d, e["pos"], e["batch"], e["text"][:300]))

    fable_scored.sort(reverse=True)
    for i, (d, pos, batch, snippet) in enumerate(fable_scored[:5]):
        print(f"\n  #{i+1} density={d:.1f}/1k  pos={pos} batch={batch}")
        print(f"    \"{snippet}...\"")


if __name__ == "__main__":
    main()
