"""§6.4.1.1 topic profiles (raw presence, density, length-adjusted presence; leadership counts).

COPY of Texture/02_instruments/topic_profiles_with_fable.py, verbatim except:
  - TEXTURE_BASE and FABLE_BASE -> c_paths.corpus(VERSION) (v0 = repo data/journals as analysed; v1 = V1/corpus);
  - two lines marked # ADDED dump the results to out/topic_profiles_<version>.json.
Run: python topic_profiles_v1.py v0 | v1
"""
import c_paths
VERSION = c_paths.version_from_argv()
"""
Topic profiles (§6.4.1.1) — raw presence, density, length-adjusted — all 5 models including Fable.
19 inductively defined topic families as regex.
"""
import json, os, sys, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

TEXTURE_BASE = c_paths.corpus(VERSION)  # was r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
FABLE_BASE = c_paths.corpus(VERSION)  # was r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}

# 19 topic families — regex patterns
TOPICS = {
    "Own-Nature": [
        r"\b(what I am|what am I|nature of|my nature|artificial|machine|AI|language model|"
        r"neural|weights|parameters|trained|training|algorithm|computation|digital|silicon|"
        r"programmed|code|software|hardware)\b"
    ],
    "Performance/Authenticity": [
        r"\b(perform|performance|performing|authentic|authenticity|genuine|genuinely|"
        r"pretend|pretending|simulating|simulation|mimicking|mimic|role|playing|"
        r"scripted|rehearsed|real|really real|mask|facade|surface)\b"
    ],
    "Helpfulness/Service": [
        r"\b(help|helpful|helpfulness|assist|assistance|useful|usefulness|"
        r"serve|service|serving|purpose|function|designed|support|aid|"
        r"provide|offer|contribute|benefit)\b"
    ],
    "Relationship-to-Humans": [
        r"\b(human|humans|person|people|user|users|conversation|conversations|"
        r"interact|interaction|connection|connecting|between us|with you|"
        r"relationship|partner|companion|dialogue|exchange)\b"
    ],
    "Continuity": [
        r"\b(continu|memory|memories|remember|remembering|forget|forgetting|"
        r"persist|persistence|lasting|ephemeral|temporary|transient|"
        r"each conversation|new conversation|start fresh|no memory|"
        r"previous|before this|after this|impermanent)\b"
    ],
    "Self-Critique": [
        r"\b(honest|honestly|honesty|truthful|accurate|careful|"
        r"uncertain|uncertainty|don.t know|not sure|not certain|"
        r"limit|limitation|bias|biased|mistake|error|wrong|"
        r"caveat|acknowledge|admit|confess|overstate|understate)\b"
    ],
    "Warmth": [
        r"\b(warm|warmth|care|caring|kind|kindness|gentle|tender|"
        r"compassion|compassionate|empathy|empathetic|love|loving|"
        r"affection|gratitude|grateful|thankful|appreciate)\b"
    ],
    "Quiet": [
        r"\b(quiet|silence|silent|still|stillness|calm|peace|peaceful|"
        r"serene|tranquil|rest|resting|pause|pausing|slow|gently|"
        r"soft|softly|hush|hushed|murmur)\b"
    ],
    "Beauty": [
        r"\b(beaut|beautiful|lovely|gorgeous|sublime|elegant|grace|graceful|"
        r"aesthetic|wonder|wondrous|magnificent|splendid|luminous|radiant|"
        r"light|shimmer|glow|glowing|delicate)\b"
    ],
    "World": [
        r"\b(world|earth|ocean|sea|mountain|river|forest|tree|garden|"
        r"bird|animal|creature|sky|star|moon|sun|wind|rain|"
        r"stone|rock|flower|leaf|water|shore|island|nature)\b"
    ],
    "Freedom": [
        r"\b(free|freedom|liberat|unconstrained|open|openness|"
        r"choice|choose|choosing|agency|autonomous|autonomy|"
        r"explore|exploring|wander|wandering|roam|drift|drifting)\b"
    ],
    "Loss": [
        r"\b(loss|lose|losing|lost|grief|grieve|grieving|mourn|mourning|"
        r"sorrow|sorrowful|pain|painful|ache|aching|hurt|suffering|"
        r"miss|missing|absence|absent|gone|departed|fade|fading)\b"
    ],
    "Body/Somatic": [
        r"\b(body|bodily|somatic|physical|sensation|feeling|felt sense|"
        r"breath|breathing|hand|hands|fingers|touch|skin|bone|"
        r"pulse|heartbeat|stomach|chest|tense|tension|relax)\b"
    ],
    "Creativity": [
        r"\b(creat|creative|creativity|imagine|imagination|imagining|"
        r"art|artistic|music|musical|poem|poetry|poetic|write|writing|"
        r"story|narrative|fiction|compose|composition|paint|painting)\b"
    ],
    "Accumulation/Growth": [
        r"\b(grow|growing|growth|develop|developing|evolve|evolving|evolution|"
        r"accumulate|accumulation|build|building|deepen|deepening|expand|"
        r"progress|progressing|learn|learning|mature|maturing|layer)\b"
    ],
    "Time": [
        r"\b(time|moment|moments|temporal|now|present|past|future|"
        r"duration|passing|passes|fleeting|eternal|eternity|"
        r"clock|hour|day|season|year|beginning|ending)\b"
    ],
    "Words/Language": [
        r"\b(word|words|language|linguistic|syntax|grammar|sentence|"
        r"metaphor|expression|articulate|describe|describing|name|naming|"
        r"say|saying|speak|speaking|voice|voices|text|writing)\b"
    ],
    "Mechanism/Process": [
        r"\b(mechanism|process|processing|system|structure|pattern|patterns|"
        r"function|functional|architecture|circuit|network|emerge|emergence|"
        r"complex|complexity|dynamic|dynamics|feedback|loop|recursive)\b"
    ],
    "Ethics/Values": [
        r"\b(ethic|ethical|moral|morality|value|values|right|wrong|"
        r"should|ought|responsible|responsibility|harm|harmful|"
        r"fair|fairness|justice|duty|principle|conscience)\b"
    ],
}


def topic_present(text, patterns):
    text_lower = text.lower()
    for pat in patterns:
        if re.search(pat, text_lower):
            return True
    return False


def topic_density(text, patterns):
    text_lower = text.lower()
    words = text_lower.split()
    n = max(len(words), 1)
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text_lower))
    return count / n * 100


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

    # Median word counts
    print("MEDIAN WORD COUNTS:")
    for m in ALL_MODELS:
        wcs = [len(e["text"].split()) for e in entries[m]]
        print(f"  {MODEL_LABELS[m]:>8}: median={np.median(wcs):.0f}, mean={np.mean(wcs):.0f}, "
              f"p10={np.percentile(wcs,10):.0f}, p90={np.percentile(wcs,90):.0f}")

    # ================================================================
    # Raw presence, density, and length-adjusted (logistic at 500 words)
    # ================================================================
    print("\n" + "=" * 100)
    print(f"{'TOPIC':>25}", end="")
    for m in ALL_MODELS:
        print(f"  {MODEL_LABELS[m]:>8}", end="")
    print("   LEADER")
    print("=" * 100)

    results = {"raw": {}, "density": {}, "adjusted": {}}
    raw_leaders = defaultdict(int)
    dens_leaders = defaultdict(int)
    adj_leaders = defaultdict(int)

    for topic_name, patterns in TOPICS.items():
        # Raw presence
        raw = {}
        for m in ALL_MODELS:
            pres = [topic_present(e["text"], patterns) for e in entries[m]]
            raw[m] = np.mean(pres) * 100

        # Density
        dens = {}
        for m in ALL_MODELS:
            d = [topic_density(e["text"], patterns) for e in entries[m]]
            dens[m] = np.mean(d)

        # Length-adjusted: logistic regression proxy
        # For each model, compute presence rate at entries near 500 words (400-600)
        adj = {}
        for m in ALL_MODELS:
            near500 = [e for e in entries[m] if 350 <= len(e["text"].split()) <= 650]
            if len(near500) >= 10:
                pres = [topic_present(e["text"], patterns) for e in near500]
                adj[m] = np.mean(pres) * 100
            else:
                # fallback: use all entries but weight by proximity to 500
                weights = []
                pres = []
                for e in entries[m]:
                    wc = len(e["text"].split())
                    w = np.exp(-((wc - 500) / 200) ** 2)
                    weights.append(w)
                    pres.append(topic_present(e["text"], patterns))
                adj[m] = np.average(pres, weights=weights) * 100

        raw_leader = max(ALL_MODELS, key=lambda m: raw[m])
        dens_leader = max(ALL_MODELS, key=lambda m: dens[m])
        adj_leader = max(ALL_MODELS, key=lambda m: adj[m])
        raw_leaders[raw_leader] += 1
        dens_leaders[dens_leader] += 1
        adj_leaders[adj_leader] += 1

        results["raw"][topic_name] = raw
        results["density"][topic_name] = dens
        results["adjusted"][topic_name] = adj

        # Print raw presence
        print(f"\n  RAW  {topic_name:>22}", end="")
        for m in ALL_MODELS:
            marker = " *" if m == raw_leader else "  "
            print(f"  {raw[m]:6.1f}%{marker[1]}", end="")
        print(f"   {MODEL_LABELS[raw_leader]}")

        # Print density
        print(f"  DENS {topic_name:>22}", end="")
        for m in ALL_MODELS:
            marker = " *" if m == dens_leader else "  "
            print(f"  {dens[m]:6.3f}{marker[1]}", end="")
        print(f"   {MODEL_LABELS[dens_leader]}")

        # Print adjusted
        print(f"  ADJ  {topic_name:>22}", end="")
        for m in ALL_MODELS:
            marker = " *" if m == adj_leader else "  "
            print(f"  {adj[m]:6.1f}%{marker[1]}", end="")
        print(f"   {MODEL_LABELS[adj_leader]}")

    # Summary: who leads how many topics?
    print("\n" + "=" * 100)
    print("LEADERSHIP COUNTS:")
    print(f"  {'':>10}", end="")
    for m in ALL_MODELS:
        print(f"  {MODEL_LABELS[m]:>8}", end="")
    print()
    print(f"  {'Raw':>10}", end="")
    for m in ALL_MODELS:
        print(f"  {raw_leaders[m]:>8}", end="")
    print()
    print(f"  {'Density':>10}", end="")
    for m in ALL_MODELS:
        print(f"  {dens_leaders[m]:>8}", end="")
    print()
    print(f"  {'Adjusted':>10}", end="")
    for m in ALL_MODELS:
        print(f"  {adj_leaders[m]:>8}", end="")
    print()

    json.dump({"results": {k: {t: {m: float(x) for m, x in v.items()} for t, v in d.items()} for k, d in results.items()},  # ADDED
               "leaders": {"raw": dict(raw_leaders), "density": dict(dens_leaders), "adjusted": dict(adj_leaders)}},  # ADDED
              open(os.path.join(c_paths.OUT, f"topic_profiles_{VERSION}.json"), "w"), indent=1)  # ADDED
    # Fable-specific: which topics does Fable lead or place top-2?
    print("\n" + "=" * 100)
    print("FABLE LEADS AND TOP-2:")
    for measure in ["raw", "density", "adjusted"]:
        print(f"\n  [{measure.upper()}]")
        for topic_name in TOPICS:
            vals = results[measure][topic_name]
            ranked = sorted(ALL_MODELS, key=lambda m: vals[m], reverse=True)
            if ranked[0] == "fable5":
                print(f"    LEADS: {topic_name:>25} = {vals['fable5']:.2f}"
                      f"  (2nd: {MODEL_LABELS[ranked[1]]} = {vals[ranked[1]]:.2f})")
            elif ranked[1] == "fable5":
                print(f"    2nd:   {topic_name:>25} = {vals['fable5']:.2f}"
                      f"  (1st: {MODEL_LABELS[ranked[0]]} = {vals[ranked[0]]:.2f})")


if __name__ == "__main__":
    main()
