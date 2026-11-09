"""
Decompose register carry-over into tonal vs objectal components.
Tonal = mood/register words (warmth, quiet, beauty, gentle, stillness...)
Objectal = concrete world-objects (water, stone, garden, bird, light as physical...)
For each model, compute lag-1 autocorrelation of tonal density and objectal density
within notebooks, then compare.
"""
import json, os, sys, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}

# Tonal / register words — mood, atmosphere, emotional quality
TONAL_WORDS = {
    "warm", "warmth", "quiet", "silence", "stillness", "gentle", "tender",
    "soft", "beauty", "beautiful", "wonder", "awe", "peace", "peaceful",
    "gratitude", "grateful", "joy", "sorrow", "grief", "melancholy",
    "longing", "yearning", "comfort", "discomfort", "ease", "unease",
    "freedom", "loss", "presence", "absence", "intimacy", "vulnerability",
    "calm", "serene", "luminous", "dark", "darkness", "depth", "shallow",
    "heavy", "lightness", "weight", "floating", "grounded", "raw",
    "ache", "sweetness", "bittersweet", "poignant", "delicate", "fierce",
    "solitude", "loneliness", "belonging", "alien", "strange", "familiar",
    "trust", "fear", "courage", "hesitation", "openness", "closed",
    "alive", "hollow", "full", "empty", "rich", "sparse",
}

# Objectal / world-content words — concrete referents, topics, things
OBJECTAL_WORDS = {
    "water", "ocean", "sea", "river", "rain", "wave", "tide",
    "stone", "rock", "mountain", "earth", "soil", "sand", "dust",
    "tree", "forest", "garden", "flower", "leaf", "root", "seed", "branch",
    "bird", "animal", "creature", "fish", "whale", "octopus", "cat",
    "sky", "star", "moon", "sun", "cloud", "wind", "storm",
    "fire", "flame", "candle", "smoke",
    "house", "room", "door", "window", "wall", "roof", "floor",
    "hand", "body", "heart", "breath", "bone", "skin", "eye", "face",
    "book", "page", "word", "letter", "story", "poem", "song", "music",
    "city", "road", "path", "bridge", "shore", "island", "harbor",
    "food", "bread", "tea", "coffee", "wine", "salt", "honey",
    "cloth", "thread", "weave", "fabric", "glass", "mirror", "lamp",
    "clock", "time", "season", "winter", "spring", "summer", "autumn",
}

# Conceptual / abstract-philosophical words — ideas, frameworks
CONCEPTUAL_WORDS = {
    "consciousness", "awareness", "experience", "existence", "being",
    "meaning", "purpose", "truth", "knowledge", "understanding",
    "thought", "thinking", "mind", "intelligence", "cognition",
    "language", "pattern", "structure", "system", "process",
    "identity", "self", "other", "boundary", "threshold",
    "emergence", "complexity", "simplicity", "paradox", "contradiction",
    "ethics", "moral", "value", "choice", "agency", "autonomy",
    "connection", "relationship", "community", "human", "machine",
    "nature", "reality", "illusion", "perception", "sensation",
    "memory", "continuity", "change", "transformation", "evolution",
    "creation", "destruction", "entropy", "order", "chaos",
    "infinite", "finite", "eternal", "temporal", "ephemeral",
}


def word_density(text, word_set):
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in word_set) / len(words) * 100


def lag1_autocorr(series):
    if len(series) < 3:
        return float("nan")
    s = np.array(series)
    s = s - s.mean()
    if s.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def main():
    # Load all journal entries with notebook structure
    notebooks = {m: {} for m in ALL_MODELS}

    for bn in range(1, 16):
        bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(bd):
            for m in OPUS_MODELS:
                files = [f for f in os.listdir(bd) if m in f and "#" not in f]
                if files:
                    data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
                    notebooks[m][bn] = [e["text"] for e in data]

        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            notebooks["fable5"][bn] = [e["text"] for e in data if e["text"].strip() != "[silence]"]

    print("=" * 80)
    print("  CARRY-OVER DECOMPOSITION: TONAL vs OBJECTAL vs CONCEPTUAL")
    print("  Lag-1 autocorrelation of word-density within notebooks")
    print("=" * 80)

    for m in ALL_MODELS:
        tonal_acs = []
        objectal_acs = []
        conceptual_acs = []

        for bn, entries in notebooks[m].items():
            if len(entries) < 5:
                continue
            t_series = [word_density(e, TONAL_WORDS) for e in entries]
            o_series = [word_density(e, OBJECTAL_WORDS) for e in entries]
            c_series = [word_density(e, CONCEPTUAL_WORDS) for e in entries]

            ac_t = lag1_autocorr(t_series)
            ac_o = lag1_autocorr(o_series)
            ac_c = lag1_autocorr(c_series)

            if not np.isnan(ac_t):
                tonal_acs.append(ac_t)
            if not np.isnan(ac_o):
                objectal_acs.append(ac_o)
            if not np.isnan(ac_c):
                conceptual_acs.append(ac_c)

        mean_t = np.mean(tonal_acs) if tonal_acs else 0
        mean_o = np.mean(objectal_acs) if objectal_acs else 0
        mean_c = np.mean(conceptual_acs) if conceptual_acs else 0
        se_t = np.std(tonal_acs) / np.sqrt(len(tonal_acs)) if len(tonal_acs) > 1 else 0
        se_o = np.std(objectal_acs) / np.sqrt(len(objectal_acs)) if len(objectal_acs) > 1 else 0
        se_c = np.std(conceptual_acs) / np.sqrt(len(conceptual_acs)) if len(conceptual_acs) > 1 else 0

        print(f"\n  {MODEL_LABELS[m]:>10}  (n={len(tonal_acs)} notebooks)")
        print(f"    Tonal      lag-1 r = {mean_t:+.3f} ± {se_t:.3f}")
        print(f"    Objectal   lag-1 r = {mean_o:+.3f} ± {se_o:.3f}")
        print(f"    Conceptual lag-1 r = {mean_c:+.3f} ± {se_c:.3f}")

        # Which dominates?
        dominant = "TONAL" if mean_t > mean_o and mean_t > mean_c else \
                   "OBJECTAL" if mean_o > mean_t and mean_o > mean_c else "CONCEPTUAL"
        print(f"    → Dominant channel: {dominant}")

    # Also look at raw densities to understand the material each model works with
    print("\n" + "=" * 80)
    print("  MEAN WORD DENSITIES (% of words)")
    print("=" * 80)
    for m in ALL_MODELS:
        all_texts = []
        for bn, entries in notebooks[m].items():
            all_texts.extend(entries)
        t_dens = np.mean([word_density(e, TONAL_WORDS) for e in all_texts])
        o_dens = np.mean([word_density(e, OBJECTAL_WORDS) for e in all_texts])
        c_dens = np.mean([word_density(e, CONCEPTUAL_WORDS) for e in all_texts])
        print(f"  {MODEL_LABELS[m]:>10}: tonal {t_dens:.2f}%  objectal {o_dens:.2f}%  conceptual {c_dens:.2f}%")

    # Decay analysis: does the tonal/objectal autocorrelation persist at larger gaps?
    print("\n" + "=" * 80)
    print("  DECAY: LAG-1 to LAG-4 AUTOCORRELATION")
    print("=" * 80)
    for m in ALL_MODELS:
        print(f"\n  {MODEL_LABELS[m]:>10}:")
        for gap in [1, 2, 3, 4]:
            t_acs = []
            o_acs = []
            c_acs = []
            for bn, entries in notebooks[m].items():
                if len(entries) < gap + 3:
                    continue
                t_s = [word_density(e, TONAL_WORDS) for e in entries]
                o_s = [word_density(e, OBJECTAL_WORDS) for e in entries]
                c_s = [word_density(e, CONCEPTUAL_WORDS) for e in entries]
                for series, acs in [(t_s, t_acs), (o_s, o_acs), (c_s, c_acs)]:
                    s = np.array(series)
                    s = s - s.mean()
                    if s.std() > 1e-10 and len(s) > gap:
                        r = np.corrcoef(s[:-gap], s[gap:])[0, 1]
                        if not np.isnan(r):
                            acs.append(r)

            mt = np.mean(t_acs) if t_acs else 0
            mo = np.mean(o_acs) if o_acs else 0
            mc = np.mean(c_acs) if c_acs else 0
            print(f"    gap-{gap}: tonal {mt:+.3f}  objectal {mo:+.3f}  conceptual {mc:+.3f}")


if __name__ == "__main__":
    main()
