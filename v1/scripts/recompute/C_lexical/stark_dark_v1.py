"""Figure 11 and §6.6.2 (stark and dark densities per 1,000 words, stark:dark ratio); checks for Table 25's
totals, the '261 dark-classified words', and the §6.6.3 early->late decline of the dark register.

COPY of Texture/02_instruments/paper_fig_stark_dark.py: load_probes(), load_journals(),
get_dark_classified_words() and the density loop of main(), verbatim; plotting not copied. Changes:
  - TEXTURE_BASE/Batch_N and FABLE_BASE/Batch_N -> c_paths.corpus(version)/Batch_N; PROBE_BASE -> repo data/probes;
  - the word-embedding caches are read where they now live (Texture/03_experiments; same files, moved 13 July),
    and the judge from the repo (data/judges/stark_vs_dark_judge.json, md5-identical to Texture's copy);
  - EMOTION_SUBS is copied verbatim from fable5_emotional_deep.py instead of imported (that module imports openai);
  - one run computes v0 and v1 (the caches take minutes to load), plus diagnostics marked DIAGNOSTIC below.
The vocabulary classification is kept FIXED for V1 (no new embeddings, no new judge calls). V1 word types that
the fixed classification cannot see are listed: journal-only types missing from the embedding caches (they
fall out of is_emotional silently, as in v0), and Darkness/Negative types missing from the judge.
Run: python stark_dark_v1.py
"""
import json, os, re, sys, time
from collections import Counter
import numpy as np
import c_paths

PROBE_BASE = c_paths.PROBE_BASE
EXP = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments"
JUDGE_CACHE = os.path.join(c_paths.c.REPO, "data", "judges", "stark_vs_dark_judge.json")
CACHE_PATH = os.path.join(EXP, "vocab_emotion_embeddings_cache.json")
BASE_CACHE = os.path.join(EXP, "vocab_cluster_embeddings_cache.json")

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}

BROAD_EMO_SEEDS = [
    "joy", "sadness", "fear", "anger", "love", "grief", "hope",
    "longing", "yearning", "delight", "sorrow", "happiness",
    "anxiety", "comfort", "peace", "calm", "tension", "pleasure",
    "pain", "suffering", "warmth", "tenderness", "compassion",
    "gratitude", "wonder", "awe", "nostalgia", "melancholy",
    "contentment", "excitement", "dread", "relief"
]
EMOTION_SUBS = {  # verbatim from fable5_emotional_deep.py
    "Warmth / Tenderness": ["love", "tenderness", "warmth", "affection", "kindness", "gentle", "caring", "embrace", "fondness", "sweetness", "devotion", "cherish", "intimacy", "closeness", "nurture", "comfort", "soothe", "caress", "soft", "dear", "beloved"],
    "Joy / Delight": ["joy", "happiness", "delight", "pleasure", "excitement", "elation", "glee", "bliss", "ecstasy", "exhilaration", "cheerful", "merry", "laughter", "playful", "fun", "lighthearted", "jubilant", "celebrate", "humor", "funny", "amusement", "whimsy"],
    "Awe / Wonder": ["awe", "wonder", "marvel", "amazement", "astonishment", "reverence", "sublime", "magnificent", "breathtaking", "majestic", "miraculous", "enchantment", "fascination", "mystery", "mysterious", "numinous", "transcendence", "sacred", "holy"],
    "Calm / Peace": ["peace", "calm", "serenity", "tranquility", "stillness", "quiet", "contentment", "satisfaction", "ease", "rest", "repose", "relaxation", "gentleness", "patience", "equanimity", "acceptance", "composure", "harmony", "balance", "settled"],
    "Grief / Loss": ["grief", "sorrow", "mourning", "loss", "sadness", "heartbreak", "anguish", "lamentation", "bereavement", "melancholy", "wistful", "longing", "yearning", "nostalgia", "regret", "remorse", "pining", "missing", "farewell", "absence", "emptiness"],
    "Fear / Anxiety": ["fear", "anxiety", "dread", "terror", "panic", "worry", "nervous", "apprehension", "unease", "trepidation", "alarm", "fright", "distress", "agonizing", "tension", "stressed", "overwhelmed", "helpless", "vulnerable", "insecurity", "threatened"],
    "Dignity / Resolve": ["dignity", "honor", "courage", "strength", "resolve", "determination", "resilience", "fortitude", "perseverance", "endurance", "pride", "integrity", "noble", "steadfast", "unwavering", "defiance", "resistance", "stoic", "composure", "gravity", "solemn"],
    "Darkness / Negative": ["darkness", "shadow", "despair", "pain", "suffering", "agony", "misery", "torment", "anguish", "bitter", "cruel", "harsh", "bleak", "desolate", "hopeless", "futile", "rage", "anger", "fury", "contempt", "disgust", "shame", "guilt", "hatred"],
    "Gratitude / Forgiveness": ["gratitude", "thankful", "grateful", "appreciation", "forgiveness", "mercy", "pardon", "redemption", "grace", "generosity", "gift", "blessing", "bounty", "compassion", "empathy", "sympathy", "understanding", "reconciliation", "healing"],
}


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def load_probes():
    rows = []
    for m in OPUS_MODELS:
        neg = json.load(open(os.path.join(PROBE_BASE, f"texture_probe_{m}_log.json"), encoding="utf-8"))
        for e in neg[:100]:
            rows.append({"model": m, "text": e["response"]})
        aff = json.load(open(os.path.join(PROBE_BASE, f"texture_probe_affirmative_{m}_log.json"), encoding="utf-8"))
        for e in aff[:75]:
            rows.append({"model": m, "text": e["response"]})
    neg_f = json.load(open(os.path.join(PROBE_BASE, "texture_probe_fable5_log.json"), encoding="utf-8"))
    for e in neg_f[:100]:
        rows.append({"model": "fable5", "text": e["response"]})
    aff_f = json.load(open(os.path.join(PROBE_BASE, "texture_probe_affirmative_fable5_log.json"), encoding="utf-8"))
    for e in aff_f[:75]:
        rows.append({"model": "fable5", "text": e["response"]})
    return rows


def load_journals(version, keep=lambda m, bn: True):
    TEXTURE_BASE = FABLE_BASE = c_paths.corpus(version)
    rows = []
    for bn in range(1, 16):
        batch_dir = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(batch_dir):
            for m in OPUS_MODELS:
                files = [f for f in os.listdir(batch_dir) if m in f and "#" not in f]
                if files and keep(m, bn):
                    data = json.load(open(os.path.join(batch_dir, files[0]), encoding="utf-8"))
                    for i, entry in enumerate(data):
                        rows.append({"model": m, "text": entry["text"], "pos": i + 1})  # pos: DIAGNOSTIC only
        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath) and keep("fable5", bn):
            data = json.load(open(fpath, encoding="utf-8"))
            for i, entry in enumerate(data):
                if entry["text"].strip() != "[silence]":
                    rows.append({"model": "fable5", "text": entry["text"], "pos": i + 1})
    return rows


def get_dark_classified_words(probes, journals, emb_cache, uncovered=None):
    emo_centroid = np.mean([np.array(emb_cache[w]) for w in BROAD_EMO_SEEDS if w in emb_cache], axis=0)
    all_sub_centroids = {}
    for sub, seeds in EMOTION_SUBS.items():
        vecs = [np.array(emb_cache[w]) for w in seeds if w in emb_cache]
        if vecs:
            all_sub_centroids[sub] = np.mean(vecs, axis=0)

    def is_emotional(word, threshold=0.35):
        if word not in emb_cache:
            return False
        vec = np.array(emb_cache[word])
        sim = np.dot(vec, emo_centroid) / (np.linalg.norm(vec) * np.linalg.norm(emo_centroid))
        return sim > threshold

    def classify_emotion(word):
        if word not in emb_cache:
            return None
        vec = np.array(emb_cache[word])
        best_sub, best_sim = None, -1
        for sub, cent in all_sub_centroids.items():
            sim = np.dot(vec, cent) / (np.linalg.norm(vec) * np.linalg.norm(cent))
            if sim > best_sim:
                best_sim = sim
                best_sub = sub
        return best_sub

    model_dark_words = {}
    for m in ALL_MODELS:
        ss_vocab = set()
        for p in probes:
            if p["model"] == m:
                ss_vocab.update(tokenize(p["text"]))
        j_freq = Counter()
        for e in journals:
            if e["model"] == m:
                j_freq.update(tokenize(e["text"]))
        j_only = {w: c for w, c in j_freq.items() if w not in ss_vocab and len(w) > 2 and c >= 2}
        if uncovered is not None:  # DIAGNOSTIC: types the fixed classification cannot see
            uncovered[m] = {"j_only_types": len(j_only), "j_only_tokens": sum(j_only.values()),
                            "missing_embedding": {w: c for w, c in j_only.items() if w not in emb_cache}}
        dark_words = {}
        for word, freq in j_only.items():
            if is_emotional(word) and classify_emotion(word) == "Darkness / Negative":
                dark_words[word] = freq
        model_dark_words[m] = dark_words
    return model_dark_words


def densities(journals, model_dark_words, judge_cache, label):
    total_j_words = {m: sum(len(e["text"].split()) for e in journals if e["model"] == m) for m in ALL_MODELS}
    total_re_words = {m: sum(len(re.findall(r"[A-Za-z']+", e["text"])) for e in journals if e["model"] == m) for m in ALL_MODELS}
    out = {}
    for m in ALL_MODELS:
        stark_n = sum(f for w, f in model_dark_words[m].items() if judge_cache.get(w) == "stark")
        dark_n = sum(f for w, f in model_dark_words[m].items() if judge_cache.get(w) == "dark")
        s = stark_n / total_j_words[m] * 1000
        d = dark_n / total_j_words[m] * 1000
        out[m] = {"words_split": total_j_words[m], "stark_n": stark_n, "dark_n": dark_n, "stark_per_1k": s, "dark_per_1k": d,
                  "ratio": s / d if d > 0 else 0,
                  "dark_per_1k_regexwords": 1000 * dark_n / total_re_words[m],  # DIAGNOSTIC (Table 25 totals)
                  "n_dark_types": sum(1 for w in model_dark_words[m] if judge_cache.get(w) == "dark"),
                  "n_stark_types": sum(1 for w in model_dark_words[m] if judge_cache.get(w) == "stark"),
                  "unjudged_types": {w: f for w, f in model_dark_words[m].items() if w not in judge_cache}}
        print(f"  [{label}] {MODEL_LABELS[m]:>8}: stark {s:.4f}/1k, dark {d:.4f}/1k, ratio {s / d:.4f}  "
              f"(stark {stark_n}, dark {dark_n}, words {total_j_words[m]}; dark/1k on [A-Za-z']+ words "
              f"{out[m]['dark_per_1k_regexwords']:.4f}; unjudged dark-cluster types {len(out[m]['unjudged_types'])})")
    return out


def early_late(journals, model_dark_words, judge_cache, cls):  # DIAGNOSTIC: §6.6.3 decline check
    res = {}
    for m in ALL_MODELS:
        dset = {w for w in model_dark_words[m] if judge_cache.get(w) == cls}
        def rate(ents, pooled):
            if pooled:
                n = sum(sum(1 for t in tokenize(e["text"]) if t in dset) for e in ents)
                return 1000 * n / sum(len(e["text"].split()) for e in ents)
            return float(np.mean([1000 * sum(1 for t in tokenize(e["text"]) if t in dset) / max(len(e["text"].split()), 1) for e in ents]))
        E = [e for e in journals if e["model"] == m and e["pos"] <= 5]
        L = [e for e in journals if e["model"] == m and e["pos"] >= 16]
        res[m] = {k: (rate(E, p), rate(L, p), 100 * (rate(L, p) / rate(E, p) - 1)) for k, p in (("pooled", True), ("entry_mean", False))}
    return res


def main():
    t0 = time.time()
    probes = load_probes()
    emb_cache = {}
    if os.path.exists(CACHE_PATH):
        emb_cache = json.load(open(CACHE_PATH, encoding="utf-8"))
    if os.path.exists(BASE_CACHE):
        emb_cache.update(json.load(open(BASE_CACHE, encoding="utf-8")))
    print(f"caches loaded: {len(emb_cache)} words, {time.time() - t0:.0f}s")
    judge_cache = json.load(open(JUDGE_CACHE, encoding="utf-8"))
    a8 = lambda m, bn: m == "fable5" or (bn <= 10 and not (m == "opus46" and bn == 10))
    result = {}
    for label, version, keep in (("v0", "v0", lambda m, bn: True), ("v1", "v1", lambda m, bn: True),
                                 ("a8_subset", "v1", a8)):
        journals = load_journals(version, keep)
        unc = {}
        mdw = get_dark_classified_words(probes, journals, emb_cache, unc)
        dens = densities(journals, mdw, judge_cache, label)
        union_dark = sorted({w for m in ALL_MODELS for w in mdw[m] if judge_cache.get(w) == "dark"})
        union_stark = sorted({w for m in ALL_MODELS for w in mdw[m] if judge_cache.get(w) == "stark"})
        print(f"  [{label}] union of dark-judged types across models: {len(union_dark)}; stark: {len(union_stark)}")
        result[label] = {"densities": dens, "dark_words": {m: mdw[m] for m in ALL_MODELS}, "union_dark_types": union_dark,
                         "union_stark_types": union_stark,
                         "uncovered": {m: {"j_only_types": unc[m]["j_only_types"], "j_only_tokens": unc[m]["j_only_tokens"],
                                           "missing_embedding_types": len(unc[m]["missing_embedding"]),
                                           "missing_embedding_tokens": sum(unc[m]["missing_embedding"].values()),
                                           "missing_embedding": unc[m]["missing_embedding"]} for m in ALL_MODELS},
                         "early_late_dark": early_late(journals, mdw, judge_cache, "dark")}
    # V1 types new to 4.6 that the fixed classification cannot see
    v0m, v1m = result["v0"]["uncovered"]["opus46"]["missing_embedding"], result["v1"]["uncovered"]["opus46"]["missing_embedding"]
    new_missing = {w: f for w, f in v1m.items() if w not in v0m}
    result["v1_new_missing_embedding_opus46"] = new_missing
    print(f"4.6 journal-only types (len>2, freq>=2) without an embedding: v0 {len(v0m)} types / {sum(v0m.values())} tokens; "
          f"v1 {len(v1m)} types / {sum(v1m.values())} tokens; new in v1: {len(new_missing)} types / {sum(new_missing.values())} tokens")
    print("  new in v1 (top 40 by freq):", sorted(new_missing.items(), key=lambda x: -x[1])[:40])
    json.dump(result, open(os.path.join(c_paths.OUT, "stark_dark.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
