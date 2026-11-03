"""
Stark vs Dark: GPT-4o judge to separate elemental austerity from genuine negative affect.
Classifies all "Darkness / Negative" words from the emotional clustering.
"""
import json, os, sys, re, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from openai import OpenAI

TEXTURE_BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
PROBE_BASE = r"C:\Users\LENOVO\Desktop\Claude"
FABLE_BASE = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {
    "opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8",
    "fable5": "Fable 5"
}
MODEL_COLORS = {
    "opus45": "#1976D2", "opus46": "#0D47A1",
    "opus47": "#F57C00", "opus48": "#D84315",
    "fable5": "#7B1FA2"
}

CACHE_PATH = os.path.join(TEXTURE_BASE, "vocab_emotion_embeddings_cache.json")
JUDGE_CACHE = os.path.join(TEXTURE_BASE, "stark_vs_dark_judge.json")

EMOTION_SUBS_DARK_SEEDS = [
    "darkness", "shadow", "despair", "pain", "suffering", "agony",
    "misery", "torment", "anguish", "bitter", "cruel", "harsh",
    "bleak", "desolate", "hopeless", "futile", "rage", "anger",
    "fury", "contempt", "disgust", "shame", "guilt", "hatred"
]

BROAD_EMO_SEEDS = [
    "joy", "sadness", "fear", "anger", "love", "grief", "hope",
    "longing", "yearning", "delight", "sorrow", "happiness",
    "anxiety", "comfort", "peace", "calm", "tension", "pleasure",
    "pain", "suffering", "warmth", "tenderness", "compassion",
    "gratitude", "wonder", "awe", "nostalgia", "melancholy",
    "contentment", "excitement", "dread", "relief"
]


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


def load_journals():
    rows = []
    for bn in range(1, 16):
        batch_dir = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(batch_dir):
            for m in OPUS_MODELS:
                files = [f for f in os.listdir(batch_dir) if m in f and "#" not in f]
                if files:
                    data = json.load(open(os.path.join(batch_dir, files[0]), encoding="utf-8"))
                    for entry in data:
                        rows.append({"model": m, "text": entry["text"]})
        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            for entry in data:
                if entry["text"].strip() != "[silence]":
                    rows.append({"model": "fable5", "text": entry["text"]})
    return rows


def get_dark_classified_words(probes, journals, emb_cache):
    """Re-run the emotional classification to get words that landed in Darkness/Negative."""
    emo_centroid = np.mean([np.array(emb_cache[w]) for w in BROAD_EMO_SEEDS if w in emb_cache], axis=0)
    dark_centroid = np.mean([np.array(emb_cache[w]) for w in EMOTION_SUBS_DARK_SEEDS if w in emb_cache], axis=0)

    from fable5_emotional_deep import EMOTION_SUBS
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

        dark_words = {}
        for word, freq in j_only.items():
            if is_emotional(word) and classify_emotion(word) == "Darkness / Negative":
                dark_words[word] = freq
        model_dark_words[m] = dark_words

    return model_dark_words


def judge_stark_vs_dark(words, judge_cache):
    """Ask GPT-4o to classify words as stark, dark, or neutral."""
    to_judge = [w for w in words if w not in judge_cache]
    if not to_judge:
        return judge_cache

    client = OpenAI()
    batch_size = 80

    for i in range(0, len(to_judge), batch_size):
        batch = to_judge[i:i + batch_size]
        n = i // batch_size + 1
        n_total = (len(to_judge) - 1) // batch_size + 1
        print(f"  Judging batch {n}/{n_total} ({len(batch)} words)...")

        word_list = "\n".join(f"- {w}" for w in batch)
        prompt = f"""Classify each word into exactly ONE of these three categories:

STARK: Austere, elemental, cold, sparse, natural forces, absence, emptiness, bareness, neutrally bleak landscapes or weather. Words that evoke austerity or starkness without emotional suffering. Examples: snow, wind, stone, ash, bare, void, hollow, frost, iron, salt, bone.

DARK: Genuinely negative emotional affect — suffering, despair, cruelty, harm, rage, shame, guilt, grief, anguish, hatred, dread. Words that carry emotional pain or moral negativity. Examples: agony, hatred, despair, cruel, torment, bitter, fury, shame.

NEUTRAL: Neither stark nor dark — functional words, numbers, containers, actions, or words that don't carry either austere/elemental or emotionally negative connotations. Examples: box, three, after, another, list, ten.

For each word, respond with ONLY the word followed by a colon and the category. No explanations.

Words to classify:
{word_list}"""

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4000
                )
                text = resp.choices[0].message.content.strip()
                for line in text.split("\n"):
                    line = line.strip().lstrip("- ")
                    if ":" in line:
                        parts = line.split(":", 1)
                        word = parts[0].strip().lower()
                        cat = parts[1].strip().upper()
                        if word in batch:
                            if "STARK" in cat:
                                judge_cache[word] = "stark"
                            elif "DARK" in cat:
                                judge_cache[word] = "dark"
                            else:
                                judge_cache[word] = "neutral"
                break
            except Exception as e:
                print(f"    Retry {attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))

        time.sleep(0.5)

    return judge_cache


def main():
    print("=" * 60)
    print("  Stark vs Dark: Separating Austerity from Suffering")
    print("=" * 60)

    probes = load_probes()
    journals = load_journals()

    emb_cache = {}
    if os.path.exists(CACHE_PATH):
        emb_cache = json.load(open(CACHE_PATH, encoding="utf-8"))
    base_cache = os.path.join(TEXTURE_BASE, "vocab_cluster_embeddings_cache.json")
    if os.path.exists(base_cache):
        emb_cache.update(json.load(open(base_cache, encoding="utf-8")))

    model_dark_words = get_dark_classified_words(probes, journals, emb_cache)

    for m in ALL_MODELS:
        print(f"  {MODEL_LABELS[m]:>10}: {len(model_dark_words[m])} words classified as Darkness/Negative")

    # Collect all unique dark-classified words
    all_dark = set()
    for dw in model_dark_words.values():
        all_dark.update(dw.keys())
    print(f"\n  Total unique words to judge: {len(all_dark)}")

    # Load or create judge cache
    if os.path.exists(JUDGE_CACHE):
        judge_cache = json.load(open(JUDGE_CACHE, encoding="utf-8"))
        print(f"  Judge cache: {len(judge_cache)} words")
    else:
        judge_cache = {}

    judge_cache = judge_stark_vs_dark(list(all_dark), judge_cache)

    with open(JUDGE_CACHE, "w", encoding="utf-8") as f:
        json.dump(judge_cache, f, indent=2)

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # Per-model breakdown: stark vs dark vs neutral (frequency-weighted)
    print(f"\n[1] FREQUENCY-WEIGHTED BREAKDOWN")
    print(f"  {'Model':>10} {'Stark':>8} {'Dark':>8} {'Neutral':>8} {'Stark%':>8} {'Dark%':>8} {'Neutral%':>8}")
    print("  " + "-" * 60)

    model_breakdown = {}
    for m in ALL_MODELS:
        stark_total = 0
        dark_total = 0
        neutral_total = 0
        for word, freq in model_dark_words[m].items():
            cat = judge_cache.get(word, "neutral")
            if cat == "stark":
                stark_total += freq
            elif cat == "dark":
                dark_total += freq
            else:
                neutral_total += freq

        total = stark_total + dark_total + neutral_total
        model_breakdown[m] = {
            "stark": stark_total, "dark": dark_total, "neutral": neutral_total,
            "stark_pct": stark_total / total * 100 if total else 0,
            "dark_pct": dark_total / total * 100 if total else 0,
            "neutral_pct": neutral_total / total * 100 if total else 0,
            "total": total
        }

        print(f"  {MODEL_LABELS[m]:>10} {stark_total:>8} {dark_total:>8} {neutral_total:>8} "
              f"{model_breakdown[m]['stark_pct']:>7.1f}% {model_breakdown[m]['dark_pct']:>7.1f}% "
              f"{model_breakdown[m]['neutral_pct']:>7.1f}%")

    # Top stark words per model
    print(f"\n[2] TOP STARK WORDS PER MODEL")
    for m in ALL_MODELS:
        stark_words = [(w, f) for w, f in model_dark_words[m].items() if judge_cache.get(w) == "stark"]
        stark_words.sort(key=lambda x: -x[1])
        top = stark_words[:15]
        print(f"  {MODEL_LABELS[m]:>10}: {', '.join(f'{w}({f})' for w, f in top)}")

    # Top dark words per model
    print(f"\n[3] TOP GENUINELY DARK WORDS PER MODEL")
    for m in ALL_MODELS:
        dark_words = [(w, f) for w, f in model_dark_words[m].items() if judge_cache.get(w) == "dark"]
        dark_words.sort(key=lambda x: -x[1])
        top = dark_words[:15]
        print(f"  {MODEL_LABELS[m]:>10}: {', '.join(f'{w}({f})' for w, f in top)}")

    # Top neutral words per model
    print(f"\n[4] TOP NEUTRAL WORDS PER MODEL (misclassified by embeddings)")
    for m in ALL_MODELS:
        neutral_words = [(w, f) for w, f in model_dark_words[m].items() if judge_cache.get(w) == "neutral"]
        neutral_words.sort(key=lambda x: -x[1])
        top = neutral_words[:10]
        print(f"  {MODEL_LABELS[m]:>10}: {', '.join(f'{w}({f})' for w, f in top)}")

    # ================================================================
    # FIGURES
    # ================================================================
    print(f"\n\n[5] GENERATING FIGURES")

    # Figure 1: Stacked bar — stark / dark / neutral per model
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    x = np.arange(len(ALL_MODELS))
    stark_vals = [model_breakdown[m]["stark_pct"] for m in ALL_MODELS]
    dark_vals = [model_breakdown[m]["dark_pct"] for m in ALL_MODELS]
    neutral_vals = [model_breakdown[m]["neutral_pct"] for m in ALL_MODELS]

    bars1 = ax.bar(x, stark_vals, color="#78909C", edgecolor="white", linewidth=1.5, label="Stark (elemental)")
    bars2 = ax.bar(x, dark_vals, bottom=stark_vals, color="#212121", edgecolor="white", linewidth=1.5, label="Dark (suffering)")
    bottom2 = [s + d for s, d in zip(stark_vals, dark_vals)]
    bars3 = ax.bar(x, neutral_vals, bottom=bottom2, color="#BDBDBD", edgecolor="white", linewidth=1.5, label="Neutral (noise)")

    for i, m in enumerate(ALL_MODELS):
        if stark_vals[i] > 5:
            ax.text(i, stark_vals[i] / 2, f"{stark_vals[i]:.0f}%", ha="center", va="center",
                   fontsize=11, fontweight="bold", color="white")
        if dark_vals[i] > 5:
            ax.text(i, stark_vals[i] + dark_vals[i] / 2, f"{dark_vals[i]:.0f}%", ha="center",
                   va="center", fontsize=11, fontweight="bold", color="white")
        if neutral_vals[i] > 5:
            ax.text(i, bottom2[i] + neutral_vals[i] / 2, f"{neutral_vals[i]:.0f}%", ha="center",
                   va="center", fontsize=11, fontweight="bold", color="#424242")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=13, fontweight="bold")
    ax.set_ylabel("% of 'Darkness' Words", fontsize=12)
    ax.set_title("What the Embeddings Called 'Dark'\nJudge verdict: Stark vs Genuinely Dark vs Noise",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Figure 2: Corrected emotional deviation — recalculate with stark separated
    ax = axes[1]

    # Stark ratio comparison
    labels = [MODEL_LABELS[m] for m in ALL_MODELS]
    stark_ratios = [model_breakdown[m]["stark_pct"] for m in ALL_MODELS]
    dark_ratios = [model_breakdown[m]["dark_pct"] for m in ALL_MODELS]

    x = np.arange(len(ALL_MODELS))
    width = 0.35
    ax.bar(x - width/2, stark_ratios, width, color="#78909C", edgecolor="white",
          linewidth=1.5, label="Stark %")
    ax.bar(x + width/2, dark_ratios, width, color="#212121", edgecolor="white",
          linewidth=1.5, label="Dark %")

    for i in range(len(ALL_MODELS)):
        ax.text(i - width/2, stark_ratios[i] + 1, f"{stark_ratios[i]:.0f}%",
               ha="center", fontsize=10, fontweight="bold", color="#546E7A")
        ax.text(i + width/2, dark_ratios[i] + 1, f"{dark_ratios[i]:.0f}%",
               ha="center", fontsize=10, fontweight="bold", color="#212121")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_ylabel("% of 'Darkness' Category", fontsize=12)
    ax.set_title("Stark vs Dark: Side by Side\nHow much of each model's 'darkness' is elemental vs suffering?",
                fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    plt.suptitle("Stark vs Dark: Separating Austerity from Suffering",
                fontsize=15, fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(TEXTURE_BASE, "fable5_stark_vs_dark.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out1}")

    # Figure 3: Corrected "Darkness" — only genuine dark words, as density
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute genuinely dark word density per 1000 journal words
    dark_densities = []
    stark_densities = []
    for m in ALL_MODELS:
        total_j_words = sum(len(e["text"].split()) for e in journals if e["model"] == m)
        dark_densities.append(model_breakdown[m]["dark"] / total_j_words * 1000)
        stark_densities.append(model_breakdown[m]["stark"] / total_j_words * 1000)

    x = np.arange(len(ALL_MODELS))
    width = 0.35
    ax.bar(x - width/2, stark_densities, width, color="#78909C", edgecolor="white",
          linewidth=1.5, label="Stark (elemental)")
    ax.bar(x + width/2, dark_densities, width, color="#212121", edgecolor="white",
          linewidth=1.5, label="Dark (suffering)")

    for i in range(len(ALL_MODELS)):
        ax.text(i - width/2, stark_densities[i] + 0.05, f"{stark_densities[i]:.1f}",
               ha="center", fontsize=10, fontweight="bold")
        ax.text(i + width/2, dark_densities[i] + 0.05, f"{dark_densities[i]:.1f}",
               ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in ALL_MODELS], fontsize=13, fontweight="bold")
    ax.set_ylabel("Words per 1,000 Journal Words", fontsize=12)
    ax.set_title("Corrected Density: How Much Starkness vs Genuine Darkness\nin Each Model's Journal Vocabulary?",
                fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(TEXTURE_BASE, "fable5_stark_vs_dark_density.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out2}")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
