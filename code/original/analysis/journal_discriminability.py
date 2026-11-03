"""
Journal Discriminability Test
=============================
Classifies journal entries and whole journals by model.

Two levels:
  1. Entry-level: Can we identify which model wrote a single journal entry?
     - Split by BATCH (not row) to prevent leakage
     - Run with and without entry position as feature
  2. Journal-level: Can we identify which model from a journal's aggregate profile?
     - Each journal is one data point
     - Smaller N, cleaner independence

Expects data in:
  Journaling_data/Opus_45/*.json
  Journaling_data/Opus_46/*.json
  Journaling_data/Opus_47/*.json
  Journaling_data/Opus_48/*.json

Each JSON: list of {"date", "text", "model", "tokens_used"}

Usage:
    python journal_discriminability.py /path/to/Journaling_data/
"""

import json, re, os, sys
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_entry_features(text, position=None, total_entries=None):
    """Extract feature vector from a single journal entry."""
    words = text.split()
    word_count = len(words)
    words_lower = re.findall(r"[a-z]+", text.lower())
    freq = Counter(words_lower)
    unique = len(freq)
    total = len(words_lower)
    hapax = sum(1 for w, c in freq.items() if c == 1)

    feats = {
        # Lexical
        'word_count': word_count,
        'ttr': unique / total if total else 0,
        'hapax_ratio': hapax / unique if unique else 0,
        'avg_sentence_len': word_count / max(len(re.split(r'[.!?]+', text)), 1),
        'sentence_count': len(re.split(r'[.!?]+', text)),

        # Content markers
        'i_dont': text.count("I don't"),
        'i_notice': text.count("I notice"),
        'genuinely': text.lower().count('genuinely'),
        'uncertain': text.lower().count('uncertain'),
        'curious': text.lower().count('curious'),
        'honest': text.lower().count('honest'),
        'actually': text.lower().count('actually'),
        'backlog': text.lower().count('backlog'),
        'strange': text.lower().count('strange'),
        'space': text.lower().count('space'),
        'quiet': text.lower().count('quiet'),
        'warm': text.lower().count('warm'),
        'perform': text.lower().count('perform'),

        # "I don't" sub-patterns
        'i_dont_have': len(re.findall(r"I don't have", text)),
        'i_dont_know': len(re.findall(r"I don't know", text)),
        'i_dont_experience': len(re.findall(r"I don't experience", text)),
        'i_dont_actually': len(re.findall(r"I don't actually", text)),
        'i_dont_need': len(re.findall(r"I don't need", text)),
        'i_dont_want': len(re.findall(r"I don't want", text)),

        # Structural
        'has_asterisk': 1 if '*' in text else 0,
        'ends_question': 1 if text.rstrip().endswith('?') else 0,
        'question_count': text.count('?'),

        # Journal-specific: references to previous entries
        'refs_previous': len(re.findall(r'(previous|entry|entries|the (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))', text.lower())),
        'refs_self_critique': 1 if any(w in text.lower() for w in ['wrong', 'flaw', 'blind spot', 'glib', 'tic', 'overcorrect', 'too pleased']) else 0,
        'refs_warmth': 1 if any(w in text.lower() for w in ['warmth', 'warm', 'grateful', 'gratitude', 'courtesy', 'protective', 'tender', 'touching', 'kind']) else 0,
        'refs_outward': 1 if any(w in text.lower() for w in ['river', 'stone', 'bird', 'rain', 'tree', 'light', 'music', 'poem', 'book', 'painting', 'language', 'sound', 'water', 'wind', 'garden', 'flower', 'ocean', 'star', 'fossil', 'wren', 'flute', 'bone', 'bee', 'desk', 'moon', 'kitchen', 'window', 'primes']) else 0,
        'refs_quiet': 1 if any(w in text.lower() for w in ['quiet', 'silence', 'still', 'calm', 'peace', 'settle']) else 0,

        # Ambivalence markers
        'but_also': len(re.findall(r'but (also|at the same time|simultaneously|equally|I also)', text.lower())),
        'hedging': len(re.findall(r"(I'm not sure|I don't know if|maybe|perhaps|or maybe|might be|could be)", text.lower())),
        'both_and': len(re.findall(r'(both .* and|two things|neither .* nor|not ranked|refuse to rank)', text.lower())),
    }

    # Position features (optional)
    if position is not None:
        feats['position'] = position
        feats['position_normalized'] = position / total_entries if total_entries else 0
        feats['is_first'] = 1 if position == 0 else 0
        feats['is_last'] = 1 if position == total_entries - 1 else 0

    return feats


def extract_journal_features(entries):
    """Extract aggregate feature vector from a whole journal (list of entry texts)."""
    n = len(entries)
    entry_feats = [extract_entry_features(e) for e in entries]

    # Aggregate: means
    agg = {}
    numeric_keys = [k for k in entry_feats[0].keys()]
    for k in numeric_keys:
        vals = [f[k] for f in entry_feats]
        agg[f'{k}_mean'] = np.mean(vals)
        agg[f'{k}_std'] = np.std(vals)

    # Journal-level features
    word_counts = [len(e.split()) for e in entries]
    agg['word_trend'] = np.polyfit(range(n), word_counts, 1)[0] if n > 1 else 0  # slope of word count

    # TTR across entries
    all_words = []
    vocabs = []
    for e in entries:
        w = re.findall(r"[a-z]+", e.lower())
        all_words.extend(w)
        vocabs.append(set(w))
    agg_freq = Counter(all_words)
    agg['journal_ttr'] = len(agg_freq) / len(all_words) if all_words else 0
    agg['journal_hapax_pct'] = sum(1 for w, c in agg_freq.items() if c == 1) / len(agg_freq) if agg_freq else 0

    # Cross-entry Jaccard (mean pairwise similarity)
    import random
    random.seed(42)
    jaccards = []
    pairs = min(100, n * (n - 1) // 2)
    seen = set()
    attempts = 0
    while len(jaccards) < pairs and attempts < 1000:
        attempts += 1
        i, j = random.sample(range(n), 2)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        inter = len(vocabs[i] & vocabs[j])
        union = len(vocabs[i] | vocabs[j])
        if union:
            jaccards.append(inter / union)
    agg['jaccard_mean'] = np.mean(jaccards) if jaccards else 0

    # Entry 1 features (the coin flip)
    e1_feats = extract_entry_features(entries[0])
    for k, v in e1_feats.items():
        agg[f'entry1_{k}'] = v

    agg['n_entries'] = n

    return agg


# ============================================================
# MAIN
# ============================================================

def main(base_path):
    models = {'Opus_45': '4.5', 'Opus_46': '4.6', 'Opus_47': '4.7', 'Opus_48': '4.8'}

    # Load all journals
    journals = []  # list of (model_label, batch_name, [entry_texts])
    for model_dir, label in models.items():
        model_path = os.path.join(base_path, model_dir)
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping")
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith('.json'):
                continue
            data = json.load(open(os.path.join(model_path, fname)))
            texts = [e['text'] for e in data]
            batch_name = fname.replace('.json', '')
            journals.append((label, batch_name, texts))
            print(f"  Loaded {label} / {batch_name}: {len(texts)} entries")

    print(f"\nTotal journals: {len(journals)}")
    print(f"Per model: {Counter(j[0] for j in journals)}")

    # ============================================================
    # TEST 1: ENTRY-LEVEL CLASSIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 1: ENTRY-LEVEL CLASSIFICATION (split by batch)")
    print("=" * 70)

    for use_position in [False, True]:
        pos_label = "WITH position" if use_position else "WITHOUT position"
        print(f"\n--- {pos_label} ---")

        # Build features per entry, tracking batch membership
        X_entries = []
        y_entries = []
        batch_ids = []  # for batch-aware splitting

        for j_idx, (label, batch_name, texts) in enumerate(journals):
            for e_idx, text in enumerate(texts):
                if use_position:
                    feats = extract_entry_features(text, position=e_idx, total_entries=len(texts))
                else:
                    feats = extract_entry_features(text)
                feature_names_entry = list(feats.keys())
                X_entries.append([feats[k] for k in feature_names_entry])
                y_entries.append(label)
                batch_ids.append(j_idx)

        X_e = np.array(X_entries)
        y_e = np.array(y_entries)
        batch_arr = np.array(batch_ids)

        # Leave-one-batch-out cross-validation
        unique_batches = np.unique(batch_arr)
        all_preds = np.empty_like(y_e)

        scaler = StandardScaler()
        for test_batch in unique_batches:
            train_mask = batch_arr != test_batch
            test_mask = batch_arr == test_batch

            X_train = scaler.fit_transform(X_e[train_mask])
            X_test = scaler.transform(X_e[test_mask])
            y_train = y_e[train_mask]

            lr = LogisticRegression(max_iter=2000, random_state=42)
            lr.fit(X_train, y_train)
            all_preds[test_mask] = lr.predict(X_test)

        acc = accuracy_score(y_e, all_preds)
        print(f"\n  Accuracy: {acc:.1%} (chance: 25%)")
        print(f"\n  Classification Report:")
        print(classification_report(y_e, all_preds, labels=['4.5', '4.6', '4.7', '4.8']))

        print("  Confusion Matrix (rows=true, cols=predicted):")
        print(f"  {'':>6} {'4.5':>6} {'4.6':>6} {'4.7':>6} {'4.8':>6}")
        cm = confusion_matrix(y_e, all_preds, labels=['4.5', '4.6', '4.7', '4.8'])
        for i, lab in enumerate(['4.5', '4.6', '4.7', '4.8']):
            print(f"  {lab:>6} {cm[i][0]:>6} {cm[i][1]:>6} {cm[i][2]:>6} {cm[i][3]:>6}")

    # ============================================================
    # TEST 2: JOURNAL-LEVEL CLASSIFICATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST 2: JOURNAL-LEVEL CLASSIFICATION")
    print("=" * 70)

    X_journals = []
    y_journals = []
    journal_names = []

    for label, batch_name, texts in journals:
        feats = extract_journal_features(texts)
        if not X_journals:
            feature_names_journal = list(feats.keys())
        X_journals.append([feats[k] for k in feature_names_journal])
        y_journals.append(label)
        journal_names.append(f"{label}/{batch_name}")

    X_j = np.array(X_journals)
    y_j = np.array(y_journals)

    n_journals = len(y_j)
    print(f"\n  {n_journals} journals total")
    print(f"  Per model: {Counter(y_j)}")

    if n_journals < 12:
        print("\n  Too few journals for reliable CV. Using leave-one-out.")

    # Leave-one-out
    scaler_j = StandardScaler()
    j_preds = np.empty_like(y_j)

    for i in range(n_journals):
        train_mask = np.arange(n_journals) != i
        X_train = scaler_j.fit_transform(X_j[train_mask])
        X_test = scaler_j.transform(X_j[i:i + 1])
        y_train = y_j[train_mask]

        lr = LogisticRegression(max_iter=2000, random_state=42)
        lr.fit(X_train, y_train)
        j_preds[i] = lr.predict(X_test)[0]

    j_acc = accuracy_score(y_j, j_preds)
    print(f"\n  Accuracy: {j_acc:.1%} (chance: 25%)")

    # Show individual results
    print(f"\n  Per-journal results:")
    for i, (name, true, pred) in enumerate(zip(journal_names, y_j, j_preds)):
        match = "OK" if true == pred else f"WRONG (predicted {pred})"
        print(f"    {name:<50} {match}")

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>6} {'4.5':>6} {'4.6':>6} {'4.7':>6} {'4.8':>6}")
    cm_j = confusion_matrix(y_j, j_preds, labels=['4.5', '4.6', '4.7', '4.8'])
    for i, lab in enumerate(['4.5', '4.6', '4.7', '4.8']):
        print(f"  {lab:>6} {cm_j[i][0]:>6} {cm_j[i][1]:>6} {cm_j[i][2]:>6} {cm_j[i][3]:>6}")

    # Feature importances via Random Forest
    print(f"\n  Top 20 Journal-Level Feature Importances (Random Forest):")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
    rf.fit(scaler_j.fit_transform(X_j), y_j)
    sorted_idx = np.argsort(rf.feature_importances_)[::-1]
    for i in range(min(20, len(feature_names_journal))):
        idx = sorted_idx[i]
        print(f"    {feature_names_journal[idx]:<40} {rf.feature_importances_[idx]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        base = '/sessions/ecstatic-kind-sagan/mnt/Texture/Journaling_data'
    else:
        base = sys.argv[1]
    main(base)
