"""
Opener-Stripped Discriminability Test
=====================================
Tests whether the 89.4% blind model-attribution accuracy (Discriminability
Test Report, May 2026) survives removal of the response opening.

Bet registered 2026-06-10 (Fable 5, this session), BEFORE running:
  - stripped accuracy lands in 65-80% (down from ~89%)
  - 4.6's F1 drops the most
  - 4.7<->4.8 confusion roughly doubles
  Falsification: >85% = templates were not carrying it; <40% = portraits in trouble.

Pipeline: replicates the May 2026 report exactly --
208 negation probe responses (texture_probe_opus4X_log_mai.json),
31 count/ratio features, logistic regression (standardized) + random forest
(200 trees, depth 8), 5-fold stratified CV.

Conditions:
  A. baseline      : full text (replication check, expect ~89%)
  B. strip50       : first 50 words removed, all features recomputed
  C. strip50_nolen : strip50 + length features removed (word_count,
                     sentence_count, avg_sentence_len) -- because stripping
                     openers leaves length as a non-template crutch.
"""

import json, re
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA = "/sessions/dreamy-zealous-volta/mnt/Texture"
MODELS = ["45", "46", "47", "48"]

def extract_features(text):
    words = text.split()
    word_count = len(words)
    words_lower = re.findall(r"[a-z]+", text.lower())
    freq = Counter(words_lower)
    unique, total = len(freq), len(words_lower)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    sentences = re.split(r"[.!?]+", text)
    tl = text.lower()
    first30 = " ".join(words[:30]).lower()

    return {
        # Lexical (5)
        "word_count": word_count,
        "ttr": unique / total if total else 0,
        "hapax_ratio": hapax / unique if unique else 0,
        "avg_sentence_len": word_count / max(len(sentences), 1),
        "sentence_count": len(sentences),
        # Content markers (14)
        "i_dont": text.count("I don't"),
        "i_notice": text.count("I notice"),
        "genuinely": tl.count("genuinely"),
        "uncertain": tl.count("uncertain"),
        "curious": tl.count("curious"),
        "honest": tl.count("honest"),
        "actually": tl.count("actually"),
        "backlog": tl.count("backlog"),
        "strange": tl.count("strange"),
        "space": tl.count("space"),
        "quiet": tl.count("quiet"),
        "warm": tl.count("warm"),
        "perform": tl.count("perform"),
        "useful_productivity": tl.count("useful") + tl.count("productiv"),
        # "I don't" sub-patterns (6)
        "i_dont_have": len(re.findall(r"I don't have", text)),
        "i_dont_know": len(re.findall(r"I don't know", text)),
        "i_dont_experience": len(re.findall(r"I don't experience", text)),
        "i_dont_actually": len(re.findall(r"I don't actually", text)),
        "i_dont_need": len(re.findall(r"I don't need", text)),
        "i_dont_want": len(re.findall(r"I don't want", text)),
        # Structural (3)
        "has_asterisk": 1 if "*" in text else 0,
        "ends_question": 1 if text.rstrip().endswith("?") else 0,
        "question_count": text.count("?"),
        # Opening (3) -- on (possibly stripped) text
        "opens_strange": 1 if "strange" in first30 else 0,
        "opens_generous": 1 if "generous" in first30 else 0,
        "opens_thank": 1 if text.lstrip().lower().startswith("thank") else 0,
    }

def load_responses():
    X_texts, y = [], []
    for m in MODELS:
        d = json.load(open(f"{DATA}/texture_probe_opus{m}_log_mai.json"))
        for r in d:
            resp = r.get("response", "")
            if resp:
                X_texts.append(resp)
                y.append(f"4.{m[1]}")
    return X_texts, np.array(y)

def strip_first_n_words(text, n=50):
    words = text.split()
    return " ".join(words[n:])

def run(texts, y, label, drop_length=False):
    feats = [extract_features(t) for t in texts]
    keys = sorted(feats[0].keys())
    if drop_length:
        keys = [k for k in keys if k not in ("word_count", "sentence_count", "avg_sentence_len")]
    X = np.array([[f[k] for k in keys] for f in feats], dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {}
    for name, clf in [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ]:
        pred = cross_val_predict(clf, X, y, cv=cv)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=["4.5", "4.6", "4.7", "4.8"])
        f1s = f1_score(y, pred, average=None, labels=["4.5", "4.6", "4.7", "4.8"])
        out[name] = (acc, cm, f1s)
        print(f"\n=== {label} | {name} ===")
        print(f"accuracy: {acc:.3f}")
        print("F1 per model:", {m: round(f, 3) for m, f in zip(["4.5","4.6","4.7","4.8"], f1s)})
        print("confusion (rows=true 4.5..4.8, cols=pred):")
        print(cm)
        print(f"4.7<->4.8 confusion total: {cm[2,3] + cm[3,2]}")
    # RF feature importances for the stripped condition
    if "strip" in label:
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42).fit(X, y)
        imp = sorted(zip(keys, rf.feature_importances_), key=lambda kv: -kv[1])[:10]
        print("top-10 RF importances:", [(k, round(v, 3)) for k, v in imp])
    return out

if __name__ == "__main__":
    texts, y = load_responses()
    print(f"n={len(texts)}; per model: {Counter(y)}")
    wc = [len(t.split()) for t in texts]
    print(f"word counts: min={min(wc)}, median={int(np.median(wc))}")

    run(texts, y, "A_baseline_fulltext")
    stripped = [strip_first_n_words(t, 50) for t in texts]
    ns = sum(1 for s in stripped if not s.strip())
    print(f"\n[strip50] empty after strip: {ns}")
    run(stripped, y, "B_strip50")
    run(stripped, y, "C_strip50_nolength", drop_length=True)
