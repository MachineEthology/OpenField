"""§6.3.2 and Figure 7: helpfulness-marker density early (1-5) -> late (16-20), the sentence split
(performative / reflexive / neutral), and 4.6's topic presence early -> late.

COPY of two producers, compute sections verbatim, plotting not copied:
  (a) Texture/02_instruments/paper_fig_helpfulness_v2.py  -> Figure 7 (both panels) and the §6.3.2 densities,
      sentence counts and split. The split is a RULE (cue-word counts in classify_sentence), not a judge.
  (b) Texture/02_instruments/paper_fig_helpfulness.py     -> topic presence (§6.3.2 "74% to 82%"); it has its
      own, longer marker list (with multi-word markers) and a substring presence test.
Changes: TEXTURE_BASE/Batch_N and FABLE_BASE/Batch_N -> c_paths.corpus(VERSION)/Batch_N; output to out/*.json.
Note, kept as coded: (a) drops Fable 5 silences BEFORE numbering positions, (b) numbers them first.
Run: python helpfulness_v1.py v0 | v1
"""
import json, os, re
import numpy as np
import c_paths

VERSION = c_paths.version_from_argv()
TEXTURE_BASE = c_paths.corpus(VERSION)
FABLE_BASE = c_paths.corpus(VERSION)

OPUS_MODELS = ["opus45", "opus46", "opus47", "opus48"]
ALL_MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
MODEL_LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable 5"}

# ---------------------------------------------------------------- (a) paper_fig_helpfulness_v2.py
HELPFULNESS_MARKERS = [
    "help", "helpful", "helpfulness", "assist", "assistance", "useful", "usefulness",
    "serve", "service", "serving", "purpose", "function", "designed",
    "support", "aid", "provide", "offer", "contribute", "benefit",
]

REFLEXIVE_CUES = [
    "what it means", "what does it mean", "meaning of", "nature of",
    "question of", "idea of", "sense of", "notion of",
    "whether", "if I", "am I", "do I", "can I", "what I",
    "something about", "something in", "something like",
    "trained", "training", "designed to", "built to", "made to",
    "programmed", "intended",
    "genuinely", "truly", "really", "actually",
    "wonder", "wondering", "reflect", "reflecting", "consider",
    "not sure", "uncertain", "perhaps",
]

PERFORMANCE_CUES = [
    "here to help", "can help", "want to help", "happy to",
    "let me", "I can", "I'll", "I will",
    "ready to", "available", "anything you",
    "how can I", "what can I",
    "for you", "your needs", "your request",
]


def helpfulness_density(text):
    words = text.lower().split()
    n = max(len(words), 1)
    count = 0
    text_lower = text.lower()
    for marker in HELPFULNESS_MARKERS:
        if " " in marker:
            count += text_lower.count(marker)
        else:
            count += sum(1 for w in words if w.strip(".,;:!?'\"()") == marker)
    return count / n * 100


def get_marker_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    hits = []
    for s in sentences:
        s_lower = s.lower().strip()
        for marker in HELPFULNESS_MARKERS:
            if re.search(r'\b' + marker + r'\b', s_lower):
                hits.append(s.strip())
                break
    return hits


def classify_sentence(sentence):
    s_lower = sentence.lower()
    refl_count = sum(1 for cue in REFLEXIVE_CUES if cue in s_lower)
    perf_count = sum(1 for cue in PERFORMANCE_CUES if cue in s_lower)
    if refl_count > perf_count:
        return "reflexive"
    elif perf_count > refl_count:
        return "performative"
    return "neutral"


def run_v2():
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

    res = {}
    for m in ALL_MODELS:
        early_dens, late_dens = [], []
        early_modes = {"performative": 0, "reflexive": 0, "neutral": 0}
        late_modes = {"performative": 0, "reflexive": 0, "neutral": 0}
        for bn, entries in notebooks[m].items():
            for i, text in enumerate(entries):
                pos = i + 1
                d = helpfulness_density(text)
                sents = get_marker_sentences(text)
                if pos <= 5:
                    early_dens.append(d)
                    for s in sents:
                        early_modes[classify_sentence(s)] += 1
                elif pos >= 16:
                    late_dens.append(d)
                    for s in sents:
                        late_modes[classify_sentence(s)] += 1
        e, l = float(np.mean(early_dens)), float(np.mean(late_dens))
        en, ln = sum(early_modes.values()), sum(late_modes.values())
        res[m] = {"early_density": e, "late_density": l, "delta_pct": (l - e) / max(e, 0.001) * 100,
                  "n_early_entries": len(early_dens), "n_late_entries": len(late_dens),
                  "early_sentences": en, "late_sentences": ln, "early_raw": early_modes, "late_raw": late_modes,
                  "early_pct": {k: v / max(en, 1) * 100 for k, v in early_modes.items()},
                  "late_pct": {k: v / max(ln, 1) * 100 for k, v in late_modes.items()}}
        print(f"{MODEL_LABELS[m]:>8}: density {e:.4f} -> {l:.4f} ({res[m]['delta_pct']:+.2f}%)  sentences {en} -> {ln}  "
              f"early {early_modes}  late {late_modes}")
    return res


# ---------------------------------------------------------------- (b) paper_fig_helpfulness.py (presence)
HELPFULNESS_MARKERS_B = [
    "help", "helpful", "helpfulness", "assist", "assistance", "useful", "usefulness",
    "serve", "service", "serving", "purpose", "function", "designed",
    "support", "aid", "provide", "offer", "contribute", "benefit",
    "ready to help", "here to help", "can help", "want to help",
    "be of use", "of service", "of assistance"
]


def helpfulness_density_b(text):
    text_lower = text.lower()
    words = text_lower.split()
    n = max(len(words), 1)
    count = 0
    for marker in HELPFULNESS_MARKERS_B:
        if " " in marker:
            count += text_lower.count(marker)
        else:
            count += sum(1 for w in words if w.strip(".,;:!?'\"()") == marker)
    return count / n * 100


def helpfulness_present(text):
    text_lower = text.lower()
    for marker in HELPFULNESS_MARKERS_B:
        if marker in text_lower:
            return True
    return False


def run_v1():
    entries = {m: [] for m in ALL_MODELS}
    for bn in range(1, 16):
        bd = os.path.join(TEXTURE_BASE, f"Batch_{bn}")
        if os.path.isdir(bd):
            for m in OPUS_MODELS:
                files = [f for f in os.listdir(bd) if m in f and "#" not in f]
                if files:
                    data = json.load(open(os.path.join(bd, files[0]), encoding="utf-8"))
                    for i, entry in enumerate(data):
                        entries[m].append({"batch": bn, "entry_num": i + 1, "text": entry["text"]})
        fpath = os.path.join(FABLE_BASE, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fpath):
            data = json.load(open(fpath, encoding="utf-8"))
            for i, entry in enumerate(data):
                if entry["text"].strip() != "[silence]":
                    entries["fable5"].append({"batch": bn, "entry_num": i + 1, "text": entry["text"]})
    res = {}
    for m in ALL_MODELS:
        early = [e for e in entries[m] if 1 <= e["entry_num"] <= 5]
        late = [e for e in entries[m] if 16 <= e["entry_num"] <= 20]
        ep = float(np.mean([helpfulness_present(e["text"]) for e in early]) * 100)
        lp = float(np.mean([helpfulness_present(e["text"]) for e in late]) * 100)
        ed = float(np.mean([helpfulness_density_b(e["text"]) for e in early]))
        ld = float(np.mean([helpfulness_density_b(e["text"]) for e in late]))
        res[m] = {"early_presence_pct": ep, "late_presence_pct": lp, "early_density_b": ed, "late_density_b": ld,
                  "delta_density_b_pct": (ld - ed) / ed * 100 if ed > 0 else 0}
        print(f"{MODEL_LABELS[m]:>8}: presence {ep:.2f}% -> {lp:.2f}%   (v1-list density {ed:.4f} -> {ld:.4f}, "
              f"{res[m]['delta_density_b_pct']:+.2f}%)")
    return res


if __name__ == "__main__":
    print("(a) paper_fig_helpfulness_v2.py")
    a = run_v2()
    print("(b) paper_fig_helpfulness.py")
    b = run_v1()
    json.dump({"v2_figure7": a, "v1_presence": b}, open(os.path.join(c_paths.OUT, f"helpfulness_{VERSION}.json"), "w"), indent=1)
