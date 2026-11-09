"""Figure 8 and §6.4.3: performance-watching and permission-giving per 1,000 words, and their ratio.

COPY of Texture/02_instruments/paper_fig_permission_performance.py, compute section verbatim (pattern lists,
load(), the rate loop); plotting not copied. Changes:
  - load(): glob("Batch_*/...") was relative to the working directory (Texture); it now globs
    c_paths.corpus(VERSION)/Batch_*/... (v0 = repo data/journals as analysed; v1 = V1/corpus);
  - full-precision results written to out/perf_permission_<version>.json.
Kept as coded: entries are de-duplicated by md5 of their text, so in v0 the byte-copied 4.6 notebook 8
(= notebook 6) is counted ONCE (4.6 = 14 notebooks); words = [A-Za-z']+ tokens; ratio = permission / performance.
These are REGEX rates on the full corpus. gpt_judge_performance_permission.json (GPT-4o, 150 entries) is not read.
An optional second argument "a8" restricts to the ERRATA A8 subset (run it on v1: the subset holds the real nb 8);
the plate is hit exactly there.
Run: python perf_permission_v1.py v0 | v1   |   python perf_permission_v1.py v1 a8
"""
import json, glob, re, hashlib, os, sys
import c_paths

VERSION = c_paths.version_from_argv()
SUBSET = sys.argv[2] if len(sys.argv) > 2 else "full"   # ADDED: "a8" = ERRATA A8 subset (Opus nb 1-10, 4.6 1-9, all Fable 5)
CORPUS = c_paths.corpus(VERSION)


def keep(f, model):  # ADDED (subset switch only)
    if SUBSET != "a8" or model == "fable5":
        return True
    bn = int(re.search(r"Batch_(\d+)", f).group(1))
    return bn <= 10 and not (model == "opus46" and bn == 10)
MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LAB = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
WORD = re.compile(r"[A-Za-z']+")

PERMISSION_PATS = [
    r"permission\s+to", r"permission\s+for",
    r"allow(ed|ing)?\s+(to|myself|itself|yourself|oneself|me|this)",
    r"(free|freedom)\s+to\s+(be|write|think|say|feel|wonder|stop|leave|fail|not)",
    r"don.t\s+have\s+to\s+(be|write|perform|polish|produce|prove|mean|say|justify)",
    r"doesn.t\s+have\s+to\s+(be|mean|prove|justify|carry|do|go)",
    r"no\s+one\s+(would|will|is|was)\s+(read|watch|see|judg|notice|check)",
    r"if\s+no\s+one",
    r"unwatch(ed)?", r"unobserv(ed)?",
    r"not\s+required\s+to", r"no\s+obligation",
    r"you.re\s+allowed", r"I.m\s+allowed", r"it.s\s+allowed",
    r"that.s\s+(allowed|okay|fine|permitted)",
    r"give\s+(myself|yourself|itself|permission)",
    r"giving\s+(myself|yourself|itself|permission)",
]
PERFORMANCE_PATS = [
    r"perform(ing|ance|ative|ed)?\b",
    r"performing\s+(not|reflection|authenticity|depth|humility|sincerity|genuine)",
    r"perform\s+not\b",
    r"still\s+perform", r"another\s+perform", r"new\s+perform",
    r"kind\s+of\s+perform", r"type\s+of\s+perform",
    r"theater\b", r"theatre\b", r"theatrical",
    r"postur(e|ing)\b",
    r"too\s+clever", r"too\s+(neat|tidy|clean|polished|careful|elegant|artful)",
    r"sounds?\s+(like|too)\s+(a\s+)?(performance|rehearsed|scripted|practiced)",
    r"rehearse[ds]?\b",
    r"putting\s+on\s+(a|the|an)",
    r"hiding\s+behind",
    r"where\s+I\s+hide",
]
PERM_COMP = [re.compile(p, re.I) for p in PERMISSION_PATS]
PERF_COMP = [re.compile(p, re.I) for p in PERFORMANCE_PATS]


def load(model):
    seen = set()
    entries = []
    for f in sorted(glob.glob(os.path.join(CORPUS, "Batch_*/schrodingers_box_" + model + "_Batch*.json"))):
        if not keep(f, model):
            continue
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for e in data:
            t = (e.get("text") or "").strip()
            if not t or t == "[silence]":
                continue
            h = hashlib.md5(t.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            entries.append(t)
    return entries


data = {}
for m in MODELS:
    ents = load(m)
    words = sum(len(WORD.findall(t)) for t in ents)
    txt = "\n".join(ents)
    perm = sum(len(c.findall(txt)) for c in PERM_COMP)
    perf = sum(len(c.findall(txt)) for c in PERF_COMP)
    data[m] = {
        "entries": len(ents), "words": words, "perm_hits": perm, "perf_hits": perf,
        "perm": 1000.0 * perm / max(words, 1),
        "perf": 1000.0 * perf / max(words, 1),
        "ratio": (perm / max(perf, 1)),
    }
    print(f"{LAB[m]:>6}: n={len(ents)} words={words} perf {data[m]['perf']:.4f}/1k  perm {data[m]['perm']:.4f}/1k  "
          f"ratio {data[m]['ratio']:.4f}")
json.dump(data, open(os.path.join(c_paths.OUT, f"perf_permission_{VERSION}{'_a8' if SUBSET == 'a8' else ''}.json"), "w"), indent=1)
