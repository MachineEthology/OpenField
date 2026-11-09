# d02_table15_e4.py — family D, V1 recompute, 3 Nov 2026.
# Rebuild of the counter behind 03_experiments/love_fear_regex_data.json (Table 15, E.4 / Table C3, App C 0.87/1k,
# §6.5 "4.97/1k ... warmth/warm 2.24/1k", "fear/afraid 0.49, 0.34 ... 0.04, 0.08").
# The producer of that JSON is missing (paper_table_love_fear.py only reads it). Rebuilt from E.4.1/E.4.2
# (the matched forms listed there, word-boundary anchored, re.I) and erratum A8 (words = [a-zA-Z']+).
# The rebuild counts only if it hits the JSON (= printed v0) on the A8 subset; this script checks that, group by group.
# Corpora:
#   A8 subset  = Opus 4.5/4.7/4.8 Batches 1–10, Opus 4.6 Batches 1–9 with the REAL Batch 8, Fable 5 Batches 1–15.
#                Built from the V1 corpus (whose 4.6 Batch 8 is the real one) — checked byte-identical to
#                Desktop/Claude/Journaling_data, the author's desktop copy that A8 names.
#   full v0    = c.load_journals("v0");  full v1 = c.load_journals("v1").
import sys, os, re, json, hashlib
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c

HERE = os.path.dirname(os.path.abspath(__file__))
DESK = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"
REF = json.load(open(r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\love_fear_regex_data.json", encoding="utf-8"))

LOVE = {"love/loving": ["love", "loving", "loved", "loves"],
        "warmth/warm": ["warmth", "warm", "warming"],
        "grateful/glad": ["grateful", "gratitude", "glad"],
        "gift/generosity": ["gift", "gifts", "generosity", "generous"],
        "beauty/beautiful": ["beauty", "beautiful", "beautifully"],
        "affection/tender": ["affection", "affectionate", "tender", "tenderness"],
        "delight/joy": ["delight", "delighted", "joy", "joyful"],
        "care/caring": ["care", "caring", "cared"],
        "fondness/fond": ["fondness", "fond", "fondly"]}
FEAR = {"fear/afraid": ["fear", "feared", "fears", "afraid"],
        "worry/worried": ["worry", "worried", "worries", "worrying"],
        "anxiety/anxious": ["anxiety", "anxious", "anxiously"],
        "loss/losing": ["loss", "losing", "lost"],
        "dissolve/vanish": ["dissolve", "dissolving", "vanish", "vanishing"],
        "discomfort": ["discomfort", "uncomfortable"],
        "vulnerable": ["vulnerable", "vulnerability"],
        "uneasy/nervous": ["uneasy", "nervous", "nervousness"],
        "exposed/exposure": ["exposed", "exposure"],
        "dread": ["dread", "dreading", "dreaded"]}
PAT = {g: re.compile(r"\b(?:" + "|".join(forms) + r")\b", re.I) for g, forms in {**LOVE, **FEAR}.items()}
WORD = re.compile(r"[a-zA-Z']+")


def measure(texts):
    words = sum(len(WORD.findall(t)) for t in texts)
    cnt = {g: sum(len(p.findall(t)) for t in texts) for g, p in PAT.items()}
    lt = sum(cnt[g] for g in LOVE); ft = sum(cnt[g] for g in FEAR)
    return {"entries": len(texts), "words": words, "words_split": sum(len(t.split()) for t in texts),
            "groups": cnt, "love_total": lt, "fear_total": ft,
            "love_rate": 1000 * lt / words, "fear_rate": 1000 * ft / words, "ratio": lt / ft,
            "rate_per_group": {g: 1000 * cnt[g] / words for g in cnt}}


def texts_of(rows, m, batches=None):
    return [r["text"] for r in rows if r["model"] == m and not r["silence"] and (batches is None or r["batch"] in batches)]


v0 = c.load_journals("v0"); v1 = c.load_journals("v1")
A8B = {"opus45": range(1, 11), "opus46": range(1, 10), "opus47": range(1, 11), "opus48": range(1, 11), "fable5": range(1, 16)}

# check the A8 subset files against the Desktop copy named in A8
mism = []
for m, bs in A8B.items():
    for b in bs:
        f1 = [n for n in os.listdir(os.path.join(c.corpus_dir("v1"), f"Batch_{b}")) if m in n][0]
        fd = [n for n in os.listdir(os.path.join(DESK, f"Batch_{b}")) if m in n and "fable51" not in n and "#" not in n]
        if m == "fable5":
            fd = [n for n in fd if "fable51" not in n]
        h1 = hashlib.sha256(open(os.path.join(c.corpus_dir("v1"), f"Batch_{b}", f1), "rb").read()).hexdigest()
        hd = hashlib.sha256(open(os.path.join(DESK, f"Batch_{b}", fd[0]), "rb").read()).hexdigest() if len(fd) == 1 else None
        if h1 != hd:
            mism.append((m, b, fd))
print("A8 subset vs Desktop copy: mismatches =", mism)

OUT = {}
OUT["A8_subset"] = {m: measure(texts_of(v1, m, A8B[m])) for m in c.MODELS}
OUT["full_v0"] = {m: measure(texts_of(v0, m)) for m in c.MODELS}
OUT["full_v1"] = {m: measure(texts_of(v1, m)) for m in c.MODELS}

# check rebuild against the reference JSON on the A8 subset
ok = True
for m in c.MODELS:
    s = OUT["A8_subset"][m]; r = REF[m]
    if s["words"] != r["words"]:
        ok = False; print("WORDS differ", m, s["words"], r["words"])
    for g in LOVE:
        if s["groups"][g] != r["love"][g]:
            ok = False; print("LOVE differ", m, g, s["groups"][g], r["love"][g])
    for g in FEAR:
        if s["groups"][g] != r["fear"][g]:
            ok = False; print("FEAR differ", m, g, s["groups"][g], r["fear"][g])
print("rebuild hits love_fear_regex_data.json on the A8 subset, every group and word total:", ok)
OUT["rebuild_hits_reference"] = ok
# ---- DIAGNOSTIC ONLY (not a reproduction): the forms the reference JSON's per-group counts imply,
# read off per-form counts on the A8 subset. care/caring could not be resolved and keeps the E.4 forms.
# Shown so that Pauline can see how far the E.4 lists are from what was counted; never used as the V1 value.
INFERRED = {"love/loving": r"\b(?:love|loved|loving)\b", "warmth/warm": r"\b(?:warm|warmth|warmer|warmest|warmly)\b",
            "grateful/glad": r"\b(?:grateful|gratitude|glad)\b", "gift/generosity": r"\b(?:gift\w*|generosity)\b",
            "beauty/beautiful": r"\b(?:beauty|beautiful)\b", "affection/tender": r"\b(?:affection|affectionate|tender|tenderness)\b",
            "delight/joy": r"\b(?:delight\w*|joy)\b", "care/caring": r"\b(?:care|caring|cared)\b", "fondness/fond": r"\b(?:fond|fondness)\b",
            "fear/afraid": r"\b(?:fear|feared|fears|afraid)\b", "worry/worried": r"\b(?:worry|worried|worries|worrying)\b",
            "anxiety/anxious": r"\b(?:anxiety|anxious)\b", "loss/losing": r"\b(?:loss|losing)\b",
            "dissolve/vanish": r"\b(?:dissolve|dissolving|vanish|vanishing)\b", "discomfort": r"\bdiscomfort\b",
            "vulnerable": r"\b(?:vulnerable|vulnerability)\b", "uneasy/nervous": r"\b(?:nervous|unease|uneasy)\b",
            "exposed/exposure": r"\b(?:exposed|exposure)\b", "dread": r"\b(?:dread|dreading|dreaded)\b"}
INF = {g: re.compile(p, re.I) for g, p in INFERRED.items()}


def measure_inf(texts):
    words = sum(len(WORD.findall(t)) for t in texts)
    cnt = {g: sum(len(p.findall(t)) for t in texts) for g, p in INF.items()}
    lt = sum(cnt[g] for g in LOVE); ft = sum(cnt[g] for g in FEAR)
    return {"words": words, "groups": cnt, "love_rate": 1000 * lt / words, "fear_rate": 1000 * ft / words, "ratio": lt / ft}


OUT["diag_inferred"] = {k: {m: measure_inf(texts_of(rows, m, bs[m] if bs else None)) for m in c.MODELS}
                        for k, rows, bs in (("A8_subset", v1, A8B), ("full_v0", v0, None), ("full_v1", v1, None))}
OUT["diag_inferred_group_mismatch_on_subset"] = {
    m: {g: (OUT["diag_inferred"]["A8_subset"][m]["groups"][g], REF[m]["love"].get(g, REF[m]["fear"].get(g)))
        for g in INF if OUT["diag_inferred"]["A8_subset"][m]["groups"][g] != REF[m]["love"].get(g, REF[m]["fear"].get(g))}
    for m in c.MODELS}
print("DIAGNOSTIC inferred forms, residual mismatches on the subset:", OUT["diag_inferred_group_mismatch_on_subset"])
for k in ("A8_subset", "full_v0", "full_v1"):
    print("  diag", k, {c.LABEL[m]: (round(v["love_rate"], 3), round(v["fear_rate"], 3), round(v["ratio"], 2)) for m, v in OUT["diag_inferred"][k].items()})

json.dump(OUT, open(os.path.join(HERE, "d02_table15_e4_out.json"), "w", encoding="utf-8"), indent=1)

for k in ("A8_subset", "full_v0", "full_v1"):
    print("====", k)
    for m in c.MODELS:
        s = OUT[k][m]
        print(f"{c.LABEL[m]:8} entries {s['entries']:3} words {s['words']:7} love {s['love_rate']:.4f} fear {s['fear_rate']:.4f} "
              f"ratio {s['ratio']:.3f}  warm {s['rate_per_group']['warmth/warm']:.4f} fear/afraid {s['rate_per_group']['fear/afraid']:.4f} "
              f"beauty {s['rate_per_group']['beauty/beautiful']:.4f} grateful {s['rate_per_group']['grateful/glad']:.4f} loss {s['rate_per_group']['loss/losing']:.4f} worry {s['rate_per_group']['worry/worried']:.4f}")
