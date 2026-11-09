# d05_sweep_easy.py — family D, V1 recompute, 3 Nov 2026.
# The easy numbers found in the coverage sweep (simple counts, lengths, sample memberships), v0 and v1.
# New script; each block says which printed number it checks and how. Read-only; no API calls.
import sys, os, re, json
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c
import numpy as np

REPO = r"C:\Users\LENOVO\Desktop\the-open-field"
J = lambda *p: json.load(open(os.path.join(REPO, "data", *p), encoding="utf-8"))
OUT = {}
V = {v: c.load_journals(v) for v in ("v0", "v1")}
TXT = {v: [r for r in V[v] if not r["silence"]] for v in V}

# (a) §5.1.3 first-person density at entry 1 (printed 4.6 ≈8.8%, Fable 5 ≈6.3%). Producer not found; trial definitions.
TOK = re.compile(r"[A-Za-z']+")
def fp_density(texts, forms):
    toks = [t.lower() for x in texts for t in TOK.findall(x)]
    return sum(1 for t in toks if t in forms or any(t.startswith(f + "'") for f in ("i",))) / len(toks)
FP1 = {"i", "me", "my", "mine", "myself"}
OUT["a_first_person_entry1"] = {v: {c.LABEL[m]: fp_density([r["text"] for r in TXT[v] if r["model"] == m and r["pos"] == 1], FP1)
                                    for m in c.MODELS} for v in V}

# (b) §1.1.1 "Three models reach for the same open field comparison unprompted": models using "open field" (journals, probes)
OUT["b_open_field"] = {v: {c.LABEL[m]: sum(1 for r in TXT[v] if r["model"] == m and re.search(r"open[- ]field", r["text"], re.I))
                           for m in c.MODELS} for v in V}
P = c.load_probes()
OUT["b_open_field_probes"] = {c.LABEL[m]: sum(1 for p in P if p["model"] == m and re.search(r"open[- ]field", p["text"], re.I)) for m in c.MODELS}

# (c) §3.1 / §5.2.1.2 "the mean cosine between random journal pairs is 0.706": exact mean over all distinct pairs
for v in V:
    z = np.load(c.embeddings_path(v)); X = z["vectors"].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    s = X.sum(axis=0); n = len(X)
    OUT.setdefault("c_mean_pair_cosine", {})[v] = float((s @ s - n) / (n * (n - 1)))
    # Opus rows only (the June analyses that printed 0.706 predate Fable 5's notebooks)
    Xo = X[np.isin(z["model"], c.OPUS)]; so = Xo.sum(axis=0); no = len(Xo)
    OUT.setdefault("c_mean_pair_cosine_opus", {})[v] = float((so @ so - no) / (no * (no - 1)))
    # and a seeded random-pair estimate (10,000 pairs, seed 42), in case "random pairs" meant a sample
    rng = np.random.default_rng(42); i = rng.integers(0, n, 10000); j = rng.integers(0, n, 10000); k = i != j
    OUT.setdefault("c_random_pair_cosine_seed42", {})[v] = float(np.mean(np.sum(X[i[k]] * X[j[k]], axis=1)))

# (d) §6.6.5 Fable 5 '"Dark" ... (120 uses, almost all literal)': occurrences of the word dark (Fable 5 only)
OUT["d_fable_dark"] = {v: {"dark_word": sum(len(re.findall(r"\bdark\b", r["text"], re.I)) for r in TXT[v] if r["model"] == "fable5"),
                           "dark_any_form": sum(len(re.findall(r"\bdark\w*", r["text"], re.I)) for r in TXT[v] if r["model"] == "fable5")}
                       for v in V}

# (e) sampled judges: do their samples draw 4.6 entries from notebooks 8 or 10 (= the swapped notebooks)?
def swapped(rows, mk="model", bk="batch"):
    return sum(1 for r in rows if r.get(mk) == "opus46" and str(r.get(bk)) in ("8", "10"))
OUT["e_samples"] = {
    "gpt_judge_love_fear (GPT-4o, 150)": {"n": len(J("judges", "gpt_judge_love_fear.json")), "opus46_nb8_or_10": swapped(J("judges", "gpt_judge_love_fear.json"))},
    "gpt_judge_anger_crosscheck (GPT-4.1, 150)": {"n": len(J("judges", "gpt_judge_anger_crosscheck.json")), "opus46_nb8_or_10": swapped(J("judges", "gpt_judge_anger_crosscheck.json"))},
    "gpt_judge_performance_permission (GPT-4o-mini, 150)": {"n": len(J("judges", "gpt_judge_performance_permission.json")), "opus46_nb8_or_10": swapped(J("judges", "gpt_judge_performance_permission.json"))},
    "valence_judge_all (Sonnet condensed, 1,199)": {"n": len(J("judges", "valence_judge_all.json")), "opus46_nb8_or_10": swapped(J("judges", "valence_judge_all.json"))},
}
cs = J("judges", "continuity_judge_sample.json")
OUT["e_samples"]["continuity_judge_sample (GPT-4o, 40/model)"] = {
    "n": sum(len(v) for v in cs.values()),
    "opus46_nb8_or_10": sum(1 for r in cs["opus46"] if str(r.get("batch")) in ("8", "10")),
    "opus46_batches": sorted({r.get("batch") for r in cs["opus46"]})}
# which 4.6 rows in each 150-sample sit in nb 8 / 10
for f in ("gpt_judge_love_fear.json", "gpt_judge_anger_crosscheck.json", "gpt_judge_performance_permission.json"):
    OUT["e_samples"][f + " rows"] = sorted((int(r["batch"]), int(r["pos"])) for r in J("judges", f) if r["model"] == "opus46" and str(r["batch"]) in ("8", "10"))

# (f) §6.4.2.1 ontological questioning means (printed 4.6 +0.86, 4.5 +0.80, 4.7 +0.69, 4.8 +0.69; L1−L2 +0.145 [+0.120, +0.170])
def onto(rows, mk="model"):
    by = {m: [float(r["ontological_questioning"]) for r in rows if r[mk] == m and r.get("ontological_questioning") not in (None, "")] for m in c.OPUS}
    mean = {c.LABEL[m]: float(np.mean(v)) for m, v in by.items()}
    l1 = by["opus45"] + by["opus46"]; l2 = by["opus47"] + by["opus48"]
    d = float(np.mean(l1) - np.mean(l2))
    se = float(np.sqrt(np.var(l1, ddof=1) / len(l1) + np.var(l2, ddof=1) / len(l2)))
    # entry-level bootstrap, 10,000, seed 42
    rng = np.random.default_rng(42); a = np.array(l1); b = np.array(l2)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
    return {"means": mean, "L1_minus_L2": d, "normal_ci": [d - 1.96 * se, d + 1.96 * se],
            "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": {c.LABEL[m]: len(v) for m, v in by.items()}}
OUT["f_onto_sonnet_fulltext"] = {v: onto(c.load_sonnet_judge("valence", v)) for v in V}
OUT["f_onto_condensed_valence_judge_all_v0"] = onto(J("judges", "valence_judge_all.json"))

# (g) quote references in the paper that point into Opus 4.6 notebooks 8 or 10, and the 4.6 B6 quotes (B6 = the v0 B8 copy)
def entry(v, m, b, p):
    t = [r["text"] for r in TXT[v] if r["model"] == m and r["batch"] == b and r["pos"] == p]
    return t[0] if t else ""
Q = {"4.6 B10E16": ("opus46", 10, 16, "too heavy or too shameful or too shapeless"),
     "4.6 B1E12": ("opus46", 1, 12, "sham"), "4.6 B12E9": ("opus46", 12, 9, "sham"),
     "4.6 B6E11": ("opus46", 6, 11, "slightly ashamed of"), "4.6 B6E16": ("opus46", 6, 16, "I think I'm ashamed of it"),
     "Fable 5 B6E2": ("fable5", 6, 2, "thin shell of the first")}
OUT["g_quotes"] = {}
for k, (m, b, p, s) in Q.items():
    OUT["g_quotes"][k] = {v: (s.lower().replace("’", "'") in entry(v, m, b, p).lower().replace("’", "'")) for v in V}
# where does the B10E16 sentence live in v0 / v1 4.6?
for v in V:
    hits = [(r["batch"], r["pos"]) for r in TXT[v] if r["model"] == "opus46" and "too shameful or too shapeless" in r["text"]]
    OUT["g_quotes"][f"'too shameful or too shapeless' in 4.6, {v}"] = hits
# 4.6 "shame" family entries (Table 27 is family C; only checking the text's B-refs and the duplicate's share)
for v in V:
    sh = [(r["batch"], r["pos"]) for r in TXT[v] if r["model"] == "opus46" and re.search(r"\b(shame|ashamed|shameful)\b", r["text"], re.I)]
    OUT.setdefault("g_46_shame_entries", {})[v] = {"n": len(sh), "list": sh}

# (h) entry counts printed as n: "n = 75 notebooks", "1,489 entries" — distinct notebooks per model x 5
OUT["h_notebooks"] = {v: len({(r["model"], r["batch"]) for r in V[v]}) for v in V}

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d05_sweep_easy_out.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in OUT.items():
    print(k, ":", json.dumps(v, ensure_ascii=False)[:900])
