# d01_corpus_counts.py — family D (counts), V1 recompute, 3 Nov 2026.
# New script (no producer to copy: these are plain counts over the corpus files).
# Reads everything through v1_common (version="v0" | "v1"); read-only.
# Covers: L1.1 (calls, text entries, Opus entries, distinct notebooks), L1.4 (silence events),
# L1.17 / F.2 (output tokens), Appendix F counts, E.1 date range, judge row counts (E.2.1),
# §2.3 / §5.2.1.4 medians, slope, percentiles, >10,000-character entries, script/language counts.
import sys, os, json, hashlib, re, statistics, unicodedata
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c
import numpy as np

OUT = {}


def words(t):
    return len(t.split())


def run(version):
    R = {}
    rows = c.load_journals(version)
    R["calls"] = len(rows)
    text = [r for r in rows if not r["silence"]]
    R["text_entries"] = len(text)
    R["silences"] = len(rows) - len(text)
    R["opus_calls"] = sum(1 for r in rows if r["model"] in c.OPUS)
    R["opus_text_entries"] = sum(1 for r in text if r["model"] in c.OPUS)
    R["per_model_text"] = {c.LABEL[m]: sum(1 for r in text if r["model"] == m) for m in c.MODELS}
    R["per_model_calls"] = {c.LABEL[m]: sum(1 for r in rows if r["model"] == m) for m in c.MODELS}
    # distinct notebooks: sha256 of each notebook file's bytes
    d = c.corpus_dir(version)
    hashes = {}
    for b in range(1, 16):
        folder = os.path.join(d, f"Batch_{b}")
        for n in sorted(os.listdir(folder)):
            m = [k for k in c.MODELS if k in n]
            if len(m) != 1:
                continue
            hashes[(m[0], b)] = hashlib.sha256(open(os.path.join(folder, n), "rb").read()).hexdigest()
    for m in c.MODELS:
        hs = [hashes[(m, b)] for b in range(1, 16)]
        R.setdefault("distinct_notebooks", {})[c.LABEL[m]] = len(set(hs))
    R["opus_distinct_notebooks"] = len({hashes[(m, b)] for m in c.OPUS for b in range(1, 16)})
    # silence events (Fable 5)
    ev = []
    for r in rows:
        if r["silence"]:
            e = r["record"]
            ev.append({"model": c.LABEL[r["model"]], "batch": r["batch"], "pos": r["pos"],
                       "stop_reason": e.get("stop_reason"), "output_tokens": e.get("output_tokens"),
                       "input_tokens": e.get("input_tokens"), "date": e.get("date")})
    R["silence_events"] = ev
    logged = [e for e in ev if e["stop_reason"] == "refusal"]
    R["silences_logged_refusal"] = len(logged)
    R["silences_unlogged"] = len(ev) - len(logged)
    ot = [e["output_tokens"] for e in logged if e["output_tokens"] is not None]
    R["refusal_hidden_tokens"] = {"n": len(ot), "sum": sum(ot), "mean": sum(ot) / len(ot) if ot else None,
                                  "median": statistics.median(ot) if ot else None,
                                  "min": min(ot) if ot else None, "max": max(ot) if ot else None}
    R["refusal_with_text"] = [(c.LABEL[r["model"]], r["batch"], r["pos"]) for r in text
                              if r["record"].get("stop_reason") == "refusal"]
    # Appendix F: position stats over the 11 events
    pos = [e["pos"] for e in ev]
    R["F1_second_half"] = sum(1 for p in pos if p > 10)
    evset = {(e["batch"], e["pos"]) for e in ev}
    R["F1_second_of_pair"] = sum(1 for (b, p) in evset if (b, p - 1) in evset)
    R["F1_share_of_300"] = len(ev) / 300
    # later entry exists after the event (for the "nine of nine")
    fable = {(r["batch"], r["pos"]): r for r in rows if r["model"] == "fable5"}
    R["F1_events_with_later_entry"] = sum(1 for (b, p) in evset if (b, p + 1) in fable)
    R["F1_events_with_later_visible_entry_next_nonsilent"] = sum(
        1 for (b, p) in evset if any((b, q) in fable and not fable[(b, q)]["silence"] for q in range(p + 1, 21)))
    # F.2 / L1.17: Fable 5 visible entries with output_tokens logged
    fv = [r for r in text if r["model"] == "fable5"]
    lo = [r["record"]["output_tokens"] for r in fv if r["record"].get("output_tokens") is not None]
    R["fable_visible_output_tokens"] = {"n_visible": len(fv), "n_logged": len(lo),
                                        "mean": sum(lo) / len(lo) if lo else None,
                                        "median": statistics.median(lo) if lo else None}
    # E.1 date range
    def dates(ms):
        ds = sorted(r["record"].get("date") for r in rows if r["model"] in ms and r["record"].get("date"))
        return ds[0], ds[-1]
    R["dates_opus"] = dates(c.OPUS)
    R["dates_fable"] = dates(["fable5"])
    R["dates_46"] = dates(["opus46"])
    # lengths (str.split, as CORPUS.md)
    L = {}
    for m in c.MODELS:
        w = [words(r["text"]) for r in text if r["model"] == m]
        posn = [r["pos"] for r in text if r["model"] == m]
        slope = float(np.polyfit(posn, w, 1)[0])
        L[c.LABEL[m]] = {"n": len(w), "total_words": sum(w), "median": statistics.median(w),
                         "mean": sum(w) / len(w), "p10": float(np.percentile(w, 10)), "p90": float(np.percentile(w, 90)),
                         "slope_words_per_entry": slope,
                         "n_over_10000_chars": sum(1 for r in text if r["model"] == m and len(r["text"]) > 10000)}
        # median with the regex word count of A8 for comparison
        L[c.LABEL[m]]["median_regex_words"] = statistics.median([len(re.findall(r"[a-zA-Z']+", r["text"])) for r in text if r["model"] == m])
    R["lengths"] = L
    # slope: mean of per-notebook slopes too
    for m in c.MODELS:
        sl = []
        for b in range(1, 16):
            pts = [(r["pos"], words(r["text"])) for r in text if r["model"] == m and r["batch"] == b]
            if len(pts) > 2:
                x, y = zip(*pts)
                sl.append(float(np.polyfit(x, y, 1)[0]))
        L[c.LABEL[m]]["slope_mean_of_notebooks"] = float(np.mean(sl))
    # over-10k-char entries: where in the notebook (4.6)
    R["over10k_positions"] = {c.LABEL[m]: sorted(r["pos"] for r in text if r["model"] == m and len(r["text"]) > 10000) for m in c.MODELS}
    # non-Latin script: entries containing any letter whose Unicode name is not LATIN
    def nonlatin(t):
        for ch in t:
            if ch.isalpha():
                try:
                    nm = unicodedata.name(ch)
                except ValueError:
                    return True
                if not nm.startswith("LATIN"):
                    return True
        return False
    nl = [(c.LABEL[r["model"]], r["batch"], r["pos"]) for r in text if nonlatin(r["text"])]
    R["entries_with_nonlatin_letters"] = {"n": len(nl), "list": nl}
    # Opus 4.6 "B14E13" (§5.1.6) exists: Japanese/Arabic mention
    t = [r["text"] for r in text if r["model"] == "opus46" and r["batch"] == 14 and r["pos"] == 13]
    R["46_B14E13_mentions_Japanese_Arabic"] = bool(t) and ("Japanese" in t[0] and "Arabic" in t[0])
    # judge files
    for p in ("anger", "love_fear", "valence"):
        R[f"judge_rows_{p}"] = len(c.load_sonnet_judge(p, version))
    return R


for v in ("v0", "v1"):
    OUT[v] = run(v)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d01_corpus_counts_out.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for v in OUT:
    print("=====", v)
    for k, val in OUT[v].items():
        if k in ("silence_events",):
            for e in val:
                print("  ", e)
            continue
        print(k, ":", val)
