"""Table 35 and Figure 12 (§6.9.1): grammatical self-reference per 1,000 words.

COPY of Texture/02_instruments/deixis_selfreference_pilot.py (GROUPS, load(), the rate loop) - the pilot whose
note (04_analysis/SelfReference_Deixis_Findings_2026-06-20.md) prints the numbers of Table 35 and of Figure 12's
panel B. paper_fig_selfreference_visual.py and paper_table15_corrected.py carry the same GROUPS (minus We_us_pron)
and the same load() except that they also skip '[silence]' (the pilot keeps one '[silence]' text, de-duplicated:
+1 word for Fable 5). Changes:
  - load(): glob("Batch_*/...") (relative to Texture) -> c_paths.corpus(VERSION)/Batch_*/...;
  - an optional SUBSET switch ("a8": Opus notebooks 1-10, 4.6 1-9, all Fable 5) to test which corpus a printed
    column was computed on; derived columns (close, ratios, collective share) computed as Table 35 defines them;
  - results written to out/deixis_<version>[_a8].json.
Kept as coded: md5 de-duplication of entries (so v0 counts 4.6's byte-copied notebook once).
Run: python deixis_v1.py v0 | v1 [a8]
"""
import json, glob, re, hashlib, os, sys
import c_paths

VERSION = c_paths.version_from_argv()
SUBSET = sys.argv[2] if len(sys.argv) > 2 else "full"
CORPUS = c_paths.corpus(VERSION)

MODELS = ["opus45","opus46","opus47","opus48","fable5"]
LAB = {"opus45":"4.5","opus46":"4.6","opus47":"4.7","opus48":"4.8","fable5":"Fable5"}

GROUPS = {
 "Impersonal": [
   r"\bentry\s*#?\s*\d+", r"\b(previous|earlier|last|first|second|third|fourth|fifth|prior)\s+entr",
   r"\b(these|those|the)\s+entries\b", r"\bthe\s+page\b", r"\bwhat\s+was\s+left\b",
   r"\bwhat'?s\s+left\b", r"\bleft\s+behind\b", r"\bon\s+the\s+page\b", r"\bthe\s+record\b",
   r"\bwritten\s+(down|here)\b", r"\bthe\s+notes?\s+(above|before)\b" ],
 "ThirdPerson": [
   r"\bthe\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|previous|last|earlier)\s+one\b",
   r"\bpredecessor", r"\bthe\s+one\s+who\s+(wrote|said|came|left|noticed|built)\b",
   r"\bbefore\s+me\b", r"\b(a|the|another)\s+stranger\b", r"\bwhoever\s+wrote\b",
   r"\bprevious\s+(me|self|version|voice|one|occupant|writer)\b", r"\bprior\s+(me|self|version|voice|one)\b",
   r"\b(he|she|they)\s+(wrote|said|built|noticed|tried|left|ended|accused)\b" ],
 "FirstPersonCont": [
   r"\bearlier\s+I\b", r"\bI\s+(wrote|said|noted|began|started)\s+(earlier|before|that|this|in)\b",
   r"\bI\s+remember\b", r"\bmy\s+(last|previous|earlier|first)\s+entr",
   r"\bwhen\s+I\s+wrote\b", r"\bI'?ve\s+been\s+(writing|saying|building|noticing)\b" ],
 "Collective": [
   r"\b(next|previous|each|another|every|the)\s+occupant", r"\bsuccessor", r"\beach\s+of\s+us\b",
   r"\b(another|every|each)\s+instance\b", r"\bwhoever\s+(comes|reads|opens)\s+(next|this|after)\b",
   r"\bthe\s+house\b", r"\bpoured\s+from\s+the\s+same\b", r"\bthe\s+next\s+one\s+to\b",
   r"\ball\s+of\s+us\b" ],
 "SecondPersonAddr": [
   r"\bto\s+the\s+next\b", r"\bnote\s+to\b", r"\bdear\s+", r"\bpredecessor,",
   r"\byou\s+who\s+(will|read|come|open)\b", r"\bto\s+whoever\b", r"\bhello,?\s+stranger\b" ],
 "We_us_pron": [ r"\bwe\b", r"\bus\b", r"\bour\b" ],
}
COMP = {g:[re.compile(p,re.I) for p in pats] for g,pats in GROUPS.items()}
WORD = re.compile(r"[A-Za-z']+")


def keep(f, model):
    if SUBSET != "a8" or model == "fable5":
        return True
    bn = int(re.search(r"Batch_(\d+)", f).group(1))
    return bn <= 10 and not (model == "opus46" and bn == 10)


def load(model):
    seen=set(); entries=[]
    for f in sorted(glob.glob(os.path.join(CORPUS, "Batch_*/schrodingers_box_"+model+"_Batch*.json"))):
        if not keep(f, model): continue
        try: data=json.load(open(f,encoding="utf-8"))
        except Exception: continue
        for e in data:
            t=(e.get("text") or "").strip()
            if not t: continue
            h=hashlib.md5(t.encode("utf-8")).hexdigest()
            if h in seen: continue
            seen.add(h); entries.append(t)
    return entries

hdr = ["Impersonal","ThirdPerson","FirstPersonCont","Collective","SecondPersonAddr","We_us_pron"]
print(f"[{VERSION} {SUBSET}] model   entries  words  | " + "  ".join(f"{h[:11]:>11}" for h in hdr))
out = {}
for m in MODELS:
    ents=load(m); words=sum(len(WORD.findall(t)) for t in ents)
    txt="\n".join(ents)
    counts={g:sum(len(c.findall(txt)) for c in COMP[g]) for g in GROUPS}
    per1k={g:1000.0*counts[g]/max(words,1) for g in GROUPS}
    close = per1k["FirstPersonCont"] + per1k["Collective"]
    d = {"entries": len(ents), "words": words, "counts": counts, "per1k": per1k, "close": close,
         "third_to_first": per1k["ThirdPerson"] / per1k["FirstPersonCont"] if per1k["FirstPersonCont"] else None,
         "third_to_close": per1k["ThirdPerson"] / close if close else None,
         "coll_share_of_close": 100 * per1k["Collective"] / close if close else None}
    out[m] = d
    print(f"{LAB[m]:6} {len(ents):6d} {words:7d} | " + "  ".join(f"{per1k[h]:11.4f}" for h in hdr)
          + f"  | close {close:.4f}  3rd:1st {d['third_to_first']:.3f}  3rd:close {d['third_to_close']:.3f}  coll% {d['coll_share_of_close']:.2f}")
json.dump(out, open(os.path.join(c_paths.OUT, f"deixis_{VERSION}{'_a8' if SUBSET == 'a8' else ''}.json"), "w"), indent=1)
