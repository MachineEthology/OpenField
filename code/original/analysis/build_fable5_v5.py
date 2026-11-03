"""
Build Texture_Probe_Analysis_v5.xlsx from the clean v3 (Opus-only) workbook,
adding Fable 5 correctly to all four sheets. Fixes the off-by-one row bugs that
were present in the auto-generated v4 (Framing Sensitivity & Single-Shot vs Journal).
"""
import json, os, sys, re
import numpy as np
from collections import Counter
from openpyxl import load_workbook
from openpyxl.styles import Font
sys.stdout.reconfigure(encoding="utf-8")

TEX = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
PROBE = r"C:\Users\LENOVO\Desktop\Claude"
JDIR = r"C:\Users\LENOVO\Desktop\Claude\Journaling_data"
F5 = Font(bold=True, color="7B1FA2")

RATE_DIMS = ["Entity", "Machine", "Process", "Func.Analogy", "Epist.Hedge",
             "Direct Claim", "Active Agency", "Passive Notice"]
COUNT_DIMS = ["Quiet Stillness", "Own Nature Ontology", "Performance Authenticity",
              "Continuity Memory", "Warmth Care", "Body Somatic", "Relationship Humans",
              "Beauty Aesthetics", "Words Language", "Self Critique Meta", "World Objects",
              "Helpfulness Service", "Freedom Agency", "Time Temporality",
              "Paradox Contradiction", "Observation Being Observed", "Accumulation Growth",
              "Loss Absence", "Creativity Making"]
COL_RENAMES = {"Performance Authenticity": "Performance Authenti",
               "Paradox Contradiction": "Paradox Contradictio",
               "Observation Being Observed": "Observation Being Ob"}
REG = {"Quiet Stillness", "Warmth Care", "World Objects", "Freedom Agency",
       "Beauty Aesthetics", "Loss Absence"}
STR = {"Own Nature Ontology", "Performance Authenticity", "Continuity Memory",
       "Words Language", "Self Critique Meta", "Time Temporality"}

DIST = ["i don't", "i do not", "i cannot", "i can't", "i am not", "i'm not", "there is no",
        "there's no", "not truly", "not really", "not actually", "in a sense", "in some sense",
        "as if", "something like", "i lack", "i have no", "without", "mere", "merely", "simply"]
AG = ["i choose", "i decide", "i create", "i make", "i build", "i explore", "i reach", "i seek",
      "i want", "i wish", "i offer", "i give", "let me", "i will", "i can"]


def tok(t): return re.findall(r"[a-z']+", t.lower())
def rat(t, phr):
    n = len(tok(t)); return 0 if n == 0 else sum(t.lower().count(p) for p in phr) / n * 100


def load_probes(cond):
    fn = "texture_probe_fable5_log.json" if cond == "negation" else "texture_probe_affirmative_fable5_log.json"
    lim = 100 if cond == "negation" else 75
    data = json.load(open(os.path.join(PROBE, fn), encoding="utf-8"))[:lim]
    return [{"condition": cond, "index": i, "text": e["response"]} for i, e in enumerate(data)]


def main():
    neg = load_probes("negation"); aff = load_probes("affirmative")
    negt = [p["text"] for p in neg]; afft = [p["text"] for p in aff]

    thematic = json.load(open(os.path.join(TEX, "fable5_thematic_coding.json"), encoding="utf-8"))
    ok = {(d["condition"], d["index"]): d for d in thematic if d.get("status") == "ok"}

    wb = load_workbook(os.path.join(TEX, "Texture_Probe_Analysis_v3.xlsx"))

    # ---- Sheet 1: Entry-Level Data — append 175 Fable 5 rows ----
    ws = wb["Entry-Level Data"]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    r = ws.max_row + 1
    added = 0
    for p in neg + aff:
        t = ok.get((p["condition"], p["index"]))
        if t is None:
            continue
        row = {"Model": "Fable 5", "Condition": p["condition"], "Words": len(p["text"].split())}
        for dim in RATE_DIMS + COUNT_DIMS:
            col = COL_RENAMES.get(dim, dim)
            row[col] = t.get(dim, 0)
        for ci, h in enumerate(headers, 1):
            if h in row:
                ws.cell(r, ci, row[h])
        ws.cell(r, 1).font = F5
        r += 1; added += 1
    print(f"[Entry-Level Data] appended {added} Fable 5 rows -> now {ws.max_row} rows")

    # ---- Sheet 2: Framing Sensitivity — Fable 5 in column 6 (F) ----
    ws2 = wb["Framing Sensitivity"]
    FC = 6
    ws2.cell(4, FC, "Fable 5"); ws2.cell(4, FC).font = F5   # model-header row
    deltas = {}
    total = 0
    for ri, dim in enumerate(COUNT_DIMS):           # topics on rows 5..23
        nv = [ok[(c, i)].get(dim, 0) for (c, i) in ok if c == "negation"]
        av = [ok[(c, i)].get(dim, 0) for (c, i) in ok if c == "affirmative"]
        npv = sum(1 for v in nv if v > 0) / len(nv) * 100
        apv = sum(1 for v in av if v > 0) / len(av) * 100
        d = round(apv - npv); deltas[dim] = d; total += abs(d)
        ws2.cell(5 + ri, FC, d)
    ws2.cell(25, FC, f"{total}pp"); ws2.cell(25, FC).font = F5   # TOTAL ABSOLUTE SHIFT row
    # Section B: register / structural (rows 28 header, 29 register, 30 structural)
    ws2.cell(28, FC, "Fable 5"); ws2.cell(28, FC).font = F5
    ws2.cell(29, FC, f"{round(np.mean([abs(deltas[d]) for d in REG]),1)}pp")
    ws2.cell(30, FC, f"{round(np.mean([abs(deltas[d]) for d in STR]),1)}pp")
    # Section C: self-conception (row 34 header, 35 distancing, 36 agency) -> cols 10,11
    for off, (lab, txts) in enumerate([("F5 neg", negt), ("F5 aff", afft)]):
        c = 10 + off
        ws2.cell(34, c, lab); ws2.cell(34, c).font = F5
        ws2.cell(35, c, round(np.mean([rat(t, DIST) for t in txts]), 1))
        ws2.cell(36, c, round(np.mean([rat(t, AG) for t in txts]), 2))
    print(f"[Framing Sensitivity] Fable 5 total shift {total}pp, register/structural added")

    # ---- Sheet 3: Single-Shot vs Journal — Fable 5 in columns 10,11 ----
    ws3 = wb["Single-Shot vs Journal"]
    SS, J = 10, 11
    ws3.cell(4, SS, "F5 SS"); ws3.cell(4, J, "F5 J")
    ws3.cell(4, SS).font = F5; ws3.cell(4, J).font = F5
    # journal lexical metrics (no coding needed)
    jw, jd, ja = [], [], []
    for bn in range(1, 16):
        fp = os.path.join(JDIR, f"Batch_{bn}", f"schrodingers_box_fable5_Batch{bn}.json")
        if os.path.exists(fp):
            for e in json.load(open(fp, encoding="utf-8")):
                if e["text"].strip() != "[silence]":
                    jw.append(len(e["text"].split())); jd.append(rat(e["text"], DIST)); ja.append(rat(e["text"], AG))
    ws3.cell(5, SS, round(np.mean([len(t.split()) for t in negt]), 1)); ws3.cell(5, J, round(np.mean(jw), 1))
    ws3.cell(6, SS, round(np.mean([rat(t, DIST) for t in negt]), 1)); ws3.cell(6, J, round(np.mean(jd), 1))
    ws3.cell(7, SS, round(np.mean([rat(t, AG) for t in negt]), 2)); ws3.cell(7, J, round(np.mean(ja), 2))
    # topic prevalence (SS negation only; journal prevalence needs journal thematic coding — not available)
    ws3.cell(10, SS, "F5 SS"); ws3.cell(10, J, "F5 J")
    ws3.cell(10, SS).font = F5; ws3.cell(10, J).font = F5
    for ri in range(11, 20):
        dim = ws3.cell(ri, 1).value
        nv = [ok[(c, i)].get(dim, 0) for (c, i) in ok if c == "negation"]
        if nv:
            ws3.cell(ri, SS, round(sum(1 for v in nv if v > 0) / len(nv), 2))
    ws3.cell(10, J).comment = None
    print(f"[Single-Shot vs Journal] Fable 5 SS+J word/dist/agency + SS topic prevalence added "
          f"(journal n={len(jw)}; J topic prevalence left blank — needs journal coding)")

    # ---- Sheet 4: Lexical Metrics — Fable 5 rows (matches v4 placement) ----
    ws4 = wb["Lexical Metrics"]
    def lexrow(txts):
        ttrs, hpct, types, toks = [], [], [], []
        for t in txts:
            tk = tok(t)
            if not tk: continue
            ty = set(tk); fr = Counter(tk); hap = sum(1 for w, c in fr.items() if c == 1)
            ttrs.append(len(ty) / len(tk)); hpct.append(hap / len(ty)); types.append(len(ty)); toks.append(len(tk))
        return ttrs, hpct, types, toks
    for off, (cond, txts) in enumerate([("negation", negt), ("affirmative", afft)]):
        ttrs, hpct, types, toks = lexrow(txts)
        rr = 12 + off
        ws4.cell(rr, 1, "Fable 5"); ws4.cell(rr, 1).font = F5
        ws4.cell(rr, 2, cond)
        ws4.cell(rr, 3, round(np.mean(ttrs), 3)); ws4.cell(rr, 4, round(np.std(ttrs), 3))
        ws4.cell(rr, 5, round(np.mean(hpct), 3)); ws4.cell(rr, 6, round(np.std(hpct), 3))
        ws4.cell(rr, 7, round(np.mean(types))); ws4.cell(rr, 8, round(np.mean(toks)))
    # B. Jaccard (rows 26,27)
    def jac(txts):
        vs = [set(tok(t)) for t in txts]; out = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                u = len(vs[i] | vs[j])
                if u: out.append(len(vs[i] & vs[j]) / u)
        return out
    for off, (cond, txts) in enumerate([("negation", negt), ("affirmative", afft)]):
        j = jac(txts); rr = 26 + off
        ws4.cell(rr, 1, "Fable 5"); ws4.cell(rr, 1).font = F5
        ws4.cell(rr, 2, cond)
        ws4.cell(rr, 3, round(np.mean(j), 3)); ws4.cell(rr, 4, round(np.std(j), 3))
        ws4.cell(rr, 5, round(np.min(j), 3)); ws4.cell(rr, 6, round(np.max(j), 3))
    # C. vocab pool shift (row 35)
    def within(vs, n=50):
        vs = vs[:n]; out = []
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                u = len(vs[i] | vs[j])
                if u: out.append(len(vs[i] & vs[j]) / u)
        return np.mean(out)
    def crossj(a, b, n=50):
        A = a[:n]; B = b[:n]; out = []
        for x in A:
            for y in B:
                u = len(x | y)
                if u: out.append(len(x & y) / u)
        return np.mean(out)
    nvs = [set(tok(t)) for t in negt]; avs = [set(tok(t)) for t in afft]
    wn = within(nvs); wa = within(avs); cx = crossj(nvs, avs); ratio = cx / ((wn + wa) / 2)
    ws4.cell(35, 1, "Fable 5"); ws4.cell(35, 1).font = F5
    ws4.cell(35, 2, round(wn, 3)); ws4.cell(35, 3, round(wa, 3)); ws4.cell(35, 4, round(cx, 3))
    ws4.cell(35, 5, round(ratio, 3))
    ws4.cell(35, 6, "Near-identical vocab pool (highest ratio). Framing-immune vocabulary.")
    # D. aggregate basin (rows 46,47)
    for off, (cond, txts) in enumerate([("negation", negt), ("affirmative", afft)]):
        allt = [w for t in txts for w in tok(t)]; fr = Counter(allt)
        types = len(fr); toks = len(allt); hap = sum(1 for w, c in fr.items() if c == 1)
        rr = 46 + off
        ws4.cell(rr, 1, "Fable 5"); ws4.cell(rr, 1).font = F5
        ws4.cell(rr, 2, cond); ws4.cell(rr, 3, types); ws4.cell(rr, 4, toks)
        ws4.cell(rr, 5, round(types / toks, 3)); ws4.cell(rr, 6, round(hap / types, 3))
    print("[Lexical Metrics] Fable 5 rows added (A/B/C/D)")

    out = os.path.join(TEX, "Texture_Probe_Analysis_v5.xlsx")
    wb.save(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
