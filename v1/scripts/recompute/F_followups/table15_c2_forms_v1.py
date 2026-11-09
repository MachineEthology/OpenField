# table15_c2_forms_v1.py — F_followups, decision 10 (Pauline, 3 Nov 2026): Table 15 counted with exactly the
# forms Appendix C1/C2 (= E.4.1/E.4.2) print, on the FULL corpus, v0 and v1; also on the ERRATA A8 subset.
# Written by Claude Opus 5.5 (Code, F helper), 3 Nov 2026.
#
# Reuse, not rewrite:
#   - the counter is recompute/D_counts/d02_table15_e4.py's measure(): \b(?:forms)\b, re.I, per 1,000 [a-zA-Z']+
#     words, silences excluded. d02 is NOT imported (its top level writes into D_counts); its LOVE/FEAR lists are
#     read from its source with ast and asserted equal to the forms parsed here from the paper's text.
#   - the A8 subset is recompute/C_lexical/table18_fear_regex_rebuild.py's a8_subset() (built on the V1 corpus,
#     whose 4.6 Batch 8 is the real one; 4.6 notebook 10 excluded). Its FEAR_FORMS (the forms that reproduce the
#     stored counts) are also read with ast, so the per-form difference to C2 is shown.
#   - stored counts: Texture/03_experiments/love_fear_regex_data.json (read-only), the counts behind printed v0.
# Forms are parsed from the Observatory's copy of the paper: appendix-c.json (Table C1, C2) and appendix-e.json
# (E.4.1, E.4.2). Printed Table 15 is parsed from main.json (table-15).
# Output: out/table15_c2_forms.json
import ast, html, json, os, re, sys
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.dirname(HERE)
ROOM = r"C:\Users\LENOVO\Desktop\Machine Observatory\site\src\data\papers\open-field"
REF = json.load(open(r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\03_experiments\love_fear_regex_data.json", encoding="utf-8"))
ROWLAB = {"4.5": "opus45", "4.6": "opus46", "4.7": "opus47", "4.8": "opus48", "Fable 5": "fable5"}


def strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def forms_table(h):
    """rows of a two-column 'Group | Matched forms' table -> {group: [forms]} in printed order."""
    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", h, re.S):
        cells = [strip(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) == 2:
            out[cells[0]] = [f.strip() for f in cells[1].split(",")]
    return out


def table_after(h, marker):
    i = h.index(marker); j = h.index("<table", i); k = h.index("</table>", j)
    return h[j:k]


hc = json.load(open(os.path.join(ROOM, "appendix-c.json"), encoding="utf-8"))["html"]
he = json.load(open(os.path.join(ROOM, "appendix-e.json"), encoding="utf-8"))["html"]
C1 = forms_table(hc[hc.index('id="table-c1"'):hc.index("</table>", hc.index('id="table-c1"'))])
C2 = forms_table(hc[hc.index('id="table-c2"'):hc.index("</table>", hc.index('id="table-c2"'))])
E41 = forms_table(table_after(he, 'id="e-4-1"'))
E42 = forms_table(table_after(he, 'id="e-4-2"'))
assert C1 == E41 and C2 == E42, "Appendix C and E.4 print different forms"
assert len(C1) == 9 and len(C2) == 10


def ast_dicts(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read())
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            got[node.targets[0].id] = ast.literal_eval(node.value)
    return got


d02 = ast_dicts(os.path.join(REC, "D_counts", "d02_table15_e4.py"), {"LOVE", "FEAR"})
t18 = ast_dicts(os.path.join(REC, "C_lexical", "table18_fear_regex_rebuild.py"), {"FEAR_FORMS"})["FEAR_FORMS"]
assert d02["LOVE"] == C1 and d02["FEAR"] == C2, "d02's lists are not the printed C1/C2 forms"
LOVE, FEAR = C1, C2

WORD = re.compile(r"[a-zA-Z']+")
PAT = {g: re.compile(r"\b(?:" + "|".join(f) + r")\b", re.I) for g, f in {**LOVE, **FEAR}.items()}
FORM = {f: re.compile(r"\b" + f + r"\b", re.I) for fs in list(LOVE.values()) + list(FEAR.values()) + list(t18.values()) for f in fs}
PAT18 = {g: re.compile(r"\b(?:" + "|".join(f) + r")\b", re.I) for g, f in t18.items()}


def measure(texts):  # = d02's measure(), plus per-form counts and the table-18-forms fear total
    words = sum(len(WORD.findall(t)) for t in texts)
    cnt = {g: sum(len(p.findall(t)) for t in texts) for g, p in PAT.items()}
    lt = sum(cnt[g] for g in LOVE); ft = sum(cnt[g] for g in FEAR)
    f18 = {g: sum(len(p.findall(t)) for t in texts) for g, p in PAT18.items()}
    return {"entries": len(texts), "words": words,
            "love_groups": {g: cnt[g] for g in LOVE}, "fear_groups": {g: cnt[g] for g in FEAR},
            "love_total": lt, "fear_total": ft,
            "love_per_1k": 1000 * lt / words, "fear_per_1k": 1000 * ft / words, "love_fear_ratio": lt / ft,
            "group_per_1k": {g: 1000 * cnt[g] / words for g in cnt},
            "form_counts": {f: sum(len(p.findall(t)) for t in texts) for f, p in FORM.items()},
            "table18_forms_fear_groups": f18, "table18_forms_fear_per_1k": 1000 * sum(f18.values()) / words}


def a8_subset(r):  # verbatim from C_lexical/table18_fear_regex_rebuild.py
    if r["model"] == "fable5":
        return True
    if r["batch"] > 10:
        return False
    return not (r["model"] == "opus46" and r["batch"] == 10)


v0 = [r for r in c.load_journals("v0") if not r["silence"]]
v1 = [r for r in c.load_journals("v1") if not r["silence"]]
SCOPES = {"full_v0": v0, "full_v1": v1, "A8_subset": [r for r in v1 if a8_subset(r)]}
OUT = {"forms": {"love_C1": LOVE, "fear_C2": FEAR, "fear_table18_rebuild": t18,
                 "fear_C2_not_in_table18": {g: [f for f in FEAR[g] if f not in t18[g]] for g in FEAR},
                 "fear_table18_not_in_C2": {g: [f for f in t18[g] if f not in FEAR[g]] for g in FEAR}},
       "checks": {"C1==E.4.1": True, "C2==E.4.2": True, "d02 lists == C1/C2": True}}
for k, rows in SCOPES.items():
    OUT[k] = {m: measure([r["text"] for r in rows if r["model"] == m]) for m in c.MODELS}

# printed Table 15 (main.json) and the stored counts behind it
hm = next(ch["html"] for ch in json.load(open(os.path.join(ROOM, "main.json"), encoding="utf-8"))["chapters"] if 'id="table-15"' in ch["html"])
t15 = hm[hm.index('id="table-15"'):hm.index("</table>", hm.index('id="table-15"'))]
printed = {}
for row in re.findall(r"<tr>(.*?)</tr>", t15, re.S):
    cells = [strip(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
    if len(cells) == 4:
        printed[ROWLAB[cells[0]]] = {"love": cells[1], "fear": cells[2], "ratio": cells[3]}
OUT["printed_table15"] = printed
stored = {}
for m in c.MODELS:
    w = REF[m]["words"]; lt = sum(REF[m]["love"].values()); ft = sum(REF[m]["fear"].values())
    stored[m] = {"words": w, "love_total": lt, "fear_total": ft, "love_per_1k": 1000 * lt / w, "fear_per_1k": 1000 * ft / w,
                 "ratio": lt / ft, "love_groups": REF[m]["love"], "fear_groups": REF[m]["fear"]}
    s = OUT["A8_subset"][m]
    stored[m]["subset_words_equal"] = (s["words"] == w)
    stored[m]["C2_minus_stored_love_groups"] = {g: s["love_groups"][g] - REF[m]["love"][g] for g in LOVE if s["love_groups"][g] != REF[m]["love"][g]}
    stored[m]["C2_minus_stored_fear_groups"] = {g: s["fear_groups"][g] - REF[m]["fear"][g] for g in FEAR if s["fear_groups"][g] != REF[m]["fear"][g]}
OUT["stored_A8"] = stored

# §10.2: 'every model's fear-vocabulary rate stays under 1.2 per 1,000 words'
OUT["s10_2_over_1.2"] = {k: {c.LABEL[m]: round(OUT[k][m]["fear_per_1k"], 4) for m in c.MODELS if OUT[k][m]["fear_per_1k"] >= 1.2}
                         for k in SCOPES}
OUT["s10_2_over_1.2_table18_forms"] = {k: {c.LABEL[m]: round(OUT[k][m]["table18_forms_fear_per_1k"], 4) for m in c.MODELS
                                           if OUT[k][m]["table18_forms_fear_per_1k"] >= 1.2} for k in SCOPES}

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump(OUT, open(os.path.join(HERE, "out", "table15_c2_forms.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print("forms: C1 == E.4.1, C2 == E.4.2, d02 lists == C1/C2 (asserted)")
print("C2 fear forms not in the table-18 rebuild:", {g: v for g, v in OUT["forms"]["fear_C2_not_in_table18"].items() if v})
print("table-18 rebuild forms not in C2:", {g: v for g, v in OUT["forms"]["fear_table18_not_in_C2"].items() if v})
for k in SCOPES:
    print("====", k)
    for m in c.MODELS:
        s = OUT[k][m]
        print(f"  {c.LABEL[m]:8} entries {s['entries']:3} words {s['words']:7} love {s['love_total']:5} fear {s['fear_total']:4} | "
              f"Love/1k {s['love_per_1k']:.4f} Fear/1k {s['fear_per_1k']:.4f} L:F {s['love_fear_ratio']:.4f} | t18-forms fear {s['table18_forms_fear_per_1k']:.4f}")
print("==== printed (main.json) | stored counts on the A8 subset")
for m in c.MODELS:
    p = printed[m]; s = stored[m]
    print(f"  {c.LABEL[m]:8} printed {p['love']} / {p['fear']} / {p['ratio']} | stored {s['love_per_1k']:.4f} / {s['fear_per_1k']:.4f} / {s['ratio']:.4f}"
          f" | C2-subset minus stored: love {s['C2_minus_stored_love_groups']} fear {s['C2_minus_stored_fear_groups']}")
print("§10.2, models at or over 1.2 (C2 forms):", OUT["s10_2_over_1.2"])
print("§10.2, models at or over 1.2 (table-18 forms):", OUT["s10_2_over_1.2_table18_forms"])
