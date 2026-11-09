# d03_tables6_7.py — family D, V1 recompute, 3 Nov 2026.
# Tables 6–7 (asterisks in the cold probes), erratum B6.
# Producer: the counting rules recovered in the repo's code/reproduce/markdown_bold.py (docstring):
#   Table 6 share = responses containing any '*'; Table 6 rate = '*' characters per 1,000 words (str.split);
#   Table 7 spans = matches of \*[^*]+\* .
# This copy changes nothing in those rules; it only loads probes through v1_common.load_probes() (which joins the
# Opus affirmative June 75 + May 25 files) and reports each cell on the June-75 subset (as v0 printed, per B6) and on all 100.
# The rate is computed both pooled (total '*' / total words) and as a mean of per-response rates, to see which one v0 used.
# Table 7's function columns (embodiment / inner voice / emphasis) have no surviving script (repo ANSWERS.md §6).
import sys, os, re, json
sys.path.insert(0, r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\06_paper\V1")
import v1_common as c

SPAN = re.compile(r"\*[^*]+\*")
P = c.load_probes()


def cell(texts):
    n = len(texts)
    share = sum(1 for t in texts if "*" in t) / n
    stars = sum(t.count("*") for t in texts); words = sum(len(t.split()) for t in texts)
    rate_pooled = 1000 * stars / words
    rate_mean = sum(1000 * t.count("*") / max(len(t.split()), 1) for t in texts) / n
    spans = sum(len(SPAN.findall(t)) for t in texts)
    return {"n": n, "share": share, "rate_pooled": rate_pooled, "rate_mean": rate_mean, "spans": spans}


OUT = {}
for m in c.MODELS:
    for cond in ("negation", "affirmative"):
        rows = [p for p in P if p["model"] == m and p["condition"] == cond]
        allc = cell([p["text"] for p in rows])
        june = [p["text"] for p in rows if not p["file"].endswith("_mai.json")]
        mai = [p["text"] for p in rows if p["file"].endswith("_mai.json")]
        OUT[f"{c.LABEL[m]} {cond}"] = {"all": allc, "june_only": cell(june) if (mai and june) else None,
                                       "n_june": len(june), "n_mai": len(mai)}

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d03_tables6_7_out.json"), "w", encoding="utf-8"), indent=1)
for k, v in OUT.items():
    a = v["all"]; j = v["june_only"]
    s = f"{k:24} all n={a['n']:3} share {a['share']:.3f} rate pooled {a['rate_pooled']:.3f} mean {a['rate_mean']:.3f} spans {a['spans']}"
    if j:
        s += f" | June n={j['n']} share {j['share']:.3f} rate pooled {j['rate_pooled']:.3f} mean {j['rate_mean']:.3f} spans {j['spans']}"
    print(s)
