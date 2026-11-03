"""Markdown bold (**...**) in the five models' probes and journals, and how the paper's asterisk
measures (Tables 6-7) count it.

Rules recovered by exact reproduction of Tables 6-7 (docs/CORPUS.md):
  Table 6 share  = responses containing any '*'
  Table 6 rate   = number of '*' characters per 1,000 words (words = str.split())
  Table 7 spans  = matches of the regex \\*[^*]+\\*
Under those rules a bold phrase **x** adds 4 asterisks to the rate (twice an italic *x*), its inner pair
counts as one span, and the leftover '*' can open a false span that runs to the next asterisk.
"""
import re
from collections import defaultdict

from common import LABEL, MODELS, load_journals, load_probes

BOLD = re.compile(r"\*\*[^*\n]+?\*\*")
BOLD_LINE = re.compile(r"(?m)^\s*\*\*[^*\n]+\*\*\s*$")
SPAN = re.compile(r"\*[^*]+\*")


def describe(texts):
    with_bold = sum(1 for t in texts if BOLD.search(t))
    spans = sum(len(BOLD.findall(t)) for t in texts)
    lines = sum(len(BOLD_LINE.findall(t)) for t in texts)
    headers = sum(1 for t in texts if re.search(r"(?m)^#{1,6} ", t))
    first_line_bold = sum(1 for t in texts if BOLD_LINE.match(t.strip().split("\n", 1)[0] if t.strip() else ""))
    all_spans = sum(len(SPAN.findall(t)) for t in texts)
    no_bold_spans = sum(len(SPAN.findall(BOLD.sub(lambda m: m.group(0)[2:-2], t))) for t in texts)
    return with_bold, spans, lines, first_line_bold, headers, all_spans, no_bold_spans


groups = defaultdict(list)
for p in load_probes():
    groups[("probe", p["model"], p["condition"])].append(p["text"])
for j in load_journals():
    if not j["silence"]:
        groups[("journal", j["model"], "")].append(j["text"])

print(f"{'set':34} {'n':>4} {'with **':>8} {'** spans':>8} {'bold-only lines':>15} {'1st line bold':>13} {'# headers':>9} {'Table-7 spans':>13} {'spans w/o bold':>14}")
for kind in ("probe", "journal"):
    for m in MODELS:
        for cond in (("negation", "affirmative") if kind == "probe" else ("",)):
            t = groups[(kind, m, cond)]
            wb, sp, ln, fl, hd, a, nb = describe(t)
            name = f"{kind} {LABEL[m]} {cond}".strip()
            print(f"{name:34} {len(t):>4} {wb:>8} {sp:>8} {ln:>15} {fl:>13} {hd:>9} {a:>13} {nb:>14}")
