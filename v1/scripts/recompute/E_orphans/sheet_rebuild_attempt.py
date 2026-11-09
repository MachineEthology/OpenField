"""Rebuild attempt for D-O9 (Section 7.1, amplification ratio 0.72) and D-O10's Section 7.2 '4% by a gap of five'
(helper E, 3 Nov 2026). Read-only.

Both printed numbers trace to Texture/04_analysis/Journal_Deep_Dive_v2.xlsx, sheet 'Context Sensitivity'
(written by openpyxl on 5 June 2026, values only, no formulas; the generating script was not found):
  B. COUPLING DECAY WITH DISTANCE   4.8: gap 1 0.809, gap 2 0.785, gap 5 0.775, decay 4.3%
  D. VARIANCE AMPLIFICATION          4.6: early spread 0.044, late spread 0.031, ratio 0.72
The sheet's own 'Entry-Level Data' holds per-entry counts for 19 topic families (4.6 notebooks 1-7, the other
models 1-10). This script tries the plain readings of 'coupling' (mean cosine of entry vectors k apart within a
notebook) and 'spread' (mean cosine distance of per-notebook early / late centroids from their mean) on those
counts. A rebuild counts only if it hits the sheet at its precision (brief): none of the variants below does,
so D-O9 and the 4% stay NO_PRODUCER. No variant was tuned beyond this list.
"""
import sys
import warnings

import numpy as np
import openpyxl

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
SHEET = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture\04_analysis\Journal_Deep_Dive_v2.xlsx"
TARGET_COUPLING = {"4.5": (0.809, 0.821, 0.812), "4.6": (0.867, 0.870, 0.863),
                   "4.7": (0.782, 0.782, 0.767), "4.8": (0.809, 0.785, 0.775)}
TARGET_SPREAD = {"4.5": (0.069, 0.079), "4.6": (0.044, 0.031), "4.7": (0.082, 0.104), "4.8": (0.083, 0.100)}

rows = list(openpyxl.load_workbook(SHEET, data_only=True, read_only=True)["Entry-Level Data"].iter_rows(values_only=True))
hdr, data = rows[0], [r for r in rows[1:] if r[0]]
COLSETS = {"19 topics": list(range(17, 36)), "identity + 19 topics": list(range(13, 36)),
           "all count columns from Back-refs": list(range(4, 36))}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def feats(cols, mode):
    X = np.array([[r[i] or 0 for i in cols] for r in data], float)
    W = np.array([max(r[3], 1) for r in data], float)
    return {"count": X, "per 100 words": 100 * X / W[:, None], "log1p": np.log1p(X), "presence": (X > 0) * 1.0}[mode]


for cname, cols in COLSETS.items():
    for mode in ("count", "per 100 words", "log1p", "presence"):
        V = feats(cols, mode)
        line_c, line_s = [], []
        for m in ("4.5", "4.6", "4.7", "4.8"):
            nb = {}
            for i, r in enumerate(data):
                if r[0] == m:
                    nb.setdefault(r[1], []).append(i)
            for b in nb:
                nb[b].sort(key=lambda i: data[i][2])
            coup = [np.nanmean([cos(V[es[k]], V[es[k + g]]) for es in nb.values() for k in range(len(es) - g)])
                    for g in (1, 2, 5)]
            sp = []
            for lo, hi in ((1, 5), (16, 20)):
                cents = np.array([V[[i for i in es if lo <= data[i][2] <= hi]].mean(0) for es in nb.values()])
                g = cents.mean(0)
                sp.append(np.mean([1 - cos(c, g) for c in cents]))
            line_c.append(f"{m} {coup[0]:.3f}/{coup[1]:.3f}/{coup[2]:.3f} ({100 * (coup[0] - coup[2]) / coup[0]:+.1f}%)")
            line_s.append(f"{m} {sp[0]:.3f}->{sp[1]:.3f} (x{sp[1] / sp[0]:.2f})")
        print(f"[{cname} | {mode}]")
        print("   coupling gap1/2/5 (decay):", "  ".join(line_c))
        print("   spread 1-5 -> 16-20 (ratio):", "  ".join(line_s))
print("\nsheet: coupling 4.5 0.809/0.821/0.812 (-0.4%)  4.6 0.867/0.870/0.863 (+0.4%)  4.7 0.782/0.782/0.767 (1.9%)  4.8 0.809/0.785/0.775 (4.3%)")
print("sheet: spread   4.5 0.069->0.079 (1.16)  4.6 0.044->0.031 (0.72)  4.7 0.082->0.104 (1.28)  4.8 0.083->0.100 (1.20)")
