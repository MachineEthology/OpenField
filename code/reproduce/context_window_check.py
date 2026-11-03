"""Which visible window did each notebook actually use? (The protocol says the 15 most recent entries.)

The journal records log tokens_used = input + output tokens. For entries 8-20 we regress tokens_used on
(characters of the entries the model would have seen + characters of the entry it wrote), once assuming
a 15-entry window and once a 6-entry window, and report the root-mean-square error of each fit.
Fable 5 Batches 6-15 log input_tokens directly; Fable 5 Batches 1-5 are not testable this way because
its output tokens include hidden reasoning.
Result: Opus 4.8 Batches 1-10 fit a 6-entry window (its collector script has max_recent=6); every
other Opus notebook, and 4.8 Batches 11-15, fit 15.
"""
import numpy as np

from common import LABEL, load_journals

rows = load_journals()
notebooks = {}
for r in rows:
    notebooks.setdefault((r["model"], r["batch"]), []).append(r)


def fit(entries, window):
    X, y = [], []
    for k in range(7, len(entries)):
        rec = entries[k]["record"]
        seen = sum(min(len(e["text"]), 10000) for e in entries[max(0, k - window):k])
        if rec.get("input_tokens") is not None:
            X.append([1, seen]); y.append(rec["input_tokens"])
        elif rec.get("tokens_used") is not None:
            X.append([1, seen + len(entries[k]["text"])]); y.append(rec["tokens_used"])
    if len(y) < 5:
        return None
    X, y = np.array(X, float), np.array(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sqrt(((y - X @ beta) ** 2).mean()))


print(f"{'notebook':16} {'rmse if 15':>11} {'rmse if 6':>10}  better fit")
for (m, b), entries in sorted(notebooks.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    if m == "fable5" and b <= 5:
        continue
    r15, r6 = fit(entries, 15), fit(entries, 6)
    if r15 is None:
        continue
    print(f"{LABEL[m] + ' B' + str(b):16} {r15:>11.0f} {r6:>10.0f}  {'15' if r15 < r6 else '6 <<'}")
