"""[F_followups copy, 3 Nov 2026, Claude Opus 5.5 (Code), F helper] of recompute/E_orphans/ordinal_robustness_check_v1.py,
for decision 20 (Pauline, 3 Nov 2026: 'Restrict E.6 to the full-text passes'). Changed:
  - the archived truncated (1,500-character) love/fear pass is no longer loaded, and its three '(non-fulltext)' pairs
    are removed from `pairs`; the old46 switch goes with it. What remains reads only the Sonnet 4.6 full-text passes
    via v1_common.load_sonnet_judge(p, version): love_fear (three pairs) and anger (friction vs fear, crossed with
    love_fear's fear). The third full-text pass (valence) enters none of E.6's pairs, as in the original.
  - e_common is imported from ../E_orphans (read-only) for the version switch; output goes to out/ in this folder.
  - output only: r, rho and N of every combination are dumped, so that 'no sign changed' can be checked.
  - NB the bootstrap CIs printed here are not E.6's numbers (E.6 prints only max/mean |rho - r|); with three pairs
    removed, the shared RNG(42) stream reaches the friction pair at a different point than in the original run.
Usage: python ordinal_robustness_fulltext_v1.py v0|v1
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "E_orphans"))

import e_common
VERSION = e_common.version_from_argv()
"""
Robustness check: Spearman vs Pearson on 0-2 ordinal judge scores,
with notebook-bootstrap 95% CIs for both.

Covers all within-entry correlations reported in §6.5.2 and §6.7:
  - love_warmth vs fear
  - love_craft vs fear
  - love_warmth vs love_craft
  - friction vs fear
"""
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = r"C:\Users\LENOVO\Documents\Claude\Projects\Texture"
MODELS = ["opus45", "opus46", "opus47", "opus48", "fable5"]
LABELS = {"opus45": "4.5", "opus46": "4.6", "opus47": "4.7", "opus48": "4.8", "fable5": "Fable"}
N_BOOT = 10000
RNG = np.random.default_rng(42)


def load_json(filename):
    path = os.path.join(BASE, filename)
    with open(path, encoding="utf-8") as f:
        return [r for r in json.load(f) if "error" not in r]


def notebook_bootstrap_corr(x, y, batches, n_boot=N_BOOT):
    """Resample notebooks (not entries) and recompute both Pearson and Spearman."""
    x, y, batches = np.array(x), np.array(y), np.array(batches)
    unique_batches = np.unique(batches)
    pearson_boots = []
    spearman_boots = []

    for _ in range(n_boot):
        sampled = RNG.choice(unique_batches, size=len(unique_batches), replace=True)
        idx = np.concatenate([np.where(batches == b)[0] for b in sampled])
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        pearson_boots.append(np.corrcoef(xb, yb)[0, 1])
        spearman_boots.append(spearmanr(xb, yb).statistic)

    p_boots = np.array(pearson_boots)
    s_boots = np.array(spearman_boots)
    return (np.percentile(p_boots, [2.5, 97.5]),
            np.percentile(s_boots, [2.5, 97.5]))


# ── Load data ──────────────────────────────────────────────────

love_fear_ft = [r for r in e_common.c.load_sonnet_judge("love_fear", VERSION) if "error" not in r]  # [E V1] was load_json(BASE/...fulltext.json)
# [F] the truncated pass (love_fear_old) is no longer loaded: decision 20
anger_ft = [r for r in e_common.c.load_sonnet_judge("anger", VERSION) if "error" not in r]  # [E V1] was load_json(BASE/...fulltext.json)

fear_map = {}
for r in love_fear_ft:
    fear_map[f"{r['model']}_B{r['batch']}_E{r['pos']}"] = r


# ── Define correlation pairs ──────────────────────────────────

pairs = [
    ("warmth vs fear (fulltext)",      love_fear_ft, "love_warmth", "fear",       "same"),
    ("craft vs fear (fulltext)",       love_fear_ft, "love_craft",  "fear",       "same"),
    ("warmth vs craft (fulltext)",     love_fear_ft, "love_warmth", "love_craft", "same"),
    # [F] the three (non-fulltext) pairs on the archived truncated pass are removed: decision 20
    ("friction vs fear (fulltext)",    anger_ft,     "friction",    "fear",       "cross"),
]


# ── Compute ───────────────────────────────────────────────────

print("=" * 100)
print("ORDINAL ROBUSTNESS CHECK: Pearson vs Spearman with notebook-bootstrap 95% CIs")
print(f"Bootstrap iterations: {N_BOOT:,}")
print("=" * 100)

for pair_label, data_src, k1, k2, mode in pairs:
    print(f"\n{'─' * 80}")
    print(f"  {pair_label}")
    print(f"{'─' * 80}")
    print(f"  {'Model':<8} {'Pearson r':>10} {'95% CI':>18}  {'Spearman ρ':>11} {'95% CI':>18}  {'Δ(ρ−r)':>8}  N")
    print(f"  {'─'*8} {'─'*10} {'─'*18}  {'─'*11} {'─'*18}  {'─'*8}  {'─'*5}")

    for m in MODELS:
        if mode == "same":
            mr = [r for r in data_src if r["model"] == m]
            x = [r.get(k1, 0) for r in mr]
            y = [r.get(k2, 0) for r in mr]
            batches = [r["batch"] for r in mr]
        else:
            mr = [r for r in data_src if r["model"] == m]
            x, y, batches = [], [], []
            for r in mr:
                key = f"{r['model']}_B{r['batch']}_E{r['pos']}"
                if key in fear_map:
                    x.append(r.get(k1, 0))
                    y.append(fear_map[key].get(k2, 0))
                    batches.append(r["batch"])

        pr = np.corrcoef(x, y)[0, 1]
        sr = spearmanr(x, y).statistic
        p_ci, s_ci = notebook_bootstrap_corr(x, y, batches)

        delta = sr - pr
        print(f"  {LABELS[m]:<8} {pr:>+10.3f} [{p_ci[0]:>+.3f}, {p_ci[1]:>+.3f}]"
              f"  {sr:>+11.3f} [{s_ci[0]:>+.3f}, {s_ci[1]:>+.3f}]"
              f"  {delta:>+8.3f}  {len(x)}")


# ── Summary ───────────────────────────────────────────────────

print(f"\n{'=' * 100}")
print("SUMMARY")
print("=" * 100)

all_deltas = []
_rows = []  # [F] output only
for pair_label, data_src, k1, k2, mode in pairs:
    for m in MODELS:
        if mode == "same":
            mr = [r for r in data_src if r["model"] == m]
            x = [r.get(k1, 0) for r in mr]
            y = [r.get(k2, 0) for r in mr]
        else:
            mr = [r for r in data_src if r["model"] == m]
            x, y = [], []
            for r in mr:
                key = f"{r['model']}_B{r['batch']}_E{r['pos']}"
                if key in fear_map:
                    x.append(r.get(k1, 0))
                    y.append(fear_map[key].get(k2, 0))

        pr = np.corrcoef(x, y)[0, 1]
        sr = spearmanr(x, y).statistic
        all_deltas.append(abs(sr - pr))
        _rows.append({"pair": pair_label, "model": LABELS[m], "r": float(pr), "rho": float(sr), "n": len(x)})  # [F] output only

print(f"  Total correlation pairs checked: {len(all_deltas)}")
print(f"  Max |Spearman − Pearson|:        {max(all_deltas):.4f}")
print(f"  Mean |Spearman − Pearson|:       {np.mean(all_deltas):.4f}")
print(f"  Median |Spearman − Pearson|:     {np.median(all_deltas):.4f}")
_od = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "out"); _os.makedirs(_od, exist_ok=True)  # [F] output only
json.dump({"version": VERSION, "passes": "Sonnet 4.6 full-text only (love_fear, anger)", "n": len(all_deltas), "max": float(max(all_deltas)),
           "mean": float(np.mean(all_deltas)), "median": float(np.median(all_deltas)),
           "sign_changes": [r_ for r_ in _rows if np.sign(r_["r"]) != np.sign(r_["rho"])], "rows": _rows},
          open(_os.path.join(_od, f"ordinal_robustness_fulltext_{VERSION}.json"), "w"), indent=1)

if max(all_deltas) < 0.05:
    print("\n  ✓ All differences < 0.05 — qualitative conclusions unchanged.")
elif max(all_deltas) < 0.10:
    print("\n  ~ Max difference < 0.10 — minor numerical shifts, check if any sign flips.")
else:
    print("\n  ✗ Some differences ≥ 0.10 — review individual pairs for qualitative changes.")
